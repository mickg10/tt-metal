# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Top-level model runner for GLM-4.7-REAP-218B on TT hardware.

Provides Glm4MoeTT: a frozen dataclass with create() factory that loads all 92 layers
of weights, creates decoder layers, embedding, final norm, and LM head, and exposes
decode() and prefill() entry points.

Key architecture:
- 92 layers (num_hidden_layers=92)
- Layers 0-2: dense MLP (first_k_dense_replace=3)
- Layers 3-91: MoE (96 routed experts EP=32 + 1 shared expert TP=8)
- Standard GQA attention (96Q/8KV heads, head_dim=128, NOT MLA)
- Galaxy Wormhole: Mesh(8,4), TP=8 (axis 0), EP=32, DP=4 (axis 1)
"""

from __future__ import annotations

import gc
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import torch

import ttnn
from loguru import logger

from models.common.rmsnorm import RMSNorm
from models.demos.glm4_moe.tt.config import Glm4MoeHParams
from models.demos.glm4_moe.tt.decoder_layer_tt import Glm4MoeDecoderLayer, _sharded_rms_norm
from models.demos.glm4_moe.tt.layer_weights import (
    DecoderLayerTTWeights,
    _env_dense_dtype,
    _tp_axis_and_size,
    _tp_mesh_mapper,
    _linear_weight_tt,
    _upcast_fp8,
    _dequant_weight,
    convert_decoder_layer_weights,
)
from models.demos.glm4_moe.tt.ccl import CCL
from models.demos.glm4_moe.tt.moe_tt import create_moe_runtime, Glm4MoeMoERuntime
from models.demos.glm4_moe.tt.trace_retainer import (
    TraceRetainer,
    set_trace_retainer,
    clear_trace_retainer,
    _dealloc,
)
from models.demos.glm4_moe.tt.tt_embedding import (
    convert_embedding_weight_to_tt,
    run_tt_embedding,
)
from models.demos.glm4_moe.tt.weights import (
    load_glm_lazy_state_dict,
)


# ---------------------------------------------------------------------------
# RoPE utilities (adapted from glm4_moe_lite/layer0_tt.py for GQA head_dim)
# ---------------------------------------------------------------------------

def _rot_transformation_mat_torch() -> torch.Tensor:
    """Transformation matrix for ttnn.experimental.rotary_embedding_llama."""
    dhead = 32
    rot = torch.zeros(1, 1, dhead, dhead, dtype=torch.float32)
    rot[..., torch.arange(0, dhead, 2), torch.arange(1, dhead, 2)] = 1.0
    rot[..., torch.arange(1, dhead, 2), torch.arange(0, dhead, 2)] = -1.0
    return rot


def _rope_cos_sin_torch(*, seq_len: int, dim: int, base: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cos/sin matrices in NeoX half-rotation form: [1,1,seq_len,dim].

    Last dim layout: [t0, t1, ..., t_{d/2-1}, t0, t1, ..., t_{d/2-1}]
    to match NeoX-style rotate_half(x) = cat(-x[..., d//2:], x[..., :d//2]).
    GLM-4.7 uses NeoX-style RoPE (confirmed from HuggingFace transformers glm4_moe).
    """
    if dim % 2 != 0:
        raise ValueError(f"rope dim must be even, got dim={dim}")
    half = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) * (2.0 / dim)))
    positions = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    cos = torch.cat((cos, cos), dim=-1)
    sin = torch.cat((sin, sin), dim=-1)
    return cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)


def _make_rope_tensors(
    *,
    device,
    seq_len: int,
    rope_dim: int,
    rope_theta: float,
) -> dict[str, object]:
    """Create RoPE cos/sin/trans tensors for GQA with partial RoPE.

    For GLM-4.7-REAP: partial_rotary_factor=0.5, so rope_dim=64 (half of head_dim=128).
    RoPE tables are created at rope_dim (the portion that gets rotated).
    """
    cos_t, sin_t = _rope_cos_sin_torch(seq_len=seq_len, dim=rope_dim, base=rope_theta)
    trans_t = _rot_transformation_mat_torch().to(dtype=torch.bfloat16)

    is_mesh_device = device.__class__.__name__ == "MeshDevice"
    cos_host = cos_t.to(dtype=torch.bfloat16).cpu()
    sin_host = sin_t.to(dtype=torch.bfloat16).cpu()

    cos = ttnn.from_torch(
        cos_t.to(dtype=torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
    )
    sin = ttnn.from_torch(
        sin_t.to(dtype=torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
    )
    trans = ttnn.from_torch(
        trans_t,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
    )
    return {
        "cos_matrix": cos,
        "sin_matrix": sin,
        "trans_matrix": trans,
        "cos_matrix_host": cos_host,
        "sin_matrix_host": sin_host,
    }


# ---------------------------------------------------------------------------
# Decode RoPE input preparation (per-step cos/sin for batch of positions)
# ---------------------------------------------------------------------------

def _prepare_decode_rope_and_positions_tt(
    *, device, rope: dict, positions: torch.Tensor, dp_shard_axis: int | None = None,
) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
    """Prepare per-step RoPE cos/sin and position tensors for decode.

    positions: [B] int32 tensor of current positions.
    dp_shard_axis: If set, shard cos/sin along dim=1 (batch) across this mesh axis
        instead of replicating.  Used on TG (Galaxy) where DP groups each process
        a subset of the batch.  The mesh axis must evenly divide the batch size.
    Returns: (tt_positions, cos_batch, sin_batch) on device.
    """
    is_mesh_device = device.__class__.__name__ == "MeshDevice"
    batch = int(positions.shape[0])

    # Upload positions to device.
    # When dp_shard_axis is set, shard positions across DP groups (each group gets
    # only its subset of positions).
    if is_mesh_device and dp_shard_axis is not None:
        mesh_shape = list(device.shape)
        pos_dims = [None, None]
        pos_dims[dp_shard_axis] = 0  # shard tensor dim=0 across dp_shard_axis
        pos_mapper = ttnn.ShardTensor2dMesh(device, dims=tuple(pos_dims), mesh_shape=mesh_shape)
    elif is_mesh_device:
        pos_mapper = ttnn.ReplicateTensorToMesh(device)
    else:
        pos_mapper = None
    tt_positions = ttnn.from_torch(
        positions.view(-1).contiguous().to(torch.int32),
        device=device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=pos_mapper,
    )

    # Gather per-position cos/sin from host-side RoPE tables.
    cos_host = rope["cos_matrix_host"]  # [1, 1, max_seq_len, rope_dim]
    sin_host = rope["sin_matrix_host"]
    rope_dim = int(cos_host.shape[3])

    positions_cpu = positions.to(torch.long).clamp(min=0, max=int(cos_host.shape[2]) - 1)
    cos_batch_t = cos_host[0, 0, positions_cpu, :].reshape(1, batch, 1, rope_dim).to(torch.bfloat16)
    sin_batch_t = sin_host[0, 0, positions_cpu, :].reshape(1, batch, 1, rope_dim).to(torch.bfloat16)

    # Determine mesh mapper for cos/sin.
    if is_mesh_device and dp_shard_axis is not None:
        # Shard batch dim across DP groups so each group gets only its positions' cos/sin.
        mesh_shape = list(device.shape)
        dims = [None, None]
        dims[dp_shard_axis] = 1  # shard tensor dim=1 (batch) across dp_shard_axis
        rope_mapper = ttnn.ShardTensor2dMesh(device, dims=tuple(dims), mesh_shape=mesh_shape)
    elif is_mesh_device:
        rope_mapper = ttnn.ReplicateTensorToMesh(device)
    else:
        rope_mapper = None

    cos_batch = ttnn.from_torch(
        cos_batch_t,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rope_mapper,
    )
    sin_batch = ttnn.from_torch(
        sin_batch_t,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rope_mapper,
    )

    return tt_positions, cos_batch, sin_batch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_mesh_device(device: Any) -> bool:
    return device.__class__.__name__ == "MeshDevice"


def _tt_to_torch_for_vllm_output(*, tensor: ttnn.Tensor, device: Any) -> torch.Tensor:
    """Convert a TT tensor to torch for vLLM outputs.

    On a mesh, reads back device 0 (replicated model bring-up mode).
    """
    if not _is_mesh_device(device):
        return ttnn.to_torch(tensor.cpu())
    device_tensors = ttnn.get_device_tensors(tensor)
    if not device_tensors:
        raise RuntimeError("ttnn.get_device_tensors returned an empty list for a mesh tensor")
    return ttnn.to_torch(device_tensors[0].cpu())


def _load_hparams_from_snapshot(snapshot_dir: Path) -> Glm4MoeHParams:
    cfg = json.loads((Path(snapshot_dir) / "config.json").read_text())
    hparams = Glm4MoeHParams.from_hf_config(SimpleNamespace(**cfg))
    hparams.validate()
    return hparams


def _torch_dtype_to_ttnn(dtype: torch.dtype) -> ttnn.DataType:
    """Map vLLM torch dtype to TT KV cache dtype, with env override."""
    override = os.environ.get("GLM4_MOE_KV_CACHE_TT_DTYPE", "").strip().lower()
    if override:
        if override in {"bf8", "bfloat8_b"}:
            return ttnn.bfloat8_b
        if override in {"bf16", "bfloat16"}:
            return ttnn.bfloat16
        if override in {"f32", "fp32", "float32"}:
            return ttnn.float32
        raise ValueError(f"Invalid GLM4_MOE_KV_CACHE_TT_DTYPE={override!r}")
    return ttnn.bfloat8_b


@dataclass
class _DecodeTraceState:
    """Per-bucket state for batch-bucketed decode traces."""
    trace_id: Any | None = None
    batch: int = 0
    page_table_width: int = 0
    tokens_tt: ttnn.Tensor | None = None
    positions_tt: ttnn.Tensor | None = None
    cos_batch_tt: ttnn.Tensor | None = None
    sin_batch_tt: ttnn.Tensor | None = None
    trans_matrix_tt: ttnn.Tensor | None = None
    page_table_tt: ttnn.Tensor | None = None
    logits_tt: ttnn.Tensor | None = None
    top1_values_tt: ttnn.Tensor | None = None
    top1_indices_tt: ttnn.Tensor | None = None
    embed_tt: ttnn.Tensor | None = None
    tokens_tt: ttnn.Tensor | None = None  # Device token IDs for in-trace embedding
    retained_intermediates: list = field(default_factory=list)  # Trace address guard

    # MTP trace state (for traced combined decode+MTP)
    mtp_hidden_tt: Any = None       # [1,1,B,hidden] clone from main trace
    mtp_positions_tt: Any = None    # [B] int32
    mtp_cos_batch_tt: Any = None    # [1,B,1,rope_dim] bf16
    mtp_sin_batch_tt: Any = None    # [1,B,1,rope_dim] bf16
    mtp_page_table_tt: Any = None   # [B,W] int32
    mtp_embed_tt: Any = None        # [1,1,B,hidden]
    mtp_trace_id: int | None = None
    mtp_logits_tt: Any = None       # MTP logits


# ---------------------------------------------------------------------------
# Model Runner
# ---------------------------------------------------------------------------

@dataclass
class Glm4MoeTT:
    """TT runner for GLM-4.7-REAP-218B (92 layers, GQA attention, MoE).

    Loaded via `Glm4MoeTT.create(...)` factory.
    """

    device: Any
    snapshot_dir: Path
    cache_dir: Path
    max_seq_len: int

    hparams: Glm4MoeHParams
    state: Any  # LazyStateDict
    embed_w: Optional[ttnn.Tensor]
    embed_w_cpu: torch.Tensor
    rope: dict[str, Any]
    final_norm: RMSNorm
    lm_head_w: ttnn.Tensor
    lm_head_sharded_vocab: bool
    lm_head_tp_axis: int | None
    lm_head_tp_size: int
    lm_head_vocab_per_shard: int
    layer_weights: dict[int, DecoderLayerTTWeights]
    decoder_layers: dict[int, Glm4MoeDecoderLayer]
    num_layers_to_run: int
    enable_moe: bool
    moe_runtime: Glm4MoeMoERuntime | None
    configuration: dict[str, Any]
    tt_ccl: Any | None

    # MTP (Multi-Token Prediction) — loaded when GLM4_MOE_MTP=1
    mtp_enabled: bool = False
    mtp_enorm: Any | None = None
    mtp_hnorm: Any | None = None
    mtp_eh_proj_e_w: Any | None = None
    mtp_eh_proj_h_w: Any | None = None
    mtp_shared_head_norm: Any | None = None
    mtp_shared_head_w: Any | None = None
    mtp_decoder_layer: Any | None = None
    mtp_max_batch: int = 16

    _decode_trace_states: dict[int, _DecodeTraceState] = field(init=False, default_factory=dict)
    _decode_traces_stale: bool = field(init=False, default=False)
    _post_prefill_eager_remaining: int = field(init=False, default=0)
    _last_draft_token_ids: torch.Tensor | None = field(init=False, default=None)
    _prev_main_ids: torch.Tensor | None = field(init=False, default=None)

    @classmethod
    def create(
        cls,
        *,
        device: Any,
        snapshot_dir: Path,
        cache_dir: Path,
        max_seq_len: int,
        max_batch_size: int = 32,
        hparams: Optional[Glm4MoeHParams] = None,
    ) -> "Glm4MoeTT":
        snapshot_dir = Path(snapshot_dir)
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        hparams = _load_hparams_from_snapshot(snapshot_dir) if hparams is None else hparams

        # Include MTP layer (layer 92) in state dict when MTP is enabled
        _mtp_flag = os.environ.get("GLM4_MOE_MTP", "").strip() == "1"
        _num_layers_for_state = int(hparams.num_hidden_layers) + (1 if _mtp_flag else 0)
        state = load_glm_lazy_state_dict(snapshot_dir, num_layers=_num_layers_for_state)

        # Embedding.
        embed_w_cpu = _dequant_weight(state, "model.embed_tokens.weight").clone().to(torch.bfloat16)
        # Device embedding enables in-trace ttnn.embedding (4-byte token IDs vs 40KB embeddings per H2D).
        # Cost: ~371 MB DRAM per device (37888 vocab × 5120 hidden × 2 bytes BF16, replicated).
        # This eliminates the biggest source of host overhead (92ms → ~30ms per token).
        _device_embed = os.environ.get("GLM4_MOE_DEVICE_EMBED", "1").strip() == "1"
        if _device_embed and _is_mesh_device(device):
            logger.info("Loading embedding to device (~371 MB/device)")
            embed_w = ttnn.from_torch(
                embed_w_cpu.unsqueeze(0).unsqueeze(0),  # [1, 1, vocab, hidden]
                device=device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            )
        else:
            embed_w = None

        # RoPE for partial rotary (rope_dim = head_dim * partial_rotary_factor = 64).
        rope_dim = int(hparams.head_dim * hparams.partial_rotary_factor)
        rope = _make_rope_tensors(
            device=device,
            seq_len=int(max_seq_len),
            rope_dim=rope_dim,
            rope_theta=float(hparams.rope_theta),
        )

        # Final norm + LM head.
        dense_dtype = _env_dense_dtype()
        final_norm = RMSNorm(
            device=device,
            dim=int(hparams.hidden_size),
            eps=float(hparams.rms_norm_eps),
            state_dict=state,
            state_dict_prefix="model.",
            weight_key="norm",
            weight_cache_path=cache_dir,
            weight_dtype=ttnn.bfloat16,
            is_distributed=False,
        )

        # LM head sharding: shard vocab across TP devices (axis 0), replicate across DP.
        lm_head_mapper = None
        lm_head_variant = ""
        lm_head_sharded_vocab = False
        lm_head_tp_axis = None
        lm_head_tp_size = 1
        lm_head_vocab_per_shard = int(hparams.vocab_size)
        num_devices = 1
        if _is_mesh_device(device):
            num_devices = int(device.get_num_devices())

        tp_axis, tp_size_detected = _tp_axis_and_size(device)
        if tp_size_detected > 1 and num_devices > 1:
            vocab = int(hparams.vocab_size)
            if vocab % tp_size_detected == 0:
                # Shard vocab across TP axis only, replicate across DP.
                lm_head_mapper = _tp_mesh_mapper(device, shard_dim=3)
                lm_head_variant = f"tp{tp_size_detected}_shard_v1"
                lm_head_sharded_vocab = True
                lm_head_tp_axis = tp_axis
                lm_head_tp_size = tp_size_detected
                lm_head_vocab_per_shard = vocab // tp_size_detected
            else:
                logger.warning(
                    "LM head vocab {} not divisible by TP={} devices, replicating instead of sharding",
                    vocab, tp_size_detected,
                )

        lm_head_w = _linear_weight_tt(
            device=device,
            torch_weight_out_in=_dequant_weight(state, "lm_head.weight"),
            cache_file=cache_dir / f"lm_head_w_{lm_head_variant}" if lm_head_variant else cache_dir / "lm_head_w",
            dtype=dense_dtype,
            mesh_mapper=lm_head_mapper,
        )

        # Layers.
        num_layers_env = os.environ.get("GLM4_MOE_NUM_LAYERS", "").strip()
        if num_layers_env and os.environ.get("GLM4_MOE_DEBUG_ALLOW_PARTIAL_LAYERS", "").strip() != "1":
            raise ValueError(
                "GLM4_MOE_NUM_LAYERS is debug-only. "
                "Set GLM4_MOE_DEBUG_ALLOW_PARTIAL_LAYERS=1 to run a partial model."
            )
        num_layers_to_run = int(num_layers_env) if num_layers_env else int(hparams.num_hidden_layers)
        num_layers_to_run = max(1, min(num_layers_to_run, int(hparams.num_hidden_layers)))

        enable_moe = os.environ.get("GLM4_MOE_ENABLE_MOE", "1").strip() != "0"
        moe_runtime = None
        if enable_moe:
            moe_runtime = create_moe_runtime(device=device, hparams=hparams)

        # Configuration dict for attention and decoder layers.
        mesh_rows = int(device.shape[0]) if _is_mesh_device(device) else 1
        mesh_cols = int(device.shape[1]) if _is_mesh_device(device) else 1
        tp_axis, tp_size = _tp_axis_and_size(device)  # uses module-level import
        configuration = {
            "num_devices": mesh_rows * mesh_cols,
            "tp_size": tp_size,
            "tp_axis": tp_axis if tp_axis is not None else 0,
            "max_batch_size": max_batch_size,
            "MAX_QKV_MM_SEQ_LEN": 4096,
            "ccl_dtype": ttnn.bfloat16,
        }

        # CCL for attention all-reduce/all-gather (async semaphore management).
        if _is_mesh_device(device):
            logger.info("Glm4MoeTT.create: initializing CCL semaphores for mesh {}", list(device.shape))
            tt_ccl = CCL(device)
        else:
            tt_ccl = None

        # Enable program cache so compiled kernels are reused across layers.
        # Without this, every layer recompiles from scratch (~minutes per layer).
        device.enable_program_cache()
        logger.info("Glm4MoeTT.create: program cache enabled")

        # Load decoder layer weights lazily (one at a time to save host memory).
        layer_weights_dict: dict[int, DecoderLayerTTWeights] = {}
        decoder_layers_dict: dict[int, Glm4MoeDecoderLayer] = {}

        logger.info(
            "Glm4MoeTT.create: loading {} layers (total={}), moe={}, devices={}, max_seq_len={}",
            num_layers_to_run, hparams.num_hidden_layers, enable_moe, num_devices, max_seq_len,
        )

        # DEBUG: test device sync before loading layers
        if os.environ.get("GLM4_MOE_DEBUG_SYNC", "0") != "0":
            import sys
            print("  [DEBUG MODEL] synchronizing before layer loading ...", flush=True, file=sys.stderr)
            ttnn.synchronize_device(device)
            print("  [DEBUG MODEL] synchronize before layers OK", flush=True, file=sys.stderr)

        for layer_idx in range(num_layers_to_run):
            t0 = time.perf_counter()
            logger.info("  [DEBUG] Starting layer {} weight conversion", layer_idx)
            # Disable ttnn weight cache to prevent OOM on 566 GB host.
            # Cache writes allocate host buffers (~7 GB/layer) that persist in anon RSS,
            # totaling ~525 GB for 92 layers and triggering OOM at layer ~78.
            _layer_cache_dir = None if os.environ.get("GLM4_MOE_DISABLE_WEIGHT_CACHE", "0") != "0" else cache_dir / "layers"
            lw = convert_decoder_layer_weights(
                device=device,
                state=state,
                layer_idx=layer_idx,
                hparams=hparams,
                cache_dir=_layer_cache_dir,
                enable_moe=enable_moe and (layer_idx >= int(hparams.first_k_dense_replace)),
            )
            layer_weights_dict[layer_idx] = lw
            logger.info("  [DEBUG] Layer {} weights converted, creating decoder layer", layer_idx)

            dl = Glm4MoeDecoderLayer(
                mesh_device=device,
                tt_ccl=tt_ccl,
                hparams=hparams,
                layer_weights=lw,
                configuration=configuration,
                paged_attention_config=None,
                moe_runtime=moe_runtime,
            )
            decoder_layers_dict[layer_idx] = dl

            elapsed = time.perf_counter() - t0
            if layer_idx == 0 or (layer_idx + 1) % 10 == 0 or (layer_idx + 1) == num_layers_to_run:
                logger.info("  Layer {}/{} loaded ({:.1f}s)", layer_idx + 1, num_layers_to_run, elapsed)

            # Clear LazyStateDict cache + GC to prevent OOM during 92-layer loading.
            # Without this, the cache accumulates ~416 GB of BF16 torch tensors
            # (92 layers × 96 experts × 3 projections × ~15.7 MB each).
            if hasattr(state, '_cache'):
                state._cache.clear()
            gc.collect()

            # DEBUG: sync after each layer to find which one hangs the device
            if os.environ.get("GLM4_MOE_DEBUG_SYNC", "0") != "0":
                print(f"  [DEBUG MODEL] sync after layer {layer_idx} ...", flush=True, file=sys.stderr)
                ttnn.synchronize_device(device)
                print(f"  [DEBUG MODEL] sync after layer {layer_idx} OK", flush=True, file=sys.stderr)

        # DEBUG: test device sync after model init
        if os.environ.get("GLM4_MOE_DEBUG_SYNC", "0") != "0":
            import sys
            print("  [DEBUG MODEL] synchronizing after all layers loaded ...", flush=True, file=sys.stderr)
            ttnn.synchronize_device(device)
            print("  [DEBUG MODEL] synchronize after layers OK", flush=True, file=sys.stderr)

        # ---- MTP Layer (optional, for GLM-4.7 Full with num_nextn_predict_layers=1) ----
        _mtp_enabled = os.environ.get("GLM4_MOE_MTP", "").strip() == "1"
        _mtp_max_batch = int(os.environ.get("GLM4_MOE_MTP_MAX_BATCH", "16") or "16")
        _mtp_enorm = None
        _mtp_hnorm = None
        _mtp_eh_proj_e_w = None
        _mtp_eh_proj_h_w = None
        _mtp_shared_head_norm = None
        _mtp_shared_head_w = None
        _mtp_decoder_layer = None

        if _mtp_enabled:
            mtp_layer_idx = int(hparams.num_hidden_layers)  # 92
            logger.info("MTP enabled: loading layer {} weights (mtp_max_batch={})", mtp_layer_idx, _mtp_max_batch)
            _hidden = int(hparams.hidden_size)

            # enorm + hnorm: RMSNorm(hidden_size)
            _mtp_enorm = RMSNorm(
                device=device, dim=_hidden, eps=float(hparams.rms_norm_eps),
                state_dict=state, state_dict_prefix=f"model.layers.{mtp_layer_idx}.",
                weight_key="enorm", weight_cache_path=cache_dir / "mtp",
                weight_dtype=ttnn.bfloat16, is_distributed=False,
            )
            _mtp_hnorm = RMSNorm(
                device=device, dim=_hidden, eps=float(hparams.rms_norm_eps),
                state_dict=state, state_dict_prefix=f"model.layers.{mtp_layer_idx}.",
                weight_key="hnorm", weight_cache_path=cache_dir / "mtp",
                weight_dtype=ttnn.bfloat16, is_distributed=False,
            )

            # eh_proj: Linear(2*hidden -> hidden). Split into two halves.
            eh_key = f"model.layers.{mtp_layer_idx}.eh_proj.weight"
            eh_full = state[eh_key]
            if hasattr(eh_full, 'item') or not isinstance(eh_full, torch.Tensor):
                eh_full = torch.tensor(eh_full) if not isinstance(eh_full, torch.Tensor) else eh_full
            # HF layout: [out_features=hidden, in_features=2*hidden]
            # ttnn.linear(x, w) computes x @ w, so w must be [in_features, out_features]
            eh_full_t = eh_full.t().contiguous()  # [2*hidden, hidden]
            eh_e = eh_full_t[:_hidden, :].contiguous()  # [hidden, hidden] (embed half)
            eh_h = eh_full_t[_hidden:, :].contiguous()   # [hidden, hidden] (hidden half)
            mapper = ttnn.ReplicateTensorToMesh(device) if _is_mesh_device(device) else None
            _mtp_eh_proj_e_w = ttnn.from_torch(
                eh_e.to(torch.bfloat16), device=device, dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )
            _mtp_eh_proj_h_w = ttnn.from_torch(
                eh_h.to(torch.bfloat16), device=device, dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )

            # shared_head norm + LM head (same vocab sharding as main lm_head)
            _mtp_shared_head_norm = RMSNorm(
                device=device, dim=_hidden, eps=float(hparams.rms_norm_eps),
                state_dict=state, state_dict_prefix=f"model.layers.{mtp_layer_idx}.shared_head.",
                weight_key="norm", weight_cache_path=cache_dir / "mtp",
                weight_dtype=ttnn.bfloat16, is_distributed=False,
            )
            sh_key = f"model.layers.{mtp_layer_idx}.shared_head.head.weight"
            sh_w = state[sh_key].to(torch.bfloat16)
            # HF: [vocab, hidden]. ttnn.linear needs [hidden, vocab] (w transposed).
            sh_w = sh_w.t().contiguous()
            _mtp_shared_head_w = ttnn.from_torch(
                sh_w, device=device, dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )

            # Full decoder layer for MTP (layer 92)
            mtp_lw = convert_decoder_layer_weights(
                device=device, state=state, layer_idx=mtp_layer_idx,
                hparams=hparams, cache_dir=cache_dir / "mtp",
            )
            _mtp_decoder_layer = Glm4MoeDecoderLayer(
                mesh_device=device, hparams=hparams, layer_weights=mtp_lw,
                configuration=configuration, moe_runtime=moe_runtime, tt_ccl=tt_ccl,
            )
            logger.info("MTP layer {} loaded", mtp_layer_idx)

        return cls(
            device=device,
            snapshot_dir=snapshot_dir,
            cache_dir=cache_dir,
            max_seq_len=int(max_seq_len),
            hparams=hparams,
            state=state,
            embed_w=embed_w,
            embed_w_cpu=embed_w_cpu,
            rope=rope,
            final_norm=final_norm,
            lm_head_w=lm_head_w,
            lm_head_sharded_vocab=lm_head_sharded_vocab,
            lm_head_tp_axis=lm_head_tp_axis,
            lm_head_tp_size=lm_head_tp_size,
            lm_head_vocab_per_shard=lm_head_vocab_per_shard,
            layer_weights=layer_weights_dict,
            decoder_layers=decoder_layers_dict,
            num_layers_to_run=num_layers_to_run,
            enable_moe=enable_moe,
            moe_runtime=moe_runtime,
            configuration=configuration,
            tt_ccl=tt_ccl,
            mtp_enabled=_mtp_enabled,
            mtp_enorm=_mtp_enorm,
            mtp_hnorm=_mtp_hnorm,
            mtp_eh_proj_e_w=_mtp_eh_proj_e_w,
            mtp_eh_proj_h_w=_mtp_eh_proj_h_w,
            mtp_shared_head_norm=_mtp_shared_head_norm,
            mtp_shared_head_w=_mtp_shared_head_w,
            mtp_decoder_layer=_mtp_decoder_layer,
            mtp_max_batch=_mtp_max_batch,
        )

    # -------------------------------------------------------------------
    # Decode
    # -------------------------------------------------------------------

    @torch.no_grad()
    def decode(
        self,
        *,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        page_table: torch.Tensor,
        kv_cache: list,
        sampling_params: Any | None = None,
        enable_trace: bool = False,
    ) -> Any:
        """Run a single-token decode step through all layers.

        Args:
            tokens: [B, 1] int32
            start_pos: [B] int32 (padded with -1 for inactive slots)
            page_table: [B, W] int32
            kv_cache: list of [cache_k, cache_v] per layer
            sampling_params: if not None, do on-device greedy sampling
            enable_trace: use trace capture/replay for decode

        Returns:
            logits [active, 1, vocab] as torch float32 when sampling_params is None,
            or next token ids [active] as torch int32 when sampling_params is not None.
        """
        if tokens.ndim != 2 or tokens.shape[1] != 1:
            raise ValueError(f"expected tokens [B,1], got {tuple(tokens.shape)}")
        if start_pos.ndim != 1:
            raise ValueError(f"expected start_pos [B], got {tuple(start_pos.shape)}")

        start_pos = start_pos.to(torch.int32)
        active = int((start_pos >= 0).sum().item())
        if active <= 0:
            return torch.zeros((0, 1, int(self.hparams.vocab_size)), dtype=torch.float32)

        # Traces are always released before prefill (WH pattern). If _decode_traces_stale
        # is set, it means prefill ran but release failed — force release now.
        if enable_trace and self._decode_traces_stale:
            logger.info("Stale traces detected — releasing before decode")
            self._release_all_decode_traces()
            self._decode_traces_stale = False

        if enable_trace:
            return self._decode_trace(
                tokens=tokens[:active].to(torch.int32),
                positions=start_pos[:active].to(torch.int32),
                page_table=page_table[:active].to(torch.int32),
                kv_cache=kv_cache,
                sampling_params=sampling_params,
            )

        return self._decode_eager(
            tokens=tokens[:active].to(torch.int32),
            positions=start_pos[:active].to(torch.int32),
            page_table=page_table[:active].to(torch.int32),
            kv_cache=kv_cache,
            sampling_params=sampling_params,
        )

    def _decode_eager(
        self,
        *,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        page_table: torch.Tensor,
        kv_cache: list,
        sampling_params: Any | None = None,
    ) -> Any:
        """Eager (non-traced) decode step."""
        active = int(tokens.shape[0])
        hidden = int(self.hparams.hidden_size)
        is_mesh = _is_mesh_device(self.device)
        logger.info("[DBG] _decode_eager entry: active={}, hidden={}, positions={}, page_table_shape={}, is_mesh={}",
                    active, hidden, positions.tolist()[:8], list(page_table.shape), is_mesh)

        # Prepare inputs.
        tokens = tokens.contiguous().clone()
        positions = positions.contiguous().clone()
        page_table = page_table.contiguous().clone()

        # On TG (2D mesh) with multi-user batch, shard batch-dimension inputs
        # across DP groups (page_table, positions, cos/sin).
        dp_shard_axis = None
        dp_batch_mapper = ttnn.ReplicateTensorToMesh(self.device) if is_mesh else None
        if is_mesh:
            tp_axis, _ = _tp_axis_and_size(self.device)
            dp_axis = 1 - (tp_axis if tp_axis is not None else 0)
            dp_size = int(self.device.shape[dp_axis])
            if dp_size > 1 and active > 1 and active % dp_size == 0:
                dp_shard_axis = dp_axis
                mesh_shape = list(self.device.shape)
                dims = [None, None]
                dims[dp_axis] = 0  # shard tensor dim 0 (batch) across dp_axis
                dp_batch_mapper = ttnn.ShardTensor2dMesh(
                    self.device, dims=tuple(dims), mesh_shape=mesh_shape,
                )

        page_table_tt = ttnn.from_torch(
            page_table,
            device=self.device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=dp_batch_mapper,
        )

        tt_positions, cos_batch, sin_batch = _prepare_decode_rope_and_positions_tt(
            device=self.device, rope=self.rope, positions=positions,
            dp_shard_axis=dp_shard_axis,
        )

        # Embedding: host-side lookup to avoid device-side tile conversion hang on TG mesh.
        embed_torch = self.embed_w_cpu[tokens[:, 0].long()]  # [B, hidden]
        x = ttnn.from_torch(
            embed_torch.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),  # [1, 1, B, hidden]
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device) if is_mesh else None,
        )

        # RoPE mats for attention (cos, sin, trans_matrix).
        rot_mats = (cos_batch, sin_batch, self.rope["trans_matrix"])

        # DEBUG: verify embedding correctness across devices
        import sys as _dbg_sys
        _dbg_decode = os.environ.get("GLM4_MOE_DEBUG_DECODE", "0") != "0"
        if _dbg_decode and is_mesh:
            dev_tensors = ttnn.get_device_tensors(x)
            d0_embed = ttnn.to_torch(dev_tensors[0].cpu())  # device 0
            d1_embed = ttnn.to_torch(dev_tensors[1].cpu())  # device 1 (different TP col)
            d4_embed = ttnn.to_torch(dev_tensors[4].cpu()) if len(dev_tensors) > 4 else None  # different DP row
            print(f"  [DBG EMBED] d0 shape={list(d0_embed.shape)}, d0[:8]={d0_embed[0,0,0,:8].tolist()}", flush=True, file=_dbg_sys.stderr)
            print(f"  [DBG EMBED] d1 shape={list(d1_embed.shape)}, d1[:8]={d1_embed[0,0,0,:8].tolist()}", flush=True, file=_dbg_sys.stderr)
            if d4_embed is not None:
                print(f"  [DBG EMBED] d4 shape={list(d4_embed.shape)}, d4[:8]={d4_embed[0,0,0,:8].tolist()}", flush=True, file=_dbg_sys.stderr)
            # Compare host embedding vs device 0
            host_ref = embed_torch[0, :8].float().tolist()
            dev0_vals = d0_embed[0,0,0,:8].float().tolist()
            match = all(abs(a-b) < 0.1 for a,b in zip(host_ref, dev0_vals))
            print(f"  [DBG EMBED] host[:8]={host_ref}, match_d0={match}", flush=True, file=_dbg_sys.stderr)

        # Decoder stack.
        logger.info("[DBG] _decode_eager: entering decoder stack ({} layers)", self.num_layers_to_run)
        for layer_idx in range(self.num_layers_to_run):
            if not hasattr(self, '_decode_layers_logged') or not self._decode_layers_logged:
                logger.info("  [DECODE] Processing layer {}/{}", layer_idx + 1, self.num_layers_to_run)
            logger.info("[DBG] _decode_eager: before layer {} x.shape={}", layer_idx, list(x.shape))
            dl = self.decoder_layers[layer_idx]
            x_next = dl.forward(
                x, tt_positions, rot_mats, page_table_tt, kv_cache[layer_idx], mode="decode",
                active_batch=active,
            )
            logger.info("[DBG] _decode_eager: after layer {} x_next.shape={}", layer_idx, list(x_next.shape))
            _dealloc(x, force=False)
            x = x_next
            if not hasattr(self, '_decode_layers_logged') or not self._decode_layers_logged:
                if layer_idx == self.num_layers_to_run - 1:
                    self._decode_layers_logged = True
                    logger.info("  [DECODE] All layers processed (program compilation complete)")

            # DEBUG: compare hidden states across TP devices after each layer
            if _dbg_decode and is_mesh:
                dev_tensors = ttnn.get_device_tensors(x)
                d0_h = ttnn.to_torch(dev_tensors[0].cpu())  # col 0 (TP 0)
                d1_h = ttnn.to_torch(dev_tensors[1].cpu())  # col 1 (TP 1)
                d4_h = ttnn.to_torch(dev_tensors[4].cpu()) if len(dev_tensors) > 4 else None  # row 1, col 0
                diff_01 = (d0_h - d1_h).abs().max().item()
                diff_04 = (d0_h - d4_h).abs().max().item() if d4_h is not None else -1
                print(f"  [DBG L{layer_idx}] d0[:8]={d0_h[0,0,0,:8].float().tolist()}", flush=True, file=_dbg_sys.stderr)
                print(f"  [DBG L{layer_idx}] d1[:8]={d1_h[0,0,0,:8].float().tolist()}", flush=True, file=_dbg_sys.stderr)
                print(f"  [DBG L{layer_idx}] d0-d1 max_diff={diff_01:.6f}, d0-d4 max_diff={diff_04:.6f}", flush=True, file=_dbg_sys.stderr)
                if diff_01 > 0.5:
                    print(f"  [DBG L{layer_idx}] WARNING: TP devices diverged! All-reduce may be broken.", flush=True, file=_dbg_sys.stderr)

        # Save pre-norm hidden for MTP (before final_norm destroys it)
        mtp_hidden = x if self.mtp_enabled else None

        # Final norm + LM head (sharded norm to avoid L1 overflow with hidden=5120).
        x = _sharded_rms_norm(x, self.final_norm, int(self.hparams.hidden_size))
        logits_tt = ttnn.linear(x, self.lm_head_w)  # [1, 1, B, vocab]

        if sampling_params is not None:
            return self._sample_greedy(logits_tt, active, x, tt_positions, cos_batch, sin_batch, page_table_tt)

        # Return full logits on host.
        vocab = int(self.hparams.vocab_size)
        result = self._logits_to_host(logits_tt, active, vocab)

        # MTP: run speculative decode after main decode
        if self.mtp_enabled and mtp_hidden is not None:
            try:
                # Get main output token IDs for MTP embedding (argmax of logits)
                main_ids = result.reshape(active, -1).argmax(dim=-1).to(torch.int32)
                # MTP positions = main positions + 1
                mtp_positions = positions[:active] + 1
                self._last_draft_token_ids = self._mtp_forward_eager(
                    main_token_ids=main_ids,
                    hidden_state=mtp_hidden,
                    mtp_positions=mtp_positions,
                    page_table=page_table[:active],
                    kv_cache=kv_cache,
                )
            except Exception as e:
                logger.warning("MTP forward failed (non-fatal): {}", e)
                self._last_draft_token_ids = None

        _dealloc(logits_tt, force=False)
        _dealloc(x, force=False)
        _dealloc(tt_positions, force=False)
        _dealloc(cos_batch, force=False)
        _dealloc(sin_batch, force=False)
        _dealloc(page_table_tt, force=False)

        # Reset CCL semaphore counters after eager decode.
        if self.tt_ccl is not None:
            self.tt_ccl.reset_sem_counters()

        return result

    def _sample_greedy(
        self,
        logits_tt: ttnn.Tensor,
        active: int,
        x: ttnn.Tensor,
        tt_positions: ttnn.Tensor,
        cos_batch: ttnn.Tensor,
        sin_batch: ttnn.Tensor,
        page_table_tt: ttnn.Tensor,
    ) -> torch.Tensor:
        """On-device greedy sampling (argmax) from logits."""
        vocab = int(self.hparams.vocab_size)

        if not (self.lm_head_sharded_vocab and _is_mesh_device(self.device)):
            logits_rm = ttnn.to_layout(logits_tt, ttnn.ROW_MAJOR_LAYOUT)
            logits_rm_view = ttnn.slice(logits_rm, [0, 0, 0, 0], [1, 1, active, vocab])
            logits_rm_tight = ttnn.clone(logits_rm_view, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            max_out = ttnn.max(logits_rm_tight, dim=3, keepdim=True)
            if isinstance(max_out, tuple):
                local_max_tt, next_ids_tt = max_out
                _dealloc(local_max_tt, force=False)
            else:
                next_ids_tt = ttnn.argmax(logits_rm_tight, dim=3, keepdim=False, use_multicore=True)
            _dealloc(logits_rm_tight, force=False)

            next_ids_torch = _tt_to_torch_for_vllm_output(tensor=next_ids_tt, device=self.device)
            next_ids_flat = next_ids_torch.reshape(-1).to(dtype=torch.int32).cpu()
            _dealloc(logits_rm, force=False)
            _dealloc(logits_tt, force=False)
            _dealloc(next_ids_tt, force=False)
        else:
            # Vocab-sharded: per-device max+argmax, reduce on host.
            logits_rm = ttnn.to_layout(logits_tt, ttnn.ROW_MAJOR_LAYOUT)
            vocab_per_shard = int(self.lm_head_vocab_per_shard)
            logits_rm_view = ttnn.slice(logits_rm, [0, 0, 0, 0], [1, 1, active, vocab_per_shard])
            max_out = ttnn.max(logits_rm_view, dim=3, keepdim=True)
            if isinstance(max_out, tuple):
                local_max_tt, local_argmax_tt = max_out
            else:
                local_max_tt = max_out
                local_argmax_tt = ttnn.argmax(logits_rm_view, dim=3, keepdim=False, use_multicore=True)

            num_devices = int(self.device.get_num_devices())
            local_max_dts = ttnn.get_device_tensors(local_max_tt)
            local_idx_dts = ttnn.get_device_tensors(local_argmax_tt)

            # Pick one device per TP shard (skip DP duplicates).
            tp_size = self.lm_head_tp_size
            tp_axis = self.lm_head_tp_axis if self.lm_head_tp_axis is not None else 0
            mesh_cols = int(self.device.shape[1])
            dp_stride = mesh_cols if tp_axis == 0 else 1

            next_ids = torch.empty((active,), dtype=torch.int32)
            for b in range(active):
                best_val = None
                best_global = None
                for tp_idx in range(tp_size):
                    shard_idx = tp_idx * dp_stride
                    max_val = float(ttnn.to_torch(local_max_dts[shard_idx].cpu()).reshape(-1)[b].item())
                    local_idx = int(ttnn.to_torch(local_idx_dts[shard_idx].cpu()).reshape(-1)[b].item())
                    global_idx = tp_idx * vocab_per_shard + local_idx
                    if global_idx >= vocab:
                        continue
                    if best_val is None or max_val > best_val:
                        best_val = max_val
                        best_global = global_idx
                if best_global is None:
                    best_global = max(0, vocab - 1)
                next_ids[b] = best_global
            next_ids_flat = next_ids

            _dealloc(local_max_tt, force=False)
            _dealloc(local_argmax_tt, force=False)
            _dealloc(logits_rm, force=False)
            _dealloc(logits_tt, force=False)

        _dealloc(x, force=False)
        _dealloc(tt_positions, force=False)
        _dealloc(cos_batch, force=False)
        _dealloc(sin_batch, force=False)
        _dealloc(page_table_tt, force=False)

        # Reset CCL semaphore counters after eager decode (sampling path).
        if self.tt_ccl is not None:
            self.tt_ccl.reset_sem_counters()

        return next_ids_flat

    def _sample_from_trace_logits(self, logits_tt: ttnn.Tensor, active: int) -> torch.Tensor:
        """Device-side greedy sampling from trace-owned logits (OUTSIDE trace).

        Called after execute_trace + synchronize_device. The logits_tt tensor
        is trace-owned and must NOT be deallocated.

        For TP-sharded vocab: per-shard max+argmax on device, then pick global
        best on host. Transfers only ~64 bytes instead of 9.6 MB.
        """
        vocab = int(self.hparams.vocab_size)

        if self.lm_head_sharded_vocab and _is_mesh_device(self.device):
            # Vocab-sharded: per-device max+argmax, reduce on host.
            logits_rm = ttnn.to_layout(logits_tt, ttnn.ROW_MAJOR_LAYOUT)
            vocab_per_shard = int(self.lm_head_vocab_per_shard)
            logits_rm_view = ttnn.slice(logits_rm, [0, 0, 0, 0], [1, 1, active, vocab_per_shard])
            max_out = ttnn.max(logits_rm_view, dim=3, keepdim=True)
            if isinstance(max_out, tuple):
                local_max_tt, local_argmax_tt = max_out
            else:
                local_max_tt = max_out
                local_argmax_tt = ttnn.argmax(logits_rm_view, dim=3, keepdim=False, use_multicore=True)

            num_devices = int(self.device.get_num_devices())
            tp_size = self.lm_head_tp_size
            local_max_dts = ttnn.get_device_tensors(local_max_tt)
            local_idx_dts = ttnn.get_device_tensors(local_argmax_tt)

            # Skip DP duplicates: pick one shard per TP position.
            # For tp_axis=0 (rows): TP shards spaced mesh_cols apart in flat device order.
            # For tp_axis=1 (cols): TP shards are adjacent (stride=1).
            tp_axis = self.lm_head_tp_axis if self.lm_head_tp_axis is not None else 0
            mesh_cols = int(self.device.shape[1])
            dp_stride = mesh_cols if tp_axis == 0 else 1

            next_ids = torch.empty((active,), dtype=torch.int32)
            for b in range(active):
                best_val = None
                best_global = None
                for tp_idx in range(tp_size):
                    shard_idx = tp_idx * dp_stride
                    max_val = float(ttnn.to_torch(local_max_dts[shard_idx].cpu()).reshape(-1)[b].item())
                    local_idx = int(ttnn.to_torch(local_idx_dts[shard_idx].cpu()).reshape(-1)[b].item())
                    global_idx = tp_idx * vocab_per_shard + local_idx
                    if global_idx >= vocab:
                        continue
                    if best_val is None or max_val > best_val:
                        best_val = max_val
                        best_global = global_idx
                if best_global is None:
                    best_global = max(0, vocab - 1)
                next_ids[b] = best_global

            # Clean up temporaries (NOT logits_tt — trace-owned)
            _dealloc(local_max_tt, force=True)
            _dealloc(local_argmax_tt, force=True)
            _dealloc(logits_rm_view, force=False)
            _dealloc(logits_rm, force=True)
        else:
            # Non-sharded: single argmax
            logits_rm = ttnn.to_layout(logits_tt, ttnn.ROW_MAJOR_LAYOUT)
            logits_rm_view = ttnn.slice(logits_rm, [0, 0, 0, 0], [1, 1, active, vocab])
            logits_rm_tight = ttnn.clone(logits_rm_view, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            max_out = ttnn.max(logits_rm_tight, dim=3, keepdim=True)
            if isinstance(max_out, tuple):
                _, next_ids_tt = max_out
            else:
                next_ids_tt = ttnn.argmax(logits_rm_tight, dim=3, keepdim=False, use_multicore=True)

            next_ids = _tt_to_torch_for_vllm_output(tensor=next_ids_tt, device=self.device)
            next_ids = next_ids.reshape(-1).to(dtype=torch.int32).cpu()

            _dealloc(next_ids_tt, force=True)
            _dealloc(logits_rm_tight, force=True)
            _dealloc(logits_rm_view, force=False)
            _dealloc(logits_rm, force=True)

        return next_ids

    def _host_argmax_from_trace_logits(
        self, logits_tt: ttnn.Tensor, active: int, vocab: int,
    ) -> torch.Tensor:
        """Host-side argmax from trace-owned logits, optimized for minimal work.

        Instead of concatenating all TP shards into full vocab then running argmax,
        does per-shard argmax on host and picks global best. Avoids the full
        torch.cat allocation.

        Does NOT deallocate logits_tt (trace-owned).
        """
        if self.lm_head_sharded_vocab and _is_mesh_device(self.device):
            shards = ttnn.get_device_tensors(logits_tt)
            num_shards = len(shards)
            tp_size = self.lm_head_tp_size
            vocab_per_shard = int(self.lm_head_vocab_per_shard)

            # Select one shard per TP position (skip DP duplicates).
            # For tp_axis=0 (rows): TP shards spaced mesh_cols apart in flat device order.
            # For tp_axis=1 (cols): TP shards are adjacent (stride=1).
            tp_axis = self.lm_head_tp_axis if self.lm_head_tp_axis is not None else 0
            mesh_cols = int(self.device.shape[1])
            dp_stride = mesh_cols if tp_axis == 0 else 1

            next_ids = torch.empty((active,), dtype=torch.int32)

            # Transfer shards and do per-shard argmax on host
            shard_maxvals = []
            shard_argmaxes = []
            for tp_idx in range(tp_size):
                shard_idx = tp_idx * dp_stride
                shard_torch = ttnn.to_torch(shards[shard_idx].cpu())
                # Slice to valid batch and vocab range
                shard_torch = shard_torch[..., :active, :vocab_per_shard]
                shard_flat = shard_torch.reshape(active, -1).to(torch.float32)
                shard_maxvals.append(shard_flat.max(dim=-1))
                shard_argmaxes.append(shard_flat.argmax(dim=-1))

            # Pick global best across shards
            for b in range(active):
                best_val = None
                best_global = None
                for tp_idx in range(tp_size):
                    max_val = float(shard_maxvals[tp_idx].values[b].item())
                    local_idx = int(shard_argmaxes[tp_idx][b].item())
                    global_idx = tp_idx * vocab_per_shard + local_idx
                    if global_idx >= vocab:
                        continue
                    if best_val is None or max_val > best_val:
                        best_val = max_val
                        best_global = global_idx
                if best_global is None:
                    best_global = max(0, vocab - 1)
                next_ids[b] = best_global

            return next_ids
        else:
            # Non-sharded: simple host transfer + argmax
            logits_host = self._logits_to_host(logits_tt, active, vocab)
            return logits_host.reshape(active, -1)[:, :vocab].argmax(dim=-1).to(torch.int32)

    def _logits_to_host(
        self,
        logits_tt: ttnn.Tensor,
        active: int,
        vocab: int,
    ) -> torch.Tensor:
        """Convert logits TT tensor to host torch [active, 1, vocab]."""
        if self.lm_head_sharded_vocab and _is_mesh_device(self.device):
            shards = ttnn.get_device_tensors(logits_tt)
            num_shards = len(shards)
            tp_size = self.lm_head_tp_size
            # Select one shard per TP position (skip DP duplicates).
            if num_shards == tp_size:
                # Already one per TP device (ShardTensor2dMesh collapsed DP).
                tp_shards = list(shards)
            elif num_shards > tp_size:
                # All mesh devices returned; pick one shard per TP position.
                # For tp_axis=0 (rows): TP shards are spaced mesh_cols apart in flat order.
                # For tp_axis=1 (cols): TP shards are adjacent in flat order.
                tp_axis = self.lm_head_tp_axis if self.lm_head_tp_axis is not None else 0
                mesh_cols = int(self.device.shape[1])
                dp_stride = mesh_cols if tp_axis == 0 else 1
                tp_shards = [shards[i * dp_stride] for i in range(tp_size)]
            else:
                tp_shards = list(shards)
            logits_shards = [ttnn.to_torch(t.cpu())[..., :int(t.shape[-1])] for t in tp_shards]
            logits_full = torch.cat(logits_shards, dim=-1)[..., :vocab]
            # Slice off TILE_SIZE padding — decode pads batch to 32 but we only need `active`.
            logits_full = logits_full[..., :active, :]
            logits_flat = logits_full.reshape(-1, vocab)
        else:
            logits_torch = _tt_to_torch_for_vllm_output(tensor=logits_tt, device=self.device)
            logits_torch = logits_torch[..., :vocab]
            # Slice off TILE_SIZE padding — decode pads batch to 32 but we only need `active`.
            logits_torch = logits_torch[..., :active, :]
            logits_flat = logits_torch.reshape(-1, vocab)

        if logits_flat.shape[0] != active:
            raise RuntimeError(
                f"decode logits shape mismatch: expected {active} rows, got {int(logits_flat.shape[0])}"
            )

        return logits_flat.reshape(active, 1, vocab).to(dtype=torch.float32).cpu()

    # -------------------------------------------------------------------
    # Decode Trace (capture/replay)
    # -------------------------------------------------------------------

    def _decode_trace(
        self,
        *,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        page_table: torch.Tensor,
        kv_cache: list,
        sampling_params: Any | None = None,
    ) -> Any:
        """Decode using trace capture/replay for performance.

        On first call with a given batch size, captures a trace.
        On subsequent calls, replays the trace with updated inputs.
        """
        active = int(tokens.shape[0])
        page_table_w = int(page_table.shape[1])
        _dbg = os.environ.get("GLM4_MOE_DEBUG_DECODE", "0") != "0"
        if _dbg:
            logger.info("[DBG] _decode_trace entry: active={}, page_table_w={}", active, page_table_w)
        # Bucket batch size to avoid capturing separate traces for every unique size.
        # With decode_trace_batch_buckets=[32], only one trace is captured.
        _DECODE_BUCKETS = [1, 4, 8, 16, 32]
        _buckets = getattr(self, "tt_config", {}).get("decode_trace_batch_buckets", _DECODE_BUCKETS)
        bucket = next((b for b in _buckets if b >= active), active)
        if _dbg:
            logger.info("[DBG] _decode_trace: bucket={}", bucket)

        # Pad inputs to bucket size if needed.
        if active < bucket:
            pad = bucket - active
            tokens = torch.cat([tokens, tokens[-1:].expand(pad, -1)], dim=0)
            positions = torch.cat([positions, positions[-1:].expand(pad)], dim=0)
            page_table = torch.cat([page_table, page_table[-1:].expand(pad, -1)], dim=0)

        state = self._decode_trace_states.get(bucket)

        # E5: track capture vs replay calls for debugging
        _debug_trace_verify = int(os.environ.get("GLM4_MOE_DEBUG_TRACE_VERIFY", "") or "0")
        _is_capture = state is None or state.trace_id is None
        if _dbg:
            logger.info("[DBG] _decode_trace: is_capture={}, bucket={}", _is_capture, bucket)

        if _is_capture:
            # Capture a new trace for this batch bucket.
            try:
                state = self._capture_decode_trace(
                    tokens=tokens,
                    positions=positions,
                    page_table=page_table,
                    kv_cache=kv_cache,
                    sampling_params=sampling_params,
                    active=bucket,
                )
                self._decode_trace_states[bucket] = state
            except RuntimeError as e:
                if "trace_region_size" in str(e) or "trace buffers" in str(e).lower():
                    logger.warning(
                        "Trace capture failed for batch={} (trace region too small). "
                        "Falling back to eager decode for this bucket.", active,
                    )
                    return self._decode_eager(
                        tokens=tokens,
                        positions=positions,
                        page_table=page_table,
                        kv_cache=kv_cache,
                        sampling_params=sampling_params,
                    )
                raise
        else:
            # Update persistent inputs and replay.
            import time as _time
            _t0 = _time.perf_counter()

            self._update_trace_inputs(state, tokens, positions, page_table)
            _t1 = _time.perf_counter()

            if self.tt_ccl is not None:
                self.tt_ccl.reset_sem_counters()
            _t2 = _time.perf_counter()

            ttnn.synchronize_device(self.device)
            _t3 = _time.perf_counter()

            # DEBUG: trace quality investigation experiments (E0/E1/E3)
            if _debug_trace_verify >= 1:
                # E0: Read-back verification — check page_table data after copy
                # For replicated tensors on mesh, read from first device only
                try:
                    check_pt = None
                    if _is_mesh_device(self.device):
                        try:
                            pt_shard = ttnn.get_device_tensors(state.page_table_tt)[0]
                            check_pt = ttnn.to_torch(pt_shard)
                        except (AttributeError, RuntimeError) as _e0err:
                            logger.warning("DEBUG E0: get_device_tensors failed ({}), trying direct to_torch", _e0err)
                            check_pt = ttnn.to_torch(state.page_table_tt)
                    else:
                        check_pt = ttnn.to_torch(state.page_table_tt)
                    expected_pt = page_table
                    _debug_pt_width = int(os.environ.get("GLM4_MOE_DEBUG_PT_WIDTH", "") or "0")
                    if _debug_pt_width > 0 and expected_pt.shape[1] > _debug_pt_width:
                        expected_pt = expected_pt[:, :_debug_pt_width]
                    actual_flat = check_pt.reshape(-1)[:expected_pt.numel()].to(torch.int32)
                    expected_flat = expected_pt.reshape(-1).to(torch.int32)
                    if not torch.equal(actual_flat, expected_flat):
                        logger.error("DEBUG E0: page_table MISMATCH! expected={} got={}", expected_flat.tolist(), actual_flat.tolist())
                    else:
                        logger.info("DEBUG E0: page_table readback OK: {}", actual_flat.tolist())
                    ttnn.synchronize_device(self.device)
                except Exception as e:
                    logger.error("DEBUG E0: readback failed: {}", e)
            if _debug_trace_verify >= 2:
                # E1: Sleep injection — test timing race hypothesis
                import time
                time.sleep(0.05)
            if _debug_trace_verify >= 3:
                # E3: Dummy read barrier — force blocking read from page_table DRAM
                try:
                    _barrier_ok = False
                    if _is_mesh_device(self.device):
                        try:
                            for dev_t in ttnn.get_device_tensors(state.page_table_tt):
                                _ = ttnn.to_torch(dev_t)
                            _barrier_ok = True
                        except (AttributeError, RuntimeError):
                            _ = ttnn.to_torch(state.page_table_tt)
                            _barrier_ok = True
                    else:
                        _ = ttnn.to_torch(state.page_table_tt)
                        _barrier_ok = True
                    if _barrier_ok:
                        ttnn.synchronize_device(self.device)
                except Exception:
                    pass

            _t4 = _time.perf_counter()
            ttnn.execute_trace(self.device, state.trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(self.device)
            _t5 = _time.perf_counter()

            # Performance instrumentation (log every 50th token to avoid spam)
            if not hasattr(self, '_replay_count'):
                self._replay_count = 0
            self._replay_count += 1
            if self._replay_count <= 3 or self._replay_count % 50 == 0:
                logger.warning(
                    "PERF REPLAY #{}: update_inputs={:.1f}ms sem_reset={:.1f}ms pre_sync={:.1f}ms "
                    "execute+sync={:.1f}ms TOTAL={:.1f}ms",
                    self._replay_count,
                    (_t1 - _t0) * 1000, (_t2 - _t1) * 1000, (_t3 - _t2) * 1000,
                    (_t5 - _t4) * 1000, (_t5 - _t0) * 1000,
                )

        # E5: Log CAPTURE vs REPLAY and top-1 token for quality tracking
        if _debug_trace_verify >= 5 and state.logits_tt is not None:
            try:
                _e5_label = "CAPTURE" if _is_capture else "REPLAY"
                if _is_mesh_device(self.device):
                    try:
                        _e5_dev_t = ttnn.get_device_tensors(state.logits_tt)[0]
                        _e5_logits = ttnn.to_torch(_e5_dev_t).float()
                    except (AttributeError, RuntimeError):
                        _e5_logits = ttnn.to_torch(state.logits_tt).float()
                else:
                    _e5_logits = ttnn.to_torch(state.logits_tt).float()
                _e5_flat = _e5_logits.reshape(-1)
                _e5_top_val, _e5_top_idx = _e5_flat.topk(5)
                if not hasattr(self, '_e5_call_count'):
                    self._e5_call_count = 0
                self._e5_call_count += 1
                logger.warning("DEBUG E5 [{} #{}]: top5_ids={} top5_vals={} logits_stats=(mean={:.2f}, std={:.2f}, min={:.2f}, max={:.2f})",
                    _e5_label, self._e5_call_count,
                    _e5_top_idx.tolist(), [f'{v:.2f}' for v in _e5_top_val.tolist()],
                    _e5_flat.mean().item(), _e5_flat.std().item(),
                    _e5_flat.min().item(), _e5_flat.max().item())
                ttnn.synchronize_device(self.device)
            except Exception as e:
                logger.error("DEBUG E5: failed: {}", e)

        # Read outputs.
        _t6 = _time.perf_counter() if '_t0' in dir() else None
        _output = None
        if sampling_params is not None and state.top1_indices_tt is not None:
            _output = _tt_to_torch_for_vllm_output(
                tensor=state.top1_indices_tt, device=self.device
            ).reshape(-1).to(dtype=torch.int32).cpu()
        elif state.logits_tt is not None:
            if sampling_params is not None:
                if _is_mesh_device(self.device):
                    vocab = int(self.hparams.vocab_size)
                    _output = self._host_argmax_from_trace_logits(state.logits_tt, active, vocab)
                else:
                    _output = self._sample_from_trace_logits(state.logits_tt, active)
            else:
                vocab = int(self.hparams.vocab_size)
                _output = self._logits_to_host(state.logits_tt, active, vocab)

        if _output is None:
            _output = torch.zeros((active, 1, int(self.hparams.vocab_size)), dtype=torch.float32)

        if _t6 is not None and hasattr(self, '_replay_count') and (self._replay_count <= 3 or self._replay_count % 50 == 0):
            logger.warning("PERF REPLAY #{}: output_read={:.1f}ms", self._replay_count, (_time.perf_counter() - _t6) * 1000)

        # MTP: run speculative decode AFTER reading main output
        if not _is_capture and self.mtp_enabled:
            self._run_mtp_after_trace_replay(state, _output, active, tokens, positions, page_table, kv_cache)

        return _output

    def _mtp_decode_step_tt(
        self,
        *,
        state: _DecodeTraceState,
        kv_cache: list,
    ) -> ttnn.Tensor:
        """MTP decode step using persistent device tensors (TRACE-SAFE).

        All ops are compute kernels that record into the trace command buffer.
        Returns MTP logits on device [1,1,B,vocab].
        """
        batch = int(state.batch)
        hidden = int(self.hparams.hidden_size)
        mtp_layer_idx = int(self.hparams.num_hidden_layers)  # 92

        # 1. Embed: use persistent buffer (uploaded by _copy_mtp_trace_inputs)
        x_embed = state.mtp_embed_tt

        # 2. enorm(embedded), hnorm(hidden_state from main trace)
        enorm_out = _sharded_rms_norm(x_embed, self.mtp_enorm, hidden)
        hnorm_out = _sharded_rms_norm(state.mtp_hidden_tt, self.mtp_hnorm, hidden)

        # 3. Split-matmul projection
        proj_e = ttnn.linear(enorm_out, self.mtp_eh_proj_e_w)
        _dealloc(enorm_out, force=False)
        proj_h = ttnn.linear(hnorm_out, self.mtp_eh_proj_h_w)
        _dealloc(hnorm_out, force=False)
        proj = ttnn.add(proj_e, proj_h)
        _dealloc(proj_e, force=False)
        _dealloc(proj_h, force=False)

        # 4. RoPE from persistent cos/sin (BH 3-tuple, no sin_neg)
        rot_mats = (
            state.mtp_cos_batch_tt,
            state.mtp_sin_batch_tt,
            self.rope["trans_matrix"],
        )

        # 5. MTP decoder layer forward
        x = self.mtp_decoder_layer.forward(
            proj, state.mtp_positions_tt, rot_mats,
            state.mtp_page_table_tt if state.mtp_page_table_tt is not None else state.page_table_tt,
            kv_cache[mtp_layer_idx],
            mode="decode",
            active_batch=batch,
        )
        _dealloc(proj, force=False)

        # 6. shared_head norm + LM head
        x = _sharded_rms_norm(x, self.mtp_shared_head_norm, hidden)
        logits_tt = ttnn.linear(x, self.mtp_shared_head_w)
        _dealloc(x, force=False)

        return logits_tt

    def _run_mtp_after_trace_replay(self, state, result, active, tokens, positions, page_table, kv_cache):
        """Run MTP eagerly after trace replay to produce draft tokens."""
        if not self.mtp_enabled or active > self.mtp_max_batch:
            return
        if state.mtp_hidden_tt is None:
            return  # No hidden state saved (trace captured without MTP)
        try:
            # Get main token IDs from result (logits or token IDs)
            if result.dim() >= 2 and result.shape[-1] > 1:
                main_ids = result.reshape(active, -1).argmax(dim=-1).to(torch.int32)
            else:
                main_ids = result.reshape(-1)[:active].to(torch.int32)
            mtp_positions = positions[:active] + 1
            self._last_draft_token_ids = self._mtp_forward_eager(
                main_token_ids=main_ids,
                hidden_state=state.mtp_hidden_tt,  # trace-owned clone of pre-norm hidden
                mtp_positions=mtp_positions,
                page_table=page_table[:active],
                kv_cache=kv_cache,
            )
            if self._last_draft_token_ids is not None:
                if not hasattr(self, '_mtp_call_count'):
                    self._mtp_call_count = 0
                self._mtp_call_count += 1
                if self._mtp_call_count <= 3 or self._mtp_call_count % 50 == 0:
                    logger.warning("MTP #{}: draft_ids={}", self._mtp_call_count,
                                   self._last_draft_token_ids[:4].tolist())
        except Exception as e:
            logger.warning("MTP after trace replay failed (non-fatal): {}", e)
            import traceback; traceback.print_exc()
            self._last_draft_token_ids = None

    def _capture_decode_trace(
        self,
        *,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        page_table: torch.Tensor,
        kv_cache: list,
        sampling_params: Any | None,
        active: int,
    ) -> _DecodeTraceState:
        """Capture a decode trace for the given batch size."""
        logger.info("[DBG] _capture_decode_trace entry: active={}, tokens.shape={}, page_table.shape={}",
                    active, list(tokens.shape), list(page_table.shape))
        hidden = int(self.hparams.hidden_size)
        is_mesh = _is_mesh_device(self.device)
        mapper = ttnn.ReplicateTensorToMesh(self.device) if is_mesh else None

        # On TG (2D mesh) with multi-user batch, shard batch-dimension inputs (cos/sin,
        # page_table, positions) across DP groups so each group gets only its subset.
        # Required because attention slices QKV per DP group: Q/K have
        # logical batch = batch_per_group (not full batch).
        dp_shard_axis = None
        dp_batch_mapper = mapper  # replicate by default
        if is_mesh:
            tp_axis, _ = _tp_axis_and_size(self.device)
            dp_axis = 1 - (tp_axis if tp_axis is not None else 0)
            dp_size = int(self.device.shape[dp_axis])
            if dp_size > 1 and active > 1 and active % dp_size == 0:
                dp_shard_axis = dp_axis
                mesh_shape = list(self.device.shape)
                dims = [None, None]
                dims[dp_axis] = 0
                dp_batch_mapper = ttnn.ShardTensor2dMesh(
                    self.device, dims=tuple(dims), mesh_shape=mesh_shape,
                )

        # DEBUG: optionally truncate page_table width for trace bug investigation.
        # GLM4_MOE_DEBUG_PT_WIDTH=2 forces width=2 (St=4), matching MML=128 config.
        _debug_pt_width = int(os.environ.get("GLM4_MOE_DEBUG_PT_WIDTH", "") or "0")
        if _debug_pt_width > 0 and page_table.shape[1] > _debug_pt_width:
            logger.warning("DEBUG: truncating page_table from width={} to {} for trace capture",
                           page_table.shape[1], _debug_pt_width)
            page_table = page_table[:, :_debug_pt_width].contiguous()

        # Create persistent input tensors (BEFORE trace capture AND warm-up).
        page_table_tt = ttnn.from_torch(
            page_table.contiguous().clone(),
            device=self.device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=dp_batch_mapper,
        )

        tt_positions, cos_batch, sin_batch = _prepare_decode_rope_and_positions_tt(
            device=self.device, rope=self.rope, positions=positions,
            dp_shard_axis=dp_shard_axis,
        )

        # Pre-allocate persistent token buffer on device (outside trace).
        tokens_tt = ttnn.from_torch(
            tokens.contiguous().clone().to(torch.int32),
            device=self.device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

        # Pre-allocate embedding tensor BEFORE compile warm-up!
        # This ensures the memory allocator state exactly matches during trace capture.
        if self.embed_w is not None:
            # Device embedding: allocate persistent token ID buffer.
            # In-trace: ttnn.embedding(tokens_tt, embed_w) runs on device.
            # On replay: only 4 bytes/user H2D instead of 40KB/user.
            tokens_device = ttnn.from_torch(
                tokens[:, 0].contiguous().to(torch.int32),
                device=self.device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device) if is_mesh else None,
            )
            embed_tt = ttnn.embedding(tokens_device, self.embed_w, layout=ttnn.TILE_LAYOUT)
            embed_tt = ttnn.reshape(embed_tt, [1, 1, active, -1])
        else:
            tokens_device = None
            embed_torch = self.embed_w_cpu[tokens[:, 0].long()]  # [B, hidden]
            embed_tt = ttnn.from_torch(
                embed_torch.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),  # [1, 1, B, hidden]
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device) if is_mesh else None,
            )

        rot_mats = (cos_batch, sin_batch, self.rope["trans_matrix"])
        logger.info("[DBG] _capture_decode_trace: persistent inputs created (embed_tt.shape={}, page_table_tt.shape={}, positions_tt.shape={})",
                    list(embed_tt.shape), list(page_table_tt.shape), list(tt_positions.shape))

        # Reset CCL semaphore counters before compile warm-up to avoid stale state.
        if self.tt_ccl is not None:
            self.tt_ccl.reset_sem_counters()

        # Run forward once (compile warm-up) using the persistent tensors.
        logger.info("  [COMPILE] Starting compile warm-up ({} layers)", self.num_layers_to_run)
        x = embed_tt
        for layer_idx in range(self.num_layers_to_run):
            _t_layer = time.time()
            dl = self.decoder_layers[layer_idx]
            x_next = dl.forward(x, tt_positions, rot_mats, page_table_tt, kv_cache[layer_idx], mode="decode",
                                active_batch=active)
            # Match trace capture exactly: skip deallocation of embed_tt!
            if layer_idx > 0:
                _dealloc(x, force=False)
            x = x_next
            if layer_idx < 5 or layer_idx % 10 == 0 or layer_idx == self.num_layers_to_run - 1:
                logger.info("  [COMPILE] Layer {}/{} ({:.1f}s)", layer_idx + 1, self.num_layers_to_run, time.time() - _t_layer)

        x = _sharded_rms_norm(x, self.final_norm, int(self.hparams.hidden_size))
        logits_tt = ttnn.linear(x, self.lm_head_w)

        # DEBUG: Check compile warm-up logits
        if int(os.environ.get("GLM4_MOE_DEBUG_TRACE_VERIFY", "0")) >= 5:
            try:
                if _is_mesh_device(self.device):
                    _wu_dev = ttnn.get_device_tensors(logits_tt)[0]
                    _wu_logits = ttnn.to_torch(_wu_dev).float()
                else:
                    _wu_logits = ttnn.to_torch(logits_tt).float()
                _wu_flat = _wu_logits.reshape(-1)
                _wu_top_val, _wu_top_idx = _wu_flat.topk(5)
                logger.warning("DEBUG WARMUP LOGITS: top5_ids={} top5_vals={} stats=(mean={:.2f}, std={:.2f}, max={:.2f})",
                    _wu_top_idx.tolist(), [f'{v:.2f}' for v in _wu_top_val.tolist()],
                    _wu_flat.mean().item(), _wu_flat.std().item(), _wu_flat.max().item())
            except Exception as e:
                logger.error("DEBUG WARMUP LOGITS: failed: {}", e)

        # On TG mesh, skip sampling in compile warm-up (matches trace capture pattern).
        # On non-TG, include sampling to compile those programs too.
        top1_values_tt = None
        top1_indices_tt = None
        if not is_mesh and sampling_params is not None:
            vocab = int(self.hparams.vocab_size)
            logits_rm = ttnn.to_layout(logits_tt, ttnn.ROW_MAJOR_LAYOUT)
            logits_rm_view = ttnn.slice(logits_rm, [0, 0, 0, 0], [1, 1, active, vocab])
            logits_rm_tight = ttnn.clone(logits_rm_view, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            max_out = ttnn.max(logits_rm_tight, dim=3, keepdim=True)
            if isinstance(max_out, tuple):
                top1_values_tt, top1_indices_tt = max_out
            else:
                top1_values_tt = max_out
                top1_indices_tt = ttnn.argmax(logits_rm_tight, dim=3, keepdim=False, use_multicore=True)
            _dealloc(logits_rm_tight, force=False)
            _dealloc(logits_rm, force=False)

        # Synchronize device to drain all async ops from compile-forward
        # before starting trace capture.
        logger.info("  [COMPILE] synchronize_device (draining async ops)...")
        _t_sync = time.time()
        ttnn.synchronize_device(self.device)
        logger.info("  [COMPILE] synchronize_device done ({:.1f}s)", time.time() - _t_sync)

        # Explicitly free compile-forward outputs.
        _dealloc(x, force=True)
        _dealloc(logits_tt, force=True)
        x = logits_tt = top1_values_tt = top1_indices_tt = None
        ttnn.synchronize_device(self.device)

        # Re-copy persistent inputs before trace capture (like GLM4-Flash)
        # to ensure no inadvertent in-place mutation broke them.
        if self.embed_w is not None and tokens_device is not None:
            # Device embed: re-copy token IDs
            host_tokens = ttnn.from_torch(
                tokens[:, 0].contiguous().to(torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mapper,
            )
            ttnn.copy_host_to_device_tensor(host_tokens, tokens_device)
        else:
            embed_torch = self.embed_w_cpu[tokens[:, 0].long()]
            host_embed = ttnn.from_torch(
                embed_torch.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=mapper,
            )
            ttnn.copy_host_to_device_tensor(host_embed, embed_tt)

        # Now capture trace.
        logger.info("[DBG] _capture_decode_trace: compile warmup done, starting trace capture for batch={}", active)
        logger.info("Capturing decode trace for batch={}", active)
        if self.tt_ccl is not None:
            self.tt_ccl.reset_sem_counters()
        # Retain inter-layer hidden state tensors during trace capture to prevent
        # DRAM address reuse after capture. Without this, freed intermediate
        # addresses get reused by prefill, and trace replay corrupts them.
        # NOTE: We only retain the layer-to-layer x tensors (~91 refs), NOT all
        # intra-layer intermediates — retaining everything OOMs during capture.
        # The _dealloc() calls pass through to ttnn.deallocate during capture
        # (no active retainer), keeping DRAM pressure manageable.
        _trace_retained = []

        trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        logger.info("[DBG] _capture_decode_trace: begin_trace_capture returned trace_id={}", trace_id)

        # Start trace with embedding lookup.
        # Device embed: ttnn.embedding runs inside trace (recorded in command buffer).
        # On replay, updated tokens_device produces new embeddings automatically.
        # Host embed: trace reads from pre-computed embed_tt (updated by H2D copy before replay).
        if self.embed_w is not None and tokens_device is not None:
            x = ttnn.embedding(tokens_device, self.embed_w, layout=ttnn.TILE_LAYOUT)
            x = ttnn.reshape(x, [1, 1, active, -1])
        else:
            x = embed_tt

        for layer_idx in range(self.num_layers_to_run):
            _t_layer = time.time()
            dl = self.decoder_layers[layer_idx]
            x_next = dl.forward(x, tt_positions, rot_mats, page_table_tt, kv_cache[layer_idx], mode="decode",
                                active_batch=active)
            # Retain x instead of deallocating — keeps its DRAM address occupied
            # so the allocator cannot reuse it for prefill buffers.
            if layer_idx > 0:
                _trace_retained.append(x)
            x = x_next
            if layer_idx < 3 or layer_idx % 20 == 0 or layer_idx == self.num_layers_to_run - 1:
                logger.info("  [TRACE_CAPTURE] Layer {}/{} ({:.1f}s)", layer_idx + 1, self.num_layers_to_run, time.time() - _t_layer)

        # MTP: copy hidden state before final_norm for MTP decoder layer input.
        # ttnn.clone() is NOT trace-safe (dispatches outside trace command stream).
        # Use multiply(x, 1.0) which is a compute kernel that records into the trace.
        # This is the same pattern used by WH Galaxy (model_tt.py).
        mtp_hidden_tt = None
        if self.mtp_enabled:
            mtp_hidden_tt = ttnn.multiply(x, 1.0, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        x = _sharded_rms_norm(x, self.final_norm, int(self.hparams.hidden_size))
        logits_tt = ttnn.linear(x, self.lm_head_w)

        # On TG mesh, sampling ops (to_layout, slice, max, argmax) inside trace
        # produce wrong results. Do sampling on host instead (read logits from trace output).
        if not _is_mesh_device(self.device) and sampling_params is not None:
            vocab = int(self.hparams.vocab_size)
            logits_rm = ttnn.to_layout(logits_tt, ttnn.ROW_MAJOR_LAYOUT)
            logits_rm_view = ttnn.slice(logits_rm, [0, 0, 0, 0], [1, 1, active, vocab])
            logits_rm_tight = ttnn.clone(logits_rm_view, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            max_out = ttnn.max(logits_rm_tight, dim=3, keepdim=True)
            if isinstance(max_out, tuple):
                top1_values_tt, top1_indices_tt = max_out
            else:
                top1_values_tt = max_out
                top1_indices_tt = ttnn.argmax(logits_rm_tight, dim=3, keepdim=False, use_multicore=True)
            _dealloc(logits_rm_tight, force=False)
            _dealloc(logits_rm, force=False)

        ttnn.end_trace_capture(self.device, trace_id, cq_id=0)
        logger.info("Decode trace captured for batch={}, retained {} intermediates", active, len(_trace_retained))

        # TG mesh: logits_tt for host-side sampling; non-TG: use on-device sampling results
        use_logits_output = _is_mesh_device(self.device)
        return _DecodeTraceState(
            trace_id=trace_id,
            batch=active,
            page_table_width=int(page_table.shape[1]),
            tokens_tt=tokens_device if (self.embed_w is not None) else None,
            positions_tt=tt_positions,
            cos_batch_tt=cos_batch,
            sin_batch_tt=sin_batch,
            trans_matrix_tt=self.rope["trans_matrix"],
            page_table_tt=page_table_tt,
            logits_tt=logits_tt if use_logits_output else (logits_tt if sampling_params is None else None),
            top1_values_tt=top1_values_tt if not use_logits_output else None,
            top1_indices_tt=top1_indices_tt if not use_logits_output else None,
            embed_tt=embed_tt,
            retained_intermediates=_trace_retained,
            mtp_hidden_tt=mtp_hidden_tt,
        )

    def _update_trace_inputs(
        self,
        state: _DecodeTraceState,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        page_table: torch.Tensor,
    ) -> None:
        """Update persistent trace input tensors with new values."""
        is_mesh = _is_mesh_device(self.device)
        mapper = ttnn.ReplicateTensorToMesh(self.device) if is_mesh else None
        batch = int(tokens.shape[0])
        logger.info("[DBG] _update_trace_inputs: batch={}, tokens.shape={}, positions.shape={}, page_table.shape={}",
                    batch, list(tokens.shape), list(positions.shape), list(page_table.shape))

        # On TG with multi-user batch, batch-dimension inputs (positions, cos/sin,
        # page_table) were sharded across DP groups during trace capture.
        # Must use the same sharding here for copy_host_to_device_tensor.
        dp_batch_mapper = mapper  # for page_table (dim=0), positions (dim=0)
        rope_mapper = mapper      # for cos/sin (dim=1)
        if is_mesh:
            tp_axis, _ = _tp_axis_and_size(self.device)
            dp_axis = 1 - (tp_axis if tp_axis is not None else 0)
            dp_size = int(self.device.shape[dp_axis])
            if dp_size > 1 and batch > 1 and batch % dp_size == 0:
                mesh_shape = list(self.device.shape)
                dims = [None, None]
                dims[dp_axis] = 0
                dp_batch_mapper = ttnn.ShardTensor2dMesh(
                    self.device, dims=tuple(dims), mesh_shape=mesh_shape,
                )
                rope_dims = [None, None]
                rope_dims[dp_axis] = 1
                rope_mapper = ttnn.ShardTensor2dMesh(
                    self.device, dims=tuple(rope_dims), mesh_shape=mesh_shape,
                )

        # Update embedding input.
        if self.embed_w is not None and state.tokens_tt is not None:
            # Device embed: only copy 4-byte token IDs (not 40KB embedding vectors)
            host_tokens = ttnn.from_torch(
                tokens[:, 0].contiguous().to(torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mapper,
            )
            ttnn.copy_host_to_device_tensor(host_tokens, state.tokens_tt)
        elif state.embed_tt is not None:
            embed_torch = self.embed_w_cpu[tokens[:, 0].long()]  # [B, hidden]
            host_embed = ttnn.from_torch(
                embed_torch.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),  # [1, 1, B, hidden]
                dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=mapper,
        )
        ttnn.copy_host_to_device_tensor(host_embed, state.embed_tt)


        # Update positions (host tensor, then copy).
        host_positions = ttnn.from_torch(
            positions.view(-1).contiguous().to(torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=dp_batch_mapper,
        )
        ttnn.copy_host_to_device_tensor(host_positions, state.positions_tt)

        # Update cos/sin for new positions.
        cos_host = self.rope["cos_matrix_host"]
        sin_host = self.rope["sin_matrix_host"]
        rope_dim = int(cos_host.shape[3])
        positions_cpu = positions.to(torch.long).clamp(min=0, max=int(cos_host.shape[2]) - 1)
        cos_batch_t = cos_host[0, 0, positions_cpu, :].reshape(1, batch, 1, rope_dim).to(torch.bfloat16)
        sin_batch_t = sin_host[0, 0, positions_cpu, :].reshape(1, batch, 1, rope_dim).to(torch.bfloat16)

        host_cos = ttnn.from_torch(
            cos_batch_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            mesh_mapper=rope_mapper,
        )
        host_sin = ttnn.from_torch(
            sin_batch_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            mesh_mapper=rope_mapper,
        )
        ttnn.copy_host_to_device_tensor(host_cos, state.cos_batch_tt)
        ttnn.copy_host_to_device_tensor(host_sin, state.sin_batch_tt)

        # DEBUG: truncate page_table to match capture-time width (if debug enabled).
        _debug_pt_width = int(os.environ.get("GLM4_MOE_DEBUG_PT_WIDTH", "") or "0")
        if _debug_pt_width > 0 and page_table.shape[1] > _debug_pt_width:
            page_table = page_table[:, :_debug_pt_width].contiguous()

        # Update page table (host tensor, then copy).
        host_pt = ttnn.from_torch(
            page_table.contiguous().clone(),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=dp_batch_mapper,
        )
        ttnn.copy_host_to_device_tensor(host_pt, state.page_table_tt)

    def _invalidate_decode_traces(self) -> None:
        """Mark decode traces as stale (legacy path, kept for PRESERVE_TRACE=1 compat).

        With PRESERVE_TRACE=0 (recommended), prefill() calls _release_all_decode_traces()
        directly and this method is never reached.
        """
        if not self._decode_trace_states:
            return
        self._decode_traces_stale = True
        logger.info("Decode traces marked stale")

    def _release_all_decode_traces(self) -> None:
        """Release ALL bucket decode traces and deallocate trace output tensors."""
        if not self._decode_trace_states:
            return
        ttnn.synchronize_device(self.device)
        for bucket, state in list(self._decode_trace_states.items()):
            if state.trace_id is not None:
                try:
                    ttnn.release_trace(self.device, state.trace_id)
                except Exception:
                    pass
                state.trace_id = None
            for t in (
                state.logits_tt, state.top1_values_tt, state.top1_indices_tt,
                state.tokens_tt, state.positions_tt, state.cos_batch_tt,
                state.sin_batch_tt, state.trans_matrix_tt, state.page_table_tt,
            ):
                if t is not None:
                    try:
                        _dealloc(t, force=True)
                    except Exception:
                        pass
            # Free retained trace intermediates (dealloc was suppressed during capture)
            for t in state.retained_intermediates:
                try:
                    _dealloc(t, force=True)
                except Exception:
                    pass
            state.retained_intermediates.clear()
        self._decode_trace_states.clear()
        self._decode_traces_stale = False
        self._post_prefill_eager_remaining = 0
        # Reset CCL semaphore counters — decode trace replay advances counters,
        # and the next prefill's all_reduce needs them starting at 0.
        if self.tt_ccl is not None:
            self.tt_ccl.reset_sem_counters()
        ttnn.synchronize_device(self.device)

    # -------------------------------------------------------------------
    # MTP (Multi-Token Prediction)
    # -------------------------------------------------------------------

    def _mtp_forward_eager(
        self,
        *,
        main_token_ids: torch.Tensor,     # [B] int32 — main model's predicted token IDs
        hidden_state: ttnn.Tensor,         # [1,1,B,hidden] TILE on device — pre-final_norm hidden
        mtp_positions: torch.Tensor,       # [B] int32 (= main start_pos + 1)
        page_table: torch.Tensor,          # [B,W] int32
        kv_cache: list,                    # full list including kv_cache[mtp_layer_idx]
    ) -> torch.Tensor:
        """Run MTP layer eagerly. Returns draft_token_ids [B] int32 on CPU."""
        if not self.mtp_enabled:
            return None
        batch = int(main_token_ids.shape[0])
        if batch > self.mtp_max_batch:
            return None
        hidden = int(self.hparams.hidden_size)  # 5120
        is_mesh = _is_mesh_device(self.device)
        mtp_layer_idx = int(self.hparams.num_hidden_layers)  # 92

        # 1. Embed main model's predicted tokens (HOST-SIDE)
        embed_torch = self.embed_w_cpu[main_token_ids.long()]  # [batch, hidden]
        hidden_batch = int(hidden_state.shape[-2])
        if batch < hidden_batch:
            pad = torch.zeros(hidden_batch - batch, hidden, dtype=embed_torch.dtype)
            embed_torch = torch.cat([embed_torch, pad], dim=0)
        x_embed = ttnn.from_torch(
            embed_torch.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device) if is_mesh else None,
        )

        # 2. enorm(embedded), hnorm(hidden_state)
        enorm_out = _sharded_rms_norm(x_embed, self.mtp_enorm, hidden)
        _dealloc(x_embed, force=False)
        hnorm_out = _sharded_rms_norm(hidden_state, self.mtp_hnorm, hidden)

        # 3. Split-matmul projection (avoids ttnn.concat which fails on TG mesh)
        proj_e = ttnn.linear(enorm_out, self.mtp_eh_proj_e_w)
        _dealloc(enorm_out, force=False)
        proj_h = ttnn.linear(hnorm_out, self.mtp_eh_proj_h_w)
        _dealloc(hnorm_out, force=False)
        proj = ttnn.add(proj_e, proj_h)
        _dealloc(proj_e, force=False)
        _dealloc(proj_h, force=False)

        # 4. Prepare RoPE for MTP positions (= main_position + 1)
        # BH uses dynamic dp_axis based on mesh topology
        from models.demos.glm4_moe.tt.layer_weights import _tp_axis_and_size
        tp_ax, _ = _tp_axis_and_size(self.device)
        dp_ax = 1 - tp_ax
        dp_shard_axis = None
        dp_batch_mapper = ttnn.ReplicateTensorToMesh(self.device) if is_mesh else None
        if is_mesh:
            mesh_shape = list(self.device.shape)
            dp_size = mesh_shape[dp_ax]
            if batch > 1 and batch % dp_size == 0:
                dp_shard_axis = dp_ax
                dp_batch_mapper = ttnn.ShardTensor2dMesh(
                    self.device, dims=(None if dp_ax == 1 else 0, 0 if dp_ax == 1 else None),
                    mesh_shape=mesh_shape,
                )

        mtp_pos_clamped = mtp_positions[:batch].to(torch.int32).clamp(
            min=0, max=max(0, int(self.max_seq_len) - 1)
        )
        # BH returns 3 values (no sin_neg)
        tt_positions, cos_batch, sin_batch = _prepare_decode_rope_and_positions_tt(
            device=self.device, rope=self.rope, positions=mtp_pos_clamped,
            dp_shard_axis=dp_shard_axis,
        )

        # Page table for MTP
        pt = page_table[:batch].to(torch.int32).contiguous()
        page_table_tt = ttnn.from_torch(
            pt, device=self.device, dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=dp_batch_mapper,
        )

        rot_mats = (cos_batch, sin_batch, self.rope["trans_matrix"])

        # 5. Run MTP decoder layer
        x = self.mtp_decoder_layer.forward(
            proj, tt_positions, rot_mats, page_table_tt,
            kv_cache[mtp_layer_idx],
            mode="decode",
            active_batch=batch,
        )
        _dealloc(proj, force=False)

        # 6. shared_head: norm + LM head
        x = _sharded_rms_norm(x, self.mtp_shared_head_norm, hidden)
        logits_tt = ttnn.linear(x, self.mtp_shared_head_w)
        _dealloc(x, force=False)

        # 7. Host-side argmax
        draft_token_ids = self._host_argmax_from_trace_logits(
            logits_tt, hidden_batch, int(self.hparams.vocab_size)
        )

        # Cleanup
        _dealloc(logits_tt, force=False)
        _dealloc(tt_positions, force=False)
        _dealloc(cos_batch, force=False)
        _dealloc(sin_batch, force=False)
        _dealloc(page_table_tt, force=False)

        return draft_token_ids[:batch]

    # -------------------------------------------------------------------
    # Prefill
    # -------------------------------------------------------------------

    @torch.no_grad()
    def prefill(
        self,
        *,
        tokens: torch.Tensor,
        prompt_lens: list[int],
        page_table: torch.Tensor,
        kv_cache: list,
        start_pos: torch.Tensor | None = None,
        seq_pad_multiple: int = 128,
    ) -> torch.Tensor:
        """Compute logits for the last prompt token for each request and fill KV caches.

        Processes each request independently (B loop). For long prompts (>16K tokens),
        uses chunked prefill to avoid activation memory OOM.

        Supports prefix caching: when start_pos[i] > 0, the first start_pos[i] tokens
        are already in the KV cache. We skip embedding/processing those tokens and
        offset RoPE positions accordingly.

        Args:
            tokens: [B, S] int32
            prompt_lens: length B, actual prompt lengths
            page_table: [B, W] int32
            kv_cache: list of [cache_k, cache_v] per layer
            start_pos: [B] int32, number of cached tokens per request (0 = no caching)
            seq_pad_multiple: pad prompt lengths to this multiple

        Returns:
            Logits [B, 1, vocab] as torch float32
        """
        # Release decode traces before prefill to avoid buffer corruption.
        # Prefill allocates device buffers that can overlap with trace-owned buffers,
        # producing garbled output on subsequent trace replays.
        # With synchronize_device before trace capture (line ~931), re-capture is safe.
        #
        # Phase 1 optimization: when GLM4_MOE_PRESERVE_TRACE_AFTER_PREFILL=1, skip the
        # expensive release+synchronize and just mark traces stale. The first decode
        # after prefill runs eagerly (no trace), then traces are lazily recaptured.
        # This saves 10-14s on TTFT by eliminating the release/recapture blocking path.
        _preserve = os.environ.get("GLM4_MOE_PRESERVE_TRACE_AFTER_PREFILL", "1") == "1"
        if _preserve and self._decode_trace_states:
            self._invalidate_decode_traces()
        else:
            self._release_all_decode_traces()

        if tokens.ndim != 2:
            raise ValueError(f"expected tokens [B,S], got {tuple(tokens.shape)}")
        if page_table.ndim != 2:
            raise ValueError(f"expected page_table [B,W], got {tuple(page_table.shape)}")
        batch, seq_total = tokens.shape
        if len(prompt_lens) != int(batch):
            raise ValueError(f"prompt_lens length {len(prompt_lens)} != batch {batch}")

        hidden = int(self.hparams.hidden_size)
        vocab = int(self.hparams.vocab_size)
        rope_dim = int(self.hparams.head_dim * self.hparams.partial_rotary_factor)
        pad_multiple = max(128, int(seq_pad_multiple))
        is_mesh = _is_mesh_device(self.device)

        # Prefill chunk size (16K tokens, matching DSv3 pattern for activation memory).
        PREFILL_CHUNK_SIZE = int(os.environ.get("GLM4_MOE_PREFILL_CHUNK_SIZE", "").strip() or "16384")

        if start_pos is None:
            start_pos = torch.zeros(int(batch), dtype=torch.int32)

        out_logits: list[torch.Tensor] = []

        # Batched prefill: process multiple users in one forward pass.
        # Gated behind env var — falls back to sequential if disabled or mixed lengths.
        _batched_prefill = os.environ.get("GLM4_MOE_BATCHED_PREFILL", "0").strip() == "1"
        _max_batch_prefill = int(os.environ.get("GLM4_MOE_MAX_BATCHED_PREFILL", "16"))
        if _batched_prefill and 1 < int(batch) <= _max_batch_prefill:
            # Check if all users have same padded length (required for Phase 1)
            _all_cached = [max(0, min(int(start_pos[i].item()), int(prompt_lens[i]) - 1)) for i in range(int(batch))]
            _all_new = [int(prompt_lens[i]) - _all_cached[i] for i in range(int(batch))]
            _all_padded = [((n + pad_multiple - 1) // pad_multiple) * pad_multiple for n in _all_new]
            _unique_padded = set(_all_padded)
            if len(_unique_padded) == 1 and all(c == 0 for c in _all_cached):
                # All same length, no prefix caching — use batched path
                return self._prefill_batched(
                    tokens=tokens, prompt_lens=prompt_lens, page_table=page_table,
                    kv_cache=kv_cache, start_pos=start_pos, padded_len=_all_padded[0],
                    hidden=hidden, vocab=vocab, rope_dim=rope_dim,
                    PREFILL_CHUNK_SIZE=PREFILL_CHUNK_SIZE,
                )
            else:
                logger.info("Batched prefill: mixed lengths or prefix cache, falling back to sequential")

        for i in range(int(batch)):
            prompt_len = int(prompt_lens[i])
            if prompt_len <= 0:
                out_logits.append(torch.zeros((1, vocab), dtype=torch.float32))
                continue

            num_cached = int(start_pos[i].item()) if start_pos is not None else 0
            num_cached = max(0, min(num_cached, prompt_len - 1))  # Must process at least 1 token
            num_new_tokens = prompt_len - num_cached

            # Pad the NEW tokens (not the full prompt) to pad_multiple.
            padded_new_len = ((num_new_tokens + pad_multiple - 1) // pad_multiple) * pad_multiple
            padded_new_len = min(padded_new_len, int(self.max_seq_len) - num_cached)

            # Extract only the new (uncached) tokens.
            new_token_ids = tokens[i, num_cached:prompt_len].to(torch.int32).cpu()
            input_padded = torch.zeros((1, padded_new_len), dtype=torch.int32)
            input_padded[0, :num_new_tokens] = new_token_ids

            logger.info(
                "prefill user {}: prompt_len={}, num_cached={}, new_tokens={}, padded_new={}, start_pos_raw={}",
                i, prompt_len, num_cached, num_new_tokens, padded_new_len,
                int(start_pos[i].item()) if start_pos is not None else "None",
            )

            # Page table for this request.
            page_row = page_table[i : i + 1, :].to(torch.int32)
            page_table_tt = ttnn.from_torch(
                page_row,
                device=self.device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device) if is_mesh else None,
            )

            # RoPE tables: offset by num_cached so positions are correct.
            # For prefix caching, new tokens start at position num_cached.
            rope_start = num_cached
            rope_end = num_cached + padded_new_len
            rope_slices_owned = True
            cos_matrix = ttnn.slice(
                self.rope["cos_matrix"], [0, 0, rope_start, 0], [1, 1, rope_end, rope_dim]
            )
            sin_matrix = ttnn.slice(
                self.rope["sin_matrix"], [0, 0, rope_start, 0], [1, 1, rope_end, rope_dim]
            )

            # DEBUG: sync checkpoints for prefill pipeline
            _dbg = os.environ.get("GLM4_MOE_DEBUG_SYNC", "0") != "0"
            def _psync(label):
                if _dbg:
                    import sys
                    print(f"  [DEBUG PREFILL-MODEL] {label} ...", flush=True, file=sys.stderr)
                    ttnn.synchronize_device(self.device)
                    print(f"  [DEBUG PREFILL-MODEL] {label} OK", flush=True, file=sys.stderr)

            _psync("after page_table + rope_slice")

            # Embedding: only embed the NEW tokens (cached tokens already in KV cache).
            embed_torch = self.embed_w_cpu[input_padded[0].long()]  # [padded_new_len, hidden]
            x = ttnn.from_torch(
                embed_torch.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),  # [1, 1, padded_new_len, hidden]
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device) if is_mesh else None,
            )
            _psync("after embedding (host)")

            rot_mats = (cos_matrix, sin_matrix, self.rope["trans_matrix"])

            _psync("before layer loop")

            # Build chunk_page_table for the new tokens' KV cache region.
            # chunk_start_idx is the absolute position where new tokens start in the KV cache.
            chunk_start_idx = num_cached if num_cached > 0 else None
            block_size = kv_cache[0][0].shape[2]  # block_size from KV cache shape

            if chunk_start_idx is not None:
                # Build chunk page table: pages covering [num_cached, num_cached + padded_new_len)
                start_block = num_cached // block_size
                end_block = (num_cached + padded_new_len + block_size - 1) // block_size
                chunk_page_table_torch = page_row[:, start_block:end_block]
                chunk_page_table_tt = ttnn.from_torch(
                    chunk_page_table_torch.to(torch.int32),
                    device=self.device,
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.device) if is_mesh else None,
                )
            else:
                chunk_page_table_tt = None

            # Chunked prefill: process PREFILL_CHUNK_SIZE tokens at a time.
            if padded_new_len > PREFILL_CHUNK_SIZE:
                x = self._prefill_chunked(
                    x=x, padded_len=padded_new_len, prompt_len=num_new_tokens,
                    page_table_tt=page_table_tt, page_row=page_row,
                    kv_cache=kv_cache,
                    rot_mats=rot_mats, user_id=i,
                    chunk_size=PREFILL_CHUNK_SIZE,
                    chunk_start_idx=chunk_start_idx,
                )
            else:
                # Single-pass prefill.
                for layer_idx in range(self.num_layers_to_run):
                    dl = self.decoder_layers[layer_idx]
                    x_next = dl.forward(
                        x, None, rot_mats, page_table_tt, kv_cache[layer_idx], mode="prefill",
                        chunk_page_table=chunk_page_table_tt,
                        chunk_start_idx=chunk_start_idx,
                    )
                    _dealloc(x, force=False)
                    x = x_next

            # Extract last token logits. num_new_tokens is relative to x (new tokens only).
            x_last = ttnn.slice(x, [0, 0, num_new_tokens - 1, 0], [1, 1, num_new_tokens, hidden])
            _dealloc(x, force=False)

            x_last = _sharded_rms_norm(x_last, self.final_norm, int(self.hparams.hidden_size))
            logits_tt = ttnn.linear(x_last, self.lm_head_w)

            # Assemble vocab-sharded logits on host (NOT all_gather — all_gather
            # corrupts shard ordering on 2D mesh, inflating Chinese token scores).
            if self.lm_head_sharded_vocab and _is_mesh_device(self.device):
                shards = ttnn.get_device_tensors(logits_tt)
                num_shards = len(shards)
                tp_size = self.lm_head_tp_size
                if num_shards == tp_size:
                    tp_shards = list(shards)
                elif num_shards > tp_size:
                    tp_axis = self.lm_head_tp_axis if self.lm_head_tp_axis is not None else 0
                    mesh_cols = int(self.device.shape[1])
                    dp_stride = mesh_cols if tp_axis == 0 else 1
                    tp_shards = [shards[i * dp_stride] for i in range(tp_size)]
                else:
                    tp_shards = list(shards)
                logits_shards = [ttnn.to_torch(t.cpu())[..., :int(t.shape[-1])] for t in tp_shards]
                logits_torch = torch.cat(logits_shards, dim=-1)[..., :vocab]
                logits_flat = logits_torch.reshape(-1, vocab)
            else:
                logits_torch = _tt_to_torch_for_vllm_output(tensor=logits_tt, device=self.device)
                logits_torch = logits_torch[..., :vocab]
                logits_flat = logits_torch.reshape(-1, vocab)
            logits_i = logits_flat.to(dtype=torch.float32).cpu()
            out_logits.append(logits_i)

            _dealloc(logits_tt, force=False)
            _dealloc(x_last, force=False)
            if rope_slices_owned:
                _dealloc(cos_matrix, force=False)
                _dealloc(sin_matrix, force=False)
            _dealloc(page_table_tt, force=False)
            if chunk_page_table_tt is not None:
                _dealloc(chunk_page_table_tt, force=False)

            # Reset CCL semaphore counters after each prefill to keep state consistent.
            if self.tt_ccl is not None:
                self.tt_ccl.reset_sem_counters()

        return torch.stack(out_logits, dim=0)  # [B, 1, vocab]

    def _prefill_batched(
        self,
        *,
        tokens: torch.Tensor,
        prompt_lens: list[int],
        page_table: torch.Tensor,
        kv_cache: list,
        start_pos: torch.Tensor,
        padded_len: int,
        hidden: int,
        vocab: int,
        rope_dim: int,
        PREFILL_CHUNK_SIZE: int,
    ) -> torch.Tensor:
        """Process multiple users' prefill in one batched forward pass.

        All matmuls and MoE ops are token-wise — concatenating U users' T-length
        embeddings into [1, 1, U*T, hidden] just creates a larger matmul.
        Only attention needs user-awareness (per-user KV cache fill + causal mask).

        Requires: all users have same padded_len and start_pos=0 (no prefix cache).
        """
        batch = int(tokens.shape[0])
        is_mesh = _is_mesh_device(self.device)
        mapper = ttnn.ReplicateTensorToMesh(self.device) if is_mesh else None

        logger.info("Batched prefill: {} users × {} tokens = {} total", batch, padded_len, batch * padded_len)

        # 1. Concatenate embeddings for all users: [U*padded_len, hidden]
        embed_parts = []
        for i in range(batch):
            prompt_len = int(prompt_lens[i])
            tok_ids = tokens[i, :prompt_len].to(torch.int32).cpu()
            padded = torch.zeros(padded_len, dtype=torch.int32)
            padded[:prompt_len] = tok_ids
            embed_parts.append(self.embed_w_cpu[padded.long()])
        embed_cat = torch.cat(embed_parts, dim=0)  # [U*padded_len, hidden]
        x = ttnn.from_torch(
            embed_cat.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),  # [1, 1, U*T, hidden]
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

        # 2. Page table for all users: [U, W]
        page_table_tt = ttnn.from_torch(
            page_table[:batch].to(torch.int32),
            device=self.device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

        # 3. RoPE: cover the full concatenated sequence length.
        # All users start at position 0. The concatenated tensor has U*padded_len tokens.
        # RoPE positions repeat for each user: [0..T-1, 0..T-1, ...]. For causal SDPA
        # with non-paged attention (first pass), this is correct since RoPE is applied
        # per-token and the causal mask prevents cross-user attention.
        # For paged attention (chunked prefill), the RoPE is applied to the chunk tokens
        # which are always within [0..T-1] per user.
        total_seq = batch * padded_len
        cos_matrix = ttnn.slice(self.rope["cos_matrix"], [0, 0, 0, 0], [1, 1, padded_len, rope_dim])
        sin_matrix = ttnn.slice(self.rope["sin_matrix"], [0, 0, 0, 0], [1, 1, padded_len, rope_dim])
        rot_mats = (cos_matrix, sin_matrix, self.rope["trans_matrix"])

        # 4. Forward through all layers
        # For Phase 1: process as flat [1,1,U*T,hidden]. Attention handles per-user
        # KV fill via user_id parameter. SDPA sees the full concatenated sequence
        # with causal masking (cross-user attention is masked by causal constraint
        # since user i's tokens come before user i+1's in the concatenated sequence).
        #
        # NOTE: This is correct ONLY when all users have the same padded_len and
        # start from position 0, because the causal mask naturally separates users
        # (user 0 tokens [0..T-1] can't attend to user 1 tokens [T..2T-1] since
        # the latter have higher positions in the causal ordering).
        for layer_idx in range(self.num_layers_to_run):
            dl = self.decoder_layers[layer_idx]
            x_next = dl.forward(
                x, None, rot_mats, page_table_tt, kv_cache[layer_idx], mode="prefill",
                user_id=0, batch_size=batch,
            )
            _dealloc(x, force=False)
            x = x_next

        # 5. Extract last token for each user and compute logits.
        # x is [1, 1, U*padded_len, hidden]. Use actual tensor dims for safe slicing.
        x_dim2 = int(x.shape[-2])
        x_dim3 = int(x.shape[-1])
        out_logits = []
        for i in range(batch):
            prompt_len = int(prompt_lens[i])
            offset = i * padded_len + prompt_len - 1
            assert offset + 1 <= x_dim2, f"Slice offset {offset+1} > tensor dim {x_dim2}"
            x_last = ttnn.slice(x, [0, 0, offset, 0], [1, 1, offset + 1, x_dim3])
            x_last_normed = _sharded_rms_norm(x_last, self.final_norm, hidden)
            logits_tt = ttnn.linear(x_last_normed, self.lm_head_w)

            # Assemble vocab-sharded logits on host
            if self.lm_head_sharded_vocab and _is_mesh_device(self.device):
                shards = ttnn.get_device_tensors(logits_tt)
                tp_size = self.lm_head_tp_size
                num_shards = len(shards)
                if num_shards == tp_size:
                    tp_shards = list(shards)
                elif num_shards > tp_size:
                    tp_axis = self.lm_head_tp_axis if self.lm_head_tp_axis is not None else 0
                    mesh_cols = int(self.device.shape[1])
                    dp_stride = mesh_cols if tp_axis == 0 else 1
                    tp_shards = [shards[j * dp_stride] for j in range(tp_size)]
                else:
                    tp_shards = list(shards)
                logits_shards = [ttnn.to_torch(t.cpu())[..., :int(t.shape[-1])] for t in tp_shards]
                logits_torch = torch.cat(logits_shards, dim=-1)[..., :vocab]
                logits_flat = logits_torch.reshape(-1, vocab)
            else:
                logits_torch = _tt_to_torch_for_vllm_output(tensor=logits_tt, device=self.device)
                logits_flat = logits_torch[..., :vocab].reshape(-1, vocab)

            out_logits.append(logits_flat.to(dtype=torch.float32).cpu())
            _dealloc(logits_tt, force=False)
            _dealloc(x_last, force=False)
            _dealloc(x_last_normed, force=False)

        _dealloc(x, force=False)
        _dealloc(cos_matrix, force=False)
        _dealloc(sin_matrix, force=False)
        _dealloc(page_table_tt, force=False)

        if self.tt_ccl is not None:
            self.tt_ccl.reset_sem_counters()

        logger.info("Batched prefill complete: {} users", batch)
        return torch.stack(out_logits, dim=0)  # [B, 1, vocab]

    def _prefill_chunked(
        self,
        *,
        x: ttnn.Tensor,
        padded_len: int,
        prompt_len: int,
        page_table_tt: ttnn.Tensor,
        page_row: torch.Tensor,
        kv_cache: list,
        rot_mats: tuple,
        user_id: int,
        chunk_size: int,
        chunk_start_idx: int | None = None,
    ) -> ttnn.Tensor:
        """Run prefill in chunks for long sequences to avoid activation OOM.

        Processes chunk_size tokens at a time through all layers, writing KV cache
        incrementally. Returns the full hidden state [1,1,padded_len,hidden].

        Pre-computes per-chunk RoPE slices, KV-fill page tables, and growing SDPA
        page tables once before the layer loop (same for every layer).

        When chunk_start_idx is set (prefix caching), the absolute KV cache position
        offset is chunk_start_idx + local chunk offset.
        """
        hidden = int(self.hparams.hidden_size)
        num_chunks = (padded_len + chunk_size - 1) // chunk_size
        base_offset = chunk_start_idx if chunk_start_idx is not None else 0

        # Pre-compute per-chunk RoPE slices, page tables, and chunk_start_idx.
        # These are the same for every layer, so compute once outside the layer loop.
        rope_dim = rot_mats[0].shape[-1]
        is_mesh = _is_mesh_device(self.device)

        # Determine block_size from KV cache shape: kv_cache[0] = [keys, values],
        # keys.shape[2] = block_size (number of positions per page/block).
        block_size = kv_cache[0][0].shape[2]

        chunk_infos = []
        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, padded_len)

            # Per-chunk RoPE: slice cos/sin for correct positions within the
            # new-token range. rot_mats are already offset by num_cached in prefill().
            cos_chunk = ttnn.slice(rot_mats[0], [0, 0, start, 0], [1, 1, end, rope_dim])
            sin_chunk = ttnn.slice(rot_mats[1], [0, 0, start, 0], [1, 1, end, rope_dim])
            chunk_rot_mats = (cos_chunk, sin_chunk, rot_mats[2])

            # Per-chunk fill page table: the fill kernel maps input tile 0 -> page_table[0],
            # so for chunk N starting at absolute position base_offset+start, we need
            # page_table entries starting at (base_offset+start) // block_size.
            abs_start = base_offset + start
            abs_end = base_offset + end
            start_page = abs_start // block_size
            chunk_len = end - start
            num_chunk_pages = chunk_len // block_size
            # Clip to actually allocated pages: vLLM allocates ceil(prompt_len/block_size)
            # pages but padded_len can exceed prompt_len, referencing unallocated pages.
            max_valid_pages = (base_offset + prompt_len + block_size - 1) // block_size
            num_chunk_pages = min(num_chunk_pages, max_valid_pages - start_page)
            chunk_page_row = page_row[:, start_page : start_page + num_chunk_pages].to(torch.int32)
            chunk_page_table_tt = ttnn.from_torch(
                chunk_page_row,
                device=self.device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device) if is_mesh else None,
            )

            # Growing SDPA page table: SDPA needs to read all KV from position 0
            # through abs_end-1, so pass pages covering [0, abs_end).
            # NOTE: Do NOT clip to max_valid_pages here — the SDPA op requires
            # K_len >= Q_len + chunk_start_idx, which equals abs_end (the padded end).
            # Pages beyond prompt_len may reference unallocated blocks (index 0 or stale),
            # but the causal mask prevents them from affecting actual token positions.
            end_page = abs_end // block_size
            sdpa_page_row = page_row[:, :end_page].to(torch.int32)
            sdpa_page_table_tt = ttnn.from_torch(
                sdpa_page_row,
                device=self.device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device) if is_mesh else None,
            )

            # Absolute chunk start for SDPA chunked attention
            abs_chunk_start = abs_start if abs_start > 0 else None

            chunk_infos.append({
                "start": start,
                "end": end,
                "rot_mats": chunk_rot_mats,
                "chunk_page_table": chunk_page_table_tt,
                "sdpa_page_table": sdpa_page_table_tt,
                "chunk_start_idx": abs_chunk_start,
            })

        for layer_idx in range(self.num_layers_to_run):
            dl = self.decoder_layers[layer_idx]
            x_next_chunks = []

            for chunk_idx in range(num_chunks):
                ci = chunk_infos[chunk_idx]
                start = ci["start"]
                end = ci["end"]

                x_chunk = ttnn.slice(x, [0, 0, start, 0], [1, 1, end, hidden])

                # Pass chunk-specific RoPE, page tables, and start index
                x_chunk_out = dl.forward(
                    x_chunk, None, ci["rot_mats"],
                    ci["sdpa_page_table"], kv_cache[layer_idx], mode="prefill",
                    chunk_page_table=ci["chunk_page_table"],
                    chunk_start_idx=ci["chunk_start_idx"],
                )
                x_next_chunks.append(x_chunk_out)
                _dealloc(x_chunk, force=False)

            _dealloc(x, force=False)
            if len(x_next_chunks) == 1:
                x = x_next_chunks[0]
            else:
                x = ttnn.concat(x_next_chunks, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                for chunk_t in x_next_chunks:
                    _dealloc(chunk_t, force=False)

        return x
