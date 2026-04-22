# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0


import os
from pathlib import Path

import torch
from loguru import logger
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.glm_moe_dsa.utils.abstract_module import AbstractModule
from models.demos.glm_moe_dsa.utils.config_dataclass import FromWeightConfig, LinearConfig, MeshDeviceStub, MulConfig
from models.demos.glm_moe_dsa.utils.config_helpers import (
    COMPUTE_KERNEL_CONFIG_HIFI2,
    COMPUTE_KERNEL_CONFIG_LOFI,
    even_int_div,
    get_dequantized_tensor,
    shard_and_save,
)
from models.demos.glm_moe_dsa.utils.run_config import (
    ModelDecodeConfig,
    ModelPrefillConfig,
    RunDecodeConfig,
    RunPrefillConfig,
    WeightConfig,
)


class Experts(AbstractModule):
    """Experts layer for Mixture-of-Experts (MoE) module."""

    WEIGHT_TORCH_DTYPE = torch.bfloat16

    @classmethod
    def _get_num_experts_per_device(cls, hf_config: PretrainedConfig, mesh_device: ttnn.Device) -> int:
        """Calculate the number of experts per device based on the total number of experts and the device shape."""
        return even_int_div(hf_config.n_routed_experts, mesh_device.get_num_devices())

    @classmethod
    def convert_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dicts: tuple[dict[str, torch.Tensor] | None, ...],
        output_path: Path,
        mesh_device: ttnn.Device,
    ) -> WeightConfig:
        assert hf_config.n_routed_experts % mesh_device.get_num_devices() == 0, (
            f"Number of experts ({hf_config.n_routed_experts}) must be divisible by the number of devices "
            f"({mesh_device.get_num_devices()})"
        )
        (state_dict,) = state_dicts
        assert state_dict is not None

        def _load_expert_weight(hf_name: str) -> torch.Tensor:
            weight_name = f"{hf_name}.weight"
            expert_weights: list[torch.Tensor] = []
            for expert_id in range(hf_config.n_routed_experts):
                full_weight_name = f"experts.{expert_id}.{weight_name}"
                expert_weights.append(
                    get_dequantized_tensor(state_dict, full_weight_name, dtype=cls.WEIGHT_TORCH_DTYPE)
                )

            return torch.stack(expert_weights)

        sparse_mode = os.getenv("GLM5_SPARSE_EXPERTS", "0") == "1"

        if sparse_mode:
            from models.demos.glm_moe_dsa.tt.sparse_experts import convert_expert_weights_sparse

            logger.info("Loading expert weights in SPARSE (WIDTH_SHARDED) format")
            num_experts_per_device = cls._get_num_experts_per_device(hf_config, mesh_device)
            result = {}
            for hf_name, ttnn_name in [
                ("gate_proj", "w1_experts"),
                ("down_proj", "w2_experts"),
                ("up_proj", "w3_experts"),
            ]:
                weights = _load_expert_weight(hf_name).unsqueeze(0).transpose(-1, -2).contiguous()
                dtype = ttnn.bfloat8_b if hf_name == "down_proj" else ttnn.bfloat4_b
                expert_tensors = convert_expert_weights_sparse(
                    weights,
                    num_experts_per_device=num_experts_per_device,
                    mesh_device=mesh_device,
                    dtype=dtype,
                    cache_path=output_path / "sparse",
                    tag=ttnn_name,
                )
                result[ttnn_name] = {"sparse_tensors": expert_tensors}
            return result

        # Dense (default) path: batched interleaved DRAM
        return {
            ttnn_name: {
                "input_tensor_b": shard_and_save(
                    output_path / f"{ttnn_name}.input_tensor_b",
                    _load_expert_weight(hf_name).unsqueeze(0).transpose(-1, -2).contiguous(),
                    shard_dims=(1, 1),
                    mesh_device=mesh_device,
                    dtype=ttnn.bfloat8_b if hf_name == "down_proj" else ttnn.bfloat4_b,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            }
            for hf_name, ttnn_name in [
                ("gate_proj", "w1_experts"),
                ("down_proj", "w2_experts"),
                ("up_proj", "w3_experts"),
            ]
        }

    @classmethod
    def is_device_supported(cls, mesh_device: ttnn.Device) -> bool:
        """
        As we only support 1D tensor parallelism, we only support 1D mesh devices.

        Args:
            mesh_device: The mesh device to check.

        Returns:
            True if the device is supported, False otherwise.
        """
        return mesh_device.shape[1] in (4, 8)

    @classmethod
    def _create_model_config(
        cls, hf_config: PretrainedConfig, mesh_device: ttnn.Device, mode: str
    ) -> ModelPrefillConfig | ModelDecodeConfig:
        num_experts_per_device = cls._get_num_experts_per_device(hf_config, mesh_device)

        # Calculate dimensions
        hidden_size = hf_config.hidden_size
        moe_intermediate_size = hf_config.moe_intermediate_size

        sparse_mode = os.getenv("GLM5_SPARSE_EXPERTS", "0") == "1"

        # Calculate input and output memory configurations
        if mode == "decode":
            input_memory_config = ttnn.L1_MEMORY_CONFIG
            output_memory_config = ttnn.L1_MEMORY_CONFIG
        else:
            input_memory_config = ttnn.DRAM_MEMORY_CONFIG
            output_memory_config = ttnn.DRAM_MEMORY_CONFIG

        # Construct the config
        base_config = {
            "mesh_device": MeshDeviceStub(mesh_device.shape),
            "sparse_mode": sparse_mode,
            "hidden_size": hidden_size,
            "moe_intermediate_size": moe_intermediate_size,
            "input_memory_config": input_memory_config,
            "output_memory_config": output_memory_config,
            "num_experts_per_device": num_experts_per_device,
        }

        if sparse_mode:
            # Sparse mode: No LinearConfig needed. Expert compute uses DRAMStreamingExpertsMatmul.
            # The "sparse_tensors" placeholders get filled by config merge from convert_weights.
            base_config.update({
                "w1_experts": {"sparse_tensors": None},
                "w2_experts": {"sparse_tensors": None},
                "w3_experts": {"sparse_tensors": None},
            })
            return base_config
            # Dense mode: standard batched matmul with LinearConfig
            base_config.update({
                "w1_experts": LinearConfig(
                    input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                    memory_config=output_memory_config,
                    compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI,
                ),
                "w2_experts": LinearConfig(
                    input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                    memory_config=output_memory_config,
                    compute_kernel_config=COMPUTE_KERNEL_CONFIG_HIFI2,
                ),
                "w3_experts": LinearConfig(
                    input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                    memory_config=output_memory_config,
                    compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI,
                ),
                "mul_experts": MulConfig(
                    memory_config=output_memory_config,
                    input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                ),
            })

        return base_config

    @classmethod
    def decode_model_config(cls, hf_config: PretrainedConfig, mesh_device: ttnn.Device) -> ModelDecodeConfig:
        """Generate decode configuration for this module.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device the model will be placed later on

        Returns:
            ModelPrefillConfig containing operator configurations for prefill mode
        """
        return cls._create_model_config(hf_config, mesh_device, "decode")

    @classmethod
    def prefill_model_config(cls, hf_config: PretrainedConfig, mesh_device: ttnn.Device) -> ModelPrefillConfig:
        """Generate prefill configuration for this module.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device the model will be placed later on

        Returns:
            ModelPrefillConfig containing operator configurations for prefill mode
        """
        return cls._create_model_config(hf_config, mesh_device, "prefill")

    @classmethod
    def _forward(cls, x: ttnn.Tensor, cfg: RunDecodeConfig) -> ttnn.Tensor:
        assert x.memory_config() == cfg["input_memory_config"], f"{x.memory_config()} != {cfg['input_memory_config']}"

        _, _, num_tokens, hidden_size = x.shape

        debug_experts = os.getenv("DEEPSEEK_V3_DEBUG_EXPERTS") == "1" and num_tokens > 8192

        def _log_expert_stats(name: str, tensor: ttnn.Tensor) -> None:
            if not debug_experts:
                return
            try:
                mesh_device = cfg.get("mesh_device")
                if mesh_device is not None:
                    tensor_torch = ttnn.to_torch(
                        tensor,
                        mesh_composer=ttnn.ConcatMesh2dToTensor(
                            mesh_device, dims=(-2, -1), mesh_shape=mesh_device.shape
                        ),
                    )
                else:
                    tensor_torch = ttnn.to_torch(tensor)
                finite_mask = torch.isfinite(tensor_torch)
                numel = tensor_torch.numel()
                finite_count = finite_mask.sum().item()
                nan_count = torch.isnan(tensor_torch).sum().item()
                inf_count = torch.isinf(tensor_torch).sum().item()
                logger.info(
                    f"DEBUG experts {name}: shape={tensor_torch.shape}, "
                    f"mean={tensor_torch.mean():.4f}, std={tensor_torch.std():.4f}, "
                    f"max={tensor_torch.abs().max():.4f}, "
                    f"finite={finite_count}/{numel}, nan={nan_count}, inf={inf_count}"
                )
            except Exception as exc:
                logger.warning(f"DEBUG experts {name}: failed to extract stats: {exc}")

        # SPARSE EXPERT: use pre-sliced per-expert weight tensors
        sparse_expert_hack = os.getenv("GLM5_SPARSE_EXPERT_HACK") == "1"
        if sparse_expert_hack:
            # Pre-slice weights on FIRST call (cached in cfg for subsequent calls)
            if "per_expert_w1" not in cfg:
                logger.info("Pre-slicing expert weights (one-time cost)...")
                w1_w = cfg["w1_experts"].input_tensor_b
                w3_w = cfg["w3_experts"].input_tensor_b
                w2_w = cfg["w2_experts"].input_tensor_b
                num_e = cfg["num_experts_per_device"]
                cfg["per_expert_w1"] = [ttnn.slice(w1_w, [0, e, 0, 0], [1, e+1, w1_w.shape[2], w1_w.shape[3]]) for e in range(num_e)]
                cfg["per_expert_w3"] = [ttnn.slice(w3_w, [0, e, 0, 0], [1, e+1, w3_w.shape[2], w3_w.shape[3]]) for e in range(num_e)]
                cfg["per_expert_w2"] = [ttnn.slice(w2_w, [0, e, 0, 0], [1, e+1, w2_w.shape[2], w2_w.shape[3]]) for e in range(num_e)]
                logger.info(f"Pre-sliced {num_e} experts for w1, w3, w2")
        per_expert_w1 = cfg.get("per_expert_w1")
        per_expert_w3 = cfg.get("per_expert_w3")
        per_expert_w2 = cfg.get("per_expert_w2")
        if sparse_expert_hack and per_expert_w1 is not None:
            # x is [1, 8, T, K] — slice input to expert 0 only
            x_1 = ttnn.slice(x, [0, 0, 0, 0], [1, 1, x.shape[2], x.shape[3]])
            # Use pre-sliced expert 0 weights (no DRAM copy!)
            w1_out = ttnn.linear(x_1, input_tensor_b=per_expert_w1[0],
                                  memory_config=cfg["w1_experts"].memory_config,
                                  compute_kernel_config=cfg["w1_experts"].compute_kernel_config)
            w3_out = ttnn.linear(x_1, input_tensor_b=per_expert_w3[0],
                                  memory_config=cfg["w3_experts"].memory_config,
                                  compute_kernel_config=cfg["w3_experts"].compute_kernel_config)
            ttnn.deallocate(x_1)
            activated = ttnn.mul(w1_out, w3_out, **cfg["mul_experts"])
            ttnn.deallocate(w1_out)
            ttnn.deallocate(w3_out)
            output = ttnn.linear(activated, input_tensor_b=per_expert_w2[0],
                                  memory_config=cfg["w2_experts"].memory_config,
                                  compute_kernel_config=cfg["w2_experts"].compute_kernel_config)
            ttnn.deallocate(activated)
            # Repeat output to match expected [1, 8, T, hidden]
            output = ttnn.repeat(output, ttnn.Shape((1, cfg["num_experts_per_device"], 1, 1)))
            output = ttnn.permute(output, (1, 0, 2, 3))
            output = ttnn.reshape(output, shape=(1, cfg["num_experts_per_device"], num_tokens, hidden_size))
            return output

        # Gate and up projections
        w1_out = ttnn.linear(x, **cfg["w1_experts"])
        w3_out = ttnn.linear(x, **cfg["w3_experts"])
        _log_expert_stats("w1_out", w1_out)
        _log_expert_stats("w3_out", w3_out)

        # Apply activation and multiply
        activated = ttnn.mul(w1_out, w3_out, **cfg["mul_experts"])
        ttnn.deallocate(w1_out)
        ttnn.deallocate(w3_out)
        _log_expert_stats("activated", activated)

        # Down projection
        output = ttnn.linear(activated, **cfg["w2_experts"])
        ttnn.deallocate(activated)
        _log_expert_stats("w2_out", output)

        # Reshape for output
        output = ttnn.permute(output, (1, 0, 2, 3))
        output = ttnn.reshape(output, shape=(1, cfg["num_experts_per_device"], num_tokens, hidden_size))

        assert output.memory_config() == cfg["output_memory_config"]
        return output

    @classmethod
    def forward_decode(cls, x: ttnn.Tensor, cfg: RunDecodeConfig) -> ttnn.Tensor:
        return cls._forward(x, cfg)

    @classmethod
    def forward_prefill(cls, x: ttnn.Tensor, cfg: RunPrefillConfig) -> ttnn.Tensor:
        return cls._forward(x, cfg)
