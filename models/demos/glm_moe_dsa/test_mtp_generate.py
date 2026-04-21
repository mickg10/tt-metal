#!/usr/bin/env python3
"""Test MTP generation throughput on GLM-5.1 BH Galaxy.

Bypasses vLLM — uses generator directly for accurate MTP throughput measurement.
"""

import os
import sys
import time
import torch
from pathlib import Path
from loguru import logger
from transformers import AutoConfig, AutoTokenizer

os.environ["DEEPSEEK_V3_MTP"] = "1"

import ttnn
from models.demos.glm_moe_dsa.tt.generator import DeepseekGenerator


def main():
    model_path = os.environ.get("DEEPSEEK_V3_HF_MODEL", "/home/mick/models/GLM-5.1")
    cache_dir = os.environ.get("DEEPSEEK_V3_CACHE", "/root/.cache/ttnn/models/glm5_bf16")
    max_seq_len = 4096

    logger.info(f"Model: {model_path}, Cache: {cache_dir}")

    # Register custom model type (not in standard transformers)
    # Import the config from the vLLM-installed package
    try:
        from vllm.transformers_utils.configs.glm_moe_dsa import GlmMoeDsaConfig
        AutoConfig.register("glm_moe_dsa", GlmMoeDsaConfig)
    except Exception as e:
        logger.warning(f"Config registration: {e}")
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Open mesh
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING, ttnn.FabricReliabilityMode.STRICT_INIT)
    mesh_device = ttnn.open_mesh_device(
        ttnn.MeshShape(8, 4),
        dispatch_core_config=ttnn.DispatchCoreConfig(),
        trace_region_size=150000000,
    )
    logger.info(f"Mesh: {mesh_device.get_num_devices()} devices")

    with DeepseekGenerator(
        hf_config=hf_config,
        mesh_device=mesh_device,
        model_path=model_path,
        cache_dir=cache_dir,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        enable_mtp=True,
        enable_trace=True,
    ) as gen:
        prompt = "Count from 1 to 100."
        logger.info(f"Generating with MTP, prompt: {prompt}")

        start = time.perf_counter()
        generations, stats = gen.generate(
            prompts=[prompt],
            max_new_tokens=100,
            early_print_first_user=True,
        )
        elapsed = time.perf_counter() - start

        tokens = len(generations[0])
        tok_s = tokens / elapsed
        logger.info(f"Generated {tokens} tokens in {elapsed:.2f}s = {tok_s:.2f} tok/s")

        output = tokenizer.decode(generations[0], skip_special_tokens=True)
        logger.info(f"Output: {output[:200]}")

        # Print MTP stats
        for k, v in stats.items():
            if isinstance(v, (int, float)):
                logger.info(f"  {k}: {v}")

        logger.info(f"RESULT: {tok_s:.2f} tok/s at bs=1 with MTP")

    ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    main()
