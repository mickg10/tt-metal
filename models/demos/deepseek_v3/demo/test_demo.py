# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import pytest

from models.demos.deepseek_v3.demo.demo import load_prompts_from_json, run_demo

MODEL_PATH = Path(os.getenv("DEEPSEEK_V3_HF_MODEL", "/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528"))
CACHE_DIR = Path(os.getenv("DEEPSEEK_V3_CACHE", "/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-Cache/CI"))


@pytest.mark.parametrize(
    "repeat_batches, max_new_tokens, override_num_layers, max_prompts, profile_decode",
    [
        pytest.param(1, 10, 5, 56, False, id="short_demo"),
        pytest.param(1, 13, 5, 1, True, id="profile_decode", marks=pytest.mark.timeout(1800)),
    ],
)
def test_demo(repeat_batches, max_new_tokens, override_num_layers, max_prompts, profile_decode):
    """
    Stress test the DeepSeek v3 demo with prompts loaded from JSON file.
    - short_demo: Quick CI test with 5 layers
    - profile_decode: Profile decode for non-moe and moe layers
    """
    # Path to the external JSON file containing prompts
    json_path = "models/demos/deepseek_v3/demo/test_prompts.json"

    # Load prompts from JSON file
    prompts = load_prompts_from_json(json_path, max_prompts=max_prompts)

    # Run demo
    results = run_demo(
        prompts=prompts,
        model_path=MODEL_PATH,
        cache_dir=CACHE_DIR,
        random_weights=False,
        max_new_tokens=max_new_tokens,
        override_num_layers=override_num_layers,
        repeat_batches=repeat_batches,
        enable_trace=True,
        profile_decode=profile_decode,
        signpost=True,
    )

    # Check output
    assert len(results["generations"][0]["tokens"]) == max_new_tokens
