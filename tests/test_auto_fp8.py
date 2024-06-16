import os
import shutil

import pytest
import safetensors.torch
from transformers import AutoTokenizer

from auto_fp8 import AutoFP8ForCausalLM, BaseQuantizeConfig

MODELS = [
    "facebook/opt-125m",
    "Qwen/Qwen2-0.5B-Instruct",
]

@pytest.mark.parametrize("model_id", MODELS)
def test_dynamic_quantization(model_id):
    quantized_model_dir = model_id.split("/")[-1] + "-fp8-dynamic"

    quantize_config = BaseQuantizeConfig(
        quant_method="fp8", activation_scheme="dynamic"
    )

    model = AutoFP8ForCausalLM.from_pretrained(model_id, quantize_config)
    model.model.to("cpu")

    model.quantize()
    model.save_quantized(quantized_model_dir)

    # Measure checkpoint size and cleanup
    model_size = os.path.getsize(f"{quantized_model_dir}/model.safetensors")
    shutil.rmtree(quantized_model_dir)

    # We expect the model to be < 160MB
    target_size = 160 * (1024 * 1024)
    assert model_size < target_size


@pytest.mark.parametrize("model_id", MODELS)
def test_static_quantization(model_id):
    quantized_model_dir = model_id.split("/")[-1] + "-fp8-static"

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    examples = ["auto-fp8 is an easy-to-use model quantization library"]
    examples = tokenizer(examples, return_tensors="pt")

    quantize_config = BaseQuantizeConfig(quant_method="fp8", activation_scheme="static")

    model = AutoFP8ForCausalLM.from_pretrained(model_id, quantize_config)
    model.model.to("cpu")

    model.quantize(examples)
    model.save_quantized(quantized_model_dir)

    # Measure checkpoint size and cleanup
    model_size = os.path.getsize(f"{quantized_model_dir}/model.safetensors")
    shutil.rmtree(quantized_model_dir)

    # We expect the model to be < 160MB
    target_size = 160 * (1024 * 1024)
    assert model_size < target_size

@pytest.mark.parametrize("model_id", MODELS)
def test_kv_cache_static_quantization(model_id):
    quantized_model_dir = model_id.split("/")[-1] + "-fp8-static-kv"

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    examples = ["auto-fp8 is an easy-to-use model quantization library"]
    examples = tokenizer(examples, return_tensors="pt")

    quantize_config = BaseQuantizeConfig(
        quant_method="fp8",
        activation_scheme="static",
        kv_cache_quant_targets=("k_proj", "v_proj"),
    )

    model = AutoFP8ForCausalLM.from_pretrained(model_id, quantize_config)
    model.model.to("cpu")

    model.quantize(examples)
    model.save_quantized(quantized_model_dir)

    tensors = safetensors.torch.load_file(f"{quantized_model_dir}/model.safetensors")
    proj_linear_count = 0
    output_scale_count = 0
    for name, _ in tensors.items():
        if name.endswith("k_proj") or name.endswith("v_proj"):
            proj_linear_count += 1
        if name.endswith("k_proj.output_scale") or name.endswith("v_proj.output_scale"):
            output_scale_count += 1
    assert proj_linear_count == output_scale_count

    # Measure checkpoint size and cleanup
    model_size = os.path.getsize(f"{quantized_model_dir}/model.safetensors")
    shutil.rmtree(quantized_model_dir)

    # We expect the model to be < 160MB
    target_size = 160 * (1024 * 1024)
    assert model_size < target_size
