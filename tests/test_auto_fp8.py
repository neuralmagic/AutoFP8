import os
import shutil

<<<<<<< HEAD
<<<<<<< HEAD
import pytest
=======
>>>>>>> 3ee9283 (Support calibrating kv cache scales)
=======
import pytest
>>>>>>> 2739d61 (Add Qwen test)
import safetensors.torch
from transformers import AutoTokenizer

from auto_fp8 import AutoFP8ForCausalLM, BaseQuantizeConfig

MODELS = [
<<<<<<< HEAD
<<<<<<< HEAD
    ("facebook/opt-125m", 160),
    ("Qwen/Qwen2-0.5B-Instruct", 620),
]

<<<<<<< HEAD
@pytest.mark.parametrize("model_id,target_size", MODELS)
def test_dynamic_quantization(model_id, target_size):
    quantized_model_dir = model_id.split("/")[-1] + "-fp8-dynamic"
=======
def test_dynamic_quantization():
    model_id = "facebook/opt-125m"
    quantized_model_dir = "opt-125m-fp8-dynamic"
>>>>>>> 3ee9283 (Support calibrating kv cache scales)
=======
    "facebook/opt-125m",
    "Qwen/Qwen2-0.5B-Instruct",
=======
    ("facebook/opt-125m", 160),
<<<<<<< HEAD
    ("Qwen/Qwen2-0.5B-Instruct", 600),
>>>>>>> 415c0b7 (Add fixed target sizes)
=======
    ("Qwen/Qwen2-0.5B-Instruct", 620),
>>>>>>> 93c0d54 (Fix proj linear count)
]

@pytest.mark.parametrize("model_id,target_size", MODELS)
def test_dynamic_quantization(model_id, target_size):
    quantized_model_dir = model_id.split("/")[-1] + "-fp8-dynamic"
>>>>>>> 2739d61 (Add Qwen test)

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

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> c3acdee (Switch from output_scale to kv_scale)
    # We expect the quantized model to be a certain size
    target_size = target_size * (1024 * 1024)
    assert model_size < target_size


@pytest.mark.parametrize("model_id,target_size", MODELS)
def test_static_quantization(model_id, target_size):
    quantized_model_dir = model_id.split("/")[-1] + "-fp8-static"
=======
    # We expect the model to be < 160MB
    target_size = 160 * (1024 * 1024)
    assert model_size < target_size


<<<<<<< HEAD
def test_static_quantization():
    model_id = "facebook/opt-125m"
    quantized_model_dir = "opt-125m-fp8-static"
>>>>>>> 3ee9283 (Support calibrating kv cache scales)
=======
@pytest.mark.parametrize("model_id", MODELS)
def test_static_quantization(model_id):
=======
    # We expect the model to be a certain size
    target_size = target_size * (1024 * 1024)
    assert model_size < target_size


@pytest.mark.parametrize("model_id,target_size", MODELS)
def test_static_quantization(model_id, target_size):
>>>>>>> 415c0b7 (Add fixed target sizes)
    quantized_model_dir = model_id.split("/")[-1] + "-fp8-static"
>>>>>>> 2739d61 (Add Qwen test)

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

<<<<<<< HEAD
<<<<<<< HEAD
    # We expect the quantized model to be a certain size
    target_size = target_size * (1024 * 1024)
    assert model_size < target_size

@pytest.mark.parametrize("model_id,target_size", MODELS)
def test_kv_cache_static_quantization(model_id, target_size):
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
    kv_scale_count = 0
    for name, _ in tensors.items():
        if name.endswith("k_proj.weight") or name.endswith("v_proj.weight"):
            proj_linear_count += 1
        if name.endswith("kv_scale"):
            kv_scale_count += 1
    assert proj_linear_count // 2 == kv_scale_count

    # Measure checkpoint size and cleanup
    model_size = os.path.getsize(f"{quantized_model_dir}/model.safetensors")
    shutil.rmtree(quantized_model_dir)

    # We expect the quantized model to be a certain size
=======
    # We expect the model to be < 160MB
>>>>>>> 415c0b7 (Add fixed target sizes)
=======
    # We expect the quantized model to be a certain size
>>>>>>> c3acdee (Switch from output_scale to kv_scale)
    target_size = target_size * (1024 * 1024)
    assert model_size < target_size

@pytest.mark.parametrize("model_id,target_size", MODELS)
def test_kv_cache_static_quantization(model_id, target_size):
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
    kv_scale_count = 0
    for name, _ in tensors.items():
        if name.endswith("k_proj.weight") or name.endswith("v_proj.weight"):
            proj_linear_count += 1
        if name.endswith("kv_scale"):
            kv_scale_count += 1
    assert proj_linear_count // 2 == kv_scale_count

    # Measure checkpoint size and cleanup
    model_size = os.path.getsize(f"{quantized_model_dir}/model.safetensors")
    shutil.rmtree(quantized_model_dir)

    # We expect the quantized model to be a certain size
    target_size = target_size * (1024 * 1024)
    assert model_size < target_size
