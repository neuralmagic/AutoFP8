import os
import shutil

import pytest
import safetensors.torch
from datasets import load_dataset
from transformers import AutoTokenizer

from auto_fp8 import AutoFP8ForCausalLM, BaseQuantizeConfig

MODELS = [
    ("facebook/opt-125m", 160),
    # ("Qwen/Qwen2-0.5B-Instruct", 620),
]

@pytest.mark.parametrize("model_id,target_size", MODELS)
def test_dynamic_quantization(model_id, target_size):
    quantized_model_dir = model_id.split("/")[-1] + "-fp8-dynamic"

    quantize_config = BaseQuantizeConfig(
        quant_method="fp8", activation_scheme="dynamic"
    )

    model = AutoFP8ForCausalLM.from_pretrained(model_id, quantize_config)
    model.quantize()
    model.save_quantized(quantized_model_dir)

    # Measure checkpoint size and cleanup
    model_size = os.path.getsize(f"{quantized_model_dir}/model.safetensors")
    shutil.rmtree(quantized_model_dir)

    # We expect the quantized model to be a certain size
    target_size = target_size * (1024 * 1024)
    assert model_size < target_size


@pytest.mark.parametrize("model_id,target_size", MODELS)
def test_static_quantization(model_id, target_size):
    quantized_model_dir = model_id.split("/")[-1] + "-fp8-static"

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    ds = load_dataset("mgoin/ultrachat_2k", split="train_sft").select(range(2))
    def preprocess(example):
        example = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        return tokenizer(
            example,
            padding=False,
            max_length=32,
            truncation=True,
            add_special_tokens=False,
        )
    ds = ds.map(preprocess, remove_columns=ds.column_names)

    quantize_config = BaseQuantizeConfig(quant_method="fp8", activation_scheme="static")

    model = AutoFP8ForCausalLM.from_pretrained(model_id, quantize_config)
    model.quantize(ds)
    model.save_quantized(quantized_model_dir)

    # Measure checkpoint size and cleanup
    model_size = os.path.getsize(f"{quantized_model_dir}/model.safetensors")
    shutil.rmtree(quantized_model_dir)

    # We expect the quantized model to be a certain size
    target_size = target_size * (1024 * 1024)
    assert model_size < target_size

# @pytest.mark.parametrize("model_id,target_size", MODELS)
# def test_kv_cache_static_quantization(model_id, target_size):
#     quantized_model_dir = model_id.split("/")[-1] + "-fp8-static-kv"

#     tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
#     examples = ["auto-fp8 is an easy-to-use model quantization library"]
#     examples = tokenizer(examples, return_tensors="pt")

#     quantize_config = BaseQuantizeConfig(
#         quant_method="fp8",
#         activation_scheme="static",
#         kv_cache_quant_targets=("k_proj", "v_proj"),
#     )

#     model = AutoFP8ForCausalLM.from_pretrained(model_id, quantize_config)
#     model.model.to("cpu")

#     model.quantize(examples)
#     model.save_quantized(quantized_model_dir)

#     tensors = safetensors.torch.load_file(f"{quantized_model_dir}/model.safetensors")
#     proj_linear_count = 0
#     kv_scale_count = 0
#     for name, _ in tensors.items():
#         if name.endswith("k_proj.weight") or name.endswith("v_proj.weight"):
#             proj_linear_count += 1
#         if name.endswith("kv_scale"):
#             kv_scale_count += 1
#     assert proj_linear_count // 2 == kv_scale_count

#     # Measure checkpoint size and cleanup
#     model_size = os.path.getsize(f"{quantized_model_dir}/model.safetensors")
#     shutil.rmtree(quantized_model_dir)

#     # We expect the quantized model to be a certain size
#     target_size = target_size * (1024 * 1024)
#     assert model_size < target_size