import os
import shutil

from transformers import AutoTokenizer

from auto_fp8 import AutoFP8ForCausalLM, BaseQuantizeConfig


def test_quantization():
    model_id = "facebook/opt-125m"
    quantized_model_dir = "opt-125m-fp8"

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    examples = ["auto-fp8 is an easy-to-use model quantization library"]
    examples = tokenizer(examples, return_tensors="pt")

    quantize_config = BaseQuantizeConfig(quant_method="fp8", activation_scheme="static")

    model = AutoFP8ForCausalLM.from_pretrained(
        model_id, quantize_config=quantize_config
    )
    model.model.to("cpu")

    model.quantize(examples)
    model.save_quantized(quantized_model_dir)

    # Measure checkpoint size and cleanup
    model_size = os.path.getsize(f"{quantized_model_dir}/model.safetensors")
    shutil.rmtree(quantized_model_dir)

    # We expect the model to be < 160MB
    target_size = 160 * (1024 * 1024)
    assert model_size < target_size
