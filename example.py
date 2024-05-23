from transformers import AutoTokenizer

from auto_fp8 import AutoFP8ForCausalLM, BaseQuantizeConfig

pretrained_model_dir = "meta-llama/Meta-Llama-3-8B-Instruct"
quantized_model_dir = "Meta-Llama-3-8B-Instruct-FP8"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
examples = ["auto_fp8 is an easy-to-use model quantization library"]
examples = tokenizer(examples, return_tensors="pt").to("cuda")

ignore_patterns = ["re:.*gate"]

quantize_config = BaseQuantizeConfig(
    quant_method="fp8",
    activation_scheme="dynamic",  # or "static"
    ignore_patterns=ignore_patterns,
)

model = AutoFP8ForCausalLM.from_pretrained(
    pretrained_model_dir, quantize_config=quantize_config
)
model.quantize(examples)
model.save_quantized(quantized_model_dir)
