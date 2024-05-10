from transformers import AutoTokenizer
from auto_fp8 import AutoFP8ForCausalLM, BaseQuantizeConfig

pretrained_model_dir = "facebook/opt-125m"
quantized_model_dir = "opt-125m-fp8"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
examples = ["auto-fp8 is an easy-to-use model quantization library"]
examples = tokenizer(examples, return_tensors="pt").to("cuda")

quantize_config = BaseQuantizeConfig(quant_method="fp8", activation_scheme="static")

model = AutoFP8ForCausalLM.from_pretrained(
    pretrained_model_dir, quantize_config=quantize_config
)
model.quantize(examples)
model.save_quantized(quantized_model_dir)
