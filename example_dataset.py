from datasets import load_dataset
from transformers import AutoTokenizer

from auto_fp8 import AutoFP8ForCausalLM, BaseQuantizeConfig

pretrained_model_dir = "facebook/opt-125m"
quantized_model_dir = "opt-125m-FP8"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

ds = load_dataset("mgoin/ultrachat_2k", split="train_sft").select(range(512))
def preprocess(example):
    example = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    return tokenizer(example, max_length=2048, truncation=True, add_special_tokens=False)
ds = ds.map(preprocess, remove_columns=ds.column_names)

quantize_config = BaseQuantizeConfig(quant_method="fp8", activation_scheme="static")

model = AutoFP8ForCausalLM.from_pretrained(pretrained_model_dir, quantize_config)
model.quantize(ds)
model.save_quantized(quantized_model_dir)
