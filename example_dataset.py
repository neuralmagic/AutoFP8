from datasets import load_dataset
from transformers import AutoTokenizer

from auto_fp8 import AutoFP8ForCausalLM, BaseQuantizeConfig

pretrained_model_dir = "meta-llama/Meta-Llama-3-8B-Instruct"
quantized_model_dir = "Meta-Llama-3-8B-Instruct-FP8"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)

DATASET_ID = "mgoin/ultrachat_2k"
DATASET_SPLIT = "train_sft"
ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
ds = ds.map(
    lambda batch: {
        "text": tokenizer.apply_chat_template(batch["messages"], tokenize=False)
    }
)
examples = [sample["text"] for sample in ds]
tokenizer.pad_token = tokenizer.eos_token
examples = tokenizer(examples, padding=True, truncation=True, return_tensors="pt").to(
    "cuda"
)

quantize_config = BaseQuantizeConfig(
    quant_method="fp8", activation_scheme="static"
)  # or "static"

model = AutoFP8ForCausalLM.from_pretrained(
    pretrained_model_dir, quantize_config=quantize_config
)
model.quantize(examples)
model.save_quantized(quantized_model_dir)
