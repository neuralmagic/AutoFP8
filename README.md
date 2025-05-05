# AutoFP8

**ATTENTION: AutoFP8 has been deprecated in preference of [`llm-compressor`](https://github.com/vllm-project/llm-compressor), a library for producing all sorts of model compression in addition to FP8. Check out the [FP8 example here](https://github.com/vllm-project/llm-compressor/tree/main/examples/quantization_w8a8_fp8).**

<details>
<summary>Old content</summary>
  
Open-source FP8 quantization library for producing compressed checkpoints for running in vLLM - see https://github.com/vllm-project/vllm/pull/4332 for details on the implementation for inference. This library focuses on providing quantized weight, activation, and kv cache scales for FP8_E4M3 precision.

[FP8 Model Collection from Neural Magic](https://huggingface.co/collections/neuralmagic/fp8-llms-for-vllm-666742ed2b78b7ac8df13127) with many accurate (<1% accuracy drop) FP8 checkpoints ready for inference with vLLM. 

<p align="center">
  <img src="https://github.com/neuralmagic/AutoFP8/assets/3195154/c6bb9ddb-1bc9-48df-bf5f-9d7916dbd1f9" width="40%" />
  <img src="https://github.com/neuralmagic/AutoFP8/assets/3195154/2e30d4c0-340a-4527-8ff7-e8d48a8807ca" width="40%" />
</p>

## Installation

Clone this repo and install it from source:
```bash
git clone https://github.com/neuralmagic/AutoFP8.git
pip install -e AutoFP8
```

A stable release will be published.

## Quickstart

This package introduces the `AutoFP8ForCausalLM` and `BaseQuantizeConfig` objects for managing how your model will be compressed.

Once you load your `AutoFP8ForCausalLM`, you can tokenize your data and provide it to the `model.quantize(tokenized_text)` function to calibrate+compress the model.

Finally, you can save your quantized model in a compressed checkpoint format compatible with vLLM using `model.save_quantized("my_model_fp8")`.

Here is a full example covering that flow:

```python
from transformers import AutoTokenizer
from auto_fp8 import AutoFP8ForCausalLM, BaseQuantizeConfig

pretrained_model_dir = "meta-llama/Meta-Llama-3-8B-Instruct"
quantized_model_dir = "Meta-Llama-3-8B-Instruct-FP8"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
examples = ["auto_fp8 is an easy-to-use model quantization library"]
examples = tokenizer(examples, return_tensors="pt").to("cuda")

quantize_config = BaseQuantizeConfig(quant_method="fp8", activation_scheme="dynamic")

model = AutoFP8ForCausalLM.from_pretrained(
    pretrained_model_dir, quantize_config=quantize_config
)
model.quantize(examples)
model.save_quantized(quantized_model_dir)
```

Finally, load it into vLLM for inference! Support began in v0.4.2 (`pip install vllm>=0.4.2`). Note that hardware support for FP8 tensor cores must be available in the GPU you are using (Ada Lovelace, Hopper, and newer).

```python
from vllm import LLM

model = LLM("Meta-Llama-3-8B-Instruct-FP8")
# INFO 05-10 18:02:40 model_runner.py:175] Loading model weights took 8.4595 GB

print(model.generate("Once upon a time"))
# [RequestOutput(request_id=0, prompt='Once upon a time', prompt_token_ids=[128000, 12805, 5304, 264, 892], prompt_logprobs=None, outputs=[CompletionOutput(index=0, text=' there was a man who fell in love with a woman. The man was so', token_ids=[1070, 574, 264, 893, 889, 11299, 304, 3021, 449, 264, 5333, 13, 578, 893, 574, 779], cumulative_logprob=-21.314169232733548, logprobs=None, finish_reason=length, stop_reason=None)], finished=True, metrics=RequestMetrics(arrival_time=1715378569.478381, last_token_time=1715378569.478381, first_scheduled_time=1715378569.480648, first_token_time=1715378569.7070432, time_in_queue=0.002267122268676758, finished_time=1715378570.104807), lora_request=None)]
```

## How to run FP8 quantized models

[vLLM](https://github.com/vllm-project/vllm) has full support for FP8 models quantized with this package. Install vLLM with: `pip install vllm>=0.4.2`

Then simply pass the quantized checkpoint directly to vLLM's entrypoints! It will detect the checkpoint format using the `quantization_config` in the `config.json`.
```python
from vllm import LLM
model = LLM("neuralmagic/Meta-Llama-3-8B-Instruct-FP8")
# INFO 05-06 10:06:23 model_runner.py:172] Loading model weights took 8.4596 GB

outputs = model.generate("Once upon a time,")
print(outputs[0].outputs[0].text)
# ' there was a beautiful princess who lived in a far-off kingdom. She was kind'
```

## Checkpoint structure explanation

Here we detail the experimental structure for the fp8 checkpoints.

The following is added to config.json
```python
"quantization_config": {
    "quant_method": "fp8",
    "activation_scheme": "static" or "dynamic"
  },
```

Each quantized layer in the state_dict will have:

If the config has `"activation_scheme": "static"`:
```
model.layers.0.mlp.down_proj.weight              < F8_E4M3
model.layers.0.mlp.down_proj.input_scale         < F32
model.layers.0.mlp.down_proj.weight_scale        < F32
```
If config has `"activation_scheme": "dynamic"`:
```
model.layers.0.mlp.down_proj.weight              < F8_E4M3
model.layers.0.mlp.down_proj.weight_scale        < F32
```

</details>
