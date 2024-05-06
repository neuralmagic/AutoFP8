# AutoFP8

Open-source FP8 quantization project for producing compressed checkpoints for running in vLLM - see https://github.com/vllm-project/vllm/pull/4332 for implementation.

# How to run quantized models

Install vLLM: `pip install vllm>=0.4.2`

Then simply pass the quantized checkpoint directly to vLLM's entrypoints! It will detect the checkpoint format using the `quantization_config` in the `config.json`.
```python
from vllm import LLM
model = LLM("nm-testing/Meta-Llama-3-8B-Instruct-FP8")
# INFO 05-06 10:06:23 model_runner.py:172] Loading model weights took 8.4596 GB

outputs = model.generate("Once upon a time,")
print(outputs[0].outputs[0].text)
# ' there was a beautiful princess who lived in a far-off kingdom. She was kind'
```

## How to quantize a model

Example model with static scales for activations and weights: https://huggingface.co/nm-testing/Meta-Llama-3-8B-Instruct-FP8

Command to produce:
```bash
python quantize.py --model-id meta-llama/Meta-Llama-3-8B-Instruct --save-dir Meta-Llama-3-8B-Instruct-FP8
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
model.layers.0.mlp.down_proj.act_scale           < F32
model.layers.0.mlp.down_proj.weight_scale        < F32
```
If config has `"activation_scheme": "dynamic"`:
```
model.layers.0.mlp.down_proj.weight              < F8_E4M3
model.layers.0.mlp.down_proj.weight_scale        < F32
```
