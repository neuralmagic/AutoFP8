# AutoFP8

Open-source FP8 quantization project for producing compressed checkpoints for running in vLLM - see https://github.com/vllm-project/vllm/pull/4332 for implementation.

## How to quantize a model

Install this repo's requirements:
```bash
pip install -r requirements.txt
```

Command to produce a `Meta-Llama-3-8B-Instruct-FP8` quantized LLM:
```bash
python quantize.py --model-id meta-llama/Meta-Llama-3-8B-Instruct --save-dir Meta-Llama-3-8B-Instruct-FP8
```

Example model checkpoint with FP8 static scales for activations and weights: https://huggingface.co/nm-testing/Meta-Llama-3-8B-Instruct-FP8

All arguments available for `quantize.py`:
```
usage: quantize.py [-h] [--model-id MODEL_ID] [--save-dir SAVE_DIR] [--activation-scheme {static,dynamic}] [--num-samples NUM_SAMPLES] [--max-seq-len MAX_SEQ_LEN]

options:
  -h, --help            show this help message and exit
  --model-id MODEL_ID
  --save-dir SAVE_DIR
  --activation-scheme {static,dynamic}
  --num-samples NUM_SAMPLES
  --max-seq-len MAX_SEQ_LEN
```

## How to run FP8 quantized models

[vLLM](https://github.com/vllm-project/vllm) has full support for FP8 models quantized with this package. Install vLLM with: `pip install vllm>=0.4.2`

Then simply pass the quantized checkpoint directly to vLLM's entrypoints! It will detect the checkpoint format using the `quantization_config` in the `config.json`.
```python
from vllm import LLM
model = LLM("nm-testing/Meta-Llama-3-8B-Instruct-FP8")
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
model.layers.0.mlp.down_proj.act_scale           < F32
model.layers.0.mlp.down_proj.weight_scale        < F32
```
If config has `"activation_scheme": "dynamic"`:
```
model.layers.0.mlp.down_proj.weight              < F8_E4M3
model.layers.0.mlp.down_proj.weight_scale        < F32
```
