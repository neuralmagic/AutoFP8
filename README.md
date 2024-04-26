# AutoFP8

Example model with static scales for activations and weights: https://huggingface.co/nm-testing/Meta-Llama-3-8B-Instruct-FP8

Command to produce:
```bash
python quantize.py --model-id meta-llama/Meta-Llama-3-8B-Instruct --save-dir Meta-Llama-3-8B-Instruct-FP8
```

## Checkpoint structure

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
