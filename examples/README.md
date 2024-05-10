## FP8 Quantization

This folder holds the original `quantize.py` example.

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