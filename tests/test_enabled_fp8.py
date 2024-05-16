"""Tests whether FP8 computation is enabled correctly.

Run `pytest tests/test_fp8.py`.
"""
import torch

capability = torch.cuda.get_device_capability()
capability = capability[0] * 10 + capability[1]

if capability < 90:
    print(
        "FP8 is not supported on this GPU type.The model can not be run properly.(Networks with FP8 Q/DQ layers "
        "require hardware)")
