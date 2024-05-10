import argparse
import gc
import re
from typing import Tuple

import torch
import transformers
import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# HACK: override the dtype_byte_size function in transformers to support float8 types
# Fix is posted upstream https://github.com/huggingface/transformers/pull/30488
def new_dtype_byte_size(dtype):
    if dtype == torch.bool:
        return 1 / 8
    bit_search = re.search(r"[^\d](\d+)_?", str(dtype))
    if bit_search is None:
        raise ValueError(f"`dtype` is not a valid dtype: {dtype}.")
    bit_size = int(bit_search.groups()[0])
    return bit_size // 8


transformers.modeling_utils.dtype_byte_size = new_dtype_byte_size


def cleanup_memory():
    gc.collect()
    torch.cuda.empty_cache()


def per_tensor_quantize(tensor: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """Quantize a tensor using per-tensor static scaling factor.
    Args:
        tensor: The input tensor.
    """
    finfo = torch.finfo(torch.float8_e4m3fn)
    # Calculate the scale as dtype max divided by absmax.
    # Since .abs() creates a new tensor, we use aminmax to get
    # the min and max first and then calculate the absmax.
    if tensor.numel() == 0:
        # Deal with empty tensors (triggered by empty MoE experts)
        min_val, max_val = (
            torch.tensor(0.0, dtype=tensor.dtype),
            torch.tensor(1.0, dtype=tensor.dtype),
        )
    else:
        min_val, max_val = tensor.aminmax()
    amax = min_val.abs().max(max_val.abs())
    scale = finfo.max / amax.clamp(min=1e-12)
    # scale and clamp the tensor to bring it to
    # the representative range of float8 data type
    # (as default cast is unsaturated)
    qweight = (tensor * scale).clamp(min=finfo.min, max=finfo.max)
    # Return both float8 data and the inverse scale (as float),
    # as both required as inputs to torch._scaled_mm
    qweight = qweight.to(torch.float8_e4m3fn)
    scale = scale.float().reciprocal()
    return qweight, scale


def fp8_gemm(A, A_scale, B, B_scale, bias, out_dtype):
    cuda_compute_capability = torch.cuda.get_device_capability()
    if cuda_compute_capability >= (9, 0):
        output, _ = torch._scaled_mm(
            A,
            B.t(),
            out_dtype=out_dtype,
            scale_a=A_scale,
            scale_b=B_scale,
            bias=bias,
        )
    else:
        output = torch.nn.functional.linear(
            A.to(out_dtype) * A_scale,
            B.to(out_dtype) * B_scale.to(out_dtype),
            bias=bias,
        )
    return output


class FP8StaticLinearQuantizer(torch.nn.Module):
    def __init__(self, qweight, weight_scale, bias):
        super().__init__()
        self.weight = torch.nn.Parameter(qweight, requires_grad=False)
        self.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
        self.act_scale = None
        self.bias = bias

    def forward(self, x):
        # Dynamically quantize
        qinput, x_act_scale = per_tensor_quantize(x)

        # Update scale if needed.
        if self.act_scale is None:
            self.act_scale = torch.nn.Parameter(x_act_scale)
        elif x_act_scale > self.act_scale:
            self.act_scale = torch.nn.Parameter(x_act_scale)

        # Pass quantized to next layer so it has realistic data.
        output = fp8_gemm(
            A=qinput,
            A_scale=self.act_scale,
            B=self.weight,
            B_scale=self.weight_scale,
            bias=self.bias,
            out_dtype=x.dtype,
        )
        return output


class FP8StaticLinear(torch.nn.Module):
    def __init__(self, qweight, weight_scale, bias, act_scale=0.0):
        super().__init__()
        self.weight = torch.nn.Parameter(qweight, requires_grad=False)
        self.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
        self.act_scale = torch.nn.Parameter(act_scale, requires_grad=False)
        self.bias = bias

    def per_tensor_quantize(
        self, tensor: torch.Tensor, inv_scale: float
    ) -> torch.Tensor:
        # Scale and clamp the tensor to bring it to
        # the representative range of float8 data type
        # (as default cast is unsaturated)
        finfo = torch.finfo(torch.float8_e4m3fn)
        qweight = (tensor / inv_scale).clamp(min=finfo.min, max=finfo.max)
        return qweight.to(torch.float8_e4m3fn)

    def forward(self, x):
        qinput = self.per_tensor_quantize(x, inv_scale=self.act_scale)
        output = fp8_gemm(
            A=qinput,
            A_scale=self.act_scale,
            B=self.weight,
            B_scale=self.weight_scale,
            bias=self.bias,
            out_dtype=x.dtype,
        )
        return output


class FP8DynamicLinear(torch.nn.Module):
    def __init__(self, qweight, scale, bias):
        super().__init__()
        self.weight = torch.nn.Parameter(qweight, requires_grad=False)
        self.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
        self.bias = bias

    def forward(self, x):
        qinput, x_scale = per_tensor_quantize(x)
        output = fp8_gemm(
            A=qinput,
            A_scale=x_scale,
            B=self.weight,
            B_scale=self.weight_scale,
            bias=self.bias,
            out_dtype=x.dtype,
        )
        return output


def replace_module(model, name, new_module):
    if "." in name:
        parent_name = name.rsplit(".", 1)[0]
        child_name = name[len(parent_name) + 1 :]
        parent = model.model.get_submodule(parent_name)
    else:
        parent_name = ""
        parent = model.model
        child_name = name
    setattr(parent, child_name, new_module)


def quantize_weights(model):
    for name, linear in model.model.named_modules():
        # if "gate" in name or not isinstance(linear, torch.nn.Linear):
        if not isinstance(linear, torch.nn.Linear):
            continue
        quant_weight, quant_scale = per_tensor_quantize(linear.weight)
        quant_linear = FP8DynamicLinear(quant_weight, quant_scale, linear.bias)
        replace_module(model, name, quant_linear)
        del linear
    cleanup_memory()


def quantize_activations(model, calibration_tokens):
    # Replace layers with quantizer.
    for name, dynamic_quant_linear in model.model.named_modules():
        # if "gate" in name or not isinstance(dynamic_quant_linear, FP8DynamicLinear):
        if not isinstance(dynamic_quant_linear, FP8DynamicLinear):
            continue
        quantizer = FP8StaticLinearQuantizer(
            dynamic_quant_linear.weight,
            dynamic_quant_linear.weight_scale,
            dynamic_quant_linear.bias,
        )
        replace_module(model, name, quantizer)
        del dynamic_quant_linear
    cleanup_memory()

    # Calibration.
    with tqdm.tqdm(total=calibration_tokens.shape[0], desc="Calibrating") as pbar:
        for row_idx in range(calibration_tokens.shape[0]):
            model(calibration_tokens[row_idx].reshape(1, -1))
            torch.cuda.empty_cache()
            pbar.update(1)

    # Replace quantizer with StaticLayer.
    for name, quantizer in model.model.named_modules():
        # if "gate" in name or not isinstance(quantizer, FP8StaticLinearQuantizer):
        if not isinstance(quantizer, FP8StaticLinearQuantizer):
            continue
        static_proj = FP8StaticLinear(
            quantizer.weight,
            quantizer.weight_scale,
            quantizer.bias,
            quantizer.act_scale,
        )
        replace_module(model, name, static_proj)
        del quantizer
    cleanup_memory()


def save_quantized_model(model, activation_scheme, save_dir):
    print(model)
    print(f"Saving the model to {save_dir}")
    static_q_dict = {
        "quantization_config": {
            "quant_method": "fp8",
            "activation_scheme": activation_scheme,
        }
    }
    model.config.update(static_q_dict)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str)
    parser.add_argument("--save-dir", type=str)
    parser.add_argument(
        "--activation-scheme", type=str, default="static", choices=["static", "dynamic"]
    )
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=512)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    sample_input_tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is your name?"}],
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    ds = ds.shuffle(seed=42).select(range(args.num_samples))
    ds = ds.map(
        lambda batch: {
            "text": tokenizer.apply_chat_template(batch["messages"], tokenize=False)
        }
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    calibration_tokens = tokenizer(
        ds["text"],
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=args.max_seq_len,
        add_special_tokens=False,
    ).input_ids.to("cuda")
    print("Calibration tokens:", calibration_tokens.shape)

    # Load and test the model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype="auto", device_map="auto"
    )
    print("Original model graph:\n", model)
    output = model.generate(input_ids=sample_input_tokens, max_new_tokens=20)
    print("ORIGINAL OUTPUT:\n", tokenizer.decode(output[0]), "\n\n")

    # Quantize weights.
    quantize_weights(model)
    print("Weight-quantized model graph:\n", model)
    output = model.generate(input_ids=sample_input_tokens, max_new_tokens=20)
    print("WEIGHT QUANT OUTPUT:\n", tokenizer.decode(output[0]), "\n\n")

    if args.activation_scheme in "dynamic":
        print("Exporting model with static weights and dynamic activations")
        save_quantized_model(model, args.activation_scheme, args.save_dir)
    else:
        assert args.activation_scheme in "static"
        # Quantize activations.
        quantize_activations(model, calibration_tokens=calibration_tokens)
        print("Weight and activation quantized model graph:\n", model)
        output = model.generate(input_ids=sample_input_tokens, max_new_tokens=20)
        print("ACT QUANT OUTPUT:\n", tokenizer.decode(output[0]), "\n\n")

        print("Exporting model with static weights and static activations")
        save_quantized_model(model, args.activation_scheme, args.save_dir)
