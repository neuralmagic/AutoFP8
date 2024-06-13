import gc
import re
from typing import List, Tuple

import torch
import tqdm
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import BaseQuantizeConfig


# HACK: Override the dtype_byte_size function in transformers to support float8 types
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
            torch.tensor(-16.0, dtype=tensor.dtype),
            torch.tensor(16.0, dtype=tensor.dtype),
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
    if A.numel() == 0:
        # Deal with empty tensors (triggeted by empty MoE experts)
        return torch.empty(size=(0, B.shape[0]), dtype=out_dtype, device=A.device)
    
    native_fp8_support = (
        torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)
    )
    if native_fp8_support:
        need_reshape = A.dim() == 3
        if need_reshape:
            batch_size = A.shape[0]
            A_input = A.reshape(-1, A.shape[-1])
        else:
            batch_size = None
            A_input = A
        output, _ = torch._scaled_mm(
            A_input,
            B.t(),
            out_dtype=out_dtype,
            scale_a=A_scale,
            scale_b=B_scale,
            bias=bias,
        )
        if need_reshape:
            output = output.reshape(
                batch_size, output.shape[0] // batch_size, output.shape[1]
            )
    else:
        output = torch.nn.functional.linear(
            A.to(out_dtype) * A_scale,
            B.to(out_dtype) * B_scale.to(out_dtype),
            bias=bias,
        )
    return output


class FP8StaticLinearQuantizer(torch.nn.Module):
    def __init__(
        self, qweight: torch.Tensor, weight_scale: torch.Tensor, bias: torch.Tensor
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(qweight, requires_grad=False)
        self.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
        self.input_scale = None
        self.bias = bias

    def forward(self, x):
        qinput, x_input_scale = per_tensor_quantize(x)
        if self.input_scale is None:
            self.input_scale = torch.nn.Parameter(x_input_scale)
        elif x_input_scale > self.input_scale:
            self.input_scale = torch.nn.Parameter(x_input_scale)
        output = fp8_gemm(
            A=qinput,
            A_scale=self.input_scale,
            B=self.weight,
            B_scale=self.weight_scale,
            bias=self.bias,
            out_dtype=x.dtype,
        )
        return output


class FP8StaticLinear(torch.nn.Module):
    def __init__(
        self,
        qweight: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor,
        input_scale: float = 1.0,
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(qweight, requires_grad=False)
        self.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
        self.input_scale = torch.nn.Parameter(input_scale, requires_grad=False)
        self.bias = bias

    def per_tensor_quantize(
        self, tensor: torch.Tensor, inv_scale: float
    ) -> torch.Tensor:
        finfo = torch.finfo(torch.float8_e4m3fn)
        qweight = (tensor / inv_scale).clamp(min=finfo.min, max=finfo.max)
        return qweight.to(torch.float8_e4m3fn)

    def forward(self, x):
        qinput = self.per_tensor_quantize(x, inv_scale=self.input_scale)
        output = fp8_gemm(
            A=qinput,
            A_scale=self.input_scale,
            B=self.weight,
            B_scale=self.weight_scale,
            bias=self.bias,
            out_dtype=x.dtype,
        )
        return output


class FP8DynamicLinear(torch.nn.Module):
    def __init__(self, qweight: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor):
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


def replace_module(model: AutoModelForCausalLM, name: str, new_module: torch.nn.Module):
    if "." in name:
        parent_name = name.rsplit(".", 1)[0]
        child_name = name[len(parent_name) + 1 :]
        parent = model.get_submodule(parent_name)
    else:
        parent_name = ""
        parent = model
        child_name = name
    setattr(parent, child_name, new_module)


def quantize_weights(
    model: AutoModelForCausalLM,
    quantize_config: BaseQuantizeConfig,
    ignored_layers: List[str] = [],
):
    named_modules = list(model.named_modules())
    for name, linear in tqdm.tqdm(named_modules, desc="Quantizing weights"):
        if (
            not isinstance(linear, torch.nn.Linear)
            or name in quantize_config.ignored_layers
        ):
            continue
        quant_weight, quant_scale = per_tensor_quantize(linear.weight.clone())
        bias = linear.bias.clone() if linear.bias is not None else None
        quant_linear = FP8DynamicLinear(quant_weight, quant_scale, bias)
        replace_module(model, name, quant_linear)
        del linear.weight
        del linear.bias
        del linear
    cleanup_memory()


def quantize_activations(
    model: AutoModelForCausalLM,
    quantize_config: BaseQuantizeConfig,
    calibration_tokens,
    ignored_layers: List[str] = [],
):
    # Replace weight quantizer with a dynamic activation quantizer observer
    for name, dynamic_quant_linear in model.named_modules():
        if (
            not isinstance(dynamic_quant_linear, FP8DynamicLinear)
            or name in quantize_config.ignored_layers
        ):
            continue
        quantizer = FP8StaticLinearQuantizer(
            dynamic_quant_linear.weight,
            dynamic_quant_linear.weight_scale,
            dynamic_quant_linear.bias,
        )
        replace_module(model, name, quantizer)
        del dynamic_quant_linear
    cleanup_memory()

    # Pass through calibration data to measure activation scales
    with tqdm.tqdm(total=calibration_tokens.shape[0], desc="Calibrating activation scales") as pbar:
        for row_idx in range(calibration_tokens.shape[0]):
            model(calibration_tokens[row_idx].reshape(1, -1))
            cleanup_memory()
            pbar.update(1)

    # Replace dynamic quantizer observer with StaticLinear for export
    for name, quantizer in model.named_modules():
        if (
            not isinstance(quantizer, FP8StaticLinearQuantizer)
            or name in quantize_config.ignored_layers
        ):
            continue
        static_proj = FP8StaticLinear(
            quantizer.weight,
            quantizer.weight_scale,
            quantizer.bias,
            quantizer.input_scale,
        )
        replace_module(model, name, static_proj)
        del quantizer
    cleanup_memory()


def save_quantized_model(
    model: AutoModelForCausalLM,
    quant_config: BaseQuantizeConfig,
    save_dir: str,
    ignored_layers: List[str] = [],
):
    print(model)
    print(f"Saving the model to {save_dir}")
    static_q_dict = {
        "quantization_config": {
            "quant_method": "fp8",
            "activation_scheme": quant_config.activation_scheme,
            "ignored_layers": quant_config.ignored_layers,
        }
    }
    model.config.update(static_q_dict)
    model.save_pretrained(save_dir)
    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
    tokenizer.save_pretrained(save_dir)
