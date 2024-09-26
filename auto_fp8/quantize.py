import gc
import re
from typing import Optional, Tuple
import copy

import torch
import tqdm
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.dbrx.modeling_dbrx import DbrxExpertGLU

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
    amax = torch.maximum(min_val.abs(), max_val.abs())
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


def static_per_tensor_quantize(tensor: torch.Tensor, inv_scale: float) -> torch.Tensor:
    finfo = torch.finfo(torch.float8_e4m3fn)
    qweight = (tensor / inv_scale).clamp(min=finfo.min, max=finfo.max)
    return qweight.to(torch.float8_e4m3fn)


def fp8_gemm(A, A_scale, B, B_scale, bias, out_dtype):
    if A.numel() == 0:
        # Deal with empty tensors (triggeted by empty MoE experts)
        return torch.empty(size=(0, B.shape[0]), dtype=out_dtype, device=A.device)

    # TODO: Disable native fp8 gemm for now, always just dequantize
    # native_fp8_support = (
    #     torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)
    # )
    native_fp8_support = False
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

class FP8DbrxExpertGLU(torch.nn.Module):
    def __init__(
        self,
        original_module: DbrxExpertGLU,
    ):
        super().__init__()
        self.hidden_size = original_module.hidden_size
        self.ffn_hidden_size = original_module.ffn_hidden_size
        self.moe_num_experts = original_module.moe_num_experts
        self.activation_fn = original_module.activation_fn
        self.cnt = 0
        self.w1 = torch.empty_like(original_module.w1, 
                                   dtype=torch.float8_e4m3fn)
        self.v1 = torch.empty_like(original_module.v1, 
                                   dtype=torch.float8_e4m3fn)
        self.w2 = torch.empty_like(original_module.w2, 
                                   dtype=torch.float8_e4m3fn)
        
        self.w1_weight_scale = torch.ones(self.moe_num_experts, 
                                          dtype=torch.float32)
        self.v1_weight_scale = torch.ones(self.moe_num_experts, 
                                          dtype=torch.float32)
        self.w2_weight_scale = torch.ones(self.moe_num_experts, 
                                          dtype=torch.float32)
        
        self.w1_input_scale = torch.zeros(self.moe_num_experts, 
                                          dtype=torch.float32)
        self.v1_input_scale = torch.zeros(self.moe_num_experts, 
                                          dtype=torch.float32)
        self.w2_input_scale = torch.zeros(self.moe_num_experts, 
                                          dtype=torch.float32)

        self._quantize_weights(original_module)

    def _quantize_weights(self, 
                          original_module: DbrxExpertGLU):

        w1_ = self.w1.view(self.moe_num_experts, self.ffn_hidden_size, 
                           self.hidden_size)
        v1_ = self.v1.view(self.moe_num_experts, self.ffn_hidden_size, 
                           self.hidden_size)
        w2_ = self.w2.view(self.moe_num_experts, self.ffn_hidden_size, 
                           self.hidden_size)

        ow1_ = original_module.w1.view(self.moe_num_experts, 
                                       self.ffn_hidden_size, self.hidden_size)
        ov1_ = original_module.v1.view(self.moe_num_experts, 
                                       self.ffn_hidden_size, self.hidden_size)
        ow2_ = original_module.w2.view(self.moe_num_experts, 
                                       self.ffn_hidden_size, self.hidden_size)
        
        # quantize each expert's weight
        for expert_id in range(self.moe_num_experts):
            w1_[expert_id], self.w1_weight_scale[expert_id] = \
                per_tensor_quantize(ow1_[expert_id])
            v1_[expert_id], self.v1_weight_scale[expert_id] = \
                per_tensor_quantize(ov1_[expert_id])
            w2_[expert_id], self.w2_weight_scale[expert_id] = \
                per_tensor_quantize(ow2_[expert_id])
            
        # register the parameter
        self.w1_weight = torch.nn.Parameter(self.w1, 
                                            requires_grad=False)
        self.v1_weight = torch.nn.Parameter(self.v1, 
                                            requires_grad=False)
        self.w2_weight = torch.nn.Parameter(self.w2, 
                                            requires_grad=False)
        
        self.w1_weight_scale = torch.nn.Parameter(self.w1_weight_scale, 
                                                  requires_grad=False)
        self.v1_weight_scale = torch.nn.Parameter(self.v1_weight_scale, 
                                                  requires_grad=False)
        self.w2_weight_scale = torch.nn.Parameter(self.w2_weight_scale, 
                                                  requires_grad=False)
        
    # For static scheme
    def register_input_scale(self):

        self.w1_input_scale = torch.nn.Parameter(self.w1_input_scale, 
                                                 requires_grad=False)
        self.v1_input_scale = torch.nn.Parameter(self.v1_input_scale, 
                                                 requires_grad=False)
        self.w2_input_scale = torch.nn.Parameter(self.w2_input_scale, 
                                                 requires_grad=False)

    def forward(self, 
                x: torch.Tensor,
                expert_w1: torch.Tensor,
                expert_v1: torch.Tensor,
                expert_w2: torch.Tensor):

        qinput, x_scale = per_tensor_quantize(x)
        self.w1_input_scale[self.cnt] = max(self.w1_input_scale[self.cnt], 
                                            x_scale)
        self.v1_input_scale[self.cnt] = max(self.v1_input_scale[self.cnt], 
                                            x_scale)
        gate_proj = fp8_gemm(qinput, x_scale, expert_w1, 
                             self.w1_weight_scale[self.cnt], None, x.dtype)
        up_proj = fp8_gemm(qinput, x_scale, expert_v1, 
                           self.v1_weight_scale[self.cnt], None, x.dtype)
        gate_proj = self.activation_fn(gate_proj)
        intermediate_states = gate_proj * up_proj
        
        qinput, x_scale = per_tensor_quantize(intermediate_states)
        self.w2_input_scale[self.cnt] = max(self.w2_input_scale[self.cnt], 
                                            x_scale)
        down_proj = fp8_gemm(qinput, x_scale, expert_w2.t(), 
                             self.w2_weight_scale[self.cnt], None, x.dtype)
        
        # Since DbrxExpert's forward function does not pass the export id 
        # when calling DbrxExpertGLU's forward function, use self.cnt to 
        # represent the expert id it is using.
        self.cnt = ((self.cnt + 1) % self.moe_num_experts)
        return down_proj
        

# Class responsible for quantizing weights
class FP8DynamicLinear(torch.nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.nn.Parameter,
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
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


# Module responsible for taking already quantized weights, and recording input scales (and possibly output scales) using an activation observer
class FP8StaticLinearQuantizer(torch.nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.nn.Parameter,
        quantize_output: bool = False,
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
        self.bias = bias
        self.input_scale = None
        self.output_scale = None
        self.quantize_output = quantize_output

    def forward(self, x):
        qinput, x_input_scale = per_tensor_quantize(x)
        if self.input_scale is None:
            self.input_scale = torch.nn.Parameter(x_input_scale, requires_grad=False)
        elif x_input_scale > self.input_scale:
            self.input_scale = torch.nn.Parameter(x_input_scale, requires_grad=False)
        output = fp8_gemm(
            A=qinput,
            A_scale=self.input_scale,
            B=self.weight,
            B_scale=self.weight_scale,
            bias=self.bias,
            out_dtype=x.dtype,
        )

        # Optionally, quantize output and record scale
        if self.quantize_output:
            qoutput, output_scale = per_tensor_quantize(output)
            if self.output_scale is None:
                self.output_scale = torch.nn.Parameter(output_scale, requires_grad=False)
            elif output_scale > self.output_scale:
                self.output_scale = torch.nn.Parameter(output_scale, requires_grad=False)
            output = qoutput.to(output.dtype) * output_scale

        return output


# Module responsible for representing the final checkpoint representation
class FP8StaticLinear(torch.nn.Module):
    def __init__(
        self,
        weight: torch.nn.Parameter,
        weight_scale: torch.nn.Parameter,
        bias: torch.nn.Parameter,
        input_scale: torch.nn.Parameter,
        output_scale: Optional[torch.nn.Parameter] = None,
    ):
        super().__init__()
        self.weight = weight
        self.weight_scale = weight_scale
        self.bias = bias
        self.input_scale = input_scale
        self.output_scale = output_scale

    def forward(self, x):
        qinput = static_per_tensor_quantize(x, self.input_scale)
        output = fp8_gemm(
            A=qinput,
            A_scale=self.input_scale,
            B=self.weight,
            B_scale=self.weight_scale,
            bias=self.bias,
            out_dtype=x.dtype,
        )

        if self.output_scale:
            qoutput = static_per_tensor_quantize(output, self.output_scale)
            output = qoutput.to(output.dtype) * self.output_scale

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
):
    named_modules = list(model.named_modules())
    for name, linear in tqdm.tqdm(named_modules, desc="Quantizing weights"):
        if (
            not isinstance(linear, torch.nn.Linear)
            or name in quantize_config.ignored_layers
        ):
            continue
        quant_weight, weight_scale = per_tensor_quantize(linear.weight)
        bias = copy.deepcopy(linear.bias) if linear.bias is not None else None
        quant_linear = FP8DynamicLinear(
            weight=quant_weight, weight_scale=weight_scale, bias=bias
        )
        replace_module(model, name, quant_linear)
        del linear.weight
        del linear.bias
        del linear
    cleanup_memory()

    # For dbrx moe
    for name, module in tqdm.tqdm(named_modules, desc="Quantizing weights"):
        if (
            not isinstance(module, DbrxExpertGLU)
            or name in quantize_config.ignored_layers
        ):
            continue
        quant_module = FP8DbrxExpertGLU(module)
        replace_module(model, name, quant_module)
        del module.w1
        del module.v1
        del module.w2
        del module
    cleanup_memory()

def quantize_activations(
    model: AutoModelForCausalLM,
    quantize_config: BaseQuantizeConfig,
    calibration_tokens,
):
    # Replace weight quantizer with a dynamic activation quantizer observer
    for name, dynamic_quant_linear in model.named_modules():
        if isinstance(dynamic_quant_linear, FP8DbrxExpertGLU):
            dynamic_quant_linear.register_input_scale()
            continue
        if (
            not isinstance(dynamic_quant_linear, FP8DynamicLinear)
            or name in quantize_config.ignored_layers
        ):
            continue
        quantizer = FP8StaticLinearQuantizer(
            weight=dynamic_quant_linear.weight,
            weight_scale=dynamic_quant_linear.weight_scale,
            bias=dynamic_quant_linear.bias,
            quantize_output=(
                hasattr(quantize_config, "kv_cache_quant_layers")
                and name in quantize_config.kv_cache_quant_layers
            ),
        )
        replace_module(model, name, quantizer)
        del dynamic_quant_linear
    cleanup_memory()

    # Pass through calibration data to measure activation scales
    with torch.inference_mode():
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
            weight=quantizer.weight,
            weight_scale=quantizer.weight_scale,
            bias=quantizer.bias,
            input_scale=quantizer.input_scale,
            output_scale=quantizer.output_scale,
        )
        replace_module(model, name, static_proj)
        del quantizer
    cleanup_memory()

    # Post-process step for kv cache scales to take the k/v module
    # `output_scale` parameters, and store them in the parent attention
    # module as `k_scale` and `v_scale`
    if hasattr(quantize_config, "kv_cache_quant_layers"):
        # Assumes that list is ordered such that [layer0.k_proj, layer0.v_proj, layer1.k_proj, layer1.v_proj, ...]
        # so we make a list of tuples [(layer0.k_proj, layer0.v_proj), (layer1.k_proj, layer1.v_proj), ...]
        kv_proj_pairs = zip(*[iter(quantize_config.kv_cache_quant_layers)]*2)
        for k_proj_name, v_proj_name in kv_proj_pairs:
            parent_module_name = ".".join(k_proj_name.split(".")[:-1])
            assert parent_module_name == ".".join(v_proj_name.split(".")[:-1])
            parent_module = dict(model.named_modules())[parent_module_name]

            k_proj = dict(model.named_modules())[k_proj_name]
            v_proj = dict(model.named_modules())[v_proj_name]

            parent_module.k_scale = torch.nn.Parameter(k_proj.output_scale, requires_grad=False)
            parent_module.v_scale = torch.nn.Parameter(v_proj.output_scale, requires_grad=False)

            # Remove output_scale from k_proj and v_proj
            k_proj.output_scale = None
            v_proj.output_scale = None
    cleanup_memory()


def save_quantized_model(
    model: AutoModelForCausalLM,
    quant_config: BaseQuantizeConfig,
    save_dir: str,
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
    if hasattr(quant_config, "kv_cache_quant_layers"):
        static_q_dict["quantization_config"]["kv_cache_scheme"] = "static"
    model.config.update(static_q_dict)
    model.save_pretrained(save_dir)
    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
    tokenizer.save_pretrained(save_dir)
