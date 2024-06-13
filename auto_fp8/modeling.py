import re
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM

from auto_fp8.config import BaseQuantizeConfig
from auto_fp8.quantize import (
    quantize_activations,
    quantize_weights,
    save_quantized_model,
)


class AutoFP8ForCausalLM:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        quantize_config: BaseQuantizeConfig,
    ):
        self.model = model
        self.model_type = self.model.config.model_type
        self.config = self.model.config

        # Gather the Linear module names that we want to ignore
        quantize_config.ignored_layers = get_layers_to_ignore(
            self.model, quantize_config.ignore_patterns
        )

        if quantize_config.kv_cache_quant_targets:
            kv_cache_quant_layers = get_kv_cache_quant_layer(
                self.model, quantize_config.kv_cache_quant_targets
            )
            if len(kv_cache_quant_layers) == 0:
                raise ValueError(
                    f"Could not find any kv cache layers using kv_cache_quant_targets={quantize_config.kv_cache_quant_targets}, please fix your argument."
                )
            quantize_config.kv_cache_quant_layers = kv_cache_quant_layers

        self.quantize_config = quantize_config

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        quantize_config: BaseQuantizeConfig,
        **model_init_kwargs,
    ):
        """Load the un-quantized pretrained model"""

        def skip(*args, **kwargs):
            pass

        torch.nn.init.kaiming_uniform_ = skip
        torch.nn.init.uniform_ = skip
        torch.nn.init.normal_ = skip

        # Parameters related to loading from Hugging Face Hub
        cache_dir = model_init_kwargs.pop("cache_dir", None)
        force_download = model_init_kwargs.pop("force_download", False)
        resume_download = model_init_kwargs.pop("resume_download", False)
        proxies = model_init_kwargs.pop("proxies", None)
        local_files_only = model_init_kwargs.pop("local_files_only", False)
        use_auth_token = model_init_kwargs.pop("use_auth_token", None)
        revision = model_init_kwargs.pop("revision", None)
        subfolder = model_init_kwargs.pop("subfolder", "")
        commit_hash = model_init_kwargs.pop("_commit_hash", None)

        cached_file_kwargs = {
            "cache_dir": cache_dir,
            "force_download": force_download,
            "proxies": proxies,
            "resume_download": resume_download,
            "local_files_only": local_files_only,
            "use_auth_token": use_auth_token,
            "revision": revision,
            "subfolder": subfolder,
            "_commit_hash": commit_hash,
        }

        torch.cuda.empty_cache()

        # Important defaults
        if "torch_dtype" not in model_init_kwargs:
            model_init_kwargs["torch_dtype"] = "auto"

        if "device_map" not in model_init_kwargs:
            model_init_kwargs["device_map"] = "auto"

        merged_kwargs = {**model_init_kwargs, **cached_file_kwargs}
        print("Loading model with the following kwargs:", merged_kwargs)
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path, **merged_kwargs
        )

        model_config = model.config.to_dict()
        seq_len_keys = ["max_position_embeddings", "seq_length", "n_positions"]
        if any(k in model_config for k in seq_len_keys):
            for key in seq_len_keys:
                if key in model_config:
                    model.seqlen = model_config[key]
                    break
        else:
            print("Can't get model's sequence length, setting to 2048.")
            model.seqlen = 2048
        model.eval()

        return cls(model, quantize_config)

    def quantize(self, calibration_tokens: Optional[torch.Tensor] = None):
        def _prepare_calibration_data(calibration_tokens):
            if hasattr(calibration_tokens, "input_ids"):
                return calibration_tokens.input_ids
            return calibration_tokens

        # Always quantize the weights as they do not require calibration data
        quantize_weights(self.model, self.quantize_config)

        if self.quantize_config.activation_scheme == "static":
            assert (
                calibration_tokens is not None
            ), "Calibration tokens required for activation quantization"
            quantize_activations(
                self.model,
                self.quantize_config,
                _prepare_calibration_data(calibration_tokens),
            )

            # import copy
            # for layer in self.model.model.layers:
            #     layer.self_attn.kv_scale = copy.deepcopy(layer.self_attn.k_proj.input_scale)

    def save_quantized(self, save_dir):
        save_quantized_model(
            self.model,
            quant_config=self.quantize_config,
            save_dir=save_dir,
        )


def get_layers_to_ignore(model, ignore_patterns) -> List[str]:
    ignored_layers = set()

    for name, linear in model.named_modules():
        if not isinstance(linear, torch.nn.Linear):
            continue

        for ignore_pattern in ignore_patterns:
            regex_prefix = "re:"
            if ignore_pattern.startswith(regex_prefix):
                # check if name matches regex and add to set if true
                regex_pattern = ignore_pattern[len(regex_prefix) :]
                if re.search(regex_pattern, name):
                    ignored_layers.add(name)
            else:
                # else, exact match
                if ignore_pattern == name:
                    ignored_layers.add(name)

    return list(ignored_layers)


def get_kv_cache_quant_layer(model, kv_cache_quant_targets: Tuple[str]) -> List[str]:
    kv_cache_quant_layers = set()

    for name, linear in model.named_modules():
        if not isinstance(linear, torch.nn.Linear):
            continue

        for output_quant_target in kv_cache_quant_targets:
            if name.endswith(output_quant_target):
                kv_cache_quant_layers.add(name)

    return list(kv_cache_quant_layers)
