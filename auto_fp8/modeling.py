import torch
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from auto_fp8.quantize import (
    quantize_weights,
    quantize_activations,
    save_quantized_model,
)
from auto_fp8.config import BaseQuantizeConfig


class AutoFP8ForCausalLM:
    def __init__(
        self,
        model: PreTrainedModel,
        quantize_config: BaseQuantizeConfig,
    ):
        # super().__init__()

        self.model = model
        self.model_type = self.model.config.model_type
        self.quantize_config = quantize_config
        self.config = self.model.config

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        quantize_config: BaseQuantizeConfig,
        **model_init_kwargs,
    ):
        """Load the un-quantized pretrained model"""

        # if not torch.cuda.is_available():
        #     raise EnvironmentError(
        #         "Load pretrained model to do quantization requires CUDA available."
        #     )

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
        if not hasattr(model_init_kwargs, "torch_dtype"):
            model_init_kwargs["torch_dtype"] = "auto"

        if not hasattr(model_init_kwargs, "device_map"):
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

    def quantize(self, calibration_tokens):
        def _prepare_calibration_data(calibration_tokens):
            if hasattr(calibration_tokens, "input_ids"):
                return calibration_tokens.input_ids
            return calibration_tokens

        # Always quantize the weights as they do not require calibration data
        quantize_weights(self.model)

        if self.quantize_config.activation_scheme == "static":
            quantize_activations(
                self.model, _prepare_calibration_data(calibration_tokens)
            )

    def save_quantized(self, save_dir):
        save_quantized_model(
            self.model,
            activation_scheme=self.quantize_config.activation_scheme,
            save_dir=save_dir,
        )
