import os
from typing import List, Optional

from transformers import AutoConfig, AutoTokenizer
from datasets import Dataset
from llmcompressor.transformers import SparseAutoModelForCausalLM
from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationType,
    QuantizationScheme,
    QuantizationStrategy,
)


class BaseQuantizeConfig:
    """Configuration for model quantization.

    Args:
        quant_method: Type/precision of quantization method to use.
            At the moment, this is just "fp8" which specifically means
            the fp8_e4m3 format in pytorch.
        activation_scheme: Choice of either "dynamic" or "static" quantization
            of activtions. If "static", then calibration samples are required
            during quantization to produce accurate per-tensor scales for
            activations of Linear modules.
        ignore_patterns: List of patterns used to ignore layers. If a string
            starts with "re:", then everything afterwards is used as python
            regex style matching i.e. re.search(), for each Linear layer.
            By default, "lm_head" is included to ignore the embedding
            Linear layer usually at the end of decoder LLMs
    """

    def __init__(
        self,
        quant_method: str = "fp8",
        activation_scheme: str = "static",
        ignore_patterns: List[str] = ["lm_head"],
    ):
        self.quant_method = quant_method
        self.activation_scheme = activation_scheme
        self.ignore_patterns = ignore_patterns


class AutoFP8ForCausalLM:
    def __init__(
        self, model: SparseAutoModelForCausalLM, quantize_config: BaseQuantizeConfig
    ):
        self.model = model
        self.model_type = self.model.config.model_type
        self.config = self.model.config
        self.quantize_config = quantize_config

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        quantize_config: BaseQuantizeConfig,
        **kwargs,
    ):
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        model = SparseAutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            config=config,
            device_map="auto",
            torch_dtype="auto",
            **kwargs,
        )
        return cls(model, quantize_config)

    def quantize(self, dataset: Optional[Dataset] = None):
        if self.quantize_config.activation_scheme == "dynamic":
            if dataset is None:
                # For dynamic activations, we don't care about calibration data
                # being provided. However, we need to pass something
                # TODO(mgoin): Remove once llmcompressor allows no dataset
                from datasets import load_dataset
                dataset = load_dataset("openai/openai_humaneval", split="test").select(range(1))
                dataset = dataset.rename_column("prompt", "text")

            FP8_W8 = QuantizationScheme(
                targets=["Linear"],
                weights=QuantizationArgs(
                    num_bits=8,
                    type=QuantizationType.FLOAT,
                    strategy=QuantizationStrategy.TENSOR,
                    symmetric=True,
                    dynamic=False,
                ),
            )

            recipe = QuantizationModifier(
                config_groups={"group_0": FP8_W8},
                ignore=self.quantize_config.ignore_patterns,
            )

            oneshot(
                model=self.model,
                dataset=dataset,
                recipe=recipe,
                num_calibration_samples=dataset.shape[0],
            )
        elif self.quantize_config.activation_scheme == "static":
            assert (
                dataset is not None
            ), "Calibration tokens required for static activation quantization"

            FP8_W8A8 = QuantizationScheme(
                targets=["Linear"],
                weights=QuantizationArgs(
                    num_bits=8,
                    type=QuantizationType.FLOAT,
                    strategy=QuantizationStrategy.TENSOR,
                    symmetric=True,
                    dynamic=False,
                ),
                input_activations=QuantizationArgs(
                    num_bits=8,
                    type=QuantizationType.FLOAT,
                    strategy=QuantizationStrategy.TENSOR,
                    symmetric=True,
                    dynamic=False,
                ),
            )

            recipe = QuantizationModifier(
                config_groups={"group_0": FP8_W8A8},
                ignore=self.quantize_config.ignore_patterns,
            )

            oneshot(
                model=self.model,
                dataset=dataset,
                recipe=recipe,
                num_calibration_samples=dataset.shape[0],
            )
        else:
            raise ValueError(
                f"Unsupported activation_scheme={self.quantize_config.activation_scheme}"
            )

    def save_quantized(self, save_directory: str):
        self.save_pretrained(save_directory, save_compressed=True)

    def save_pretrained(self, save_directory: str, save_compressed: bool = True):
        self.model.save_pretrained(save_directory, save_compressed=save_compressed)
        tokenizer = AutoTokenizer.from_pretrained(self.model.config._name_or_path)
        tokenizer.save_pretrained(save_directory)
        print(f"Saved final checkpoint to {os.path.abspath(save_directory)}")
