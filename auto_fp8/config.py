from typing import List


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
            By default, "re:.*lm_head" is included to ignore the embedding
            Linear layer usually at the end of decoder LLMs
    """

    def __init__(
        self,
        quant_method: str = "fp8",
        activation_scheme: str = "static",
        ignore_patterns: List[str] = [],
    ):
        if quant_method != "fp8":
            raise ValueError("Only FP8 quantization is supported.")
        if activation_scheme not in ["static", "dynamic"]:
            raise ValueError(
                "Invalid activation_scheme. Choose either 'static' or 'dynamic'."
            )
        self.quant_method = quant_method
        self.activation_scheme = activation_scheme
        self.ignore_patterns = ignore_patterns
        self.ignored_layers = []
