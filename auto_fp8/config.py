from typing import List


class BaseQuantizeConfig:
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
