from optimum.exporters.onnx.model_configs import MistralOnnxConfig

from optimum.exporters.onnx.config import TextDecoderOnnxConfig
from optimum.utils import NormalizedTextConfig, DummyPastKeyValuesGenerator
from packaging import version
import torch

class MiniCPMDummyPastKeyValuesGenerator(DummyPastKeyValuesGenerator):
    """
    MPT swaps the two last dimensions for the key cache compared to usual transformers
    decoder models, thus the redefinition here.
    """
    def generate(self, input_name: str, framework: str = "pt"):
        past_key_shape = (
            self.batch_size,
            self.num_attention_heads,
            self.hidden_size // self.num_attention_heads,
            self.sequence_length,
        )
        past_value_shape = (
            self.batch_size,
            self.num_attention_heads,
            self.sequence_length,
            self.hidden_size // self.num_attention_heads,
        )
        return [
            (
                self.random_float_tensor(past_key_shape, framework=framework),
                self.random_float_tensor(past_value_shape, framework=framework),
            )
            for _ in range(self.num_layers)
        ]
    

class MiniCPMOnnxConfig(TextDecoderOnnxConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (MiniCPMDummyPastKeyValuesGenerator,) + TextDecoderOnnxConfig.DUMMY_INPUT_GENERATOR_CLASSES
    DUMMY_PKV_GENERATOR_CLASS = MiniCPMDummyPastKeyValuesGenerator

    DEFAULT_ONNX_OPSET = 14  # aten::tril operator requires opset>=14
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(
        num_layers="num_hidden_layers",
        num_attention_heads="num_key_value_heads"
    )
    
    def add_past_key_values(self, inputs_or_outputs, direction: str):
        """
        Adapted from https://github.com/huggingface/optimum/blob/v1.9.0/optimum/exporters/onnx/base.py#L625
        """
        if direction not in ["inputs", "outputs"]:
            raise ValueError(f'direction must either be "inputs" or "outputs", but {direction} was given')

        if direction == "inputs":
            decoder_sequence_name = "past_sequence_length"
            name = "past_key_values"
        else:
            decoder_sequence_name = "past_sequence_length + 1"
            name = "present"

        for i in range(self._normalized_config.num_layers):
            inputs_or_outputs[f"{name}.{i}.key"] = {0: "batch_size", 3: decoder_sequence_name}
            inputs_or_outputs[f"{name}.{i}.value"] = {0: "batch_size", 2: decoder_sequence_name}


