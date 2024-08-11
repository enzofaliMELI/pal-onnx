from core.conf.settings import OnnxVersion
from core.onnx.mapper import Mapper
from core.onnx.tokenizer import Tokenizer
from onnx import ModelProto
from onnx.compose import merge_graphs
from onnx.helper import make_model, make_opsetid


class Numericalizer:
    def __init__(self, tokenizer: Tokenizer, mapper: Mapper) -> None:
        """
        Numericalizer class constructor.

        Parameters:
            tokenizer (Tokenizer): A ONNX compatible tokenizer model that converts text to tokens.
            mapper (Mapper): A ONNX compatible mapper that given a vocabulary maps token to ints.

        Returns:
            ModelProto: Numericalizer model.
        """

        self.mapper = mapper
        self.tokenizer = tokenizer

    def model(self) -> ModelProto:
        io_map = [
            (
                self.tokenizer.model().graph.output[0].name,
                self.mapper.model().graph.input[0].name,
            )
        ]

        graph = merge_graphs(
            self.tokenizer.model().graph, self.mapper.model().graph, io_map
        )

        # Specify opset versions
        onnx_model = make_model(
            graph,
            opset_imports=[
                make_opsetid("ai.onnx.ml", 1),
                make_opsetid("com.microsoft", 1),
                make_opsetid("", OnnxVersion.OPSET_VERSION.value),
            ],
            ir_version=OnnxVersion.IR_VERSION.value,
        )

        return onnx_model
