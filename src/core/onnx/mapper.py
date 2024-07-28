from core.conf.settings import OnnxVersion
from onnx import ModelProto, TensorProto
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_opsetid,
    make_tensor,
    make_tensor_value_info,
)


class Mapper:
    def __init__(
        self, vocabulary: dict, oov_token: int = 1, output_vector_length: int = 15
    ) -> None:
        """
        Mapper
        class constructor.

           Parameters:
               vocabulary (dict): A dictionary containing tokens as keys and corresponding integer identifiers as values.
               oov_token (int): Integer identifier for out-of-vocabulary tokens.
               output_vector_length (int): Length of the output vector.

           Returns:
               ModelProto: Mapper
            model.
        """

        # Vocabulary must be a non-empty dictionary
        if not isinstance(vocabulary, dict):
            raise TypeError("Vocabulary must be a dictionary.")
        if not vocabulary:
            raise ValueError("Vocabulary cannot be empty.")

        # Vocabulary int_map is 2-indexed
        if any(val <= 1 for val in vocabulary.values()):
            raise ValueError("Integer identifiers in vocabulary cannot start from 2.")

        # oov_token must be -1, 0 or 1
        if oov_token not in (-1, 0, 1):
            raise ValueError("Out-of-vocabulary token must be -1, 0 or 1.")

        # output_vector_length must be a positive integer
        if not isinstance(output_vector_length, int) or output_vector_length <= 0:
            raise ValueError("Output vector length must be a positive integer.")

        self.oov_token: int = oov_token
        self.output_length: int = output_vector_length
        self._vocab: dict = vocabulary
        self._update_vocab_properties()

    def _update_vocab_properties(self):
        self.vocab_list: list = list(self._vocab.keys())
        self.int_map: list = list(self._vocab.values())
        return self

    @property
    def vocab(self):
        return self._vocab

    @vocab.setter
    def vocab(self, val):
        self._vocab = val
        self._update_vocab_properties()

    def model(self) -> ModelProto:
        # Input string Tensor: [["my", "clean", "query", "oov"]]
        string_normalized = make_tensor_value_info(
            "string_normalized", TensorProto.STRING, [1, None]
        )

        # Category Mapper node
        # Mapper Tensor: [[1, 2, 3, -1]]
        mapper_node = make_node(
            "CategoryMapper",
            inputs=["string_normalized"],
            outputs=["numeric_output"],
            cats_strings=list(self.vocab_list),
            cats_int64s=self.int_map,
            default_int64=self.oov_token,
            domain="ai.onnx.ml",
        )

        # Pad inputs:
        # Constant value of [0, 0, 0, output_length]
        pads_data = make_tensor(
            "pads_data", TensorProto.INT64, [4], [0, 0, 0, self.output_length]
        )

        # Constant node for pads
        pads_constant_node = make_node(
            "Constant", inputs=[], outputs=["pads"], value=pads_data
        )

        # Padding node
        # Pad Tensor (pad = [0, 0, 0, 3]): [[1, 2, 3, -1, 0, 0, 0]]
        padding_node = make_node(
            "Pad",
            inputs=["numeric_output", "pads"],
            outputs=["numeric_output_padded"],
            mode="constant",
        )

        # Slice inputs:
        # Constant start value of [0, 0] (first index)
        # Constant end value of [0, output_length]
        slice_start_data = make_tensor(
            "slice_start_data", TensorProto.INT64, [2], [0, 0]
        )
        slice_end_data = make_tensor(
            "slice_end_data", TensorProto.INT64, [2], [1, self.output_length]
        )

        # Constant for slice_start
        slice_start_constant_node = make_node(
            "Constant", inputs=[], outputs=["slice_start"], value=slice_start_data
        )

        # Constant for slice_end
        slice_end_constant_node = make_node(
            "Constant", inputs=[], outputs=["slice_end"], value=slice_end_data
        )

        # Slice node
        # Slice Tensor (start=[0, 0], end=[0, 3]): [[1, 2, 3]]
        slice_node = make_node(
            "Slice",
            inputs=["numeric_output_padded", "slice_start", "slice_end"],
            outputs=["numeric_tensor"],
        )

        # Output numeric Tensor: [[1, 2, 3]]
        numeric_tensor = make_tensor_value_info(
            "numeric_tensor", TensorProto.INT64, [1, self.output_length]
        )

        # Create the graph with the nodes
        graph = make_graph(
            [
                mapper_node,
                pads_constant_node,
                padding_node,
                slice_start_constant_node,
                slice_end_constant_node,
                slice_node,
            ],
            "Mapper_graph",
            inputs=[string_normalized],
            outputs=[numeric_tensor],
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
