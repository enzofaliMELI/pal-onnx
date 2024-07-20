from core.utils.constants import OnnxVersion
from onnx import ModelProto, TensorProto
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_opsetid,
    make_tensor_value_info,
)


class Tokenizer:
    def __init__(
        self,
        case_change_action: str = "LOWER",
        separators: list[str] = [" "],
        stopwords: list[str] = [""],
    ) -> None:
        """
        Tokenizer class constructor.

        Parameters:
            action (str): Action for text normalization. Possible values: "LOWER", "UPPER", "NONE".
            separators (list): List of separators for text splitting.
            stopwords (list): List of stop words for text normalization.
        """

        # Validate case_change_action parameter
        if case_change_action not in ["LOWER", "UPPER", "NONE"]:
            raise ValueError("Invalid action. Choose from 'LOWER', 'UPPER', 'NONE'.")

        # Validate separators parameter
        if not isinstance(separators, list) or not all(isinstance(s, str) for s in separators):
            raise TypeError("Separators must be a list of strings.")

        # Validate stopwords parameter
        if not isinstance(stopwords, list) or not all(isinstance(s, str) for s in stopwords):
            raise TypeError("Stopwords must be a list of strings.")

        self.case_change_action = case_change_action
        self.separators = separators
        self.stopwords = stopwords

    def model(self) -> ModelProto:
        # Input string Tensor: ["MY CLEAN QUERY oov stopword"]
        string_input = make_tensor_value_info("string_input", TensorProto.STRING, [1])

        # String Split node
        # Split Tensor: [["MY", "CLEAN", "QUERY", "oov", "stopword"]]
        split_node = make_node(
            op_type="Tokenizer",
            inputs=["string_input"],
            outputs=["string_split"],
            mark=0,  # Mark the beginning/end character
            mincharnum=1,  # Minimum number of characters allowed
            pad_value="",  # Padding value
            separators=self.separators,
            domain="com.microsoft",
        )

        # String Normalizer node
        # Normalizer Tensor: [["my", "clean", "query", "oov"]]
        normalizer_node = make_node(
            "StringNormalizer",
            inputs=["string_split"],
            outputs=["string_normalized"],
            case_change_action=self.case_change_action,
            stopwords=self.stopwords,
        )

        # Output string Tensor: [["my", "clean", "query", "oov"]]
        string_normalized = make_tensor_value_info("string_normalized", TensorProto.STRING, [1, None])

        # Create the graph with the nodes
        graph = make_graph(
            [
                split_node,
                normalizer_node,
            ],
            "tokenizer_graph",
            inputs=[string_input],
            outputs=[string_normalized],
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
