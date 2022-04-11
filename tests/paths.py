"""Resource paths for tests."""
import typing as t


class TestPaths(t.NamedTuple):
    """Resource paths for tests."""

    model_bert: str = "pretrained_segmenter_model/2_6000_layer_model"
    model_lstm: str = (
        "pretrained_segmenter_model/128_6000_1_lstm/checkpoints/epoch=3-step=3591.ckpt"
    )
    tokenizer: str = "tokenizers/6000_subwords"
    legal_text_long: str = "tests/resources/test_legal_text_long.txt"
    legal_text_short: str = "tests/resources/test_legal_text_short.txt"

    quantized_test_model_dirname: str = "tests/temp_quantization_models"
    quantized_test_model_lstm_torch: str = "onnx_temp_torch"
    quantized_test_model_lstm_onnx: str = "onnx_temp_lstm"
    quantized_test_model_bert_onnx: str = "onnx_temp_bert"
