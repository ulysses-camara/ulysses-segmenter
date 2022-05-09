"""Resource paths for tests."""
import typing as t
import os


class TestPaths(t.NamedTuple):
    """Resource paths for tests."""

    model_bert: str = "2_layer_6000_vocab_size_bert"
    model_lstm: str = "512_hidden_dim_6000_vocab_size_1_layer_lstm"
    tokenizer: str = "6000_subword_tokenizer"
    legal_text_long: str = "tests/resources/test_legal_text_long.txt"
    legal_text_short: str = "tests/resources/test_legal_text_short.txt"
    legal_text_with_noise: str = "tests/resources/test_legal_text_with_noise.txt"

    quantized_test_model_dirname: str = "tests/temp_quantization_models"
    quantized_test_model_lstm_torch: str = "torch_temp_lstm"
    quantized_test_model_lstm_onnx: str = "onnx_temp_lstm"
    quantized_test_model_bert_torch: str = "torch_temp_bert"
    quantized_test_model_bert_onnx: str = "onnx_temp_bert"

    cache_dir_models = os.path.join(os.path.dirname(__file__), "cache/models")
    cache_dir_tokenizers = os.path.join(os.path.dirname(__file__), "cache/tokenizers")
