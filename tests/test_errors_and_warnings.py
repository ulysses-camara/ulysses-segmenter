"""Tests that are expected to fail, raise expections or warnings."""
import pytest

import segmentador
import segmentador.optimize

from . import paths


@pytest.mark.parametrize("batch_size", (0, -1, -100))
def test_invalid_batch_size(
    fixture_model_bert_2_layers: segmentador.Segmenter,
    fixture_legal_text_short: str,
    batch_size: int,
):
    with pytest.raises(ValueError):
        fixture_model_bert_2_layers(fixture_legal_text_short, batch_size=batch_size)


@pytest.mark.parametrize("moving_window_size", (0, -1, -100))
def test_invalid_moving_window_size(
    fixture_model_bert_2_layers: segmentador.Segmenter,
    fixture_legal_text_short: str,
    moving_window_size: int,
):
    with pytest.raises(ValueError):
        fixture_model_bert_2_layers(fixture_legal_text_short, moving_window_size=moving_window_size)


@pytest.mark.parametrize("window_shift_size", (0, -1, -100, 0.0, 1.001, -0.01))
def test_invalid_window_shift_size(
    fixture_model_bert_2_layers: segmentador.Segmenter,
    fixture_legal_text_short: str,
    window_shift_size: int,
):
    with pytest.raises(ValueError):
        fixture_model_bert_2_layers(fixture_legal_text_short, window_shift_size=window_shift_size)


@pytest.mark.parametrize("window_shift_size", (1025, 10000))
def test_warning_window_shift_size(
    fixture_model_bert_2_layers: segmentador.Segmenter,
    fixture_legal_text_short: str,
    window_shift_size: int,
):
    with pytest.warns(UserWarning):
        fixture_model_bert_2_layers(fixture_legal_text_short, window_shift_size=window_shift_size)


@pytest.mark.parametrize("moving_window_size", (1025, 10000))
def test_warning_moving_window_size(
    fixture_model_bert_2_layers: segmentador.Segmenter,
    fixture_legal_text_short: str,
    moving_window_size: int,
):
    with pytest.warns(UserWarning):
        fixture_model_bert_2_layers(fixture_legal_text_short, moving_window_size=moving_window_size)


@pytest.mark.parametrize("inference_pooling_operation", (None, "", "avg"))
def test_invalid_inference_pooling_operation(
    inference_pooling_operation: str, fixture_test_paths: paths.TestPaths
):
    with pytest.raises(ValueError):
        segmentador.Segmenter(
            uri_model=fixture_test_paths.model_bert,
            inference_pooling_operation=inference_pooling_operation,
            device="cpu",
            local_files_only=False,
            cache_dir_model=fixture_test_paths.cache_dir_models,
        )


@pytest.mark.skip(reason="Support for ONNX newer versions has been dropped.")
def test_invalid_quantization_output_format(fixture_model_lstm_1_layer: segmentador.LSTMSegmenter):
    with pytest.raises(ValueError):
        segmentador.optimize.quantize_model(
            model=fixture_model_lstm_1_layer,
            model_output_format="invalid",
        )


@pytest.mark.skip(reason="Support for ONNX newer versions has been dropped.")
def test_invalid_quantization_model_format(
    fixture_quantized_model_lstm_onnx: segmentador.optimize.ONNXLSTMSegmenter,
):
    with pytest.raises(TypeError):
        segmentador.optimize.quantize_model(
            model=fixture_quantized_model_lstm_onnx,
            model_output_format="onnx",
        )


def test_invalid_onnx_format_for_pruned_bert(fixture_test_paths: paths.TestPaths):
    model_pruned = segmentador.BERTSegmenter(
        uri_model=fixture_test_paths.model_bert,
        device="cpu",
        local_files_only=False,
        cache_dir_model=fixture_test_paths.cache_dir_models,
    )
    model_pruned.model.prune_heads({0: [1, 2, 5], 1: [4, 5]})

    with pytest.raises(RuntimeError):
        segmentador.optimize.quantize_model(
            model=model_pruned,
            model_output_format="onnx",
        )


@pytest.mark.skip(reason="Support for ONNX newer versions has been dropped.")
def test_repeated_file_uris_in_onnx_quantization_lstm(
    fixture_model_lstm_1_layer: segmentador.LSTMSegmenter,
):
    with pytest.raises(ValueError):
        segmentador.optimize.quantize_model(
            model=fixture_model_lstm_1_layer,
            model_output_format="onnx",
            quantized_model_filename="repeated_name",
            intermediary_onnx_model_name="repeated_name",
        )


@pytest.mark.skip(reason="Support for ONNX newer versions has been dropped.")
def test_repeated_file_uris_in_onnx_quantization_bert(
    fixture_model_bert_2_layers: segmentador.Segmenter,
):
    with pytest.raises(ValueError):
        segmentador.optimize.quantize_model(
            model=fixture_model_bert_2_layers,
            model_output_format="onnx",
            quantized_model_filename="repeated_name",
            intermediary_onnx_model_name="repeated_name",
        )
