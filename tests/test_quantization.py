"""Tests related to quantization and quantized segmenter models."""
import timeit
import os

import segmentador
import segmentador.optimize

from . import paths


def test_inference_lstm_onnx(
    fixture_quantized_model_lstm_onnx: segmentador.optimize.ONNXLSTMSegmenter,
    fixture_legal_text_short: str,
):
    output = fixture_quantized_model_lstm_onnx(
        fixture_legal_text_short,
        return_labels=True,
        return_logits=True,
    )

    num_segments = len(output.segments)

    assert num_segments and output.logits.shape == (output.labels.size, 4)


def test_inference_bert_onnx(
    fixture_quantized_model_bert_onnx: segmentador.optimize.ONNXBERTSegmenter,
    fixture_legal_text_short: str,
):
    output = fixture_quantized_model_bert_onnx(
        fixture_legal_text_short,
        return_labels=True,
        return_logits=True,
    )

    num_segments = len(output.segments)

    assert num_segments and output.logits.shape == (output.labels.size, 4)


def test_model_lstm_inference_time_standard_vs_quantized_torch(
    fixture_quantized_model_lstm_torch: segmentador.LSTMSegmenter,
    fixture_model_lstm_1_layer: segmentador.LSTMSegmenter,
    fixture_legal_text_long: str,
):
    common_kwargs = dict(
        number=20,
        repeat=3,
        globals=dict(
            fixture_quantized_model_lstm_torch=fixture_quantized_model_lstm_torch,
            fixture_legal_text_long=fixture_legal_text_long,
            fixture_model_lstm_1_layer=fixture_model_lstm_1_layer,
        ),
    )

    times_quantized = timeit.repeat(
        "fixture_quantized_model_lstm_torch(fixture_legal_text_long)",
        **common_kwargs,
    )
    times_standard = timeit.repeat(
        "fixture_model_lstm_1_layer(fixture_legal_text_long)",
        **common_kwargs,
    )

    best_time_quantized = min(times_quantized)
    best_time_standard = min(times_standard)

    assert best_time_quantized < best_time_standard


def test_model_lstm_inference_time_standard_vs_quantized_onnx(
    fixture_quantized_model_lstm_onnx: segmentador.optimize.ONNXLSTMSegmenter,
    fixture_model_lstm_1_layer: segmentador.LSTMSegmenter,
    fixture_legal_text_long: str,
):
    common_kwargs = dict(
        number=20,
        repeat=3,
        globals=dict(
            fixture_quantized_model_lstm_onnx=fixture_quantized_model_lstm_onnx,
            fixture_legal_text_long=fixture_legal_text_long,
            fixture_model_lstm_1_layer=fixture_model_lstm_1_layer,
        ),
    )

    times_quantized = timeit.repeat(
        "fixture_quantized_model_lstm_onnx(fixture_legal_text_long)",
        **common_kwargs,
    )
    times_standard = timeit.repeat(
        "fixture_model_lstm_1_layer(fixture_legal_text_long)",
        **common_kwargs,
    )

    best_time_quantized = min(times_quantized)
    best_time_standard = min(times_standard)

    assert best_time_quantized < best_time_standard


def test_model_bert_inference_time_standard_vs_quantized_onnx(
    fixture_quantized_model_bert_onnx: segmentador.optimize.ONNXBERTSegmenter,
    fixture_model_bert_2_layers: segmentador.BERTSegmenter,
    fixture_legal_text_long: str,
):
    common_kwargs = dict(
        number=10,
        repeat=3,
        globals=dict(
            fixture_quantized_model_bert_onnx=fixture_quantized_model_bert_onnx,
            fixture_legal_text_long=fixture_legal_text_long,
            fixture_model_bert_2_layers=fixture_model_bert_2_layers,
        ),
    )

    times_quantized = timeit.repeat(
        "fixture_quantized_model_bert_onnx(fixture_legal_text_long)",
        **common_kwargs,
    )
    times_standard = timeit.repeat(
        "fixture_model_bert_2_layers(fixture_legal_text_long)",
        **common_kwargs,
    )

    best_time_quantized = min(times_quantized)
    best_time_standard = min(times_standard)

    assert best_time_quantized < best_time_standard


def test_create_bert_model_with_default_name_onnx(
    fixture_test_paths: paths.TestPaths,
    fixture_model_bert_2_layers: segmentador.BERTSegmenter,
):
    output_paths = segmentador.optimize.quantize_model(
        model=fixture_model_bert_2_layers,
        quantized_model_dirpath=fixture_test_paths.quantized_test_model_dirname,
        optimization_level=99,
        onnx_opset_version=15,
        model_output_format="onnx",
        check_cached=False,
        verbose=False,
    )

    assert os.path.isfile(output_paths.output_uri)

    for path in set(output_paths):
        os.remove(path)


def test_create_lstm_model_with_default_name_torch(
    fixture_test_paths: paths.TestPaths,
    fixture_model_lstm_1_layer: segmentador.LSTMSegmenter,
):
    output_paths = segmentador.optimize.quantize_model(
        model=fixture_model_lstm_1_layer,
        quantized_model_dirpath=fixture_test_paths.quantized_test_model_dirname,
        model_output_format="torch",
        check_cached=False,
        verbose=False,
    )

    assert os.path.isfile(output_paths.output_uri)

    for path in set(output_paths):
        if path:
            os.remove(path)
