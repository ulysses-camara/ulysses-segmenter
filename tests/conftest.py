"""Declare fixtures for tests."""
import os

import pytest
import onnx

import segmentador
import segmentador.optimize

from . import paths


@pytest.fixture(name="fixture_test_paths", scope="session")
def fn_fixture_test_paths() -> str:
    return paths.TestPaths()


@pytest.fixture(name="fixture_model_bert_2_layers", scope="session")
def fn_fixture_model_bert_2_layers(
    fixture_test_paths: paths.TestPaths,
) -> segmentador.BERTSegmenter:
    model = segmentador.BERTSegmenter(
        uri_model=fixture_test_paths.model_bert,
        inference_pooling_operation="assymetric-max",
        device="cpu",
        local_files_only=True,
    )

    return model


@pytest.fixture(name="fixture_model_lstm_1_layer", scope="session")
def fn_fixture_model_lstm_1_layer(fixture_test_paths: paths.TestPaths) -> segmentador.LSTMSegmenter:
    model = segmentador.LSTMSegmenter(
        uri_model=fixture_test_paths.model_lstm,
        uri_tokenizer=fixture_test_paths.tokenizer,
        inference_pooling_operation="gaussian",
        device="cpu",
        local_files_only=True,
    )

    return model


@pytest.fixture(name="fixture_legal_text_long", scope="session")
def fn_fixture_legal_text_long(fixture_test_paths: paths.TestPaths) -> str:
    with open(fixture_test_paths.legal_text_long, "r", encoding="utf-8") as f_in:
        text = f_in.read()

    return text


@pytest.fixture(name="fixture_legal_text_short", scope="session")
def fn_fixture_legal_text_short(fixture_test_paths: paths.TestPaths) -> str:
    with open(fixture_test_paths.legal_text_short, "r", encoding="utf-8") as f_in:
        text = f_in.read()

    return text


@pytest.fixture(name="fixture_quantized_model_lstm_onnx", scope="session")
def fn_fixture_quantized_model_lstm_onnx(
    fixture_test_paths: paths.TestPaths, fixture_model_lstm_1_layer: segmentador.LSTMSegmenter
) -> segmentador.optimize.ONNXLSTMSegmenter:
    output_paths = segmentador.optimize.quantize_model(
        model=fixture_model_lstm_1_layer,
        quantized_model_filename=fixture_test_paths.quantized_test_model_lstm_onnx,
        quantized_model_dirpath=fixture_test_paths.quantized_test_model_dirname,
        optimization_level=99,
        onnx_opset_version=15,
        model_output_format="onnx",
        check_cached=False,
        verbose=False,
    )

    onnx.checker.check_model(output_paths.output_uri)

    onnx_lstm_model = segmentador.optimize.ONNXLSTMSegmenter(
        uri_model=output_paths.output_uri,
        uri_tokenizer=fixture_model_lstm_1_layer.tokenizer.name_or_path,
    )

    yield onnx_lstm_model

    for path in set(output_paths):
        if path:
            os.remove(path)

    try:
        os.rmdir(fixture_test_paths.quantized_test_model_dirname)

    except OSError:
        pass


@pytest.fixture(name="fixture_quantized_model_lstm_torch", scope="session")
def fn_fixture_quantized_model_lstm_torch(
    fixture_test_paths: paths.TestPaths, fixture_model_lstm_1_layer: segmentador.LSTMSegmenter
) -> segmentador.LSTMSegmenter:
    output_paths = segmentador.optimize.quantize_model(
        model=fixture_model_lstm_1_layer,
        quantized_model_filename=fixture_test_paths.quantized_test_model_lstm_torch,
        quantized_model_dirpath=fixture_test_paths.quantized_test_model_dirname,
        model_output_format="torch",
        check_cached=False,
        verbose=False,
    )

    torch_lstm_model = segmentador.LSTMSegmenter(
        uri_model=output_paths.output_uri,
        uri_tokenizer=fixture_model_lstm_1_layer.tokenizer.name_or_path,
        from_quantized_weights=True,
        device="cpu",
    )

    yield torch_lstm_model

    for path in set(output_paths):
        if path:
            os.remove(path)

    try:
        os.rmdir(fixture_test_paths.quantized_test_model_dirname)

    except OSError:
        pass


@pytest.fixture(name="fixture_quantized_model_bert_onnx", scope="session")
def fn_fixture_quantized_model_bert_onnx(
    fixture_test_paths: paths.TestPaths, fixture_model_bert_2_layers: segmentador.BERTSegmenter
) -> segmentador.optimize.ONNXBERTSegmenter:
    output_paths = segmentador.optimize.quantize_model(
        model=fixture_model_bert_2_layers,
        quantized_model_filename=fixture_test_paths.quantized_test_model_bert_onnx,
        quantized_model_dirpath=fixture_test_paths.quantized_test_model_dirname,
        optimization_level=99,
        onnx_opset_version=15,
        model_output_format="onnx",
        check_cached=False,
        verbose=False,
    )

    # Note: will fail for onnx_opset_version <= 15, since there's no support for LayerNormalization
    # onnx.checker.check_model(output_paths.output_uri)

    onnx_bert_model = segmentador.optimize.ONNXBERTSegmenter(
        uri_model=output_paths.output_uri,
        uri_tokenizer=fixture_model_bert_2_layers.tokenizer.name_or_path,
        uri_onnx_config=output_paths.onnx_config_uri,
    )

    yield onnx_bert_model

    for path in set(output_paths):
        if path:
            os.remove(path)

    try:
        os.rmdir(fixture_test_paths.quantized_test_model_dirname)

    except OSError:
        pass
