"""Declare fixtures for tests."""
import os
import shutil

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
        inference_pooling_operation="asymmetric-max",
        device="cpu",
        local_files_only=False,
        cache_dir_model=fixture_test_paths.cache_dir_models,
        show_download_progress_bar=False,
    )

    yield model

    shutil.rmtree(os.path.join(fixture_test_paths.cache_dir_models, fixture_test_paths.model_bert))

    try:
        os.rmdir(fixture_test_paths.cache_dir_models)

    except OSError:
        pass


@pytest.fixture(name="fixture_model_lstm_1_layer", scope="session")
def fn_fixture_model_lstm_1_layer(fixture_test_paths: paths.TestPaths) -> segmentador.LSTMSegmenter:
    model = segmentador.LSTMSegmenter(
        uri_model=fixture_test_paths.model_lstm,
        uri_tokenizer=fixture_test_paths.tokenizer,
        inference_pooling_operation="gaussian",
        device="cpu",
        local_files_only=False,
        cache_dir_model=fixture_test_paths.cache_dir_models,
        cache_dir_tokenizer=fixture_test_paths.cache_dir_tokenizers,
        show_download_progress_bar=False,
    )

    yield model

    os.remove(
        os.path.join(fixture_test_paths.cache_dir_models, f"{fixture_test_paths.model_lstm}.pt")
    )
    shutil.rmtree(
        os.path.join(fixture_test_paths.cache_dir_tokenizers, fixture_test_paths.tokenizer)
    )

    try:
        os.rmdir(fixture_test_paths.cache_dir_models)

    except OSError:
        pass

    try:
        os.rmdir(fixture_test_paths.cache_dir_tokenizers)

    except OSError:
        pass


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


@pytest.fixture(name="fixture_legal_text_with_noise", scope="session")
def fn_fixture_legal_text_with_noise(fixture_test_paths: paths.TestPaths) -> str:
    with open(fixture_test_paths.legal_text_with_noise, "r", encoding="utf-8") as f_in:
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
        onnx_opset_version=17,
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
        model_output_format="torch_jit",
        check_cached=False,
        verbose=False,
    )

    torch_lstm_model = segmentador.optimize.TorchJITLSTMSegmenter(
        uri_model=output_paths.output_uri,
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
        onnx_opset_version=17,
        model_output_format="onnx",
        check_cached=False,
        verbose=False,
    )

    # Note: failing for onnx_opset_version < 17, since they had no support for LayerNormalization
    # onnx.checker.check_model(os.path.join(output_paths.output_uri, "model_quantized.onnx"))

    onnx_bert_model = segmentador.optimize.ONNXBERTSegmenter(
        uri_model=output_paths.output_uri,
        uri_tokenizer=fixture_model_bert_2_layers.tokenizer.name_or_path,
    )

    yield onnx_bert_model

    try:
        shutil.rmtree(output_paths.output_uri)

    except OSError:
        pass


@pytest.fixture(name="fixture_quantized_model_bert_torch", scope="session")
def fn_fixture_quantized_model_bert_torch(
    fixture_test_paths: paths.TestPaths, fixture_model_bert_2_layers: segmentador.BERTSegmenter
) -> segmentador.optimize.TorchJITBERTSegmenter:
    output_paths = segmentador.optimize.quantize_model(
        model=fixture_model_bert_2_layers,
        quantized_model_filename=fixture_test_paths.quantized_test_model_bert_torch,
        quantized_model_dirpath=fixture_test_paths.quantized_test_model_dirname,
        model_output_format="torch_jit",
        check_cached=False,
        verbose=False,
    )

    torch_bert_model = segmentador.optimize.TorchJITBERTSegmenter(
        uri_model=output_paths.output_uri,
    )

    yield torch_bert_model

    for path in set(output_paths):
        if path:
            os.remove(path)

    try:
        os.rmdir(fixture_test_paths.quantized_test_model_dirname)

    except OSError:
        pass
