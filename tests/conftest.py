"""Declare fixtures for tests."""
import typing as t
import pytest

import segmentador


class TestPaths(t.NamedTuple):
    model_bert: str = "pretrained_segmenter_model/2_6000_layer_model"
    model_lstm: str = (
        "pretrained_segmenter_model/128_6000_1_lstm/checkpoints/epoch=3-step=3591.ckpt"
    )
    tokenizer: str = "tokenizers/6000_subwords"
    legal_text_long: str = "tests/resources/test_legal_text_long.txt"
    legal_text_short: str = "tests/resources/test_legal_text_short.txt"


@pytest.fixture(scope="session")
def fixture_test_paths() -> str:
    return TestPaths()


@pytest.fixture(scope="session")
def fixture_model_bert_2_layers(fixture_test_paths: TestPaths) -> segmentador.BERTSegmenter:
    model = segmentador.BERTSegmenter(
        uri_model=fixture_test_paths.model_bert,
        inference_pooling_operation="assymetric-max",
        device="cpu",
        local_files_only=True,
    )

    return model


@pytest.fixture(scope="session")
def fixture_model_lstm_1_layer(fixture_test_paths: TestPaths) -> segmentador.LSTMSegmenter:
    model = segmentador.LSTMSegmenter(
        uri_model=fixture_test_paths.model_lstm,
        inference_pooling_operation="gaussian",
        device="cpu",
        local_files_only=True,
    )

    return model


@pytest.fixture(scope="session")
def fixture_legal_text_long(fixture_test_paths: TestPaths) -> str:
    with open(fixture_test_paths.legal_text_long, "r", encoding="utf-8") as f_in:
        text = f_in.read()

    return text


@pytest.fixture(scope="session")
def fixture_legal_text_short(fixture_test_paths: TestPaths) -> str:
    with open(fixture_test_paths.legal_text_short, "r", encoding="utf-8") as f_in:
        text = f_in.read()

    return text
