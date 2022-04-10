"""Declare fixtures for tests."""
import pytest

import segmentador


@pytest.fixture(scope="session")
def fixture_model_bert_2_layers() -> segmentador.Segmenter:
    model = segmentador.Segmenter(
        uri_model="pretrained_segmenter_model/2_6000_layer_model",
        inference_pooling_operation="assymetric-max",
        device="cpu",
        local_files_only=True,
    )

    return model


@pytest.fixture(scope="session")
def fixture_legal_text_long() -> str:
    with open("tests/resources/test_legal_text_long.txt", "r", encoding="utf-8") as f_in:
        text = f_in.read()

    return text


@pytest.fixture(scope="session")
def fixture_legal_text_short() -> str:
    with open("tests/resources/test_legal_text_short.txt", "r", encoding="utf-8") as f_in:
        text = f_in.read()

    return text
