import pytest

import segmentador


@pytest.fixture(scope="session")
def fixture_model_2_layers() -> segmentador.Segmenter:
    """TODO."""
    model = segmentador.Segmenter(
        uri_model="pretrained_segmenter_model/2_6000_layer_model",
        uri_tokenizer="tokenizers/6000_subwords",
        inference_pooling_operation="assymetric-max",
        device="cpu",
        local_files_only=True,
    )

    return model


@pytest.fixture(scope="session")
def fixture_legal_text() -> str:
    with open("tests/resources/test_legal_text.txt", "r") as f_in:
        text = f_in.read()

    return text
