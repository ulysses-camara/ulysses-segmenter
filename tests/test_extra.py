"""Test extra features from the package."""
import pytest
import regex

import segmentador


def test_segmenter_model_to_string(fixture_model_2_layers: segmentador.Segmenter):
    assert isinstance(str(fixture_model_2_layers), str)


def test_lstm_instantiation_infer_model_parameters():
    segmentador.LSTMSegmenter(
        uri_model="pretrained_segmenter_model/128_6000_2_lstm/checkpoints/epoch=3-step=3591.ckpt",
        uri_tokenizer="tokenizers/6000_subwords",
        device="cpu",
        local_files_only=True,
    )
    assert True


@pytest.mark.parametrize("regex_justificativa", [None, "ABC"])
def test_model_preprocessing_no_justificativa(
    fixture_model_2_layers: segmentador.Segmenter,
    fixture_legal_text_short: str,
    regex_justificativa: str,
):
    preproc_text = fixture_model_2_layers.preprocess_legal_text(
        fixture_legal_text_short,
        return_justificativa=False,
        regex_justificativa=regex_justificativa,
    )
    double_spaces = "  "
    assert preproc_text.find(double_spaces) == -1 and preproc_text.find("\n") == -1


@pytest.mark.parametrize(
    "regex_justificativa,expected_just_length",
    [
        (None, 1),
        ("ABC", 0),
        ("JUSTIFICATIVA", 1),
        (regex.compile("ABC"), 0),
        # Note: regex.IGNORECASE == 2.
        (regex.compile("justificativa", flags=2), 1),
    ],
)
def test_model_preprocessing_with_justificativa(
    fixture_model_2_layers: segmentador.Segmenter,
    fixture_legal_text_short: str,
    regex_justificativa: str,
    expected_just_length: int,
):
    preproc_text, justificativa = fixture_model_2_layers.preprocess_legal_text(
        fixture_legal_text_short, return_justificativa=True, regex_justificativa=regex_justificativa
    )
    double_spaces = "  "
    assert (
        preproc_text.find(double_spaces) == -1
        and preproc_text.find("\n") == -1
        and len(justificativa) == expected_just_length
    )
