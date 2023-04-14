"""Test extra features from the package."""
import pytest
import regex

import segmentador

from . import paths


def test_segmenter_model_to_string(fixture_model_bert_2_layers: segmentador.Segmenter):
    assert isinstance(str(fixture_model_bert_2_layers), str)


def test_lstm_instantiation_infer_model_parameters(fixture_test_paths: paths.TestPaths):
    segmentador.LSTMSegmenter(
        uri_model=fixture_test_paths.model_lstm,
        uri_tokenizer=fixture_test_paths.tokenizer,
        device="cpu",
        local_files_only=False,
        cache_dir_model=fixture_test_paths.cache_dir_models,
        cache_dir_tokenizer=fixture_test_paths.cache_dir_tokenizers,
    )
    assert True


@pytest.mark.parametrize("regex_justificativa", [None, "ABC"])
def test_model_preprocessing_no_justificativa(
    fixture_model_bert_2_layers: segmentador.Segmenter,
    fixture_legal_text_short: str,
    regex_justificativa: str,
):
    preproc_text = fixture_model_bert_2_layers.preprocess_legal_text(
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
    fixture_model_bert_2_layers: segmentador.Segmenter,
    fixture_legal_text_short: str,
    regex_justificativa: str,
    expected_just_length: int,
):
    preproc_text, justificativa = fixture_model_bert_2_layers.preprocess_legal_text(
        fixture_legal_text_short, return_justificativa=True, regex_justificativa=regex_justificativa
    )
    double_spaces = "  "
    assert (
        preproc_text.find(double_spaces) == -1
        and preproc_text.find("\n") == -1
        and len(justificativa) == expected_just_length
    )


@pytest.mark.parametrize("apply_postprocessing", [True, False])
def test_postprocessing(
    fixture_model_bert_2_layers: segmentador.Segmenter,
    fixture_legal_text_short: str,
    apply_postprocessing: bool,
):
    segs = fixture_model_bert_2_layers(
        fixture_legal_text_short,
        apply_postprocessing=apply_postprocessing,
    )

    reg_spurious_whitespaces = regex.compile(
        r"""
        \s+[\.:;)\]}]|[\[({]\s+|
        (?<=[0-9])\s+\.\s*(?=[0-9])|
        (?<=[0-9])\s*\.\s+(?=[0-9])|
        (?<=[a-zçáéíóúãẽõâêôü])\s+-\s*(?=[a-zçáéíóúãẽõâêôü])|
        (?<=[a-zçáéíóúãẽõâêôü])\s*-\s+(?=[a-zçáéíóúãẽõâêôü])
    """,
        regex.VERBOSE,  # pylint: disable='no-member'
    )

    if apply_postprocessing:
        assert all(reg_spurious_whitespaces.search(seg) is None for seg in segs)
    else:
        assert any(reg_spurious_whitespaces.search(seg) for seg in segs)
