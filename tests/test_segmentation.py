"""General tests for arguments, model instantiation and segmentation."""
import typing as t

import pytest
import pandas as pd
import datasets

import segmentador

from . import paths


def no_segmentation_at_middle_subwords(segs: t.List[str]) -> bool:
    """Check if no word has been segmented."""
    return not any(s.startswith("##") for s in segs)


@pytest.mark.parametrize("pooling_operation", ("max", "sum", "asymmetric-max", "gaussian"))
def test_inference_pooling_operation_argument_with_long_text_and_bert(
    pooling_operation: str, fixture_test_paths: paths.TestPaths, fixture_legal_text_long: str
):
    model = segmentador.Segmenter(
        uri_model=fixture_test_paths.model_bert,
        inference_pooling_operation=pooling_operation,
        device="cpu",
        local_files_only=False,
        cache_dir_model=fixture_test_paths.cache_dir_models,
    )
    segs = model(fixture_legal_text_long, batch_size=4)
    assert len(segs) >= 50 and no_segmentation_at_middle_subwords(segs)


@pytest.mark.parametrize("pooling_operation", ("max", "sum", "asymmetric-max", "gaussian"))
def test_inference_pooling_operation_argument_with_short_text_and_lstm(
    pooling_operation: str, fixture_test_paths: paths.TestPaths, fixture_legal_text_short: str
):
    model = segmentador.LSTMSegmenter(
        uri_model=fixture_test_paths.model_lstm,
        uri_tokenizer=fixture_test_paths.tokenizer,
        inference_pooling_operation=pooling_operation,
        device="cpu",
        lstm_hidden_layer_size=256,
        lstm_num_layers=1,
        local_files_only=False,
        cache_dir_model=fixture_test_paths.cache_dir_models,
        cache_dir_tokenizer=fixture_test_paths.cache_dir_tokenizers,
    )
    segs = model(fixture_legal_text_short)
    assert len(segs) >= 6 and no_segmentation_at_middle_subwords(segs)


@pytest.mark.parametrize(
    "fake_legal_text, expected_justificativa_length",
    [
        ("abc JUSTIFICATIVA abc", 1),
        ("abc JUST jk", 0),
        ("abc justificativa do documento", 0),
        ("justificativa JUSTIFICATIVA abc", 1),
        ("abc JUSTIFICAÇÃO abc", 1),
        ("justificativa justificação", 0),
        ("abc ANEXO abc", 1),
        ("abc o anexo do documento", 0),
        ("abc JUSTIFICATIVA dfe ANEXO xyz", 2),
    ],
)
def test_justificativa_regex_standard(
    fixture_model_bert_2_layers: segmentador.Segmenter,
    fake_legal_text: str,
    expected_justificativa_length: int,
):
    _, justificativa = fixture_model_bert_2_layers(fake_legal_text, return_justificativa=True)
    assert len(justificativa) == expected_justificativa_length


@pytest.mark.parametrize(
    "regex_justificativa, fake_legal_text, expected_justificativa_length",
    [
        (r"JUSTIF[iI]CATI\s*VA", "abc JUSTIFiCATI  VA abc", 1),
        (r"ABC", "documento JUSTIFICATIVA abc", 0),
        (r"ABC", "documento ANEXO abc", 0),
        (r"abc|xyz", "documento abc 000 xyz 111", 2),
    ],
)
def test_justificativa_regex_custom(
    fixture_model_bert_2_layers: segmentador.Segmenter,
    regex_justificativa: str,
    fake_legal_text: str,
    expected_justificativa_length: int,
):
    _, justificativa = fixture_model_bert_2_layers(
        fake_legal_text, return_justificativa=True, regex_justificativa=regex_justificativa
    )

    assert len(justificativa) == expected_justificativa_length


@pytest.mark.parametrize("batch_size", (1, 2, 3, 16, 10000))
def test_batch_size_with_short_text(
    fixture_model_bert_2_layers: segmentador.Segmenter,
    fixture_legal_text_short: str,
    batch_size: int,
):
    segs = fixture_model_bert_2_layers(fixture_legal_text_short, batch_size=batch_size)
    assert len(segs) == 9 and no_segmentation_at_middle_subwords(segs)


@pytest.mark.parametrize("batch_size", (1, 2, 3, 16, 10000))
def test_batch_size_with_long_text(
    fixture_model_bert_2_layers: segmentador.Segmenter,
    fixture_legal_text_long: str,
    batch_size: int,
):
    segs = fixture_model_bert_2_layers(fixture_legal_text_long, batch_size=batch_size)
    assert len(segs) >= 60 and no_segmentation_at_middle_subwords(segs)


@pytest.mark.parametrize("window_shift_size", (1024, 512, 256, 1.0, 0.5, 0.25))
def test_window_shift_size(
    fixture_model_bert_2_layers: segmentador.Segmenter,
    fixture_legal_text_long: str,
    window_shift_size: int,
):
    segs = fixture_model_bert_2_layers(fixture_legal_text_long, window_shift_size=window_shift_size)
    assert len(segs) >= 50 and no_segmentation_at_middle_subwords(segs)


@pytest.mark.parametrize("input_type_fn", [tuple, list, pd.Series])
def test_input_type(fixture_model_bert_2_layers: segmentador.Segmenter, input_type_fn):
    with pytest.warns(UserWarning):
        segs = fixture_model_bert_2_layers(
            input_type_fn(["Projeto de Lei (do XYZ).", "Artigo 10: abc xyz", "a) 0 abc b) xyz"])
        )

    assert len(segs) > 0 and no_segmentation_at_middle_subwords(segs)


def test_input_type_pandas_df(fixture_model_bert_2_layers: segmentador.Segmenter):
    with pytest.raises(TypeError):
        pandas_df = pd.DataFrame.from_dict({"a": ["Artigo 1: abc", "Artigo 2: xyz"]})
        fixture_model_bert_2_layers(pandas_df)


@pytest.mark.parametrize("return_tensors", (None, "pt", "np"))
def test_input_type_pre_tokenized(
    fixture_model_bert_2_layers: segmentador.Segmenter,
    fixture_legal_text_short: str,
    return_tensors: str,
):
    preproc_text = fixture_model_bert_2_layers.preprocess_legal_text(fixture_legal_text_short)
    tokenized_input = fixture_model_bert_2_layers.tokenizer(
        preproc_text, return_tensors=return_tensors
    )
    segs = fixture_model_bert_2_layers(tokenized_input)
    assert len(segs) == 9 and no_segmentation_at_middle_subwords(segs)


def test_input_type_huggingface_dataset(
    fixture_model_bert_2_layers: segmentador.Segmenter, fixture_legal_text_short: str
):
    preproc_text = fixture_model_bert_2_layers.preprocess_legal_text(fixture_legal_text_short)
    tokenized_input = fixture_model_bert_2_layers.tokenizer(preproc_text)
    dataset = datasets.Dataset.from_dict(tokenized_input)
    segs = fixture_model_bert_2_layers(dataset)
    assert len(segs) == 9 and no_segmentation_at_middle_subwords(segs)
