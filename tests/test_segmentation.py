"""General tests for arguments, model instantiation and segmentation."""
import pytest

import segmentador


@pytest.mark.parametrize("pooling_operation", ("max", "sum", "assymetric-max", "gaussian"))
def test_inference_pooling_operation_argument_with_long_text_and_bert(
    pooling_operation: str, fixture_legal_text_long: str
):
    model = segmentador.Segmenter(
        uri_model="pretrained_segmenter_model/2_6000_layer_model",
        inference_pooling_operation=pooling_operation,
        device="cpu",
        local_files_only=True,
    )
    segs = model(fixture_legal_text_long)
    assert len(segs) >= 50


@pytest.mark.parametrize("pooling_operation", ("max", "sum", "assymetric-max", "gaussian"))
def test_inference_pooling_operation_argument_with_short_text_and_lstm(
    pooling_operation: str, fixture_legal_text_short: str
):
    model = segmentador.LSTMSegmenter(
        uri_model="pretrained_segmenter_model/128_6000_lstm/checkpoints/epoch=3-step=3591.ckpt",
        uri_tokenizer="tokenizers/6000_subwords",
        lstm_hidden_layer_size=128,
        inference_pooling_operation=pooling_operation,
        device="cpu",
        local_files_only=True,
    )
    segs = model(fixture_legal_text_short)
    assert len(segs) == 8


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
    fixture_model_2_layers: segmentador.Segmenter,
    fake_legal_text: str,
    expected_justificativa_length: int,
):
    _, justificativa = fixture_model_2_layers(fake_legal_text, return_justificativa=True)
    return len(justificativa) == expected_justificativa_length


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
    regex_justificativa: str, fake_legal_text: str, expected_justificativa_length: int
):
    model = segmentador.Segmenter(
        uri_model="pretrained_segmenter_model/2_6000_layer_model",
        regex_justificativa=regex_justificativa,
        device="cpu",
        local_files_only=True,
    )

    _, justificativa = model(fake_legal_text, return_justificativa=True)

    return len(justificativa) == expected_justificativa_length


@pytest.mark.parametrize("batch_size", (1, 2, 3, 16, 10000))
def test_batch_size_with_short_text(
    fixture_model_2_layers: segmentador.Segmenter,
    fixture_legal_text_short: str,
    batch_size: int,
):
    segments = fixture_model_2_layers(fixture_legal_text_short, batch_size=batch_size)
    assert len(segments) == 9


@pytest.mark.parametrize("batch_size", (1, 2, 3, 16, 10000))
def test_batch_size_with_long_text(
    fixture_model_2_layers: segmentador.Segmenter,
    fixture_legal_text_long: str,
    batch_size: int,
):
    segments = fixture_model_2_layers(fixture_legal_text_long, batch_size=batch_size)
    assert len(segments) == 63


@pytest.mark.parametrize("window_shift_size", (1024, 512, 256, 1.0, 0.5, 0.25))
def test_window_shift_size(
    fixture_model_2_layers: segmentador.Segmenter,
    fixture_legal_text_long: str,
    window_shift_size: int,
):
    segments = fixture_model_2_layers(fixture_legal_text_long, window_shift_size=window_shift_size)
    assert len(segments) >= 59
