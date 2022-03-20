"""Tests that are expected to fail, raise expections or warnings."""
import pytest

import segmentador


@pytest.mark.parametrize("batch_size", (0, -1, -100))
def test_invalid_batch_size(
    fixture_model_2_layers: segmentador.Segmenter,
    fixture_legal_text_short: str,
    batch_size: int,
):
    with pytest.raises(ValueError):
        fixture_model_2_layers(fixture_legal_text_short, batch_size=batch_size)


@pytest.mark.parametrize("moving_window_size", (0, -1, -100))
def test_invalid_moving_window_size(
    fixture_model_2_layers: segmentador.Segmenter,
    fixture_legal_text_short: str,
    moving_window_size: int,
):
    with pytest.raises(ValueError):
        fixture_model_2_layers(fixture_legal_text_short, moving_window_size=moving_window_size)


@pytest.mark.parametrize("window_shift_size", (0, -1, -100, 0.0, 1.001, -0.01))
def test_invalid_window_shift_size(
    fixture_model_2_layers: segmentador.Segmenter,
    fixture_legal_text_short: str,
    window_shift_size: int,
):
    with pytest.raises(ValueError):
        fixture_model_2_layers(fixture_legal_text_short, window_shift_size=window_shift_size)


@pytest.mark.parametrize("window_shift_size", (1025, 10000))
def test_warning_window_shift_size(
    fixture_model_2_layers: segmentador.Segmenter,
    fixture_legal_text_short: str,
    window_shift_size: int,
):
    with pytest.warns(UserWarning):
        fixture_model_2_layers(fixture_legal_text_short, window_shift_size=window_shift_size)


@pytest.mark.parametrize("moving_window_size", (1025, 10000))
def test_warning_moving_window_size(
    fixture_model_2_layers: segmentador.Segmenter,
    fixture_legal_text_short: str,
    moving_window_size: int,
):
    with pytest.warns(UserWarning):
        fixture_model_2_layers(fixture_legal_text_short, moving_window_size=moving_window_size)


@pytest.mark.parametrize("inference_pooling_operation", (None, "", "avg"))
def test_invalid_inference_pooling_operation(inference_pooling_operation: str):
    with pytest.raises(ValueError):
        segmentador.Segmenter(
            uri_model="pretrained_segmenter_model/2_6000_layer_model",
            uri_tokenizer="tokenizers/6000_subwords",
            inference_pooling_operation=inference_pooling_operation,
            device="cpu",
            local_files_only=True,
        )
