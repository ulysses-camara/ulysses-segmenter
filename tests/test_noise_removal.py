"""Test noise removal (regarding 3rd and 4th class labels)."""
import typing as t

import pytest
import numpy as np

import segmentador
import segmentador.output_handlers


@pytest.mark.parametrize(
    "label_ids, expected_length_removed",
    [
        ((0, 0, 0, 1, 0, 0, 2, 3, 1, 0, 0), 1),
        ((0, 0, 3, 1, 0, 0, 2, 3, 1, 0, 0), 1),
        ((2, 0, 3, 1, 0, 0, 2, 3, 1, 0, 0), 3),
        ((2, 0, 3, 1, 0, 0, 2, 0, 1, 0, 0), 4),
        ((2, 0, 3, 1, 0, 0, 2, 1, 0, 0, 0), 3),
        ((2, 0, 3, 1, 0, 0, 2, 0, 0, 0, 0), 7),
        ([2, 0, 3, 1, 0, 0, 2, 2, 3, 0, 0], 3),
        ([2, 0, 3, 1, 0, 0, 2, 0, 2, 0, 3], 4),
        ([2, 0, 3, 1, 0, 0, 2, 0, 2, 0, 0], 5),
        ([2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 11),
        ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0),
        ([0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0], 0),
        ([0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0], 3),
        ([0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1], 2),
        ([0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 1], 1),
        ([2, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1], 9),
        ([2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 10),
    ],
)
def test_noise_removal_in_toy_examples(label_ids: t.Sequence[int], expected_length_removed: int):
    label_ids = np.asarray(label_ids, dtype=int)
    label_ids_denoised, extra_args = segmentador.output_handlers.remove_noise_subsegments(label_ids)
    expected_length = len(label_ids) - expected_length_removed

    assert not extra_args
    assert expected_length == len(label_ids_denoised)


@pytest.mark.parametrize(
    "label_ids, expected_length_removed",
    [
        ((0, 0, 0, 1, 0, 0, 2, 3, 1, 0, 0), 1),
        ((0, 0, 3, 1, 0, 0, 2, 3, 1, 0, 0), 1),
        ((2, 0, 3, 1, 0, 0, 2, 3, 1, 0, 0), 3),
        ((2, 0, 3, 1, 0, 0, 2, 0, 1, 0, 0), 4),
        ((2, 0, 3, 1, 0, 0, 2, 1, 0, 0, 0), 3),
        ((2, 0, 3, 1, 0, 0, 2, 0, 0, 0, 0), 2),
        ([2, 0, 3, 1, 0, 0, 2, 2, 3, 0, 0], 3),
        ([2, 0, 3, 1, 0, 0, 2, 0, 2, 0, 3], 4),
        ([2, 0, 3, 1, 0, 0, 2, 0, 2, 0, 0], 5),
        ([2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0),
        ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0),
        ([0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0], 0),
        ([0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0], 3),
        ([0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1], 2),
        ([0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 1], 1),
        ([2, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1], 0),
        ([2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 0),
    ],
)
def test_noise_removal_with_maximum_noise_length(
    label_ids: t.Sequence[int], expected_length_removed: int
):
    label_ids = np.asarray(label_ids, dtype=int)
    label_ids_denoised, extra_args = segmentador.output_handlers.remove_noise_subsegments(
        label_ids, maximum_noise_subsegment_length=3
    )
    expected_length = len(label_ids) - expected_length_removed

    assert not extra_args
    assert expected_length == len(label_ids_denoised)


@pytest.mark.parametrize(
    "label_ids, extra_arg_count, expected_length_removed",
    [
        ((0, 0, 0, 1, 0, 0, 2, 3, 1, 0, 0), 1, 1),
        ((0, 0, 3, 1, 0, 0, 2, 3, 1, 0, 0), 2, 1),
        ((2, 0, 3, 1, 0, 0, 2, 3, 1, 0, 0), 3, 3),
        ((2, 0, 3, 1, 0, 0, 2, 0, 1, 0, 0), 4, 4),
        ((2, 0, 3, 1, 0, 0, 2, 1, 0, 0, 0), 8, 3),
        ((2, 0, 3, 1, 0, 0, 2, 0, 0, 0, 0), 2, 7),
        ([2, 0, 3, 1, 0, 0, 2, 2, 3, 0, 0], 1, 3),
        ([2, 0, 3, 1, 0, 0, 2, 0, 2, 0, 3], 3, 4),
        ([2, 0, 3, 1, 0, 0, 2, 0, 2, 0, 0], 3, 5),
        ([2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 6, 11),
        ([2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 1, 11),
        ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 1, 0),
        ([0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0], 2, 0),
        ([0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0], 3, 3),
        ([0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1], 3, 2),
        ([0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 1], 3, 1),
        ([2, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1], 2, 9),
        ([2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 1, 10),
    ],
)
def test_noise_removal_in_toy_examples_with_extra_args(
    label_ids: t.Sequence[int], extra_arg_count: int, expected_length_removed: int
):
    label_ids = np.asarray(label_ids, dtype=int)
    extra_args = np.random.randint(-10, 11, size=(extra_arg_count, label_ids.size))
    label_ids_denoised, extra_args = segmentador.output_handlers.remove_noise_subsegments(
        label_ids, *extra_args
    )
    expected_length = len(label_ids) - expected_length_removed

    assert all(len(item) == len(label_ids_denoised) for item in extra_args)
    assert expected_length == len(label_ids_denoised)


def test_noise_removal_in_legal_text(
    fixture_model_lstm_1_layer: segmentador.LSTMSegmenter, fixture_legal_text_long: str
):
    ret_a = fixture_model_lstm_1_layer(
        fixture_legal_text_long,
        remove_noise_subsegments=False,
        return_labels=True,
        return_justificativa=True,
    )

    ret_b = fixture_model_lstm_1_layer(
        fixture_legal_text_long,
        remove_noise_subsegments=True,
        return_labels=True,
        return_justificativa=True,
    )

    segs_a = ret_a.segments
    segs_b = ret_b.segments

    justs_a = ret_a.justificativa
    justs_b = ret_b.justificativa

    noise_start_id = 2
    segs_with_noise_ids = set(np.flatnonzero(ret_a.labels == noise_start_id))

    assert segs_with_noise_ids
    assert len(ret_b.labels) <= len(ret_a.labels)
    assert all(len(seg_b) <= len(seg_a) for seg_a, seg_b in zip(segs_a, segs_b))
    assert all(just_b == just_a for just_a, just_b in zip(justs_a, justs_b))
