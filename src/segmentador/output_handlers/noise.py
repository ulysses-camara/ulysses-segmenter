"""Handle noise subsegments."""
import typing as t

import numpy.typing as npt
import numpy as np


__all__ = [
    "remove_noise_subsegments",
]


def remove_noise_subsegments(
    label_ids: npt.NDArray[np.int32],
    *args: npt.NDArray[t.Any],
    label2id: t.Optional[t.Dict[str, int]] = None,
) -> t.Tuple[npt.NDArray[np.int32], t.Tuple[npt.NDArray[t.Any], ...]]:
    """TODO"""
    label2id = label2id or {}
    seg_cls_id = label2id.get("SEG_START", 1)
    noise_start_cls_id = label2id.get("NOISE_START", 2)
    noise_end_cls_id = label2id.get("NOISE_END", 3)

    cand_noise_start_inds = np.flatnonzero(label_ids == noise_start_cls_id)

    # Note: noise ends either on segment end or right before the next 'noise_end' token.
    noise_end_inds = np.logical_or(label_ids == noise_end_cls_id, label_ids == seg_cls_id)
    noise_end_inds = np.flatnonzero(noise_end_inds)

    if cand_noise_start_inds.size == 0:
        return label_ids, args

    j = 0
    noise_subsegment_inds: t.List[t.Tuple[int, int]] = []

    # pylint: disable='consider-using-enumerate'
    for i in range(len(cand_noise_start_inds)):
        while j < len(noise_end_inds) and cand_noise_start_inds[i] > noise_end_inds[j]:
            # Note: skip spurious 'noise end' tokens.
            j += 1

        if j >= len(noise_end_inds):
            # Note: run out of explicit 'noise end'. Consider the end of document the 'noise end'.
            noise_subsegment_inds.append((cand_noise_start_inds[-1], len(label_ids)))
            break

        if i < len(cand_noise_start_inds) - 1 and cand_noise_start_inds[i + 1] < noise_end_inds[j]:
            # Note: 'noise start' token is not closest to the next 'noise end'. Skip it.
            continue

        noise_subsegment_inds.append((cand_noise_start_inds[i], noise_end_inds[j]))
        j += 1

    non_noise_mask = np.full_like(label_ids, fill_value=True, dtype=bool)
    for i_start, i_end in noise_subsegment_inds:
        non_noise_mask[i_start:i_end] = False

    label_ids = label_ids[non_noise_mask]
    ret: t.List[npt.NDArray[t.Any]] = []

    for seq in args:
        seq = seq.squeeze()
        seq = seq[non_noise_mask[: len(seq)], ...]
        ret.append(seq)

    return label_ids, tuple(ret)
