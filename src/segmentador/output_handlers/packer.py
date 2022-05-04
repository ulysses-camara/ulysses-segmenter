"""Pack segmenter results into a named tuple."""
import typing as t
import collections


def pack_results(
    keys: t.Sequence[str], vals: t.Sequence[t.Any], inclusion: t.Sequence[bool]
) -> t.Union[t.List[str], t.Tuple[t.List[t.Any], ...]]:
    """Build result tuple (if more than one value) or return segment list."""
    ret_keys: t.List[str] = []
    ret_vals: t.List[t.Any] = []

    for key, val, inc in zip(keys, vals, inclusion):
        if not inc:
            continue

        ret_keys.append(key)
        ret_vals.append(val)

    if len(ret_vals) == 1:
        segs: t.List[str] = ret_vals[0]
        return segs

    # Note: pylint config below is due to a weird, possible buggy, error that only
    # occurs with Python 3.9 @ GitHub.
    # pylint: disable=unused-variable
    ret_type = collections.namedtuple("SegmentationResults", ret_keys)  # type: ignore

    return ret_type(*ret_vals)
