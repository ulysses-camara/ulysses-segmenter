"""Pack segmenter results into a named tuple."""
import typing as t
import collections


def pack_results(
    keys: list[str], vals: list[t.Any], inclusion: list[bool]
) -> t.Union[list[str], tuple[list[t.Any], ...]]:
    """Build result tuple (if more than one value) or return segment list."""
    ret_keys: list[str] = []
    ret_vals: list[t.Any] = []

    for key, val, inc in zip(keys, vals, inclusion):
        if not inc:
            continue

        ret_keys.append(key)
        ret_vals.append(val)

    if len(ret_vals) == 1:
        segs: list[str] = ret_vals[0]
        return segs

    # Note: pylint config below is due to a weird, possible buggy, error that only
    # occurs with Python 3.9 @ GitHub.
    # pylint: disable=unused-variable
    ret_type = collections.namedtuple("SegmentationResults", ret_keys)  # type: ignore

    return ret_type(*ret_vals)
