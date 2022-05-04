"""Build minibatches for segmenter inference."""
import typing as t

import transformers
import numpy as np
import torch
import torch.nn.functional as F


def build_minibatches(
    tokens: transformers.BatchEncoding,
    num_tokens: int,
    batch_size: int,
    moving_window_size: int,
    window_shift_size: int,
    pad_id: int = 0,
) -> t.List[transformers.BatchEncoding]:
    """Break BatchEncoding items into proper smaller minibatches."""
    minibatches: t.List[transformers.BatchEncoding] = []
    minibatch = transformers.BatchEncoding()

    total_minibatches = 1 + max(
        0, int(np.ceil((num_tokens - moving_window_size) / window_shift_size))
    )

    for i in range(total_minibatches):
        i_start = i * window_shift_size
        i_end = i_start + moving_window_size

        for key, vals in tokens.items():
            slice_ = vals[..., i_start:i_end]

            minibatch.setdefault(key, [])
            minibatch[key].append(slice_)

        if (i + 1) % batch_size == 0:
            minibatches.append(minibatch)
            minibatch = transformers.BatchEncoding()

    if minibatch:
        minibatches.append(minibatch)

    for minibatch in minibatches:
        for key, vals in minibatch.items():
            if torch.is_tensor(vals):
                continue

            for i in reversed(range(len(vals))):
                cur_len = int(max(vals[i].size()))

                if cur_len >= moving_window_size:
                    break

                vals[i] = F.pad(
                    input=vals[i],
                    pad=(0, moving_window_size - cur_len),
                    mode="constant",
                    value=pad_id,
                )

            minibatch[key] = torch.vstack(vals)

    return minibatches
