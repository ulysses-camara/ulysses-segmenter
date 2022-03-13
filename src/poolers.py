import typing as t
import abc

import numpy as np


class _BasePooler(abc.ABC):
    @abc.abstractmethod
    def pool(self, logits: np.ndarray, window_shift_size: int) -> np.ndarray:
        pass

    def __call__(self, *args, **kwargs) -> np.ndarray:
        return self.pool(*args, **kwargs)


class AutoMovingWindowPooler(_BasePooler):
    def __new__(cls, pooling_operation: t.Literal["max", "avg"]):
        assert pooling_operation in {"max", "avg"}

        if pooling_operation == "max":
            return MaxMovingWindowPooler()

        return AvgMovingWindowPooler()


class MaxMovingWindowPooler(_BasePooler):
    def pool(self, logits: np.ndarray, window_shift_size: int) -> np.ndarray:
        d_batch, d_block_size, d_emb_dim = logits.shape

        if d_batch <= 1:
            return logits

        d_batch_output = d_block_size + (d_batch - 1) * window_shift_size
        pooled_logits = np.full((d_batch_output, d_emb_dim), fill_value=-np.inf)

        for i, logit_block in enumerate(logits):
            i_start = i * window_shift_size
            i_end = i_start + d_block_size

            np.maximum(
                pooled_logits[i_start:i_end, ...],
                logit_block,
                out=pooled_logits[i_start:i_end, ...],
            )

        return pooled_logits


class AvgMovingWindowPooler(_BasePooler):
    def pool(self, logits: np.ndarray, window_shift_size: int) -> np.ndarray:
        d_batch, d_block_size, d_emb_dim = logits.shape

        if d_batch <= 1:
            return logits

        d_batch_output = d_block_size + (d_batch - 1) * window_shift_size
        pooled_logits = np.zeros((d_batch_output, d_emb_dim))
        elem_count_per_pos = np.zeros_like(pooled_logits)

        for i, logit_block in enumerate(logits):
            i_start = i * window_shift_size
            i_end = i_start + d_block_size
            pooled_logits[i_start:i_end, ...] += logit_block
            elem_count_per_pos[i_start:i_end] += 1

        pooled_logits /= elem_count_per_pos

        return pooled_logits
