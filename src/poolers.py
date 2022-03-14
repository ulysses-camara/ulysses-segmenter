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
    def __new__(
        cls, pooling_operation: t.Literal["max", "avg", "gaussian", "assymetric-max"]
    ):
        assert pooling_operation in {"max", "avg", "gaussian", "assymetric-max"}

        if pooling_operation == "max":
            return MaxMovingWindowPooler()

        if pooling_operation == "avg":
            return SumMovingWindowPooler()

        if pooling_operation == "gaussian":
            return GaussianMovingWindowPooler()

        return AssymetricMaxMovingWindowPooler()


class MaxMovingWindowPooler(_BasePooler):
    def pool(self, logits: np.ndarray, window_shift_size: int) -> np.ndarray:
        d_batch, d_block_size, d_n_cls = logits.shape

        if d_batch <= 1:
            return logits

        d_batch_output = d_block_size + (d_batch - 1) * window_shift_size
        pooled_logits = np.full((d_batch_output, d_n_cls), fill_value=-np.inf)

        for i, logit_block in enumerate(logits):
            i_start = i * window_shift_size
            i_end = i_start + d_block_size

            np.maximum(
                pooled_logits[i_start:i_end, ...],
                logit_block,
                out=pooled_logits[i_start:i_end, ...],
            )

        return pooled_logits


class AssymetricMaxMovingWindowPooler(_BasePooler):
    def pool(self, logits: np.ndarray, window_shift_size: int) -> np.ndarray:
        d_batch, d_block_size, d_n_cls = logits.shape

        if d_batch <= 1:
            return logits

        d_batch_output = d_block_size + (d_batch - 1) * window_shift_size
        pooled_logits = np.full((d_batch_output, d_n_cls), fill_value=-np.inf)
        pooled_logits[..., :1] = np.inf

        for i, logit_block in enumerate(logits):
            i_start = i * window_shift_size
            i_end = i_start + d_block_size

            np.maximum(
                pooled_logits[i_start:i_end, ..., 1:],
                logit_block[..., 1:],
                out=pooled_logits[i_start:i_end, ..., 1:],
            )

            np.minimum(
                pooled_logits[i_start:i_end, ..., :1],
                logit_block[..., :1],
                out=pooled_logits[i_start:i_end, ..., :1],
            )

        return pooled_logits


class SumMovingWindowPooler(_BasePooler):
    def pool(self, logits: np.ndarray, window_shift_size: int) -> np.ndarray:
        d_batch, d_block_size, d_n_cls = logits.shape

        if d_batch <= 1:
            return logits

        d_batch_output = d_block_size + (d_batch - 1) * window_shift_size
        pooled_logits = np.zeros((d_batch_output, d_n_cls))

        for i, logit_block in enumerate(logits):
            i_start = i * window_shift_size
            i_end = i_start + d_block_size
            pooled_logits[i_start:i_end, ...] += logit_block

        return pooled_logits


class GaussianMovingWindowPooler(_BasePooler):
    @staticmethod
    def _build_gaussian_pdf_per_position(block_size: int) -> np.ndarray:
        # Note: block_size / 6 = (half_block_size) / (3 standard deviations from the mean)
        dist_std = block_size / 6.0
        dist_avg = 0.5 * (block_size - 1.0)

        norm_factor = dist_std * np.sqrt(2.0 * np.pi)

        pos_weights = np.exp(
            -0.5 * np.square((np.arange(block_size) - dist_avg) / dist_std)
        )

        return pos_weights / norm_factor

    def pool(self, logits: np.ndarray, window_shift_size: int) -> np.ndarray:
        d_batch, d_block_size, d_n_cls = logits.shape

        if d_batch <= 1:
            return logits

        d_batch_output = d_block_size + (d_batch - 1) * window_shift_size
        pooled_logits = np.zeros((d_batch_output, d_n_cls))

        pos_weights = self._build_gaussian_pdf_per_position(d_block_size)
        pos_weights = np.expand_dims(pos_weights, 1)

        for i, logit_block in enumerate(logits):
            i_start = i * window_shift_size
            i_end = i_start + d_block_size
            pooled_logits[i_start:i_end, ...] += logit_block * pos_weights

        return pooled_logits
