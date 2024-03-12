"""This module contains logits poolers used during segmentation inference.

Poolers combine logits from overlapping moving windows along the document
subword tokens. This is necessary for documents longer than 1024 subwords,
since they need to be sharded into 1024-long windows to be feed to the
segmenter model.
"""
import typing as t
import abc

import numpy as np
import numpy.typing as npt


class _BasePooler(abc.ABC):
    """Base abstract class for Inference Poolers."""

    @abc.abstractmethod
    def pool(
        self, logits: npt.NDArray[np.float64], window_shift_size: int
    ) -> npt.NDArray[np.float64]:
        """Combine logits from overlapping moving windows.

        Parameters
        ----------
        logits : npt.NDArray[np.float64] of shape (N, B, C)
            Logits to be pooled, where:
            - N: Input batch size;
            - B: Moving window size; and
            - C: Number of classes.

        window_shift_size : int
            Shift size (in subword tokens) between two adjacent windows.

        Returns
        -------
        pooled_logits : npt.NDArray[np.float64] of shape (B + (N - 1) * window_shift_size, C)
            Logits combined by the selected pooling strategy, where `B`, `N` and
            `C` are as specified in the `logits` parameter documentation.
        """

    def __call__(self, *args: t.Any, **kwargs: t.Any) -> npt.NDArray[np.float64]:
        return self.pool(*args, **kwargs)


class AutoMovingWindowPooler(_BasePooler):
    """Generate a specific pooler based on the chosen strategy."""

    def __new__(cls, pooling_operation: str):  # type: ignore
        options = {
            "asymmetric-max": AsymmetricMaxMovingWindowPooler,
            "sum": SumMovingWindowPooler,
            "gaussian": GaussianMovingWindowPooler,
            "max": MaxMovingWindowPooler,
        }

        if pooling_operation not in options:
            raise ValueError(
                "Invalid value for 'pooling_operation' parameter, which must assume a "
                f"value from: {', '.join(options.keys())} (got '{pooling_operation}')."
            )

        chosen_cls = options[pooling_operation]

        return chosen_cls()  # type: ignore

    def pool(
        self, logits: npt.NDArray[np.float64], window_shift_size: int
    ) -> npt.NDArray[np.float64]:
        return logits


class MaxMovingWindowPooler(_BasePooler):
    """Maximal Pooler.

    Chooses the largest logit element-wise.
    """

    def pool(
        self, logits: npt.NDArray[np.float64], window_shift_size: int
    ) -> npt.NDArray[np.float64]:
        """Pick the largest logits from overlapping moving windows for each token.

        Parameters
        ----------
        logits : npt.NDArray[np.float64] of shape (N, B, C)
            Logits to be pooled, where:
            - N: Input batch size;
            - B: Moving window size; and
            - C: Number of classes.

        window_shift_size : int
            Shift size (in subword tokens) between two adjacent windows.

        Returns
        -------
        pooled_logits : npt.NDArray[np.float64] of shape (B + (N - 1) * window_shift_size, C)
            Largest logits associated with each input token, where `B`, `N` and
            `C` are as specified in the `logits` parameter documentation.

        Examples
        --------
        >>> logits = np.array([
        ...    [[ 1, 0], [ 2, -1], [ 3, 2]],
        ...    [[-1, 1], [-2, -2], [-3, 1]],
        ...    [[ 0, 2], [ 5, -3], [-5, 0]],
        ... ])
        >>> pooler = MaxMovingWindowPooler()
        >>> pooler(logits, window_shift_size=1)
        array([[ 1.,  0.],
               [ 2.,  1.],
               [ 3.,  2.],
               [ 5.,  1.],
               [-5.,  0.]])
        """
        d_batch, d_window_size, d_n_cls = logits.shape

        if d_batch <= 1:
            return logits

        d_batch_output = d_window_size + (d_batch - 1) * window_shift_size
        pooled_logits = np.full((d_batch_output, d_n_cls), fill_value=-np.inf)

        for i, logit_window in enumerate(logits):
            i_start = i * window_shift_size
            i_end = i_start + d_window_size

            np.maximum(
                pooled_logits[i_start:i_end, ...],
                logit_window,
                out=pooled_logits[i_start:i_end, ...],
            )

        return pooled_logits


class AsymmetricMaxMovingWindowPooler(_BasePooler):
    """Asymmetric-Maximal Pooler.

    Chooses the largest logit element-wise for all classes except the 'No-op'
    class, that receives the smallest logit instead.
    """

    def pool(
        self, logits: npt.NDArray[np.float64], window_shift_size: int
    ) -> npt.NDArray[np.float64]:
        """Pick maximal overlapping logits for each class, except the 'No-op' class.

        The 'No-op' class receives the minimal logit instead.

        This pooling strategy facilitates actions from the segmenter model, reducing
        false negatives (false 'No-ops'), while possibly increasing false positives
        (missing true 'No-ops').

        Parameters
        ----------
        logits : npt.NDArray[np.float64] of shape (N, B, C)
            Logits to be pooled, where:
            - N: Input batch size;
            - B: Moving window size; and
            - C: Number of classes.

        window_shift_size : int
            Shift size (in subword tokens) between two adjacent windows.

        Returns
        -------
        pooled_logits : npt.NDArray[np.float64] of shape (B + (N - 1) * window_shift_size, C)
            Largest logits associated with each input token, and for all classes except
            the 'No-op' class, that receives the minimal logits instead. `B`, `N` and `C`
            are as specified in the `logits` parameter documentation.

        Examples
        --------
        >>> logits = np.array([
        ...    [[ 1, 0], [ 2, -1], [ 3, 2]],
        ...    [[-1, 1], [-2, -2], [-3, 1]],
        ...    [[ 0, 2], [ 5, -3], [-5, 0]],
        ... ])
        >>> pooler = AsymmetricMaxMovingWindowPooler()
        >>> pooler(logits, window_shift_size=1)
        array([[ 1.,  0.],
               [-1.,  1.],
               [-2.,  2.],
               [-3.,  1.],
               [-5.,  0.]])
        """
        d_batch, d_window_size, d_n_cls = logits.shape

        if d_batch <= 1:
            return logits

        d_batch_output = d_window_size + (d_batch - 1) * window_shift_size
        pooled_logits = np.full((d_batch_output, d_n_cls), fill_value=-np.inf)
        pooled_logits[..., :1] = np.inf

        for i, logit_window in enumerate(logits):
            i_start = i * window_shift_size
            i_end = i_start + d_window_size

            np.maximum(
                pooled_logits[i_start:i_end, ..., 1:],
                logit_window[..., 1:],
                out=pooled_logits[i_start:i_end, ..., 1:],
            )

            np.minimum(
                pooled_logits[i_start:i_end, ..., :1],
                logit_window[..., :1],
                out=pooled_logits[i_start:i_end, ..., :1],
            )

        return pooled_logits


class SumMovingWindowPooler(_BasePooler):
    """Sum Pooler.

    Sum overlapping logits.
    """

    def pool(
        self, logits: npt.NDArray[np.float64], window_shift_size: int
    ) -> npt.NDArray[np.float64]:
        """Sum overlapping logits for each class, except the 'No-op' class.

        Parameters
        ----------
        logits : npt.NDArray[np.float64] of shape (N, B, C)
            Logits to be pooled, where:
            - N: Input batch size;
            - B: Moving window size; and
            - C: Number of classes.

        window_shift_size : int
            Shift size (in subword tokens) between two adjacent windows.

        Returns
        -------
        pooled_logits : npt.NDArray[np.float64] of shape (B + (N - 1) * window_shift_size, C)
            Sum of logits associated with each input token, where `B`, `N` and `C`
            are as specified in the `logits` parameter documentation.

        Examples
        --------
        >>> logits = np.array([
        ...    [[ 1, 0], [ 2, -1], [ 3, 2]],
        ...    [[-1, 1], [-2, -2], [-3, 1]],
        ...    [[ 0, 2], [ 5, -3], [-5, 0]],
        ... ])
        >>> pooler = SumMovingWindowPooler()
        >>> pooler(logits, window_shift_size=1)
        array([[ 1.,  0.],
               [ 1.,  0.],
               [ 1.,  2.],
               [ 2., -2.],
               [-5.,  0.]])
        """
        d_batch, d_window_size, d_n_cls = logits.shape

        if d_batch <= 1:
            return logits

        d_batch_output = d_window_size + (d_batch - 1) * window_shift_size
        pooled_logits = np.zeros((d_batch_output, d_n_cls))

        for i, logit_window in enumerate(logits):
            i_start = i * window_shift_size
            i_end = i_start + d_window_size
            pooled_logits[i_start:i_end, ...] += logit_window

        return pooled_logits


class GaussianMovingWindowPooler(_BasePooler):
    """Gaussian Weighting Pooler.

    Apply weights from a Gaussian distribution to overlapping logits.
    """

    @staticmethod
    def _compute_gaussian_pdf_per_position(window_size: int) -> npt.NDArray[np.float64]:
        """Compute the Gaussian Probability Density associated to each position in window."""
        # Note: window_size / 6 = (half_window_size) / (3 standard deviations from the mean)
        dist_std = window_size / 6.0
        dist_avg = 0.5 * (window_size - 1.0)

        norm_factor = dist_std * float(np.sqrt(2.0 * np.pi))

        pos_weights = np.exp(-0.5 * np.square((np.arange(window_size) - dist_avg) / dist_std))

        return np.asfarray(pos_weights / norm_factor)

    def pool(
        self, logits: npt.NDArray[np.float64], window_shift_size: int
    ) -> npt.NDArray[np.float64]:
        """Apply weights from a Gaussian distribution to overlapping logits.

        The gaussian distribution is parametrized as N(window_size / 2, window_size / 6),
        such that it weights more logits next to the center of the window, and instances
        within the radius (-window_size / 6, window_size / 6) from the window center
        receives approximately 68% from the weight total, whereas doubling this radius
        yields a total weight of approximately 95%.

        Parameters
        ----------
        logits : npt.NDArray[np.float64] of shape (N, B, C)
            Logits to be pooled, where:
            - N: Input batch size;
            - B: Moving window size; and
            - C: Number of classes.

        window_shift_size : int
            Shift size (in subword tokens) between two adjacent windows.

        Returns
        -------
        pooled_logits : npt.NDArray[np.float64] of shape (B + (N - 1) * window_shift_size, C)
            Weighted sum of logits associated with each input token, where `B`, `N`
            and `C` are as specified in the `logits` parameter documentation.

        Examples
        --------
        >>> logits = np.array([
        ...    [[ 1, 0], [ 2, -1], [ 3, 2]],
        ...    [[-1, 1], [-2, -2], [-3, 1]],
        ...    [[ 0, 2], [ 5, -3], [-5, 0]],
        ... ])
        >>> pooler = GaussianMovingWindowPooler()
        >>> pooler(logits, window_shift_size=1).round(3)
        array([[ 0.108,  0.   ],
               [ 1.488, -0.69 ],
               [-1.272, -1.164],
               [ 3.665, -2.286],
               [-0.54 ,  0.   ]])

        Notes
        -----
        The total sum of weights is different than 1.0 due to the discrete nature of the
        problem, altought it is fairly next to it (~0.9973) for any window size larger than
        2. This numeric difference should not affect the segmentation output, since all
        logits associated with the same token will remain in the same scale.
        """
        d_batch, d_window_size, d_n_cls = logits.shape

        if d_batch <= 1:
            return logits

        d_batch_output = d_window_size + (d_batch - 1) * window_shift_size
        pooled_logits = np.zeros((d_batch_output, d_n_cls))

        pos_weights = self._compute_gaussian_pdf_per_position(d_window_size)
        pos_weights = np.expand_dims(pos_weights, 1)

        for i, logit_window in enumerate(logits):
            i_start = i * window_shift_size
            i_end = i_start + d_window_size
            pooled_logits[i_start:i_end, ...] += logit_window * pos_weights

        return pooled_logits
