import typing as t

import numpy as np


class AutoMovingWindowPooler:
    def __new__(self, pooling_operation: t.Literal["max", "avg"]):
        assert pooling_operation in {"max", "avg"}

        if pooling_operation == "max":
            return MaxMovingWindowPooler()

        return AvgMoxingWindowPooler()

    def pool(self, logits: np.ndarray, window_shift_size: int) -> np.ndarray:
        raise NotImplementedError("This method must be implemented by a derived class.")

    def __call__(self, *args, **kwargs) -> np.ndarray:
        return self.pool(*args, **kawrgs)


class MaxMovingWindowPooler(AutoMovingWindowPooler):
    def pool(self, logits: np.ndarray, window_shift_size: int) -> np.ndarray:
        pass


class AvgMovingWindowPooler(AutoMovingWindowPooler):
    def pool(self, logits: np.ndarray, window_shift_size: int) -> np.ndarray:
        pass
