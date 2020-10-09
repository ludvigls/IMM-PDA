from typing import Optional, Union, Tuple
from dataclasses import dataclass
from mytypes import ArrayLike
import numpy as np


@dataclass(init=False)
class GaussParams:
    """A class for holding Gaussian parameters"""

    __slots__ = ["mean", "cov"]
    mean: np.ndarray  # shape=(n,)
    cov: np.ndarray  # shape=(n, n)

    def __init__(self, mean: ArrayLike, cov: ArrayLike) -> None:
        self.mean = np.asarray(mean, dtype=float)
        self.cov = np.asarray(cov, dtype=float)

    def __iter__(self):  # in order to use tuple unpacking
        return iter((self.mean, self.cov))


@dataclass(init=False)
class GaussParamList:
    __slots__ = ["mean", "cov"]
    mean: np.ndarray  # shape=(N, n)
    cov: np.ndarray  # shape=(N, n, n)

    def __init__(self, mean=None, cov=None):
        if mean is not None and cov is not None:
            self.mean = mean
            self.cov = cov
        else:
            # container left empty
            pass

    @classmethod
    def allocate(
        cls,
        shape: Union[int, Tuple[int, ...]],  # list shape
        n: int,  # dimension
        fill: Optional[float] = None,  # fill the allocated arrays
    ) -> "GaussParamList":
        if isinstance(shape, int):
            shape = (shape,)

        if fill is None:
            return cls(np.empty((*shape, n)), np.empty((*shape, n, n)))
        else:
            return cls(np.full((*shape, n), fill), np.full((*shape, n, n), fill))

    def __getitem__(self, key):
        theCls = GaussParams if isinstance(key, int) else GaussParamList
        return theCls(self.mean[key], self.cov[key])

    def __setitem__(self, key, value):
        if isinstance(value, (GaussParams, tuple)):
            self.mean[key], self.cov[key] = value
        elif isinstance(value, GaussParamList):
            self.mean[key] = value.mean
            self.cov[key] = value.cov
        else:
            raise NotImplementedError(f"Cannot set from type {value}")

    def __len__(self):
        return self.mean.shape[0]

    def __iter__(self):
        yield from (self[k] for k in range(len(self)))
