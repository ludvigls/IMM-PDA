# %% Imports
from typing import Any, Dict, Sequence, Optional
from dataclasses import dataclass, field
from typing_extensions import Protocol

import numpy as np

# %% Measurement models interface declaration


class MeasurementModel(Protocol):
    m: int

    def h(self, x: np.ndarray, *, sensor_state: Dict[str, Any] = None) -> np.ndarray:
        ...

    def H(self, x: np.ndarray, *, sensor_state: Dict[str, Any] = None) -> np.ndarray:
        ...

    def R(
        self,
        x: np.ndarray,
        *,
        sensor_state: Dict[str, Any] = None,
        z: np.ndarray = None,
    ) -> np.ndarray:
        ...


# %% Models


@dataclass
class CartesianPosition:
    sigma: float
    m: int = 2
    state_dim: Optional[int] = None
    pos_idx: Optional[Sequence[int]] = None

    _H: np.ndarray = field(init=False, repr=False)
    _R: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.state_dim = self.state_dim or 2 * self.m
        self.pos_idx = np.asarray(self.pos_idx or np.arange(self.m), dtype=int)

        # H is a constant matrix so simply store it:
        # same as eye(m, state_dim) for pos_idx = 0:dim
        self._H = np.zeros((self.m, self.state_dim))
        self._H[self.pos_idx, self.pos_idx] = 1

        # R is a constant so store it
        self._R = self.sigma ** 2 * np.eye(self.m)

    def h(self, x: np.ndarray, *, sensor_state: Dict[str, Any] = None,) -> np.ndarray:
        """Calculate the noise free measurement location at x in sensor_state."""
        if sensor_state is not None:
            return x[: self.m] - sensor_state["pos"]
        else:
            return x[: self.m]

    def H(self, x: np.ndarray, *, sensor_state: Dict[str, Any] = None,) -> np.ndarray:
        """Calculate the measurement Jacobian matrix at x in sensor_state."""
        return self._H

    def R(
        self,
        x: np.ndarray,
        *,
        sensor_state: Dict[str, Any] = None,
        z: np.ndarray = None,
    ) -> np.ndarray:
        """Calculate the measurement covariance matrix at x in sensor_state having potentially received measurement z."""
        return self._R
