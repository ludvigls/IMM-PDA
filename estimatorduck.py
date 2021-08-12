#%%
from typing import Dict, Any, Generic, TypeVar, Optional
from typing_extensions import Protocol, runtime

from mixturedata import MixtureParameters
from gaussparams import GaussParams

import numpy as np


T = TypeVar("T")


@runtime
class StateEstimator(Protocol[T]):
    def predict(self, eststate: T, Ts: float) -> T:
        ...

    def update(
        self,
        z: np.ndarray,
        eststate: T,
        *,
        sensor_state: Optional[Dict[str, Any]] = None,
    ) -> T:
        ...

    def step(
        self,
        z: np.ndarray,
        eststate: T,
        Ts: float,
        *,
        sensor_state: Optional[Dict[str, Any]] = None,
    ) -> T:
        ...

    def estimate(self, eststate: T) -> GaussParams:
        ...

    def loglikelihood(
        self,
        z: np.ndarray,
        eststate: T,
        *,
        sensor_state: Optional[Dict[str, Any]] = None,
    ) -> float:
        ...

    def reduce_mixture(self, estimator_mixture: MixtureParameters[T]) -> T:
        ...

    def gate(
        self,
        z: np.ndarray,
        eststate: T,
        gate_size_square: float,
        *,
        sensor_state: Optional[Dict[str, Any]] = None,
    ) -> bool:
        ...
