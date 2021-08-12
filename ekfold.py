"""
Notation:
----------
x is generally used for either the state or the mean of a gaussian. It should be clear from context which it is.
P is used about the state covariance
z is a single measurement
Z are multiple measurements so that z = Z[k] at a given time step k
v is the innovation z - h(x)
S is the innovation covariance
"""
# %% Imports
# types
from typing import Union, Callable, Any, Dict, Optional, List, Sequence, Tuple, Iterable
from typing_extensions import Final

# packages
from dataclasses import dataclass, field
import numpy as np
import scipy.linalg as la
import scipy

# local
import dynamicmodels as dynmods
import measurementmodels as measmods
from gaussparams import GaussParams, GaussParamList
from mixturedata import MixtureParameters
import mixturereduction

# %% The EKF


def isPSD(arr: np.ndarray, do_print: bool = False) -> bool:
    # This block only works for positive definite matrices (no zero eigvals)
    # try:
    #     arr_chol = np.linalg.cholesky(arr)
    # except LinAlgError as e:
    #     if do_print:
    #         print(e)
    #     return False
    # return True

    return np.allclose(arr, arr.T) and np.all(np.linalg.eigvals(arr) >= 0)


@dataclass
class EKF:
    # A Protocol so duck typing can be used
    dynamic_model: dynmods.DynamicModel
    # A Protocol so duck typing can be used
    sensor_model: measmods.MeasurementModel

    # _MLOG2PIby2: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._MLOG2PIby2: Final[float] = self.sensor_model.m * np.log(2 * np.pi) / 2

    def predict(
        self,
        ekfstate: GaussParams,
        # The sampling time in units specified by dynamic_model
        Ts: float,
    ) -> GaussParams:
        """Predict the EKF state Ts seconds ahead."""
        x, P = ekfstate

        assert isPSD(P), "P input to EKF.predict not PSD"

        F = self.dynamic_model.F(x, Ts)
        Q = self.dynamic_model.Q(x, Ts)

        x_pred = self.dynamic_model.f(x, Ts)
        P_pred = F @ P @ F.T + Q

        assert np.all(np.isfinite(P_pred)) and np.all(
            np.isfinite(x_pred)
        ), "Non-finite EKF prediction."
        assert isPSD(P_pred), "P_pred calculated by EKF.predict not PSD"

        state_pred = GaussParams(x_pred, P_pred)
        return state_pred

    def innovation_mean(
        self,
        z: np.ndarray,
        ekfstate: GaussParams,
        *,
        sensor_state: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Calculate the innovation mean for ekfstate at z in sensor_state."""

        x = ekfstate.mean

        zbar = self.sensor_model.h(x, sensor_state=sensor_state)

        v = z - zbar

        return v

    def innovation_cov(
        self,
        z: np.ndarray,
        ekfstate: GaussParams,
        *,
        sensor_state: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Calculate the innovation covariance for ekfstate at z in sensorstate."""

        x, P = ekfstate
        assert isPSD(P), "P input to EKF.innovation_cov not PSD"

        H = self.sensor_model.H(x, sensor_state=sensor_state)
        R = self.sensor_model.R(x, sensor_state=sensor_state, z=z)
        S = H @ P @ H.T + R

        assert isPSD(P), "S calculated by EKF.innovation_cov not PSD"
        return S

    def innovation(
        self,
        z: np.ndarray,
        ekfstate: GaussParams,
        *,
        sensor_state: Optional[Dict[str, Any]] = None,
    ) -> GaussParams:
        """Calculate the innovation for ekfstate at z in sensor_state."""

        v = self.innovation_mean(z, ekfstate, sensor_state=sensor_state)
        S = self.innovation_cov(z, ekfstate, sensor_state=sensor_state)

        innovationstate = GaussParams(v, S)

        return innovationstate

    def update(
        self,
        z: np.ndarray,
        ekfstate: GaussParams,
        *,
        sensor_state: Optional[Dict[str, Any]] = None,
    ) -> GaussParams:
        """Update ekfstate with z in sensor_state"""

        x, P = ekfstate
        assert isPSD(P), "P input to EKF.update not PSD"

        v, S = self.innovation(z, ekfstate, sensor_state=sensor_state)

        H = self.sensor_model.H(x, sensor_state=sensor_state)
        W = P @ la.solve(S, H).T

        x_upd = x + W @ v
        I = np.eye(*P.shape)

        # standard form seem to give numerical instability causing non-PSD matrices for certain setups,
        # or that some other calculate increases it in IMM etc.
        # P_upd = P - W @ H @ P

        # Better to use the more numerically stable Joseph form
        P_upd = (I - W @ H) @ P @ (I - W @ H).T + W @ self.sensor_model.R(x) @ W.T

        ekfstate_upd = GaussParams(x_upd, P_upd)

        assert isPSD(P), "P_upd calculated by EKF.update not PSD"
        return ekfstate_upd

    def step(
        self,
        z: np.ndarray,
        ekfstate: GaussParams,
        # sampling time
        Ts: float,
        *,
        sensor_state: Optional[Dict[str, Any]] = None,
    ) -> GaussParams:
        """Predict ekfstate Ts units ahead and then update this prediction with z in sensor_state."""

        ekfstate_pred = self.predict(ekfstate, Ts)
        ekfstate_upd = self.update(z, ekfstate_pred, sensor_state=sensor_state)
        return ekfstate_upd

    def NIS(
        self,
        z: np.ndarray,
        ekfstate: GaussParams,
        *,
        sensor_state: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Calculate the normalized innovation squared for ekfstate at z in sensor_state"""

        v, S = self.innovation(z, ekfstate, sensor_state=sensor_state)

        cholS = la.cholesky(S, lower=True)

        invcholS_v = la.solve_triangular(cholS, v, lower=True)

        NIS = (invcholS_v ** 2).sum()

        # alternative:
        # NIS = v @ la.solve(S, v)
        return NIS

    @classmethod
    def estimate(cls, ekfstate: GaussParams) -> GaussParams:
        """Get the estimate from the state with its covariance. (Compatibility method)"""
        return ekfstate

    def loglikelihood(
        self,
        z: np.ndarray,
        ekfstate: GaussParams,
        *,
        sensor_state: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Calculate the log likelihood of ekfstate at z in sensor_state"""

        v, S = self.innovation(z, ekfstate, sensor_state=sensor_state)

        cholS = la.cholesky(S, lower=True)

        invcholS_v = la.solve_triangular(cholS, v, lower=True)
        NISby2 = (invcholS_v ** 2).sum() / 2
        # alternative self.NIS(...) /2 or v @ la.solve(S, v)/2

        logdetSby2 = np.log(cholS.diagonal()).sum()
        # alternative use la.slogdet(S), or np.log(la.det(S))

        ll = -(NISby2 + logdetSby2 + self._MLOG2PIby2)

        # simplest overall alternative
        # ll = scipy.stats.multivariate_normal.logpdf(v, cov=S)

        return ll

    def reduce_mixture(
        self, ekfstate_mixture: MixtureParameters[GaussParams]
    ) -> GaussParams:
        """Merge a Gaussian mixture into single mixture"""
        w = ekfstate_mixture.weights
        x = np.array([c.mean for c in ekfstate_mixture.components], dtype=float)
        P = np.array([c.cov for c in ekfstate_mixture.components], dtype=float)
        x_reduced, P_reduced = mixturereduction.gaussian_mixture_moments(w, x, P)
        return GaussParams(x_reduced, P_reduced)

    def gate(
        self,
        z: np.ndarray,
        ekfstate: GaussParams,
        gate_size_square: float,
        *,
        sensor_state: Optional[Dict[str, Any]],
    ) -> bool:
        """ Check if z is inside sqrt(gate_sized_squared)-sigma ellipse of ekfstate in sensor_state """
        NIS = self.NIS(z, ekfstate, sensor_state=sensor_state)

        raise NotImplementedError  # TODO: remove this line when implemented
        return None  # TODO: a simple comparison should suffice here
