from typing import TypeVar, Optional, Dict, Any, List, Generic
from dataclasses import dataclass, field
import numpy as np
import scipy
import scipy.special

from estimatorduck import StateEstimator
from mixturedata import MixtureParameters
from gaussparams import GaussParams

ET = TypeVar("ET")


@dataclass
class PDA(Generic[ET]):  # Probabilistic Data Association
    state_filter: StateEstimator[ET]
    clutter_intensity: float
    PD: float
    gate_size: float

    def predict(self, filter_state: ET, Ts: float) -> ET:
        """Predict state estimate Ts time units ahead"""
        return self.state_filter.predict(filter_state, Ts)

    def gate(
        self,
        # measurements of shape=(M, m)=(#measurements, dim)
        Z: np.ndarray,
        filter_state: ET,
        *,
        sensor_state: Optional[Dict[str, Any]] = None,
    ) -> List[bool]:  # gated (m x 1): gated(j) = true if measurement j is within gate
        """Gate/validate measurements: (z-h(x))'S^(-1)(z-h(x)) <= g^2."""

        M = Z.shape[0]
        g_squared = self.gate_size ** 2

        # The loop can be done using ether of these: normal loop, list comprehension or map
        gated = list(map(
            lambda z: self.state_filter.gate(z, filter_state, gate_size_square=g_squared, sensor_state=sensor_state),
            Z))

        # debug
        # count = np.sum(gated)
        # print(f"{count} out of {M} measurements used")

        return gated

    def loglikelihood_ratios(
        self,  # measurements of shape=(M, m)=(#measurements, dim)
        Z: np.ndarray,
        filter_state: ET,
        *,
        sensor_state: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:  # shape=(M + 1,), first element for no detection
        """ Calculates the posterior event loglikelihood ratios."""

        log_PD = np.log(self.PD)
        log_PND = np.log(1 - self.PD)  # P_ND = 1 - P_D
        log_clutter = np.log(self.clutter_intensity)

        # allocate
        ll = np.empty(Z.shape[0] + 1)

        # calculate log likelihood ratios
        ll[0] =  log_clutter + log_PND
        for i, z in enumerate(Z):
            ak = i+1
            ll[ak] = log_PD + self.state_filter.loglikelihood(z, filter_state, sensor_state=sensor_state)

        return ll

    def association_probabilities(
        self,
        # measurements of shape=(M, m)=(#measurements, dim)
        Z: np.ndarray,
        filter_state: ET,
        *,
        sensor_state: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:  # beta, shape=(M + 1,): the association probabilities (normalized likelihood ratios)
        """calculate the poseterior event/association probabilities."""

        # log likelihoods
        lls = self.loglikelihood_ratios(Z, filter_state, sensor_state=sensor_state)

        # lls_normalized[ak] = log(alpha) + lls[ak]
        # log(alpha) = -logsumexp(ll[:])
        log_a = -scipy.special.logsumexp(lls)
        lls_normalized = log_a + lls

        # probabilities
        beta = np.exp(lls_normalized)
        return beta

    def conditional_update(
        self,
        # measurements of shape=(M, m)=(#measurements, dim)
        Z: np.ndarray,
        filter_state: ET,
        *,
        sensor_state: Optional[Dict[str, Any]] = None,
    ) -> List[
        ET
    ]:  # Updated filter_state for all association events, first element is no detection
        """Update the state with all possible measurement associations."""

        conditional_update = []
        conditional_update.append(
                filter_state
        )
        conditional_update.extend(
                [self.state_filter.update(z, filter_state, sensor_state=sensor_state) for z in Z]
        )

        return conditional_update

    def reduce_mixture(
        self, mixture_filter_state: MixtureParameters[ET]
    ) -> ET:  # the two first moments of the mixture
        """Reduce a Gaussian mixture to a single Gaussian."""
        return self.state_filter.reduce_mixture(mixture_filter_state)


    def update(
        self,
        # measurements of shape=(M, m)=(#measurements, dim)
        Z: np.ndarray,
        filter_state: ET,
        *,
        sensor_state: Optional[Dict[str, Any]] = None,
    ) -> ET:  # The filter_state updated by approximating the data association
        """
        Perform the PDA update cycle.

        Gate -> association probabilities -> conditional update -> reduce mixture.
        """
        # remove the not gated measurements from consideration
        gated = self.gate(Z, filter_state, sensor_state=sensor_state)
        Zg = Z[gated]

        # find association probabilities
        beta = self.association_probabilities(Zg, filter_state, sensor_state=sensor_state)

        # find the mixture components
        filter_state_updated_mixture_components = self.conditional_update(Zg, filter_state, sensor_state=sensor_state)

        # make mixture
        filter_state_update_mixture = MixtureParameters(
            beta, filter_state_updated_mixture_components 
        )

        # reduce mixture
        filter_state_updated_reduced = self.reduce_mixture(filter_state_update_mixture)

        return filter_state_updated_reduced

    def step(
        self,
        # measurements of shape=(M, m)=(#measurements, dim)
        Z: np.ndarray,
        filter_state: ET,
        Ts: float,
        *,
        sensor_state: Optional[Dict[str, Any]] = None,
    ) -> ET:
        """Perform a predict update cycle with Ts time units and measurements Z in sensor_state"""
        filter_state_predicted = self.predict(filter_state, Ts, sensor_state=sensor_state)
        filter_state_updated = self.update(Z, filter_state_predicted, sensor_state=sensor_state)
        return filter_state_updated

    def estimate(self, filter_state: ET) -> GaussParams:
        """Get an estimate with its covariance from the filter state."""
        return self.state_filter.estimate(filter_state)

    def init_filter_state(
        self,
        # need to have the type required by the specified state_filter
        init_state: "ET_like",
    ) -> ET:
        """Initialize a filter state to proper form."""
        return self.state_filter.init_filter_state(init_state)
