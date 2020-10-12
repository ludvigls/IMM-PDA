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
    ) -> bool:  # gated (m x 1): gated(j) = true if measurement j is within gate
        """Gate/validate measurements: (z-h(x))'S^(-1)(z-h(x)) <= g^2."""

        gated = np.array([self.state_filter.gate(zj,filter_state,sensor_state=sensor_state,gate_size_square=self.gate_size ** 2,)for zj in Z],dtype=bool,)
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
        ll[0] = log_PND + log_clutter # missed detection
        
            #ll=np.append(ll,self.state_filter.loglikelihood( z, filter_state, sensor_state=sensor_state)+log_PD)
        
        ll[1:] = np.array(
        [
        self.state_filter.loglikelihood(
        zj, filter_state, sensor_state=sensor_state
        )
        for zj in Z
        ]
        )
        ll[1:] += log_PD
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

        # probabilities
        beta = np.exp(lls - scipy.special.logsumexp(lls))

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
        """
        conditional_update = []
        conditional_update.append(
            # TODO: missed detection
            filter_state
        )
        conditional_update.extend(
            [state_filter.update(z,  filter_state,sensor_state=sensor_state) for z in Z]
            # TODO: some loop over Z making a list of updates
        )
        """

        return [filter_state] + [ self.state_filter.update(zj, filter_state, sensor_state=sensor_state) for zj in Z]


        

    def reduce_mixture(
        self, mixture_filter_state: MixtureParameters[ET]
    ) -> ET:  # the two first moments of the mixture
        """Reduce a Gaussian mixture to a single Gaussian."""
        return self.state_filter.reduce_mixture(mixture_filter_state)

            # TODO: utilize self.state_filter to keep this working for both EKF and IMM


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
        beta = self.association_probabilities(
        Zg, filter_state, sensor_state=sensor_state
)
# find the mixture components
        filter_state_update_mixture_components = self.conditional_update(
        Zg, filter_state, sensor_state=sensor_state
)
        filter_state_update_mixture = MixtureParameters(
        beta, filter_state_update_mixture_components
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
        filter_state_predicted = self.predict(filter_state, Ts)
        filter_state_updated = self.update(
        Z, filter_state_predicted, sensor_state=sensor_state
        )

        return filter_state_updated


    def estimate(self, filter_state: ET) -> GaussParams:
        """Get an estimate with its covariance from the filter state."""
        return self.state_filter.estimate(filter_state)
# TODO: remember to use self.state_filter to keep it working for both EKF and IMM

    def init_filter_state(
        self,
        # need to have the type required by the specified state_filter
        init_state: "ET_like",
    ) -> ET:
        """Initialize a filter state to proper form."""
        return self.state_filter.init_filter_state(init_state)
