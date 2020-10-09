# %% Imports
from typing import Any, Dict, Sequence, Optional
from dataclasses import dataclass, field
from typing_extensions import Protocol

import numpy as np

# %% Measurement models interface declaration


class MeasurementModel(Protocol):
    m: int

    def h(self, x: np.ndarray, *,
          sensor_state: Dict[str, Any] = None) -> np.ndarray: ...

    def H(self, x: np.ndarray, *,
          sensor_state: Dict[str, Any] = None) -> np.ndarray: ...

    def R(self, x: np.ndarray, *,
          sensor_state: Dict[str, Any] = None, z: np.ndarray = None) -> np.ndarray: ...

# %% Models


@dataclass
class CartesianPosition:
    sigma: float
    m: int = 2
    state_dim: int = 4

    def h(self,
            x: np.ndarray,
            *,
            sensor_state: Dict[str, Any] = None,
          ) -> np.ndarray:
        """Calculate the noise free measurement location at x in sensor_state."""
        # TODO
        # x[0:2] is position
        # you do not need to care about sensor_state
        #return None

        return self.H(x, sensor_state=sensor_state) @ x

    def H(self,
            x: np.ndarray,
            *,
            sensor_state: Dict[str, Any] = None,
          ) -> np.ndarray:
        """Calculate the measurement Jacobian matrix at x in sensor_state."""
        # TODO
        # x[0:2] is position
        # you do not need to care about sensor_state
        # if you need the size of the state dimension it is in self.state_dim
        #return None
        return np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0]])

    def R(self,
            x: np.ndarray,
            *,
            sensor_state: Dict[str, Any] = None,
            z: np.ndarray = None,
          ) -> np.ndarray:
        """Calculate the measurement covariance matrix at x in sensor_state having potentially received measurement z."""
        # TODO
        # you do not need to care about sensor_state
        # sigma is available as self.sigma, and @dataclass makes it available in the init class constructor
        #return None

        return np.eye(2) * self.sigma
