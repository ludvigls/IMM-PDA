from typing import Tuple

import numpy as np


def gaussian_mixture_moments(
    w: np.ndarray,  # the mixture weights shape=(N,)
    x: np.ndarray,  # the mixture means shape(N, n)
    P: np.ndarray,  # the mixture covariances shape (N, n, n)
) -> Tuple[
    np.ndarray, np.ndarray
]:  # the mean and covariance of of the mixture shapes ((n,), (n, n))
    """Calculate the first two moments of a Gaussian mixture"""

    # mean
    xbar = np.average(x, axis=0, weights=w)

    # covariance
    # # internal covariance
    Pint = np.average(P, axis=0, weights=w)

    # # spread of means
    xdiff = x - xbar[None]
    Pext = np.average(xdiff[:, :, None] * xdiff[:, None, :], axis=0, weights=w)

    # # total
    Pbar = Pint + Pext

    return xbar, Pbar
