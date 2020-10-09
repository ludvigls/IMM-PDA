from typing import Tuple

import numpy as np


def gaussian_mixture_moments(
    w: np.ndarray,  # the mixture weights shape=(N,)
    mean: np.ndarray,  # the mixture means shape(N, n)
    cov: np.ndarray,  # the mixture covariances shape (N, n, n)
) -> Tuple[
    np.ndarray, np.ndarray
]:  # the mean and covariance of of the mixture shapes ((n,), (n, n))
    """Calculate the first two moments of a Gaussian mixture"""

    # mean
    mean_bar = None  # TODO: hint np.average using axis and weights argument

    # covariance
    # # internal covariance
    cov_int = None  # TODO: hint, also an average

    # # spread of means
    # Optional calc: mean_diff =
    cov_ext = None  # TODO: hint, also an average

    # # total covariance
    cov_bar = None  # TODO

    return mean_bar, cov_bar
