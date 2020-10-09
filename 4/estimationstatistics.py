from typing import Sequence, Optional
import numpy as np


def mahalanobis_distance_squared(
    # shape (n,)
    a: np.ndarray,
    # shape (n,)
    b: np.ndarray,
    # shape (n, n)
    psd_mat: np.ndarray,
) -> float:  # positive
    diff = a - b
    dist = diff @ np.linalg.solve(psd_mat, diff)
    return dist


def NEES(
    # shape=(n,)
    estimate: np.ndarray,
    # shape=(n,n), positive definite,
    cov: np.ndarray,
    # shape=(n,)
    true_val: np.ndarray,
    # into which part of the state to calculate NEES for
    idxs: Optional[Sequence[int]] = None,
) -> float:  # positive
    idxs = idxs if idxs is not None else np.arange(estimate.shape[-1])
    return mahalanobis_distance_squared(
        estimate[idxs], true_val[idxs], cov[np.ix_(idxs, idxs)]
    )


NEES_sequence = np.vectorize(
    NEES, otypes=[float], excluded=["idxs"], signature="(n),(m,m),(p)->()"
)


def distance_sequence(
    # shape (N, n)
    mean_seq,
    # shape (N, n)
    true_seq,
    # into which part of the state to calculate the distance for
    idxs: Optional[Sequence[int]] = None,
) -> np.ndarray:
    mean_seq_indexed = mean_seq if idxs is None else mean_seq[:, idxs]
    true_seq_indexed = true_seq if idxs is None else true_seq[:, idxs]
    dists = np.linalg.norm(mean_seq_indexed - true_seq_indexed, axis=-1)
    return dists
