#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic models to be used with eg. EKF.

@author: Lars-Christian Tokle, lars-christian.n.tokle@ntnu.no
"""
# %%
from typing import Optional, Sequence
from typing_extensions import Final, Protocol
from dataclasses import dataclass, field

import numpy as np

# %% the dynamic models interface declaration


class DynamicModel(Protocol):
    n: int

    def f(self, x: np.ndarray, Ts: float) -> np.ndarray:
        ...

    def F(self, x: np.ndarray, Ts: float) -> np.ndarray:
        ...

    def Q(self, x: np.ndarray, Ts: float) -> np.ndarray:
        ...


# %%


@dataclass
class WhitenoiseAccelleration:
    """
    A white noise accelereation model, also known as constan velocity. States are position and speed.

    The model includes the discrete prediction equation f, its Jacobian F, and
    the process noise covariance Q. This can be specified in any number of
    dimensions dim, with arbitrary indexes for position and velocity.
    It can also handle extra dimensions by either forcing them to zero or leaving them untouched.
    """

    # noise standard deviation
    sigma: float
    # number of dimensions
    dim: int = 2
    # number of states
    n: int = None
    # indexes into the state space for position. (defaults to 0:2)
    pos_idx: Optional[Sequence[int]] = None
    # indexes into the state space for velocity. (defaults to the complementary of pos_idx)
    vel_idx: Optional[Sequence[int]] = None
    # indexes to propagate, ie. not force to zero
    identity_idx: Optional[Sequence[int]] = None

    _sigma2: float = field(init=False, repr=False)
    _F_mat: np.ndarray = field(init=False, repr=False)
    _Q_mat: np.ndarray = field(init=False, repr=False)
    _state: np.ndarray = field(init=False, repr=False)
    _all_idx: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.n is None:
            self.n = 2 * self.dim

        self.pos_idx = self.pos_idx or np.arange(self.dim)
        self.vel_idx = self.vel_idx or np.array(
            [i for i in range(2 * self.dim) if i not in self.pos_idx]
        )
        self._all_idx = np.concatenate((self.pos_idx, self.vel_idx))
        if self.identity_idx is not None:
            self._all_idx = np.concatenate((self._all_idx, self.identity_idx))

        self._sigma2 = self.sigma ** 2

        self._F_mat = np.zeros((self.n, self.n))
        self._F_mat[self._all_idx, self._all_idx] = 1

        self._Q_mat = np.zeros((self.n, self.n))
        self._state = np.zeros(self.n)

    def f(self, x: np.ndarray, Ts: float,) -> np.ndarray:
        """Calculate the zero noise Ts time units transition from x."""
        x_p = self._state
        x_p[self._all_idx] = x[self._all_idx]
        x_p[self.pos_idx] += Ts * x[self.vel_idx]
        return x_p.copy()

    def F(self, x: np.ndarray, Ts: float,) -> np.ndarray:
        """Calculate the transition function jacobian for Ts time units at x."""
        F = self._F_mat
        F[self.pos_idx, self.vel_idx] = Ts
        return F.copy()

    def Q(self, x: np.ndarray, Ts: float,) -> np.ndarray:
        """Calculate the Ts time units transition Covariance."""

        Q = self._Q_mat

        # diags
        Q[self.pos_idx, self.pos_idx] = self._sigma2 * Ts ** 3 / 3
        Q[self.vel_idx, self.vel_idx] = self._sigma2 * Ts

        # off diags
        Q[self.pos_idx, self.vel_idx] = self._sigma2 * Ts ** 2 / 2
        Q[self.vel_idx, self.pos_idx] = self._sigma2 * Ts ** 2 / 2

        return Q.copy()


@dataclass
class ConstantTurnrate:
    sigma_a: float
    sigma_omgea: float
    n: int = 5
    pos_idx: np.ndarray = np.arange(2)
    vel_idx: np.ndarray = np.arange(2, 4)
    omega_idx: int = 4

    _all_idx: np.ndarray = field(init=False, repr=False)
    _sigma_a2: float = field(init=False, repr=False)
    _simga_omega2: float = field(init=False, repr=False)
    _F_mat: np.ndarray = field(init=False, repr=False)
    _Q_mat: np.ndarray = field(init=False, repr=False)
    _state: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self._sigma_a2 = self.sigma_a ** 2
        self._sigma_omega2 = self.sigma_omgea ** 2

        self._state = np.zeros(self.n)
        self._F_mat = np.zeros((self.n, self.n))
        self._Q_mat = np.zeros((self.n, self.n))
        self._all_idx = np.concatenate(
            (self.pos_idx, self.vel_idx, np.atleast_1d(self.omega_idx))
        )

    def f(self, x: np.ndarray, Ts: float) -> np.ndarray:
        xp = self._state
        xp[self._all_idx] = f_CT(x[self._all_idx], Ts)
        return xp.copy()

    def F(self, x: np.ndarray, Ts: float) -> np.ndarray:
        F = self._F_mat
        F[np.ix_(self._all_idx, self._all_idx)] = F_CT(x[self._all_idx], Ts)
        return F.copy()

    def Q(self, x: np.ndarray, Ts: float) -> np.ndarray:
        """Get the Ts time units noise covariance at x."""
        Q = self._Q_mat

        # diags
        Q[self.pos_idx, self.pos_idx] = self._sigma_a2 * Ts ** 3 / 3
        Q[self.vel_idx, self.vel_idx] = self._sigma_a2 * Ts
        Q[self.omega_idx, self.omega_idx] = self._sigma_omega2 * Ts

        # off diags
        Q[self.pos_idx, self.vel_idx] = self._sigma_a2 * Ts ** 2 / 2
        Q[self.vel_idx, self.pos_idx] = self._sigma_a2 * Ts ** 2 / 2

        return Q.copy()


def cosc(x: np.ndarray) -> np.ndarray:  # same shape as input
    """
    Calculate (1 - cos(x * pi))/(x * pi).

    The name is invented here due to similarities to sinc, and uses sinc in its calculation.

    (1 - cos(x * pi))/(x * pi) = (0.5 * x * pi) * (sin(x*pi/2) / (x*pi)) ** 2 = (0.5 * x * pi) * (sinc(x/2) ** 2)

    """
    return (0.5 * x * np.pi) * (np.sinc(x / 2) ** 2)


def diff_sinc_small(x: float) -> float:
    xpi = np.pi * x
    return (-xpi / 3 + xpi ** 3 / 30) * np.pi


def diff_sinc_larger(x: float) -> float:
    xpi = np.pi * x
    return (np.cos(xpi) - np.sinc(x)) / x


def diff_sinc(x: np.ndarray) -> np.ndarray:  # same shape as input
    """
    Calculate d np.sinc(x) / dx = (np.cos(np.pi * x) - np.sinc(x)) / (np.pi * x).

    If derivative of sin(x)/x is wanted, the usage becomes diff_sinc(x / np.pi) / np.pi.
    Uses 3rd order taylor series for abs(x) < 1e-3 as it is more accurate and avoids division by 0.
    """
    return np.piecewise(x, [np.abs(x) > 1e-3], [diff_sinc_larger, diff_sinc_small])


def diff_cosc(x: np.ndarray) -> np.ndarray:  # same shape as input
    """
    Calculate d cosc(x) / dx = np.pi * (np.sinc(x) - 0.5 * np.sinc(x/2)**2).

    If derivative of (1 - cos(x))/x is wanted, the usage becomes  diff_cosc(x / np.pi) / np.pi.
    Relies solely on sinc for calculation.
    """
    sincx = np.sinc(x)
    sincx2 = 0.5 * np.sinc(x / 2) ** 2

    return np.pi * (sincx - sincx2)


def f_CT(
    # probably only works for a single state
    x: np.ndarray,
    Ts: float,
):
    """Calculate the constant turn rate time transition for Ts time units at x."""
    x0, y0, u0, v0, omega = x
    pi = np.pi

    theta = omega * Ts

    cth = np.cos(theta)
    sth = np.sin(theta)

    sincth = np.sinc(theta / pi)  # == sin(theta)/theta
    coscth = cosc(theta / pi)  # == (1 - cos(tehta))/theta

    xp = np.array(
        [
            x0 + Ts * u0 * sincth - Ts * v0 * coscth,
            y0 + Ts * u0 * coscth + Ts * v0 * sincth,
            u0 * cth - v0 * sth,
            u0 * sth + v0 * cth,
            omega,
        ]
    )
    assert np.all(np.isfinite(xp)), f"Non finite calculation in CT predict for x={x}."
    # max_diff = np.abs(xp - f_m2_withT(x, Ts)).max()
    # clearance = 1e-5 if np.abs(x[4]) > 1e-4 else 2e-3
    # assert max_diff < clearance, "CT transition not consistent with MATLAB version"
    return xp


def F_CT(
    # probably only works for a single state
    x: np.array,
    Ts: float,
) -> np.ndarray:
    """Calculate the constant turn rate time transition jacobian for Ts time units at x."""
    x0, y0, u0, v0, omega = x
    theta = Ts * omega

    sth = np.sin(theta)
    cth = np.cos(theta)

    sincth = np.sinc(theta / np.pi)
    coscth = cosc(theta / np.pi)

    dsincth = diff_sinc(theta / np.pi) / np.pi
    dcoscth = diff_cosc(theta / np.pi) / np.pi

    F = np.array(
        [
            [1, 0, Ts * sincth, -Ts * coscth, Ts ** 2 * (u0 * dsincth - v0 * dcoscth)],
            [0, 1, Ts * coscth, Ts * sincth, Ts ** 2 * (u0 * dcoscth + v0 * dsincth)],
            [0, 0, cth, -sth, -Ts * (u0 * sth + v0 * cth)],
            [0, 0, sth, cth, Ts * (u0 * cth - v0 * sth)],
            [0, 0, 0, 0, 1],
        ]
    )
    assert np.all(np.isfinite(F)), f"Non finite calculation in CT Jacobian for x={x}."
    # max_diff = np.abs(F - Phi_m2_withT(x, Ts)).max()
    # clearance = 1e-5 if np.abs(x[4]) > 1e-4 else 2.6e-3
    # assert (
    #     max_diff < clearance
    # ), "CT transition Jacobian not consistent with MATLAB version"
    return F


def f_m2_withT(x, T):
    # MATLAB version converted
    # Should now have been readjusted for parametrisation used in PDA
    sth = np.sin(T * x[4])
    cth = np.cos(T * x[4])
    if abs(x[4]) > 0.0001:
        xout = np.array(
            [
                x[0] + sth * x[2] / x[4] - (1 - cth) * x[3] / x[4],
                x[1] + (1 - cth) * x[2] / x[4] + sth * x[3] / x[4],
                cth * x[2] - sth * x[3],
                sth * x[2] + cth * x[3],
                x[4],
            ]
        )
    else:
        xout = np.array([x[0] + T * x[2], x[1] + T * x[3], x[2], x[3], 0])
    return xout


def Phi_m2_withT(x, T):
    sth = np.sin(T * x[4])
    cth = np.cos(T * x[4])
    if abs(x[4]) > 0.0001:

        Jacobi_omega = np.array(
            [
                cth * T * x[2] / x[4]
                - sth * x[2] / x[4] ** 2
                - sth * T * x[3] / x[4]
                + (1 - cth) * x[3] / x[4] ** 2,
                # x
                sth * T * x[2] / x[4]
                - (1 - cth) * x[2] / x[4] ** 2
                + cth * T * x[3] / x[4]
                - sth * x[3] / x[4] ** 2,
                # y
                -sth * T * x[2] - cth * T * x[3],
                # v_x
                cth * T * x[2] - sth * T * x[3],
                # v_y
                1,
            ]
        )

        r = x[4]
        #%u = x(3);
        #%v = x(4);

        colX = np.array([1, 0, 0, 0, 0])
        colY = np.array([0, 1, 0, 0, 0])
        colU = np.array([sth / r, (1 - cth) / r, cth, sth, 0,])
        colV = np.array([-(1 - cth) / r, sth / r, -sth, cth, 0,])

        Linmatrix = np.stack([colX, colY, colU, colV, Jacobi_omega]).T
    else:

        Linmatrix = np.array(
            [
                [1, 0, T, 0, -(T ** 2) * x[3] / 2],
                [0, 1, 0, T, T ** 2 * x[2] / 2],
                [0, 0, 1, 0, -T * x[3]],
                [0, 0, 0, 1, T * x[2]],
                [0, 0, 0, 0, 1],
            ]
        )
    return Linmatrix
