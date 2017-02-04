# This file contains functions defining the vertical motion of the drop while in contact
# with the bath, as defined in MBI, section 3.

from typing import Tuple

import brisk
import numba
import numpy as np
from numpy import log, sin
from scipy import integrate

from constants import *


jit = numba.jit(nopython=True)


# todo the lin spring models aren't matching the paper atm, but the log model seems to work.


@jit
def weber(v_in: float) -> float:
    # For calculating the weber number dynamically.
    return (ρ * R_0 * v_in ** 2) / σ  # Weber number; Dimensionless. 0.01 - 1


@jit
def Bo_star(τ_: float) -> float:
    """Effective bond number; defined near MBI eq 3.1. Reflects effective grav in the
    # vibrating bath frame of reference.  This component: (1 + Γ * sin(Ω * τ_)) is the
    ""effective gravity."""
    # todo is tau_ included in the sin?? The notation in MBI and MBII makes it look
    # todo like it's not, but including it makes more sense?
    return Bo * (1 + Γ * sin(Ω * τ_))


# @jit
def Z_ττ(Z, τ_):
    # From MBI, equation 3.4. Something related to air drag.
    R_e = 2 * (abs(Z) / Oh_a) * (ρ_a / ρ)  # Reynolds number.

    # Eq  3.3: Re < 1
    return -Bo_star(τ_) - (9/2) * Oh_a * Z

    # Eq 3.4: 1 < Re < 50
    return -Bo_star(τ_) - (9/2) * Oh_a * Z * (1 + (1/12) * R_e)


# @jit
def F(τ_: float) -> float:
    """Dimensionless reaction force; part of MBI equation 3.8"""
    return Z_ττ(τ_) + Bo * (1 + Γ * sin(Ω) * τ_)


# @jit
def Φ(τ_C: Tuple[float, float]):
    """Impact phase, from MBI equation 3.8.  Weighted avg of the driving phase over time."""
    # todo faster, jittable analytic solution?
    # quad returns tuple of solution and precision estimate; index to 0.
    part1 = integrate.quad(lambda τ_: F(τ_) * Ω * τ_, *τ_C)[0]
    part2 = integrate.quad(F, *τ_C)[0]
    return (part1 / part2) % τ


@jit
def rhs_log_spring(y: np.ndarray, τ_: float, c1: float, c2: float, c3: float) -> Tuple:
    """For logarithmic spring model in MBI, equation 3.7 """
    Z, dZ_dτ = y

    Q = log(c1/abs(Z))
    # Breaking down MBI's equation 3.7
    part1 = (Oh * c2 * dZ_dτ) / Q
    part2 = ((3/2) * Z) / Q
    part3 = 1 + c3/Q**2

    ddZ_ddτ = (-Bo_star(τ_) - part1 - part2) / part3

    return dZ_dτ, ddZ_ddτ


def log_spring(τ_: np.array, v_in: float) -> np.ndarray:
    """Vertical drop position during contact, modelled as a logarithmic spring,
    from MBI, section 3."""

    # τ_ is dimensionless time, ie ω_D * t

    # Undisturbed bath surfaces lies at Z=-1; frame of reference centered on the oscillating
    # bath.  Contact starts when center of drop is at Z=0, and its base is at Z=-1.

    # c1 through three are experimentally-determined constants.
    c1 = 2  # nonlinearity of the spring model.
    c2 = 12.5  # 12.5 for 20cSt, 7.5 for 50cSt. Amount of viscous dissipation within the bodies.
    c3 = 1.4  # kinetic energy associated with fluid motion

    # 0 is the initial position at the start of contact, due to the coordinate system
    # described above.
    # Set initial position as near, but not at 0 to avoid a divide-by-zero error
    # when calculating Q.
    y0 = [-1e-20, -weber(v_in)]  # These initial conditions are defined in MBI, below eq 3.7.

    return integrate.odeint(rhs_log_spring, y0, τ_, args=(c1, c2, c3))


@jit
def rhs_lin_spring(y: np.ndarray, τ_: float, C_: float, D_: float) -> Tuple:
    Z, dZ_dτ = y

    # Simpler variant; equation 3.1 has an approximate analytic solution.
    # ddZ_ddτ = -Bo_star(τ_) - brisk.heaviside(-Z) * (D_ * dZ_dτ + C_ * Z)

    ddZ_ddτ = -Bo_star(τ_) + brisk.heaviside(-Z) * max(-D_*dZ_dτ - C_*Z, 0)

    return dZ_dτ, ddZ_ddτ


def lin_spring(τ_: np.array, v_in: float) -> np.ndarray:
    """For linear spring model in MBI, equation 3.6. Doesn't match experimental
    results as well as 3.7, the log spring."""

    # τ_ is dimensionless time, ie ω_D * t

    # Undisturbed bath surfaces lies at Z=-1; frame of reference centered on the oscillating
    # bath.  Contact starts when center of drop is at Z=0, and its base is at Z=-1.
    τ_C = .45  # where?
    C_R = 0.3  # Coeficient of restitution; dimensionless.  0.3 for 20cSt oil, 0.19 for 50cSt.

    C_ = (π**2 + (log(C_R))**2) / τ_C**2
    D_ = -2 * log(C_R) / τ_C  # todo another confusing order of operations issue...

    y0 = [0, -weber(v_in)]  # These initial conditions are defined in MBI, below eq 3.7.

    return integrate.odeint(rhs_lin_spring, y0, τ_, args=(C_, D_))


def lin_spring_analytic(τ_: np.array) -> np.ndarray:
    """Based on an approximate analytic solution to eq 3.1 from MBI. Less accurate than eqs  3.6 and 3.7"""
    # τ_ = ω_D * t

    # τ_ is dimensionless time, ie ω_D * t

    # Undisturbed bath surfaces lies at Z=-1; frame of reference centered on the oscillating
    # bath.  Contact starts when center of drop is at Z=0, and its base is at Z=-1.
    τ_C = .45  # where?
    C_R = 0.3  # Coeficient of restitution; dimensionless.  0.3 for 20cSt oil, 0.19 for 50cSt.

    C_ = (π**2 + (log(C_R))**2) / τ_C**2
    D_ = -2 * log(C_R) / τ_C  # todo another confusing order of operations issue...

    C_p = C_ - D_**2 / 4

    Z_τ0 = -1

    return Z_τ0 * np.exp(-D_*τ_ /2) * sin(sqrt(C_p) * τ_) / sqrt(C_p)


# @jit   # Argwhere breaks jit?
def find_exit_conditions(τ_: np.ndarray, soln: np.ndarray) -> Tuple[float, float, float]:
    """Find the (all dimensionless) height, time, and velocity when the drop exits the bath."""
    # Min and max times to look for the exit
    times = (2., 10.)

    Z, v = soln[:, 0], soln[:, 1]

    i_to_search = np.argwhere((τ_ > times[0]) & (τ_ < times[1]))
    # Index of the already-filtered indexes where the exit occurs.
    min_i = np.argmin(np.abs(Z[i_to_search]))

    # Index twice to find the solution.
    return float(τ_[i_to_search][min_i]), float(Z[i_to_search][min_i]), float(v[i_to_search][min_i])


# @jit
# def dimensionize_exit(τ_: float, Z: float, we: float) -> Tuple[float, float, float]:
#     """Add dimensions to contact time, height, and weber number(velocity). Ie, use this
#     for the exit conditions after contact."""
#     return τ_ / ω_D, Z * R_0, sqrt((we * σ) / (ρ * R_0))


@jit
def dimensionize_motion(soln: np.ndarray) -> np.ndarray:
    """Re-dimensionalize arrays of position and velocity, as returned by the lin/log integrators
    for contact time."""
    soln[:, 0] *= R_0
    soln[:, 1] = sqrt((soln[:, 1] * σ) / (ρ * R_0))
    return soln


def remove_dimensions(z: float, t: float) -> Tuple[float, float]:
    """Make vertical position and time, dimensionless, before performing the calculations."""
    Z = z / R_0
    τ_ = ω_D * t
    return Z, τ_


def add_dimensions(Z: float, τ_: float) -> Tuple[float, float]:
    """Re-dimensionalize vertical position, and time, after performing the calculations."""
    z = Z * R_0
    t = τ_ / ω_D
    return z, t
