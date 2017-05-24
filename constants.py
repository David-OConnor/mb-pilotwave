# Load this file from all others with from constants import *. in addition to constants,
# it includes standard imports like numpy and numba.

from collections import namedtuple
from typing import Iterator, Iterable, List, Tuple

import numba
import numpy as np
from numpy import pi as π, sqrt, cos, sin, tan, arctan2, exp, log
from scipy.constants import g

jit = numba.jit(nopython=True)


# an Impact is an event of the drop hitting the surface.
Impact = namedtuple('Impact', ['t', 'x', 'y', 'F'])  # Add other aspects like speed, force etc.
Point = namedtuple('Point', ['x', 'y'])  # Add other aspects like speed, force etc.


# System parameters, from MBII, Table 1, and MBI table 1.


"""Oil and air properties"""
# MBI has data for two drops:
# A: rho= 949, sigma = 20.6*10**-3, ν=20
# B: rho=960, sigma = 20.8 * 10**-3, and ν = 50cSt

ν = 50  # Drop kinematic viscocity; 10-100 cSt  MBI/I tested at 20 and 50.

"""Oil properties for the two types tested in MBI/II"""
if ν == 20:
    ρ = 949  # Silicone oil density (droplet and bed), 949 and 960 kg/m^3 tested in MBI.
    σ = 20.6e-3  # Surface tension, N/m  20-21 mN * m**-1. MBI tested at 20.6 and 20.8e-3

elif ν == 50:
    ρ = 960
    σ = 20.8e-3
else:
    raise ValueError


# Drop radius: 0.07 - 0.8mm. Smaller drops coalesce; larger ones don't do anything new at
# the driving accelerations used in MBI.
R_0 = 0.25e-3  # Undeformed drop radius.

ρ_a = 1.2  # Air density, kg/m^3
μ = 1e-2  # Drop dynamic viscocity. 10**-3 - 10**-1 kg*m^-1*s^-1
μ_a = 1.84e-5  # Air dynamic viscocity. 1.84 * 10**-5 kg*m^-1*s^-1 Constant?

ν_a = 15  # Air kinematic viscosity; 15 cSt


"""Bath oscillation properties; changes have a big impact on drop behavior."""
f = 80  # Bath shaking frequency.  40 - 200 Hz  # 80
# γ corresponds to path memory; high γ means high memory
γ = g * 2.2  # Peak bath vibration acceleration, m * s^-2 0-70  # 4.2


"""Values derived from constants."""
ω = 2*π * f  # = 2π*f Bath angular frequency.  250 - 1250 rad s^-1

ω_D = sqrt(σ / (ρ * R_0**3))  # Characteristic drop oscillation freq.  300 - 5000 s^-1
Bo = (ρ * g * R_0**2) / σ  # Bond number.  10**-3 - .04.
Oh = μ * (σ*ρ*R_0)**(-1/2)  # Drop Ohnsesorge number. 0.004-2
Oh_a = μ_a * (σ*ρ*R_0)**(-1/2)  # Air Ohnesorge number. 10**-4 - 10**-3
Ω = 2*π * f * sqrt(ρ * R_0**3 / σ)  # Vibration number.  0 - 1.4
Γ = γ / g  # Peak non-dimensional bath acceleration.  0 - 7


"""Constants from MBI / MBII that are calculated dynamically in our integrator."""
# We = (ρ * R_0 * V_in**2) / σ  # Weber number; Dimensionless. 0.01 - 1
# T_C = 10e-3  # Contact time, 1-20ms # Shown as tau elsewhere in MBII???
# V_in = 0.2  # Drop incoming speed; 0.1 - 1 m*s**-1
# V_out = 0.5  # Drop outgoing speed; 0.1 - 1 m*s**-1

"""Used to calculate the wave field; from a table in MBII, A.5"""
# Keep here instead of the surface_height function so it can be more efficient.
if ν == 20:
    if f <= 70:
        Γ_F = 2.562
    elif f <= 85:
        Γ_F = 4.220
    else:
        Γ_F = 5.159

elif ν == 50:
    if f <= 45:
        Γ_F = 2.707
    elif f <= 55:
        Γ_F = 4.028
    else:
        Γ_F = 5.514
else:
    Γ_F = 4.220

# ϵ = (Oh_e * Ω * k_F) / (3*k_F**2 + Bo)  # MBII A 41
# Γ_F = 2*ϵ * (1 + 3*k_F**2 / Bo) * (1 - (1/2) * ϵ**2)




"""More system parameters; MBII p 645. Uses version of 20 cSt drop."""
bath_depth = 9e-3  # m. Depth and diameter are from MBII, top of section II
D = 76e-3  # Cylindrical bath container diameter, m
# Effective gravity is g + γ*sin(τ*f*t)
# C is the non-dimensional drag cofficient. Depends weakly on system params.
# C ranges from .17 to .33 for walking regime, but .17 matches data from M&B paper.
# todo C should be close to 0.3, below MBII Eq 4.10??
C = .3
# ΓF  From lookup table?

m = .001  # Not in paper; temporary mass I'm using.

# Use the global lookup table for these values??

