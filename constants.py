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


# System paramters, from MBII, Table 1, and MBI table 1.

# MBI/II has data for two drops:
# A: rho= 949, sigma = 20.6*10**-3, ν=20
# B: rho=960, sigma = 20.8 * 10**-3, and ν = 50cSt

# Constants
# Drop radius: 0.07 - 0.8mm. Smaller drops coalesce; larger ones don't do anything new at
# the driving accelerations used in MBI.
R_0 = 0.40e-3  # Undeformed drop radius.
ρ = 949  # Silicone oil density (droplet and bed), 949 and 960 kg/m^3 tested in MBI.
ρ_a = 1.2  # Air density, kg/m^3
σ = 20.6e-3  # Surface tension, N/m  20-21 mN * m**-1. MBI tested at 20.6 and 28.8e-3

# Use the integrator to find incoming and outgoing speeds.
# V_in = 0.2  # Drop incoming speed; 0.1 - 1 m*s**-1
# V_out = 0.5  # Drop outgoing speed; 0.1 - 1 m*s**-1

μ = 1e-2  # Drop dynamic viscocity. 10**-3 - 10**-1 kg*m^-1*s^-1
μ_a = 1.84e-5  # Air dynamic viscocity. 1.84 * 10**-5 kg*m^-1*s^-1 Constant?
ν = 20  # Drop kinematic viscocity; 10-100 cSt  MBI tested at 20 and 50.
ν_a = 15  # Air kinematic viscosity; 15 cSt
T_C = 10e-3  # Contact time, 1-20ms # Shown as tau elsewhere in MBII???
f = 20  # Bath shaking frequency.  40 - 200 Hz  # 80
# γ corresponds to path memory; high γ means high memory
γ = g * 3.2  # Peak bath vibration acceleration, m * s^-2 0-70  # 4.2

# Derived from constants
ω = 2*π * f  # = 2π*f Bath angular frequency.  250 - 1250 rad s^-1
ω_D = sqrt(σ / (ρ * R_0**3))  # Characteristic drop oscillation freq.  300 - 5000 rad*s^-1

# Dynamically Calculate Weber number based on impact velocity
# We = (ρ * R_0 * V_in**2) / σ  # Weber number; Dimensionless. 0.01 - 1

Bo = (ρ * g * R_0**2) / σ  # Bond number.  10**-3 - .04.
Oh = μ * (σ*ρ*R_0)**(-1/2)  # Drop Ohnsesorge number. 0.004-2
Oh_a = μ_a * (σ*ρ*R_0)**(-1/2)  # Air Ohnesorge number. 10**-4 - 10**-3
Ω = 2*π * f * sqrt(ρ * R_0**3 / σ)  # Vibration number.  0 - 1.4
Γ = γ / g  # Peak non-dimensional bath acceleration.  0 - 7


# More system parameters; MBII p 645. Uses version of 20 cSt drop.
bath_depth = 9e-3  # m
D = 76e-3  # Cylindrical bath container diameter, m
# Effective gravity is g + γ*sin(τ*f*t)
# C is the non-dimensional drag cofficient. Depends weakly on system params.
# C ranges from .17 to .33 for walking regime, but .17 matches data from M&B paper.
C = .17
# ΓF  From lookup table?

m = .001  # Not in paper; temporary mass I'm using.

# Use the global lookup table for these values??

# The following are numerically derived, found in a table MBI p 645.
# Use version of 20 cSt 80hz drop for now.

# These constants are found intables in M&B
Γ_F = 4.220
k_C__k_F = .971  # ratio of k_C to k_F
τ_F__τ_D = 1.303

# I think we can do k_C = k_F: "The critical (most unstable) wavenumber k C is
# found to be close to the Faraday wavenumber k F , given by the dispersion
# relation (Benjamin & Ursell 1954): k_F**3 + Bo * k_F = (1/4) * Ω**2.
# This doesn't come out neatly solving for k_F!. but JFM has an approximation:
k_F = 1.25  # mm**-1
k_C = k_F  # k_C is the critical (most unstable) wavenumber.

# M&B also includes it as τ_D = (Oh_e * k_C**2) ** -1. Not sure what Oh_e is, so can't compare.
# todo I think we can approximate Oh_e as just Oh? In that case, this value's wack?
τ_D = 1 / 54.9  # Decay time (s) of waves without forcing, from JFM (T_D) there