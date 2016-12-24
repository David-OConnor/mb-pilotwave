from collections import namedtuple
from functools import partial
from typing import Tuple, Iterable

import brisk
import matplotlib.pyplot as plt
import numba
import numpy as np
from numpy import pi as π, sqrt, cos, sin, tan, arctan2, exp, log

from wave_reflection import find_reflection_points

jit = numba.jit(nopython=True)


τ = 2 * π

# Assume we're not modeling the up and down motions; each simulation tick
# represents the particle impacting the grid

# todo improved bounce/contact mechanics; tranfer energy to the drop from oscillation, and do more
# todo than a simple reflection.  Find the force of each impact; put into wave height eq.

# 2D grid, simulatinga a vibrating silicon/oil field. Positive values indicate
# a height above neutral; negative below.
GRID_SIZE = (200, 200)

RUN_TIME = 7400  # in ticks

dt = 1  # Seconds per tick

PARTICLE_MASS = 1  # Assumed to be a point; ie no volume.

# MBI / M&B = Malacek and Bush, 2013
# JFM = A trajectory equation for walking droplets: hydrodynamic pilot-wave theory
# by Anand U. Oza, Rodolfo R. Rosales and John W. M. Bush

# MBI has data for two drops:
# A: rho= 949, sigma = 20.6*10**-3, ν=20
# B: rho=960, sigma = 20.8 * 10**-3, and ν = 50cSt

# todo meters vs mm for distance??
# System paramters,  from MBI, Table 1, P 617.
# todo watch the metric units, ie mm vs m; papers are sometimes unclear or inconsistent
# todo ie implicity conversions in formulas vice table values.
R_0 = 0.40 * 10**-3  # Undeformed drop radius.  0.07 - 0.8mm
ρ = 949  # Silicone oil density (droplet and bed), 949 - 960 kg/m^3
ρ_a = 1.2  # Air density, kg/m^3
σ = 20.6 * 10 ** -3  # Surface tension, N/m/  20-21 mN * m**-1
g = 9.81  # Gravity, in m * s^-2.  Paper uses positive value??
V_in = 0.5  # Drop incoming speed; 0.1 - 1 m*s**-1
V_out = 0.5  # Drop outgoing speed; 0.1 - 1 m*s**-1
μ = 10**-2  # Drop dynamic viscocity. 10**-3 - 10**-1 kg*m^-1*s^-1
μ_a = 1.84 * 10 ** -5  # Air dynamic viscocity. 1.84 * 10**-5 kg*m^-1*s^-1 Constant?
ν = 20  # Drop kinematic viscocity; 10-100 cSt  MBI tested at 20 and 50.
ν_a = 15  # Air kinematic viscosity; 15 cSt
T_C = 10 * 10**-3  # Contact time, 1-20ms # Shown as tau elsewhere in MBI???
# C_r Coefficient of restitution; 0-0.4
f = 80  # Bath shaking frequency.  40 - 200 Hz
# γ corresponds to path memory; high gamma means high memory
γ = g * 4.2  # Peak bath vibration acceleration, m * s^-2 0-70
ω = τ * f  # = 2π*f Bath angular frequency.  250 - 1250 rad s^-1
# todo note: Poor notation on the paper for weber number; missing parents with rho and R_0.
ω_D = (σ / (ρ * R_0 ** 3)) ** (1 / 2)  # Characteristic drop oscillation freq.  300 - 5000s^-1
# todo weber's not coming out right...
We = ρ * R_0 * V_in**2 / σ  # Weber number; 0.01 - 1
Bo = ρ * g * R_0**2 / σ  # Bond number.  10**-3 - .04.
Oh = μ * (σ*ρ*R_0)**(-1/2)  # Drop Ohnsesorge number. 0.004-2
# todo oha not in range.
Oh_a = μ_a * (σ*ρ*R_0)**(-1/2)  # Air Ohnesorge number. 10**-4 - 10**-3
Ω = τ*f * sqrt(ρ * R_0**3 / σ)  # Vibration number.  0 - 1.4
Γ = γ / g  # Peak non-dimensional bath acceleration.  0 - 7



# More system parameters; MBI p 645. Use version of 20 cSt drop.

bath_depth = 9 * 10**-3  # mm
D = 76  # Cylindrical bath container diameter, mm  # todo should be x 10**-3 I think.
# Effective gravity is g + γ*sin(τ*f*t)
# C is the non-dimensional drag cofficient. Depends weakly on system params.
# C ranges from .17 to .33 for walking regime, but .17 matches data from M&B paper.
C = .17
# ΓF  From lookup table?

m = .001  # Not in paper; temporary mass I'm using.

# Use the global lookup table for these values??


# The following are numerically derived, found in a table MBI p 645.
# Use version of 20 cSt 80hz drop for now.

# These constants are foudd intables in M&B
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




############

# an Impact is an event of the drop hitting the surface.
Impact = namedtuple('Impact', ['t', 'x', 'y', 'F'])  # Add other aspects like speed, force etc.
Point = namedtuple('Point', ['x', 'y'])  # Add other aspects like speed, force etc.

impacts = []


class IntegrationEvent(Exception):
    """Pass the new y value as an argument."""
    pass


@jit
def effective_grav(t:  float) -> float:
    """Effective gravity in bath frame of reference. From MBI; not sure where
    to use it atm."""
    # Gravity plus the fictitious force in the vibrating bath reference frame.
    # I think this is taken care of numerically simply by subtracting bath height
    # when calculating the height below drop. Maybe.
    # Shown as  g*(t)
    return g + γ * sin(τ * f * t)


def effective_grav2(τ_:  float) -> float:
    """Effective gravity in bath frame of reference. From MBI; not sure where
    to use it atm."""
    # Shown as B0*(τ_) in MBI.
    return 1 + Γ * sin(Ω * τ_)


@jit
def drag(v: float):
    """From Molacek and Bush and JFM. The drag force """
    # D is the time-averaged drag coefficient.
    # "The first term arises from the transfer of momentum from the drop to the
    # bath during impact, and the second from the aerodynamic drag exerted on
    # the droplet during flight." - Oza, Rosales and Bush
    D = C*m*g * sqrt(ρ * R_0 / σ) + 6 * π * μ_a * R_0 * (1 + (π * ρ_a * g * R_0) / (6 * μ_a * ω))
    return -D * v


@jit
def impact_force(v: float) -> float:
    # todo superceded by momentum for now. ?
    """Calculate the force imparted by a drop bounce on the bath."""
    # We're transferring momentum from the drop to the bath. ?
    # v is drop speed at impact.

    # JFM; this is given in terms of the component of drag force, but might be
    # what we're looking for.
    C*m*g * sqrt(ρ * R_0 / σ)
    # units: kg * m * s**-2 * sqrt(kg/m^3 * m * m * N**-1)
    # kg * m * s**-2 * sqrt(m**-2 * s**2)  -->   kg * m * s**-2 * m**-1 * s**-1 = kg * s**-3??


    """ From M&B:  The drop’s change
    of momentum during impact is at most 1P ≈ 4/3 * π * ρ * R_0^3 * 2v"""
    ΔP = (2*τ / 3) * ρ * R_0**3 * 2*v


@jit
def surface_height(t: float, r: float, v: float) -> float:
    """From Molacek and Bush 'Drops Walking on a vibrating Bath'. Analytic solution for surface
    height, based on one impact. The results can be added to take into account multiple
    impacts. v is impact velocity

    This models only the standing/faraday wave; the transient wave generated by
    a  bounce does not interact with the droplet on subsequent bounces, so we
    can discard it."""

    τ_ = ω_D * t  # τ_ is dimensionless time, not 2*pi !!
    # todo what is μe??? Not defined in paper, but used.

    # Ohe is the effective Ohnesorge number. # todo what is mu e??
    # Ohe = μe / (σ*ρ*R0)**(1/2)  # μe / (σρR0)**(1/2)  # or OhD ?'
    μ_e = μ # todo ?
    Oh_e = μ_e / (σ * ρ * R_0) ** (1/2)  # μe / (σρR0)**(1/2)  # or OhD ?

    term1 = (4*sqrt(τ))/(3*sqrt(τ_)) * (k_C**2*k_F*Oh_e**(1/2))/(3*k_F**2 + Bo)


    # F is the Dimensionless reaction force acting on the drop.

    # The formula in M&B calls for an integration of F over the contact time;
    # The info clarifying that force can be approximated by a point force doens't
    # explicitly, AFAICT, say how to approximate this point force, but includes
    # this statement about change of momentum; and since integrating momentum over
    # time is force... Can we just use this?

    # "The drop’s change of momentum during impact is at most ΔP ≈ 4/3 * π *
    # ρ * R_0^3 * 2v"
    ΔP = (2 * τ / 3) * ρ * R_0**3 * 2 * v
    amplitude = ΔP * sin(Ω/2)

    term3 = cos(Ω*τ_ / 2) * exp((Γ / Γ_F - 1) * (τ_ / τ_D)) * brisk.j0(k_C * r)

    # Note: todo this is a start at the more "complete" version in the paper.
    # term1 = (4*sqrt(τ))/(3) * (k_C**2*kF*Ohe**(1/2))/(3*kF**2 + Bo)
    # term2 =
    # term3 = H(τ)/sqrt(τ) * exp((Γ/ΓF - 1) * (τ/τ_D)) * special.j0(k_C*r)

    return term1 * amplitude * term3


@jit
def height_helper(t: float, impact: Impact, point: Point) -> float:
    """Convenience function to calculate surface height at a point, given an
    impact and time."""
    Δx, Δy = point[0] - impact.x, point[1] - impact.y
    point_dist = (Δx ** 2 + Δy ** 2) ** 0.5
    return surface_height(t, point_dist, impact.F)


def net_surface_height(t: float, impacts_: Iterable, sample_pt: np.ndarray,
                       corral=False, reflectivity=1) -> float:
    """Finds the height under a sample point, taking into account multiple impacts."""
    # todo add a limiter so old impacts aren't included; they're insignificant,
    # todo and we need to computationally limit the num of impacts.
    # This could possibly be jitted by replacing the Impacts list/object with a 2d array.

    # height_below_drop includes the base height from all impacts, and their reflections.
    height_below_sample = 0

    for impact in impacts_:
        t_since_impact = t - impact.t  # We care about time since impact.
        # Exclude impacts that occur after, or simulataneously with the current time.
        if t_since_impact <= 0:
            continue

        # Find the the base, ie non-reflected-component height.
        height_below_sample += height_helper(t_since_impact, impact, sample_pt)

        corral_center = np.array([0, 0])
        # todo adjustable corral center
        if corral:
            Δx, Δy = sample_pt[0] - corral_center[0], sample_pt[1] - corral_center[1]
            dist_sample_center = (Δx ** 2 + Δy ** 2) ** 0.5
            # Points outside the circle are 0 for full reflectivity.
            if dist_sample_center > D/2:
                height_below_sample *= 1 - reflectivity
            else:
                # Add reflection adjustment; one reflection currently. Find all points
                # outside the circle that reflect onto our sample point; find their
                # heights, and add to the sample.
                # Convert impact to an array, for faster use with jited funcs.
                for reflection_pt in find_reflection_points(np.array([impact.x, impact.y]), sample_pt):
                    height_below_sample += height_helper(t_since_impact, impact, reflection_pt) * reflectivity

    return height_below_sample


def surface_height_gradient(t: float, impacts_: Iterable[Impact], x: float, y: float) -> \
        Tuple[float, float]:
    """Create a linear approximation, for finding the slope at a point.
    x and y are points.  t is the time we're taking the derivative.
    Used to calculate bounce mechanics."""
    # todo perhaps you should take into account higher order effects.
    δ = 1e-6  # Arbitrarily small

    height = partial(net_surface_height, t, impacts_)

    # Take a sample on each side of the location we're testing.
    h_x_left = height((x - δ/2, y))
    h_x_right = height((x + δ/2, y))
    h_y_left = height((x, y - δ/2))
    h_y_right = height((x, y + δ/2))

    return (h_x_right - h_x_left) / δ, (h_y_right - h_y_left) / δ


def surface_height_gradient2(t, impacts_: Iterable[Impact], x, y) -> Tuple[float, float]:
    # todo no worky with numba
    """Create a linear approximation, for finding the slope at a point.
    x and y are points.  t is the time we're taking the derivative.
    Used to calculate bounce mechanics."""
    import autograd

    height = partial(net_surface_height, t, impacts_)
    grad = autograd.grad(height)
    return grad(x, y)


@jit
def surface_oscilation(t: float) -> Tuple[float, float, float]:
    """Return the surface's height and velocity at time t due to its oscillation;
    a simple harmonic oscillator."""
    # Oscillation starts the cycle at negative amplitude.
    # Use global constants for frequency and bath acceleration.

    bath_accel = γ * cos(ω * t)  # integrate this to get velocity and position.
    bath_vel = γ * ω**-1 * sin(ω * t)  # + C
    bath_height = -γ * ω**(-2) * cos(ω*t)  # + C*t

    return bath_height, bath_vel, bath_accel


# def bounce():
#     # FT is the trangental component of the reaction force.
#     F = 0
#     FT = -F*(δh(X, τ)) / (δX)

# @jit
def bounce_v(grad_x: float, grad_y: float, vx: float, vy: float, vz: float) -> np.ndarray:
    """Calculate the outgoing velocity in x, y, and z directions after a bounce."""
    # todo atm the drop does not lose any momentum to the surface.
    # todo you coulud include bath oscillation velocity here, but may need
    # todo a fininte contact time to do so.
    v = np.array([vx, vy, vz])
    normal = np.cross(np.array([1, 0, grad_x]), np.array([0, 1, grad_y]))
    unit_normal = normal / np.linalg.norm(normal)

    reflection = v - 2*(v @ unit_normal) * unit_normal

    # todo calculate collision kinetic energy? Catchers mit keeps drop from coalescing??

    return reflection


# @jit
def rk4(f, y: Iterable, t: float, h: float, args: Tuple) -> np.ndarray:
    """Basic mechanics of Runge-Kutta 4 ODE"""
    # Convert to arrays to so we can add and multiply element-wise.
    y = np.array(y)

    k1 = np.array(f(y, t, *args)) * h
    k2 = np.array(f(y + k1/2, t + h/2, *args)) * h
    k3 = np.array(f(y + k2/2, t + h/2, *args)) * h
    k4 = np.array(f(y + k3, t + h, *args)) * h
    return y + (k1 + 2*(k2 + k3) + k4) / 6


def rhs(y: np.ndarray, t: np.ndarray) -> Tuple:
    """Right hand integration function, simplified for one drop; no bounce"""
    sx, sy, sz, vx, vy, vz = y

    # # Calculate drag force for accelerations. The z component also includes gravity.
    ax, ay = drag(vx), drag(vy)
    az = drag(vz) - g
    # az = drag(vz) - effective_grav(t) # you could do this too?

    return vx, vy, vz, ax, ay, az


def ode_standalone(t: np.ndarray, corral=False) -> Tuple:
    """Purpose-built, non-general RK4 integrator for modelling multiple drops,
    with events. Events with one drop, or multiple drops with no events work
    with more elegant solutions, like scipy's odeint or the rk4_odeint above."""
    impacts_ = []

    # todo model two separate conditions: drop in flight, and drop in contact
    # todo with surface.

    # initial drop conditions: sx, sy, sz, vx, vy, vz
    # Per MBI, initial conditions play little role in subsequent dynamics.
    drops = np.array([
        [0, 0, 1, .1, .1, 0],
        # [100, 110, 10, 0, 0, 0],
        # [100, 95, 10, 0, 0, 0],
        # [105, 100, 10, 0, 0, 0],
        # [105, 105, 10, 0, 0, 0],
        # [105, 95, 10, 0, 0, 0],
        # [95, 100, 10, 0, 0, 0],
        # [95, 105, 10, 0, 0, 0],
        # [95, 95, 10, 0, 0, 0],

    ])

    # Border format is (x1, y1, x2, y2), ie a line connecting two points.
    borders = [(0, 200, 500, 200)]

    num_drops, drop_len = drops.shape
    # The solution will have three axis: time, drop, and drop feature. (features are position, velocity etc etc)
    soln = np.empty([len(t), num_drops, drop_len])
    soln[0] = drops

    for i in range(len(t) - 1):
        t_ = t[i]  # current  time
        h = t[i + 1] - t[i]  # time step

        for j in range(num_drops):
            y_to_integrate = soln[i, j]  # todo ??
            y_drop = rk4(rhs, y_to_integrate, t_, h, args=())  # no bounce

            sx, sy, sz, vx, vy, vz = y_to_integrate

            if sz <= 10 and vz < 0:  # todo lower sz limit.
                sample_pt = np.array([sx, sy])
                height_below_drop = net_surface_height(t_, impacts_, sample_pt, corral=corral)

                # Take into account the surface oscillation; it moves the whole
                # surface uniformly.  Also, reference the bottom of the drop
                # rather than the top by shifting down half a radius.
                height_below_drop -= (surface_oscilation(t_)[0] + R_0/2)

                # An impct is detected.
                if sz <= height_below_drop:
                    grad_x, grad_y = surface_height_gradient(t_, impacts_, sx, sy)
                    # This bounce velocity change overrides the default, of last step's accel.
                    vx_bounce, vy_bounce, vz_bounce = bounce_v(grad_x, grad_y,
                                                               vx, vy, vz)

                    # todo here, or in bounce_v, goes the force imparted on the drop:
                    # todo −mg∇h(x_p, t) horizontal force??

                    # todo what next?
                    bath_v_start = surface_oscilation(t)[1]
                    bath_v_end = surface_oscilation(t_ + T_C)[1]

                    bath_a_start = surface_oscilation(t)[2]
                    bath_a_end = surface_oscilation(t_ + T_C)[2]

                    # todo is it this simple? add the bath velocity at the end of
                    # todo the bounce. Can't be, since it's net is 0; we need to add
                    # todo energy.
                    # vz_bounce += bath_v_end

                    v = sqrt(vx**2 + vy**2 + vz**2)
                    impacts_.append(Impact(t_, sx, sy, v))

                    # Overwrite the prev-calculated non-bounce integration with
                    # these calculated values.
                    y_drop = sx, sy, sz, vx_bounce, vy_bounce, vz_bounce

            soln[i+1, j] = y_drop

    return soln, impacts_


def plot_trajectories(soln: np.ndarray):
    n_drops = soln.shape[1]

    for drop_i in range(n_drops):
        plt.plot(soln[:, drop_i, 0], soln[:, drop_i, 1])

    plt.show()


def quickplot(t: np.ndarray):
    soln, impacts = ode_standalone(t)
    plot_trajectories(soln)
