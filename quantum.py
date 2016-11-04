from collections import namedtuple
from typing import Tuple, Iterable

# import PyDSTool
import brisk
import matplotlib.pyplot as plt
import numba
import numpy as np
import scikits.odes
import scikits.odes.sundials
from numpy import pi as π, sqrt, cos, sin, exp
from scikits.odes import dae
from scikits.odes.sundials import ida

jit = numba.jit(nopython=True)

# Assume we're not modeling the up and down motions; each simulation tick
# represents the particle impacting the grid

# todo Do I want a discrete grid as an array, or should I take a more precise route?
# todo What about modeling the particle's verticle motion: analytic, or numerical?
# todo ... I think it has to be numerical, for now.

# does particle height represent energy??

# 2D grid, simulatinga a vibrating silicon/oil field. Positive values indicate
# a height above neutral; negative below.
GRID_SIZE = (200, 200)

RUN_TIME = 7400  # in ticks

dt = 1  # Seconds per tick

PARTICLE_MASS = 1  # Assumed to be a point; ie no volume.

# Added from Drops Walking...
R0 = 0.39  # Undeformed drop radius.
ρ = 949  # Silicone oil density (droplet and bed), kg/m^3
ρa = 1.2  # Air density, kg/m^3
σ = 20.6e-3  # Surface tension, N/m
g = -9.81  # Gravity, in m * s^-2.  Paper uses positive value??
vis = 20  # kinematic viscocity, cSt
μ = 10**-2 # Drop dynamic viscocity. 10**-3 - 10**-1 kg*m^-1*s^-1
μa = 1.84 * 10**-5  # Air dynamic viscocity. 1.84 * 10**-5 kg*m^-1*s^-1 Constant?
bath_depth = 9  # mm
D = 76  # Cylindrical bath container diameter, mm
γ = 50  # Peak bath vibration acceleration, m * s^-2 0-70
# Effective gravity is g + γ*sin(τ*f*t)


# don't use circle tau here since it's used to mean
# something different.

f = 100  # Bath shaking frequency.  40 - 200 Hz
ω = 2 * π * f  # = 2π*f Bath angular frequency.  250 - 1250 rad s^-1
# ωD = (σ/ρR0^3)^(1/2) Characteristic drop oscillation freq.  300 - 5000s^-1
ωD = (σ/ρ*R0**3)**(1/2)
Oh = 1  # Drop Ohnsesorge number. 0.004-2
Bo = .1  # Bond number.  10**-3 - 0.4
Ω = 0.7  # Vibration number.  0 - 1.4
Γ = 3  # Peak non-dimensional bath acceleration.  0 - 7
# ΓF  From lookup table?
ΓF = 5.159

m = .001  # Not in paper; temporary mass I'm using.


# todo do I need accel? Constant down due to g, and bounce accel could be applied instantly??

# todo bouncing: logarithmic spring? from 'Drops walking on a vibrating bath'.
# todo: We're ommitting air drag for now.

############

# an Impact is an event of the drop hitting the surface.
Impact = namedtuple('Impact', ['t', 'x', 'y', 'F'])  # Add other aspects like speed, force etc.

impacts = []

class IntegrationEvent(Exception):
    """Pass the new y value as an argument."""
    pass


@jit
def surface_height(t: float, r: float, F: float) -> float:
    """From 'Drops Walking on a vibrating Bath'."""
    # Analytic solution for surface height, based on one impact.
    τ = ωD * t  # dimensionless time.
    # todo what is μe??? Not defined in paper, but used.

    # Ohe is the effective Ohnesorge number. # todo what is mu e??
    # Ohe = μe / (σ*ρ*R0)**(1/2)  # μe / (σρR0)**(1/2)  # or OhD ?
    Ohe = μ / (σ*ρ*R0)**(1/2)  # μe / (σρR0)**(1/2)  # or OhD ?

    # Use the global lookup table for these values??
    τF = 1.303  # Faraday period.
    τC = 0  # Contact time? Dimensionless?? 1-20ms ??
    τD = 1.303  # Decay time
    kC = .888  # Also given by a formula... (3.8)
    kF = .888

    # F = 0  # Dimensionless reaction force.

    term1 = (4*sqrt(2*π))/(3*sqrt(τ)) * (kC**2*kF*Ohe**(1/2))/(3*kF**2 + Bo)

    # Todo we're using instantaneous impacts as a simplification for now.
    # Term 2 represents the amplitude of the wave.
    # term2 = integrate(F(u) * sin(Ω*u/2) * du)
    # Todo I'm not sure if this is the correct way to make the integration
    # todo of the impact instant, but let's try.
    term2 = F * sin(Ω/2)

    term3 = cos(Ω/2) * exp((Γ/ΓF - 1) * (τ/τD)) * brisk.j0(kC*r)

    # Note: todo this is a start at the more "complete" version in the paper.
    # term1 = (4*sqrt(2*π))/(3) * (kC**2*kF*Ohe**(1/2))/(3*kF**2 + Bo)
    # term2 =
    # term3 = H(τ)/sqrt(τ) * exp((Γ/ΓF - 1) * (τ/τD)) * special.j0(kC*r)

    return term1 * term2 * term3


def net_surface_height(t: float, x: float, y: float, impacts_: Iterable) -> float:
    """Finds the height, taking into account multiple impacts."""
    # todo add a limiter so old impacts aren't included; they're insignificant,
    # todo and we need to computationally limit the num of impacts.
    # This could possibly be jitted by replacing the Impacts list/object with a 2d array.

    height_below_drop = 0

    for impact_ in impacts_:
        t_since_impact = t - impact_.t  # We care about time since impact.
        # Excluse impacts that occur after, or simulataneously with the current time.
        if t_since_impact <= 0:
            continue

        r = ((impact_.x - x)**2 + (impact_.y - y)**2) ** 0.5
        height_below_drop += surface_height(t_since_impact, r, impact_.F)
    return height_below_drop


def surface_height_gradient(t, x, y, impacts_: Iterable[Impact]) -> Tuple[float, float]:
    """Create a linear approximation, for finding the slope at a point.
    x and y are points.  t is the time we're taking the derivative.
    Used to calculate bounce mechanics."""
    # todo perhaps you should take into account higher order effects.
    δ = 10e-5  # Should this be fixed?

    # Take a sample on each side of the location we're testing.
    # Calculate the radiuses.

    def height(x_, y_):
        return net_surface_height(t, x_, y_, impacts_)

    h_x_left = height(x - δ/2, y)
    h_x_right = height(x + δ/2, y)
    h_y_left = height(x, y - δ/2)
    h_y_right = height(x, y + δ/2)

    return (h_x_right - h_x_left) / δ, (h_y_right - h_y_left) / δ


# def bounce():
#     # FT is the trangental component of the reaction force.
#     F = 0
#     FT = -F*(δh(X, τ)) / (δX)

# @jit
def bounce_v(grad_x: float, grad_y: float, vx: float, vy: float, vz: float) -> np.ndarray:
    """Calculate the outgoing velocity in x, y, and z directions after a bounce."""
    # todo atm the drop does not lose any momentum to the surface.
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


def rk4_odeint(f, y0: Iterable, t: np.ndarray, args: Tuple=()) -> np.ndarray:
    """Interface for RK4 ODE solver, similar to scipy.integrate.odeint."""
    y0 = np.array(y0)
    result = np.empty([len(t), len(y0)])
    y = y0
    for i in range(len(t) - 1):
        result[i] = y

        t_ = t[i]
        h = t[i+1] - t[i]
        try:
            y = rk4(f, y, t_, h, args)
        except IntegrationEvent as event:
            # Assign y to be the values you're restarting the integrator from,
            # passed from the right-hand side in the exception.
            y = event.args[0]
            # todo handle restarting after a bounce here, or just return what
            # todo we have, and continue in a loop outside?

    result[-1] = y
    return result


def drag(v: float):
    CD = .58  # really this varies! Usually between 0.2 and 0.5
    A = π * R0 ** 2  # Cross-sectional area of undeformed drop.
    return (CD * ρa * A * v**2) / 2


def ode_rhs(y: Iterable, t: float) -> Tuple:
    """Right hand integration function."""
    sx, sy, sz, vx, vy, vz = y
    ax, ay, az = 0, 0, g

    # The limit on sz is to prevent calling net_surface_height when the ball's
    # no where near bouncing; the vz check is a fudge for the ball being detected
    # a bit below the surface; dont' catch it on the way up.
    if sz <= 10 and vz < 0:  # todo lower sz limit.
        height_below_drop = net_surface_height(t, sx, sy, impacts)

        if sz <= height_below_drop:  # A bounce is detected.
            grad_x, grad_y = surface_height_gradient(t, sx, sy, impacts)
            vx, vy, vz = bounce_v(grad_x, grad_y, vx, vy, vz)

            # Add this new impact for future calculations.
            F = 3  # todo i don't know what to do here.
            impacts.append(Impact(t, sx, sy, F))
            y = sx, sy, sz, vx, vy, vz, ax, ay, az
            raise IntegrationEvent(y)

    # todo fudge factors giving an approx of air resistance.
    # ax_p = drag(vx) / m
    # ay_p = drag(vy) / m
    # az_p = drag(vz) / m

    return vx, vy, vz, ax, ay, az


def integrate_run() -> np.ndarray:
    # y0 is Drop sx, sy, sz, vx, vy, vz, ax, ay, az
    y0 = 150, 200, 10, .8, .3, 0
    t = np.linspace(0, 30, 4000)

    global impacts
    impacts = []
    # return integrate.odeint(ode_rhs, y0, t)
    return rk4_odeint(ode_rhs, y0, t), t, impacts


def skode_test():
    # y0 is Drop ss, sy, sz, vx, vy, vz, ax, ay, az
    y0 = 150, 200, 10, 0, 0, 0
    # t0 = 0
    t = np.linspace(0, 20, 200)

    def skode_rhs(t, y, ydot):
        sx, sy, sz, vx, vy, vz = y
        ax_p, ay_p, az_p = 0, 0, g

        ydot[:] = np.array([vx, vy, vz, ax_p, ay_p, az_p])

    solution = scikits.odes.ode('cvode', skode_rhs, old_api=False).solve(t, y0)
    return solution

