from collections import namedtuple
from functools import partial
from typing import Tuple, Iterable

import matplotlib.pyplot as plt
import numba
import numpy as np

from numpy import pi as π, e, sqrt, cos, sin, exp, arctan

from scipy import integrate, special


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

dt = 10e-2  # Seconds per tick

PARTICLE_MASS = 1  # Assumed to be a point; ie no volume.

# Added from Drops Walking...
R0 = 0.39  # Undeformed drop radius.
ρ = 949  # Silicone oil density (droplet and bed), kg/m^3
ρa = 1.2  # Air density, kg/m^3
σ = 20.6e-3  # Surface tension, N/m
g = -9.81  # Gravity, in m * s^-2.  Paper uses positive value??
v = 20  # kinematic viscocity, cSt
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


# @jit
def surface_height_inner(t: float, F: float) -> float:
    """From 'Drops Walking on a vibrating Bath'."""
    # Analytic solution for surface height, based on one impact.
    τ = ωD * t  # dimensionless time.
    # todo what is μe??? Not defined in paper, but used.

    # Ohe is the effective Ohnesorge number. # todo what is mu e??
    # Ohe = μe / (σ*ρ*R0)**(1/2)  # μe / (σρR0)**(1/2)  # or OhD ?
    Ohe = μ / (σ*ρ*R0)**(1/2)  # μe / (σρR0)**(1/2)  # or OhD ?


    # Use the lookup table for these values??
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

    term3 = cos(Ω/2) * exp((Γ/ΓF - 1) * (τ/τD))  # * special.j0(kC*r)

    # Note: todo this is a start at the more "complete" version in the paper.
    # term1 = (4*sqrt(2*π))/(3) * (kC**2*kF*Ohe**(1/2))/(3*kF**2 + Bo)
    # term2 =
    # term3 = H(τ)/sqrt(τ) * exp((Γ/ΓF - 1) * (τ/τD)) * special.j0(kC*r)

    return term1 * term2 * term3


def surface_height(t: float, r: float, F: float) -> float:
    """Workaround for scipy.special.j0 not working in numba; @jit the rest,
    then multiply by the j0 bessel function separately."""
    kC = .888
    return surface_height_inner(t, F) * special.j0(kC*r)


def net_surface_height(t: float, x: float, y: float, impacts_: Iterable) -> float:
    """Finds the height, taking into account multiple impacts."""
    # todo add a limiter so old impacts aren't included; they're insignificant,
    # todo and we need to computationally limit the num of impacts.
    height_below_drop = 0

    for impact_ in impacts_:
        t_since_impact = t - impact_.t  # We care about time since impact.
        r = ((impact_.x - x)**2 + (impact_.y - y)**2) **.5
        height_below_drop += surface_height(t_since_impact, r, impact_.F)
    return height_below_drop


# @jit
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
def bounce_v(grad_x, grad_y, vx, vy, vz):
    """Calculate the outgoing velocity in x, y, and z directions after the
    initial bounce."""
    # todo atm the drop does not lose any momentum to the surface.
    v = np.array([vx, vy, vz])
    normal = np.cross(np.array([1, 0, grad_x]), np.array([0, 1, grad_y]))
    unit_normal = normal / np.linalg.norm(normal)

    reflection = v - 2*(v @ unit_normal)*unit_normal

    # print(reflection, "reflect")
    print(grad_x, grad_y, vx, vy, vz, "inputs")
    # print(unit_normal, v, "normal, v")
    return reflection




def int_func(y: Iterable, t: np.ndarray) -> Tuple:
    """Right hand integration function."""
    sx, sy, sz, vx, vy, vz, xa, ya, az = y

    # waves = wave_field(t)
    # _p means prime; ie derivative
    sx_p, sy_p, sz_p = vx*dt, vy*dt, vz*dt
    vx_p, vy_p, vz_p = xa*dt, ya*dt, az*dt
    ax_p, ay_p, az_p = 0, 0, 0

    # todo only invoke bounce-detection logic if drop's below a certain height,
    # todo for computational efficiency?
    height_below_drop = net_surface_height(t, sx, sy, impacts)

    if sz <= height_below_drop:
        grad_x, grad_y = surface_height_gradient(t, sx, sy, impacts)
        vx_p, vy_p, vz_p = bounce_v(grad_x, grad_y, vx, vy, vz)

        # Add this new impact for future calculations.
        F = .1  # todo i don't know what to do here.
        impacts.append(Impact(t, sx, sy, F))

    dydt = sx_p, sy_p, sz_p, vx_p, vy_p, vz_p, ax_p, ay_p, az_p
    print("ITER")
    return dydt

def integrate_test():
    # y0 is Drop xs, ys, zs, xv, yv, zv, xa, ya, za, surface
    y0 = 150, 200, 10, 0, 0, 0, 0, 0, g
    t = np.arange(30)
    return integrate.odeint(int_func, y0, t)


def wave_field(t: float=2, origin=(250, 250)) -> np.ndarray:
    """Calculate a wave's effect on a 2d field."""
    h = np.zeros([500, 500])

    x_origin, y_origin = origin
    pixel_dist = 10

    for i in range(500):
        for j in range(500):
            r = ((y_origin-i)**2 + (x_origin-j)**2) ** .5
            r /= pixel_dist
            # Assuming we can just add the heights.
            F = 1
            h[i, j] += surface_height(t, r, F)

    # # This style uses broadcasting. Stumbling block on surface height ufunc.
    # y, x = np.mgrid[:500, :500]
    # r = sqrt((x_origin - x)**2 + (y_origin - y)**2)
    # h = surface_height(r, t)  # does surfaceheight need to be a ufunc? yes.

    return h


def plot_surface(z: np.ndarray) -> None:
    """Make a surface plot from a 2d array."""
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm

    y, x = np.mgrid[:z.shape[0], :z.shape[1]]

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(x, y, z, cmap=cm.viridis, linewidth=.2, cstride=10, rstride=10)

    plt.tight_layout(rect=(0, 0, 1, 1))
    plt.show()
