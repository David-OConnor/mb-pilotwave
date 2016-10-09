from collections import namedtuple
from functools import partial
from typing import Tuple, Iterable

import matplotlib.pyplot as plt
import numba
import numpy as np

from numpy import pi as π, e, sqrt, cos, sin, exp

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


# pos in x, y, height above neutral; respective velocities
PARTICLE_START = (150, 150, 10, 0, 0, 0, 0, 0, g)  # should probably be G for z accel.
x, y = np.mgrid[:GRID_SIZE[0], :GRID_SIZE[1]]
droplet_x, droplet_y = PARTICLE_START[0], PARTICLE_START[1]
rr = (x - droplet_x)**2 + (y - droplet_y)**2
# m/s. Confirm this is a constant. Group vs phase velocity?  Sometimes 'c' isused.
WAVE_SPEED = 10

# todo do I need accel? Constant down due to g, and bounce accel could be applied instantly??

# todo bouncing: logarithmic spring? from 'Drops walking on a vibrating bath'.

# negative z values are down.
# Instead of simpler euler method step, use RK4? Either write yourself, or use scipy.integrate??
# Perhaps use a midpoint method? (Like feynman)

class Particle:
    # The particle's position (s) values are grid indices.
    def __init__(self, mass, sx, sy, sz, vy, vx, vz, ax, ay, az):
        self.mass = mass
        self.sx = sx
        self.sy = sy
        self.sz = sz
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.ax = ax
        self.ay = ay
        self.az = az

    def step(self) -> None:
        self.sy += self.vy * dt
        self.vz += self.az * dt
        self.sz += self.vz * dt

        self.vx += self.ax * dt
        self.sx += self.vx * dt
        self.vy += self.ay * dt

    def bounce(self, surface: np.ndarray) -> None:
        """Perfectly elastic?"""
        # todo do I need to take xz and yz into account too? Probably!
        ke = 1/2 * self.mass * self.vz**2

        # todo temp kludge!!
        nearest_grid_point = int(round(self.sx)), int(round(self.sy))
        self.vz *= -1.0
        self.sz = surface.η[nearest_grid_point[0], nearest_grid_point[1]]


def spatial_derivative(axis: int, A: np.ndarray):
    """
    Compute derivative of array A using balanced finite differences
    Axis specifies direction of spatial derivative (d/dx or d/dy)

    dA[i] =  A[i+1] - A[i-1]   / 2
    ... or with grid spacing included ...
    dA[i]/dx =  A[i+1] - A[i-1]   / 2dx

    Used By:
        d_dx
        d_dy
    """
    grid_spacing = 1  # todo what is this?
    return (np.roll(A, -1, axis) - np.roll(A, 1, axis)) / (grid_spacing*2.)


d_dx = partial(spatial_derivative, 1)
d_dy = partial(spatial_derivative, 0)


class Surface:
    def __init__(self, size: Tuple[int, int]):
        # todo one dim=3 array, or multiple dim=2

        self.η = np.ones([*size])  # Columnheight/ pressure?
        self.dη_dt = np.zeros([*size])  # change in Columnehgith / pressure?

        self.u = np.zeros([*size])  # Velocity in x direction
        self.du_dt = np.zeros([*size])  # accel? in x direction

        self.v = np.zeros([*size])  # Velocity in y direction
        self.dv_dt = np.zeros([*size])  # accel? in y direction

        # self.η[rr<10**2] = 1.1 # add a perturbation in pressure surface
        self.η[100, 100] = 1.1

    def step(self) -> None:
        """Execute non-conservative shallow water equations:
        https://en.wikipedia.org/wiki/Shallow_water_equations"""
        η, u, v = self.η, self.u, self.v  # Code shortener

        H = 0  # H is the mean height of the horizontal pressure surface.
        b = 0  # b is the viscous drag coefficient.

        # todo η is used here instead of "h: the height deviation of the horizontal
        # pressure surface from its mean height H".  Why??
        self.du_dt = g * d_dx(η) - b * u
        self.dv_dt = g * d_dy(η) - b * v
        self.dη_dt = -d_dx(u * (H + η)) - d_dy(v * (H + η))

        self.η += self.dη_dt * dt
        self.u += self.du_dt * dt
        self.v += self.dv_dt * dt

    def bounce(self, p: Particle) -> None:
        """Perfectly elastic?"""
        # Impact point corresponds to a point on grid's first two dimensions.
        # todo do I need to take xz and yz into account to? Probably!

        # todo interpolate!
        nearest_grid_point = int(round(p.sx)), int(round(p.sy))

        ke = 1/2 * p.mass * p.vz**2


def step(particle: Particle, surface: Surface, time: int) -> int:
    """Execute simple euler time-stepping."""
    particle.step()
    surface.step()
    time += dt

    return time


def main():
    surface = Surface(GRID_SIZE)
    particle = Particle(PARTICLE_MASS, *PARTICLE_START)
    t = 0

    x_ = []
    y_ = []
    for i in range(RUN_TIME):
        t = step(particle, surface, t)

        # print(p.sx, p.sy, p.sz, p.vx, p.vy, p.vz, p.ax, p.ay, p.az)
        # todo interpolate, instead of taking the nearest one!!

        x_.append(t)
        y_.append(particle.sz)

        nearest_grid_point = int(round(particle.sx)), int(round(particle.sy))
        if particle.sz <= surface.η[nearest_grid_point[0], nearest_grid_point[1]]:
            surface.bounce(particle)
            particle.bounce(surface)

    # plt.plot(x, y)
    # plt.show()

    im = plt.imshow(surface.η)
    plt.colorbar(im, orientation='horizontal')
    plt.show()

    return particle, surface


def d_dt_conservative(η, u, v, g):
    """
    http://en.wikipedia.org/wiki/Shallow_water_equations#Conservative_form
    """

    dη_dt = -d_dx(η*u) -d_dy(η*v)
    du_dt = (dη_dt*u - d_dx(η*u**2 + 1./2*g*η**2) - d_dy(η*u*v)) / η
    dv_dt = (dη_dt*v - d_dx(η*u*v) - d_dy(η*v**2 + 1./2*g*η**2)) / η

    return dη_dt, du_dt, dv_dt


############

# an Impact is an event of the drop hitting the surface.
Impact = namedtuple('Impact', ['t', 'x', 'y'])  # Add other aspects like speed, force etc.


@jit
def surface_height_inner(t: float, r: float) -> float:
    """From paper. No idea what half it means!"""
    # This appears to be an analytic solution. No need for numerical??
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

    # Term 2 represents the amplitude of the wave.
    # term2 = integrate(F(u) * sin(Ω*u/2) * du)
    term2 = 1

    term3 = cos(Ω/2) * exp((Γ/ΓF - 1) * (τ/τD))  # * special.j0(kC*r)

    # Note: todo this is a start at the more "complete" version in the paper.
    # term1 = (4*sqrt(2*π))/(3) * (kC**2*kF*Ohe**(1/2))/(3*kF**2 + Bo)
    # term2 =
    # term3 = H(τ)/sqrt(τ) * exp((Γ/ΓF - 1) * (τ/τD)) * special.j0(kC*r)

    return term1 * term2 * term3


def surface_height(t: float, r: float) -> float:
    """Workaround for scipy.special.j0 not working in numba; @jit the rest,
    then multiply by the j0 bessel function separately."""
    kC = .888
    return surface_height_inner(t, r) * special.j0(kC*r)


# @jit
def surface_height_derivative(t, x, y, impact: Impact) -> Tuple[float, float]:
    """Create a linear approximation, for finding the slope at a point.
    x and y are points.  t is
    the time we're taking the derivative."""
    # todo perhaps you should take into account higher order effects.
    δ = 10e-5  # Should this be fixed?

    t_since_impact = t - impact.t  # We care about time since impact.

    x_dist, y_dist = x - impact.x, y - impact.y

    # Take a sample on each side of the location we're testing.

    # Calculate the radiuses.
    r_x_left = (y_dist ** 2 + (x_dist - δ) ** 2) ** .5
    r_x_right = (y_dist ** 2 + (x_dist + δ) ** 2) ** .5
    r_y_left = ((y_dist - δ) ** 2 + x_dist ** 2) ** .5
    r_y_right = ((y_dist + δ) ** 2 + x_dist ** 2) ** .5

    m = 10

    r_x_left /= m
    r_x_right /= m
    r_y_left /= m
    r_y_right /= m

    func = partial(surface_height, t_since_impact)

    h_x_left = func(r_x_left)
    h_x_right = func(r_x_right)
    h_y_left = func(r_y_left)
    h_y_right = func(r_y_right)

    # # Numba contingency; avoid partial. lambda ok or not?
    # h_x_left = surface_height(t_since_impact, r_x_left)
    # h_x_right = surface_height(t_since_impact, r_x_right)
    # h_y_left = surface_height(t_since_impact, r_y_left)
    # h_y_right = surface_height(t_since_impact, r_y_right)

    # todo check signs!
    return (h_x_right - h_x_left) / δ, (h_y_right - h_y_left) / δ


def wave_field(t: float=2, origin=(250, 250)) -> np.ndarray:
    """Calculate a wave's effecton a 2d field."""
    h = np.zeros([500, 500])

    x_origin, y_origin = origin
    pixel_dist = 10

    for i in range(500):
        for j in range(500):
            r = ((y_origin-i)**2 + (x_origin-j)**2) ** .5
            r /= pixel_dist
            # Assuming we can just add the heights.
            h[i, j] += surface_height(t, r)

    # # This style uses broadcasting. Stumbling block on surface height ufunc.
    # y, x = np.mgrid[:500, :500]
    # r = sqrt((x_origin - x)**2 + (y_origin - y)**2)
    # h = surface_height(r, t)  # does surfaceheight need to be a ufunc? yes.

    return h

impacts = []


def int_func(y: Iterable, t: np.ndarray):
    xs, ys, zs, xv, yv, zv, xa, ya, za, h = y

    # waves = wave_field(t)
    # _p means prime; ie derivative
    xs_p, ys_p, zs_p = xv*dt, yv*dt, zv*dt
    xv_p, yv_p, zv_p = xa*dt, ya*dt, za*dt
    xa_p, ya_p, za_p = 0, 0, 0

    # h = h * wave_field(t)
    h_p = 0

    surface_grad = np.gradient(h_p)

    # This doesn't belong here...
    if 1 == 1:
        impacts.append(Impact(t, xs, ys))

    dydt = xs_p, ys_p, zs_p, xv_p, yv_p, zv, xa_p, ya_p, za_p, h_p
    return dydt


def integrate_test():
    # y0 is Drop xs, ys, zs, xv, yv, zv, xa, ya, za, surface
    y0 = 150, 200, 10, 0, 0, 0, 0, 0, g, 69
    t = np.arange(10)
    results = integrate.odeint(int_func, y0, t)

    return results


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