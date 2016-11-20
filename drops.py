from collections import namedtuple
from functools import partial
from typing import Tuple, Iterable

# import PyDSTool
import brisk
import matplotlib.pyplot as plt
import numba
import numpy as np
import scikits.odes
import scikits.odes.sundials
from numpy import pi as π, sqrt, cos, sin, tan, arctan2, exp, log
from scikits.odes import dae
from scikits.odes.sundials import ida

jit = numba.jit(nopython=True)


τ = 2 * π

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

# todo meters vs mm for distance??
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
ω = τ * f  # = 2π*f Bath angular frequency.  250 - 1250 rad s^-1
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
Point = namedtuple('Point', ['x', 'y'])  # Add other aspects like speed, force etc.

impacts = []

class IntegrationEvent(Exception):
    """Pass the new y value as an argument."""
    pass


@jit
def surface_height(t: float, r: float, F: float) -> float:
    """From 'Drops Walking on a vibrating Bath'. Analytic solution for surface
    height, based on one impact. The results can be added to take into account multiple
    impacts."""
    #
    τ_ = ωD * t  # τ is dimensionless time, not 2*pi !!
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

    term1 = (4*sqrt(τ))/(3*sqrt(τ_)) * (kC**2*kF*Ohe**(1/2))/(3*kF**2 + Bo)

    # Todo we're using instantaneous impacts as a simplification for now.
    # Term 2 represents the amplitude of the wave.
    # term2 = integrate(F(u) * sin(Ω*u/2) * du)
    # Todo I'm not sure if this is the correct way to make the integration
    # todo of the impact instant, but let's try.
    term2 = F * sin(Ω/2)

    term3 = cos(Ω/2) * exp((Γ/ΓF - 1) * (τ_/τD)) * brisk.j0(kC*r)

    # Note: todo this is a start at the more "complete" version in the paper.
    # term1 = (4*sqrt(τ))/(3) * (kC**2*kF*Ohe**(1/2))/(3*kF**2 + Bo)
    # term2 =
    # term3 = H(τ)/sqrt(τ) * exp((Γ/ΓF - 1) * (τ/τD)) * special.j0(kC*r)

    return term1 * term2 * term3


def net_surface_height(t: float, impacts_: Iterable, point: Tuple[float, float], reflectivity=1) -> float:
    """Finds the height, taking into account multiple impacts."""
    # todo add a limiter so old impacts aren't included; they're insignificant,
    # todo and we need to computationally limit the num of impacts.
    # This could possibly be jitted by replacing the Impacts list/object with a 2d array.

    height_below_drop = 0

    for impact in impacts_:
        t_since_impact = t - impact.t  # We care about time since impact.
        # Exclude impacts that occur after, or simulataneously with the current time.

        if t_since_impact <= 0:
            continue

        # Difference between point we're examing, and impact
        Δx, Δy = point[0] - impact.x, point[1] - impact.y
        point_dist = (Δx**2 + Δy**2) ** 0.5

        # 'height' is the base, ie non-reflected-component height.
        height = surface_height(t_since_impact, point_dist, impact.F)

        # todo this doesn't even belong in this func; ie see wall reflection??
        if point_dist > 50:  # todo btw this also only works on a corral....
            height *= 1 - reflectivity

        height_below_drop += height
        if point_dist <= 50:
            # todo temp reflection calc!
            height_below_drop += wall_reflection(impact, t_since_impact, point, Δx, Δy, point_dist, reflectivity)

            pass
    return height_below_drop


@jit
def point_to_line(linept0: Point, linept1: Point, point: Point) -> float:
    """Calculate the distance between a line, defined by two points, and a point."""
    dx, dy = linept1.x - linept0.x, linept1.y - linept0.y
    num = abs(dy * point.x - dx * point.y + linept1.x * linept0.y - linept1.y * linept0.x)
    denom = sqrt(dy**2 + dx**2)
    return num/denom


# todo separate file for surface reflection calcs??

@jit
def simple_collision(impact_prime: Point, θiw: float):
    """Calculate where on the wall of a circular corral a line coming from an impact
    will hit, given an initial angle from the impact."""
    # impact_prime is in a cartesian coordinate system centered on the corral's center.
    # slope and y-intercept of the line connecting the impact to the wall
    m_ = tan(θiw)
    y_int = impact_prime.y - m_ * impact_prime.x  # Could also use point_prime here; same result.

    # print(m_, y_int)
    # D is corral radius, as used in the Molacek and Bush paper.

    # solve the system of equations:
    # y = m_ * x + b, x**2 + y**2 = D**2
    # Result: x**2 *(1+m_**2) + x*(2*b*m_) - (D**2 + b**2) = 0

    # Solve with quadratic formula:
    # Note: b here is for quadratic formula; use y_int above ie y = mx + b

    a, b, c = 1 + m_**2, 2*y_int*m_, - D**2 + y_int**2

    # if the impact's to the left of the point, use the positive root; else negative
    # This is to make sure we're always using the normal vector inside the circle.
    # todo if they're equal, you get a zero div error; fix later.
    root_sign = -1 if τ/4 <= θiw < 3*τ/4 else 1

    x_wall = (-b + root_sign * sqrt(b**2 - 4 * a * c)) / (2 * a)
    y_wall = m_ * x_wall + y_int
    return Point(x_wall, y_wall)



@jit
def cast_ray(impact: Impact, sample_pt: Point, center: Point, θiw: float):
    """Calculate the distance between a point, and a ray cast from the impact,
    bounced off one wall, in a circular corral."""
    # θiw is the angle of the impact to the wall we're trying.'
    # Circular corral only for now.
    # Create a grid system centered on the corral center; find impact and the point
    # we're examining in that grid.
    # primes are the coordinates in the coral's coord system.
    impact_prime = Point(impact.x - center.x, impact.y - center.y)

    collision_pt = simple_collision(impact_prime, θiw)

    θw = arctan2(collision_pt.y, collision_pt.x) % τ  # normal angle to the wall
    θw = (θw + τ/2) % τ
    # print("thetaw:", θw)

    unit_normal = np.array([cos(θw), sin(θw)])
    v = np.array([cos(θiw), sin(θiw)])
    reflection = v - 2*(v @ unit_normal) * unit_normal

    # print("unit normal, v", unit_normal, v)
    # print(collision_pt, θw)
    # print("Reflection:", reflection)

    reflection_pt = Point(collision_pt.x + reflection[0], collision_pt.y + reflection[1])
    return point_to_line(collision_pt, reflection_pt, sample_pt)


# todo do you need to find both solutions, and add both? Likely.
# todo you can probably think of a smarter way with interpolation...
@jit
def find_wall_collision(impact: Impact, sample_pt: Point, precision: int=τ/1000):
    """Calculate where a wave would hit the nearest wall. Cast rays, and guess"""
    # precision is in radians. Consider setting it in terms of distance to sample.
    # Find the number of iterations required to meet the specified precision.
    # You could also use a while loop instead of calcing this manually.
    n = int(log(τ / precision) / log(2))

    # Circular corral only for now.
    # Create a grid system centered on the corral center; find impact and the point
    # we're examining in that grid.
    coral_center = Point(0, 0)  # todo this only works with a circle!
    θiw_0 = 0
    θiw_1 = τ / 2
    θiw_closer = θiw_0

    θ_guesses = τ / 2
    for i in range(n):
        θ_guesses /= 2  # Angular distance between guesses; shrink this  each iteration.
        dist_ray_to_sample0 = cast_ray(impact, sample_pt, coral_center, θiw_0)
        dist_ray_to_sample1 = cast_ray(impact, sample_pt, coral_center, θiw_1)

        θiw_closer = θiw_0 if dist_ray_to_sample0 < dist_ray_to_sample1 else θiw_1

        # Make the next guesses bracket the closest of the current  ones.
        θiw_0 = θiw_closer - θ_guesses
        θiw_1 = θiw_closer + θ_guesses

    print(dist_ray_to_sample0)
    impact_prime = Point(impact.x - coral_center.x, impact.y - coral_center.y)
    return simple_collision(impact_prime, θiw_closer)


# def wall_reflection(impact, t_since_impact, point: Tuple[float, float], Δx, Δy, point_dist, reflectivity, n_reflections=1):
#     """Experimental! For reflecting  the standing faraday waves off walls."""
#     # n_reflections controls how many walls to bounce off of.
#
#
#     # Angle between the impact origin, and the location we're examining.
#     θ_point_impact = arctan2(Δy, Δx)
#
#     # Angle between coral ceenter and the location we're examining.
#     # θ_point_coral_ctr = arctan2(point[1] - coral_center[1], point[0] - coral_center[0])
#
#     collision_point, collision_angle = find_wall_collision(impact, point)
#
#     # Todo here's where my geometrical knowledge shall fail.
#
#
#
#     θ_wall = (θ_point_impact + τ/4) % τ # todo this only works with a circle!
#     # todo perhaps this works for circle, but you could generalize this now
#     # todo including wall angle.
#
#
#     # Don't calc reflections for points beyond this wall going in this direction.
#     if point_dist > wall_dist:  # todo again, this only works for a circle.
#         return 0
#
#
#
#     # Find the height at this distance, reflect it back over by adding.
#     reflection_dist = wall_dist + (wall_dist - point_dist)
#     height_reflection = surface_height(t_since_impact, reflection_dist, impact.F)
#
#     return height_reflection * reflectivity


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
def surface_oscilation(t: float) -> float:
    """Return the surface's height and velocity at time t due to its oscillation;
    a simple harmonic oscillator."""
    # Oscillation starts the cycle at negative amplitude.
    # Use global constants for frequency and bath acceleration.

    ω = τ * f
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


def drag(v: float):
    CD = .58  # really this varies! Usually between 0.2 and 0.5
    A = π * R0 ** 2  # Cross-sectional area of undeformed drop.
    return (CD * ρa * A * v**2) / 2


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
    result[0] = y

    for i in range(len(t) - 1):
        t_ = t[i]  # current  time
        h = t[i+1] - t[i]  # time step
        try:
            y = rk4(f, y, t_, h, args)
        except IntegrationEvent as event:
            # Assign y to be the values you're restarting the integrator from,
            # passed from the right-hand side in the exception.
            y = event.args[0]
            # todo handle restarting after a bounce here, or just return what
            # todo we have, and continue in a loop outside?
            # break

        result[i+1] = y
    return result


def ode_rhs(y: np.ndarray, t: float, drop_len: int) -> Tuple:
    """Right hand integration function."""
    drops = y.reshape(-1, drop_len)
    dy_dx = []

    for drop in drops:
        sx, sy, sz, vx, vy, vz = drop
        ax, ay, az = 0, 0, g

        # The limit on sz is to prevent calling net_surface_height when the ball's
        # no where near bouncing; the vz check is a fudge for the ball being detected
        # a bit below the surface; dont' catch it on the way up.
        if sz <= 10 and vz < 0:  # todo lower sz limit.
            height_below_drop = net_surface_height(t, impacts, (sx, sy))

            if sz <= height_below_drop:  # A bounce is detected.
                print('bounce', sz, height_below_drop)
                grad_x, grad_y = surface_height_gradient(t, impacts, sx, sy)
                # This bounce velocity change overrides the default, of last step's accel.
                vx_bounce, vy_bounce, vz_bounce = bounce_v(grad_x, grad_y, vx, vy, vz)

                # Add this new impact for future calculations.
                F = .01  # todo i don't know what to do here.
                impacts.append(Impact(t, sx, sy, F))

                y = sx, sy, sz, vx_bounce, vy_bounce, vz_bounce
                raise IntegrationEvent(y)

        # todo fudge factors giving an approx of air resistance.
        # ax_p = drag(vx) / m
        # ay_p = drag(vy) / m
        # az_p = drag(vz) / m

        # Append to a 1d array (or list) which we'll output.
        dy_dx.extend([vx, vy, vz, ax, ay, az])

    return dy_dx


def ode_rhs_simple(y: np.ndarray, t: np.ndarray) -> Tuple:
    """Right hand integration function, simplified for one drop; no bounce"""
    sx, sy, sz, vx, vy, vz = y
    ax, ay, az = 0, 0, g

    return vx, vy, vz, ax, ay, az


def run() -> np.ndarray:
    # y0 is Drop sx, sy, sz, vx, vy, vz
    drops_initial = np.array([
        [0, 0, 10, 30, -100, 0],
        # [100, 105, 10, 0, 0, 0],
        # [100, 95, 10, 0, 0, 0],
        # [105, 100, 10, 0, 0, 0],
        # [105, 105, 10, 0, 0, 0],
        # [105, 95, 10, 0, 0, 0],
        # [95, 100, 10, 0, 0, 0],
        # [95, 105, 10, 0, 0, 0],
        # [95, 95, 10, 0, 0, 0],

    ])

    n_drops, drop_len = drops_initial.shape

    # Integrators take 1d arrays by convention.
    y0 = drops_initial.flatten()

    t = np.linspace(0, 20, 400)

    global impacts
    impacts = []
    # return integrate.odeint(ode_rhs, y0, t)
    return rk4_odeint(ode_rhs, y0, t, args=(drop_len,)), t, impacts


def ode_standalone(t: np.ndarray, bath_oscillation=False) -> Tuple:
    """Purpose-built, non-general RK4 integrator for modelling multiple drops,
    with events. Events with one drop, or multiple drops with no events work
    with more elegant solutions, like scipy's odeint or the rk4_odeint above."""

    impacts_ = []

    # initial drop conditions.
    drops = np.array([
        [0, 0, 10, 1, 1, 0],
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

    soln = np.empty([len(t), num_drops, drop_len])
    soln[0] = drops

    for i in range(len(t) - 1):
        t_ = t[i]  # current  time
        h = t[i + 1] - t[i]  # time step

        for j in range(num_drops):
            y_to_integrate = soln[i, j]  # todo ??
            y_drop = rk4(ode_rhs_simple, y_to_integrate, t_, h, args=())  # no bounce

            sx, sy, sz, vx, vy, vz = y_to_integrate

            if sz <= 10 and vz < 0:  # todo lower sz limit.

                height_below_drop = net_surface_height(t_, impacts_, sx, sy)

                if bath_oscillation:
                    # todo need a valid bath oscilation amplitide; it's given as
                    # todo an acceleration in the paper. Integrate twice?
                    height_below_drop += surface_oscilation(1, f, t_)

                if sz <= height_below_drop:  # A bounce is detected.
                    grad_x, grad_y = surface_height_gradient(t_, impacts_, sx, sy)
                    # This bounce velocity change overrides the default, of last step's accel.
                    vx_bounce, vy_bounce, vz_bounce = bounce_v(grad_x, grad_y,
                                                               vx, vy, vz)

                    # Add this new impact for future calculations.
                    F = .02  # todo i don't know what to do here.
                    impacts_.append(Impact(t_, sx, sy, F))

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

