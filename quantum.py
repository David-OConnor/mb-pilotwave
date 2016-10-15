from collections import namedtuple
from functools import partial
from typing import Tuple, Iterable

import brisk
import matplotlib.pyplot as plt
import numba
import numpy as np

from numpy import pi as π, e, sqrt, cos, sin, exp, arctan
import PyDSTool
from scipy import integrate, special
import scikits.odes
import scikits.odes.sundials
# import scikits.odes.sundials.ida
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


@jit
def net_surface_height(t: float, x: float, y: float, impacts_: Iterable) -> float:
    """Finds the height, taking into account multiple impacts."""
    # todo add a limiter so old impacts aren't included; they're insignificant,
    # todo and we need to computationally limit the num of impacts.
    height_below_drop = 0

    for impact_ in impacts_:
        t_since_impact = t - impact_.t  # We care about time since impact.
        r = ((impact_.x - x)**2 + (impact_.y - y)**2) ** 0.5
        height_below_drop += surface_height(t_since_impact, r, impact_.F)
    return height_below_drop


@jit
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
    print(unit_normal, v, "normal, v")
    print(reflection, "REF")
    return reflection

#
# @jit
# def _add_elwise(a: Iterable, b: Iterable):
#     result = []
#     for i in zip(a, b):
#         result.append(i[0] + i[1])
#     return result
#
#
# @jit
# def _div_elwise(items: Iterable, value: float):
#     # return map(partial(mul, value), items)
#     result = []
#     for i in items:
#         result.append(i / value)
#     return result
#
#
# @jit
# def _mult_elwise(items: Iterable, value: float):
#     # return map(partial(mul, value), items)
#     result = []
#     for i in items:
#         result.append(i * value)
#     return result
#
#
# @jit
# def rk4(f, y: Iterable, t: float, h: float):
#     """Basic mechanics of Runge-Kutta 4 ODE"""
#     # todo messy messy for supporting numba and its restrictions. TBH this
#     # todo function and its methods don't work unless f is numbaed too, so forget it?
#
#     k1 = _mult_elwise(f(y, t), h)
#     k2 = _mult_elwise(f(_add_elwise(y, _div_elwise(k1, 2)), t + h/2), h)
#     k3 = _mult_elwise(f(_add_elwise(y, _div_elwise(k2, 2)), t + h/2), h)
#     k4 = _mult_elwise(f(_add_elwise(y, k3), t + h), h)
#     part1 = _add_elwise(_add_elwise(_mult_elwise(_add_elwise(k2, k3), 2), k1), k4)
#
#     return _add_elwise(y, _div_elwise(part1, 6))


# @jit
def rk4(f, y: Iterable, t: float, h: float):
    """Basic mechanics of Runge-Kutta 4 ODE"""
    y = np.array(y)

    k1 = np.array(f(y, t)) * h
    k2 = np.array(f(y + k1/2, t + h/2)) * h
    k3 = np.array(f(y + k2/2, t + h/2)) * h
    k4 = np.array(f(y + k3, t + h)) * h
    return y + (k1 + 2*(k2 + k3) + k4) / 6


def rk4_ode(f, y0, t):
    """Interface for RK4 ODE, similar to scipy.integrate.odeint."""
    y0 = np.array(y0)
    result = np.empty([len(t), len(y0)])
    y = y0
    for i in range(len(t) - 1):
        result[i] = y

        t_ = t[i]
        h = t[i+1] - t[i]
        y = rk4(f, y, t_, h)

    result[-1] = y
    return result


def int_rhs(y: Iterable, t: float) -> Tuple:
    """Right hand integration function."""
    sx, sy, sz, vx, vy, vz, xa, ya, az = y

    # _p means prime; ie derivative

    # todo only invoke bounce-detection logic if drop's below a certain height,
    # todo for computational efficiency?
    height_below_drop = net_surface_height(t, sx, sy, impacts)
    #
    # if sz <= height_below_drop:
    #     print("Bounce:", sz, height_below_drop)
    #
    #     grad_x, grad_y = surface_height_gradient(t, sx, sy, impacts)
    #     vx_new, vy_new, vz_new = bounce_v(grad_x, grad_y, vx, vy, vz)
    #
    #     # This functions outputs are assumed by the change in one unit t;
    #     # We calculated actual new velocities;  convert to dy/dt format.
    #     vx_p, vy_p, vz_p = vx_new - vx, vy_new - vy, vz_new-vz
    #
    #     # Add this new impact for future calculations.
    #     F = .1  # todo i don't know what to do here.
    #     impacts.append(Impact(t, sx, sy, F))
    #
    # else:
    #     # pass
    #     # If no impact, calculate velocities as usual.

    vx_p, vy_p, vz_p = xa * dt, ya * dt, az * dt

    sz_p = vz * dt

    sx_p, sy_p = vx*dt, vy*dt
    ax_p, ay_p, az_p = 0, 0, 0

    dydt = sx_p, sy_p, sz_p, vx_p, vy_p, vz_p, ax_p, ay_p, az_p
    return dydt


def integrate():
    # y0 is Drop ss, sy, sz, vx, vy, vz, ax, ay, az
    y0 = 150, 200, 10, 0, 0, 0, 0, 0, g
    t = np.linspace(0, 20, 200)
    return rk4_ode(int_rhs, y0, t)
    # return integrate.odeint(int_rhs, y0, t)


def wave_field(t: float=2, origin=(250, 250)) -> np.ndarray:
    """Calculate a wave's effect on a 2d field."""
    h = np.zeros([500, 500])
    F = 1

    x_origin, y_origin = origin
    pixel_dist = 10

    for i in range(500):
        for j in range(500):
            r = ((y_origin-i)**2 + (x_origin-j)**2) ** .5
            r /= pixel_dist
            # Assuming we can just add the heights.

            h[i, j] += surface_height(t, r, F)

    # # This style uses broadcasting. Stumbling block on surface height ufunc.
    # y, x = np.mgrid[:500, :500]
    # r = sqrt((x_origin - x)**2 + (y_origin - y)**2)
    # h = surface_height(r, t, F)  # does surfaceheight need to be a ufunc? yes.

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


def dst_rhs():
    sx_p = 'vx'  # todo * dt?
    sy_p = 'vy'
    sz_p = 'vz'

    vx_p = 'ax'
    vy_p = 'ay'
    vz_p = 'az'

    ax_p = '0'
    ay_p = '0'
    az_p = '0'

    # height_below_drop = net_surface_height(t, sx, sy, impacts)

    dydt = sx_p, sy_p, sz_p, vx_p, vy_p, vz_p, ax_p, ay_p, az_p#, height_below_drop
    return dydt


def dst_integrate_test():
    # initial conditions.
    y0 = {'sx': 150, 'sy': 200, 'sz': 10,
          'vx': 0, 'vy': 0, 'vz': 0,
          'ax': 0, 'ay': 0, 'az': g,
          'height_below_drop': 999}

    # pardict = {'k': 0.1, 'm': 0.5}

    # sx_p, sy_p, sz_p, vx_p, vy_p, vz_p, ax_p, ay_p, az_p, height_below_drop = dst_rhs()
    sx_p, sy_p, sz_p, vx_p, vy_p, vz_p, ax_p, ay_p, az_p = dst_rhs()

    vardict = {'sx': sx_p, 'sy': sy_p, 'sz': sz_p,
               'vx': vx_p, 'vy': vy_p, 'vz': vz_p,
               'ax': ax_p, 'ay': ay_p, 'az': az_p,
               'height_below_drop': 'net_surface_height(t, sx, sy, impacts)'}

    event_bounce = PyDSTool.makeZeroCrossEvent('sz - height_below_drop', 0,
                                               {'name': 'event_bounce',
                                                'eventtol': 1e-6,
                                                'term': False,
                                                'active': True},
                                               varnames=['sz', 'height_below_drop'],
                                               parnames=[''],
                                               targetlang='python')
        #                                        extra_funcspec_args={'ignorespecial': ['height_below_drop'],
        # 'codeinsert_start': 'height_below_drop = net_surface_height(t, sx, sy, impacts)'})

    DSargs = PyDSTool.args()  # create an empty object instance of the args class, call it DSargs
    DSargs.name = 'drops'  # name our model
    DSargs.ics = y0  # assign the icdict to the ics attribute
    # DSargs.pars = pardict  # assign nthe pardict to the pars attribute
    DSargs.tdata = [0, 20]  # declare how long we expect to integrate for
    DSargs.varspecs = vardict  # assign the vardict dictionary to the 'varspecs' attribute of DSargs
    DSargs.auxvars = 'height_below_drop'

    # For getting net_surface_height() working?:
    # DSargs.vfcodeinsert_start = 'height_below_drop = ds.height_below_drop(N)'
    # DSargs.ignorespecial = ['height_below_drop']

    DS = PyDSTool.Generator.Vode_ODEsystem(DSargs)

    # DS.height_below_drop = net_surface_height

    # DS.set(pars={'k': 0.3},
    #        ics={'x': 0.4})

    traj = DS.compute('demo')
    pts = traj.sample()

    plt.plot(pts['t'], pts['sz'], label='sz')
    plt.legend()
    plt.xlabel('t')
    return traj


def skode_rhs(t, y):
    sx, sy, sz, vx, vy, vz, xa, ya, az = y

    # _p means prime; ie derivative

    # todo only invoke bounce-detection logic if drop's below a certain height,
    # todo for computational efficiency?
    # height_below_drop = net_surface_height(t, sx, sy, impacts)
    #
    # if sz <= height_below_drop:
    #     print("Bounce:", sz, height_below_drop)
    #
    #     grad_x, grad_y = surface_height_gradient(t, sx, sy, impacts)
    #     # vx_p, vy_p, vz_p = bounce_v(grad_x, grad_y, vx, vy, vz)
    #
    #     # Add this new impact for future calculations.
    #     F = .1  # todo i don't know what to do here.
    #     impacts.append(Impact(t, sx, sy, F))
    #
    #     sz_p = 0
    #
    # else:
    #     pass
    #     # If no impact, calculate velocities as usual.

    vx_p, vy_p, vz_p = xa * dt, ya * dt, az * dt
    sz_p = vz * dt

    sx_p, sy_p = vx*dt, vy*dt
    ax_p, ay_p, az_p = 0, 0, 0

    dydt = sx_p, sy_p, sz_p, vx_p, vy_p, vz_p, ax_p, ay_p, az_p
    return dydt


def skode_test():
    # y0 is Drop ss, sy, sz, vx, vy, vz, ax, ay, az
    y0 = 150, 200, 10, 0, 0, 0, 0, 0, g
    # t0 = 0
    t = np.linspace(0, 20, 200)

    def skode_rhs2(t, y, ydot):
        sx, sy, sz, vx, vy, vz, xa, ya, az = y

        vx_p, vy_p, vz_p = xa * dt, ya * dt, az * dt
        sz_p = vz * dt

        sx_p, sy_p = vx * dt, vy * dt
        ax_p, ay_p, az_p = 0, 0, 0

        ydot[:] = np.array([sx_p, sy_p, sz_p, vx_p, vy_p, vz_p, ax_p, ay_p, az_p])

    solution = scikits.odes.ode('cvode', skode_rhs2, old_api=False).solve(t, y0)
    return solution


def skode_test_event():
    # y0 is Drop ss, sy, sz, vx, vy, vz, ax, ay, az
    y0 = 150, 200, 10, 0, 0, 0, 0, 0, g
    # t0 = 0
    t = np.linspace(0, 20, 200)

    def skode_rhs2(t, y, ydot):
        sx, sy, sz, vx, vy, vz, xa, ya, az = y

        vx_p, vy_p, vz_p = xa * dt, ya * dt, az * dt
        sz_p = vz * dt

        sx_p, sy_p = vx * dt, vy * dt
        ax_p, ay_p, az_p = 0, 0, 0

        ydot[:] = np.array([sx_p, sy_p, sz_p, vx_p, vy_p, vz_p, ax_p, ay_p, az_p])

    solution = scikits.odes.ode('cvode', skode_rhs2, old_api=False).solve(t, y0)
    return solution


class Drop:
    defg = g
    y0 = 150, 200, 10, 0, 0, 0, 0, 0, g
    t = np.linspace(0, 20, 200)

    deftend = 300
    deftstep = 1e-2

    defsx0, defsy0, defsz0, defvx0, defvy0, defvz0, defax0, defay0, defaz0 = y0

    def __init__(self, data=None):
        self.sx = Drop.defsx0
        self.sy = Drop.defsy0
        self.sz = Drop.defsz0
        self.vx = Drop.defvx0
        self.vy = Drop.defvy0
        self.vz = Drop.defvz0
        self.ax = Drop.defax0
        self.ay = Drop.defay0
        self.az = Drop.defaz0

        self.res = None
        self.jac = None

        if data is not None:
            self.tend = data.deftend
            self.tstep = data.deftstep
            self.sx0 = data.sx0
            self.sy0 = data.sy0
            self.sz0 = data.sz0
            self.vx0 = data.vx0
            self.vy0 = data.vy0
            self.vz0 = data.vz0
            self.ax0 = data.ax0
            self.ay0 = data.ay0
            self.azx0 = data.az0

            self.g = data.g

        self.stop_t = np.arange(.0, self.tend, self.tstep)

        # the index1 problem with jacobian :
        self.neq = 10
        # initial conditions
        lambdaval = 0.0
        self.z0 = np.array([self.x0, self.y0, self.x1, self.y1, 0., 0., 0.,
                         0., lambdaval, lambdaval])
        self.zprime0 = np.array([0., 0., 0., 0., -lambdaval * self.x0,
                              -lambdaval * self.y0 - self.g,
                              -lambdaval * self.x1,
                              -lambdaval * self.y1 - self.g, 0., 0.],
                             float)
        self.algvar_idx = [8, 9]
        self.algvar = np.array([1, 1, 1, 1, 1, 1, 1, 1, -1, -1])
        self.exclalg_err = False

    def set_res(self, resfunction):
        """Function to set the resisual function as required by IDA"""
        self.res = resfunction

    def set_jac(self, jacfunction):
        """Function to set the resisual function as required by IDA"""
        self.jac = jacfunction


class resindex1(scikits.odes.sundials.ida.IDA_RhsFunction):
    """ Residual function class as needed by the IDA DAE solver"""

    def set_dblpend(self, dblpend):
        """ Set the double pendulum problem to solve to have access to
            the data """
        self.dblpend = dblpend

    def evaluate(self, tres, yy, yp, result, userdata):
        m1 = self.dblpend.m1
        m2 = self.dblpend.m2
        g = self.dblpend.g

        result[0]= m1*yp[4]        - yy[9]*(yy[0] - yy[2])  - yy[0]*yy[8]
        result[1]= m1*yp[5] + g*m1 - yy[9]*(yy[1] - yy[3])  - yy[1]*yy[8]
        result[2]= m2*yp[6]        + yy[9]*(yy[0] - yy[2])
        result[3]= m2*yp[7] + g*m2 + yy[9]*(yy[1] - yy[3])
        result[4]= yp[0] - yy[4]
        result[5]= yp[1] - yy[5]
        result[6]= yp[2] - yy[6]
        result[7]= yp[3] - yy[7]
        result[8] = yy[4]**2 + yy[5]**2 + yy[8]/m1*(yy[0]**2 + yy[1]**2) \
                    - g * yy[1] + yy[9]/m1 *(yy[0]*(yy[0]-yy[2]) +
                                            yy[1]*(yy[1]-yy[3]) )
        result[9] = (yy[4]-yy[6])**2 + (yy[5]-yy[7])**2 \
                  + yy[9]*(1./m1+1./m2)*((yy[0]-yy[2])**2 + (yy[1]-yy[3])**2)\
                  + yy[8]/m1 *(yy[0]*(yy[0]-yy[2]) + yy[1]*(yy[1]-yy[3]) )
        return 0

class jacindex1(scikits.odes.sundials.ida.IDA_JacRhsFunction):

    def set_dblpend(self, dblpend):
        """ Set the double pendulum problem to solve to have access to
            the data """
        self.dblpend = dblpend

    def evaluate(self, tres, yy, yp, cj, jac):

        m1 = self.dblpend.m1
        m2 = self.dblpend.m2
        g = self.dblpend.g
        jac[:,:] = 0.
        jac[0][0] = - yy[9]   - yy[8]
        jac[0][2] =  yy[9]
        jac[0][4] = cj * m1
        jac[0][8] = - yy[0]
        jac[0][9] = - (yy[0] - yy[2])
        jac[1][1] = - yy[9] - yy[8]
        jac[1][3] = yy[9]
        jac[1][5] = cj * m1
        jac[1][8] = - yy[1]
        jac[1][9] = - (yy[1] - yy[3])
        jac[2][0] = yy[9]
        jac[2][2] = -yy[9]
        jac[2][6] = cj * m2
        jac[2][9] = (yy[0] - yy[2])
        jac[3][1] = yy[9]
        jac[3][3] = -yy[9]
        jac[3][7] = cj * m2
        jac[3][9] = (yy[1] - yy[3])
        jac[4][0] = cj
        jac[4][4] = -1
        jac[5][1] = cj
        jac[5][5] = -1
        jac[6][2] = cj
        jac[6][6] = -1
        jac[7][3] = cj
        jac[7][7] = -1
        jac[8][0] = (yy[8]+yy[9])/m1*2*yy[0] - yy[9]/m1 * yy[2]
        jac[8][1] = (yy[8]+yy[9])/m1*2*yy[1] - yy[9]/m1 * yy[3] - g
        jac[8][2] = - yy[9]/m1 * yy[0]
        jac[8][3] = - yy[9]/m1 * yy[1]
        jac[8][4] = 2*yy[4]
        jac[8][5] = 2*yy[5]
        jac[8][8] = 1./m1*(yy[0]**2 + yy[1]**2)
        jac[8][9] = 1./m1 *(yy[0]*(yy[0]-yy[2]) + yy[1]*(yy[1]-yy[3]) )
        jac[9][0] = yy[9]*(1./m1+1./m2)*2*(yy[0]-yy[2]) + \
                    yy[8]/m1 *(2*yy[0] - yy[2])
        jac[9][1] = yy[9]*(1./m1+1./m2)*2*(yy[1]-yy[3]) + \
                    yy[8]/m1 *(2*yy[1] - yy[3])
        jac[9][2] = - yy[9]*(1./m1+1./m2)*2*(yy[0]-yy[2]) - \
                    yy[8]/m1 * yy[0]
        jac[9][3] = - yy[9]*(1./m1+1./m2)*2*(yy[1]-yy[3])
        jac[9][4] = 2*(yy[4]-yy[6])
        jac[9][5] = 2*(yy[5]-yy[7])
        jac[9][6] = -2*(yy[4]-yy[6])
        jac[9][7] = -2*(yy[5]-yy[7])
        jac[9][8] = 1./m1 *(yy[0]*(yy[0]-yy[2]) + yy[1]*(yy[1]-yy[3]) )
        jac[9][9] = (1./m1+1./m2)*((yy[0]-yy[2])**2 + (yy[1]-yy[3])**2)
        return 0


#  a root function has a specific signature. Result will be of size nr_rootfns, and must be filled with the result of the
#  function that is observed to determine if a root is present.
def crosses_Y(t, yy, yp, result, user_data):
    result[0] = yy[2]


def run():
    problem = Drop()
    res = resindex1()
    jac = jacindex1()
    res.set_dblpend(problem)
    jac.set_dblpend(problem)
    solver = dae('ida', res,
                 compute_initcond='yp0',
                 first_step_size=1e-18,
                 atol=1e-10,
                 rtol=1e-8,
                 max_steps=5000,
                 jacfn=jac,
                 algebraic_vars_idx=problem.algvar_idx,
                 exclude_algvar_from_error=problem.exclalg_err,
                 rootfn=crosses_Y, nr_rootfns=1,
                 old_api=False)

    # storage of solution
    x1t = np.empty(len(problem.stop_t), float)
    y1t = np.empty(len(problem.stop_t), float)
    x2t = np.empty(len(problem.stop_t), float)
    y2t = np.empty(len(problem.stop_t), float)
    xp1t = np.empty(len(problem.stop_t), float)
    yp1t = np.empty(len(problem.stop_t), float)
    xp2t = np.empty(len(problem.stop_t), float)
    yp2t = np.empty(len(problem.stop_t), float)

    sol = solver.init_step(0., problem.z0, problem.zprime0)
    if sol.errors.t:
        print('Error in determination init condition')
        print(sol.message)
    else:
        ind = 0
        x1t[ind] = sol.values.y[0]
        y1t[ind] = sol.values.y[1]
        x2t[ind] = sol.values.y[2]
        y2t[ind] = sol.values.y[3]
        xp1t[ind] = sol.values.ydot[0]
        yp1t[ind] = sol.values.ydot[1]
        xp2t[ind] = sol.values.ydot[2]
        yp2t[ind] = sol.values.ydot[3]

    lastind = len(problem.stop_t)
    for index, time in enumerate(problem.stop_t[1:]):
        # print 'at time', time
        sol = solver.step(time)
        if sol.errors.t:
            lastind = index + 1
            print('Error in solver, breaking solution at time %g' % time)
            print(sol.message)
            break
        ind = index + 1
        x1t[ind] = sol.values.y[0]
        y1t[ind] = sol.values.y[1]
        x2t[ind] = sol.values.y[2]
        y2t[ind] = sol.values.y[3]
        xp1t[ind] = sol.values.ydot[0]
        yp1t[ind] = sol.values.ydot[1]
        xp2t[ind] = sol.values.ydot[2]
        yp2t[ind] = sol.values.ydot[3]

    energy = problem.m1 * problem.g * y1t + \
             problem.m2 * problem.g * y2t + \
             .5 * (problem.m1 * (xp1t ** 2 + yp1t ** 2)
                   + problem.m2 * (xp2t ** 2 + yp2t ** 2))
