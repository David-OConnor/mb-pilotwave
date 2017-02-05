import matplotlib.pyplot as plt

from scipy import integrate

from constants import *
from wave_reflection import find_reflection_points
import vertical, waves


jit = numba.jit(nopython=True)

# Assume we're not modeling the up and down motions; each simulation tick
# represents the particle impacting the grid

# todo improved bounce/contact mechanics; tranfer energy to the drop from oscillation, and do more
# todo than a simple reflection.  Find the force of each impact; put into wave height eq.


# MBI / MBII = Malacek and Bush companion papers, 2013
# JFM = A trajectory equation for walking droplets: hydrodynamic pilot-wave theory
# by Anand U. Oza, Rodolfo R. Rosales and John W. M. Bush


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
    return g + γ * sin(2*π * f * t)


def effective_grav2(τ:  float) -> float:
    """Effective gravity in bath frame of reference. From MBI; not sure where
    to use it atm."""
    # Shown as B0*(τ) in MBI.
    return 1 + Γ * sin(Ω * τ)


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
def surface_oscilation(t: float) -> Tuple[float, float, float]:
    """Return the surface's height and velocity at time t due to its oscillation;
    a simple harmonic oscillator."""
    # Oscillation starts the cycle at negative amplitude.
    # Use global constants for frequency and bath acceleration.

    bath_accel = γ * cos(ω * t)  # integrate this to get velocity and position.
    bath_vel = γ * ω**-1 * sin(ω * t)  # + C
    bath_height = -γ * ω**(-2) * cos(ω * t)  # + C*t

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


def rhs_airborne(y: np.ndarray, t: np.ndarray) -> Tuple:
    """Right hand integration function for drop's airborne motion, simplified for one drop;
    no bounce"""
    sx, sy, sz, vx, vy, vz = y

    # # Calculate drag force for accelerations. The z component also includes gravity.
    ax, ay = drag(vx), drag(vy)
    az = drag(vz) - g
    # az = drag(vz) - effective_grav(t) # you could do this too?

    return vx, vy, vz, ax, ay, az


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


def ode_standalone(t: np.ndarray, corral=False) -> Tuple[np.ndarray, List[Impact]]:
    """Purpose-built, non-general RK4 integrator for modelling multiple drops,
    with events. Events with one drop, or multiple drops with no events work
    with more elegant solutions, like scipy's odeint or the rk4_odeint above."""
    impacts_ = []

    # Vertical contact integration precision
    VERTICAL_PRECISON = int(1e5)

    # todo model two separate conditions: drop in flight, and drop in contact
    # todo with surface.

    # initial drop conditions: sx, sy, sz, vx, vy, vz
    # Per MBI, initial conditions play little role in subsequent dynamics.
    # Set initial height low for now; odd things happen with vert dynamics if the
    # impact velocity is too high.
    drops = np.array([
        [0, 0, .001, 0, 0, 0],
        # [100, 110, 10, 0, 0, 0],
        # [100, 95, 10, 0, 0, 0],
        # [105, 100, 10, 0, 0, 0],
        # [105, 105, 10, 0, 0, 0],
        # [105, 95, 10, 0, 0, 0],
        # [95, 100, 10, 0, 0, 0],
        # [95, 105, 10, 0, 0, 0],
        # [95, 95, 10, 0, 0, 0],

    ])
    # in_contact values correspond to the different drops. False means the drop's
    # airborne; True means it's in contact with the bath.
    in_contact = np.zeros(drops.shape[0], dtype=bool)

    # contact_motion holds the precalculated values from the vertical-motion contact
    # integrator. Axis 0: drops. Axis 1: time, Axis 2: (all dimensioned), vertical position, vert velocity
    contact_motion = np.zeros([drops.shape[0], VERTICAL_PRECISON, 2])

    # Drop exit conditions from contact: Axis 0: drops. Axis 1: (all dimensioned) time, height, velocity.
    # exit_conditions = np.zeros([drops.shape[0], 3])
    # exit_time = np.zeros(drops.shape[0])

    # contact_t is used as a reference to map values from the vert motion (contact) integrator
    # with the main one.
    # contact_t Axis 0: Drops. Axis 1: dimensioned time.
    contact_t = np.zeros([drops.shape[0], VERTICAL_PRECISON])

    # Border format is (x1, y1, x2, y2), ie a line connecting two points.
    # borders = [(0, 200, 500, 200)]

    num_drops, drop_len = drops.shape
    # The solution will have three axis: time, which drop, and drop conditions
    # (conditions are position, velocity etc etc)
    soln = np.empty([len(t), num_drops, drop_len])
    soln[0] = drops  # Set initial condition.

    for i in range(len(t) - 1):
        t_ = t[i]  # current  time
        h = t[i + 1] - t[i]  # time step

        for j in range(num_drops):
            drop_conditions = soln[i, j]
            sx, sy, sz, vx, vy, vz = drop_conditions

            # Take a value from the vertical motion in-contact integrator;
            # we've already pre-generated it.
            if in_contact[j]:
                # Find the row in contact motion that's closest to the current time.
                # todo instead of finding just the closest, interpolate for more precision.
                contact_mot_ix = np.argmin(np.abs(contact_t - t_))

                # todo sz_contact should be zero, until we include bath motion.
                sz_contact, vz_contact = contact_motion[j, contact_mot_ix]
                bath_z, bath_v = surface_oscilation(t_)[:2]

                # todo take out bath oscillation while troubleshooting.
                # Add in bath height.  The contact integrator operates relative to the
                # bath surface.  Also add in drop radius, since this integrator tracks
                # non-dim Z = 0, which is really surface + R0
                # todo examien this!
                sz_contact_bathed = sz_contact + bath_z + R_0

                # todo add back in horizontal dynamics
                sx, sy, sz, vx, vy, vz = 0, 0, sz_contact_bathed, 0, 0, vz_contact
                soln[i+1, j] = sx, sy, sz, vx, vy, vz

                # Check for an exit into the air.
                if sz_contact >= 0:  # Don't use the one that includes bath motion here.
                    print(f"Exit   t  :{round(t_, 5)} vz: {round(vz, 3)}")

                    # Leave in_contact, contact_motion, and exit_time blank while the drop's airborne.
                    in_contact[j], contact_motion[j], contact_t[j] = False, 0, 0

            # If not in contact, either use an airborne-kinematics integrator, or start
            # an in-contact integrator.
            else:
                # Limit checks for contact when the drop's high or going up, for performance reasons.
                if vz < 0:  # todo could add an az check too, ie make sure it's above a certain height.
                    # There might be an impact; calculate bath height and check.
                    sample_pt = np.array([sx, sy])
                    # The surface height, compared to a reference avg of 0.
                    surface_h_below_drop = waves.net_surface_height(t_, impacts_, sample_pt, corral=corral)

                    # Take into account the surface oscillation; it moves the whole
                    # surface uniformly.  Also,
                    surface_h_below_drop += surface_oscilation(t_)[0]

                else:
                    surface_h_below_drop = sz - 100  # This means trigger the airborne integrator.

                    # todo remove oscillation while in the bath frame. Fix this once done troubleshooting.

                # Check for an impact.
                # reference the bottom of the drop
                # rather than the top by shifting down half a radius.
                # surface_h_below_drop -= (surface_oscilation(t_)[0] + R_0)
                if sz <= surface_h_below_drop + R_0:
                    print(f"Impact   t:  {round(t_, 5)} vz: {round(vz, 3)}, sz: {round(sz, 3)}")
                    # We've found an impact.
                    in_contact[j] = True
                    # Model contact period with vertical functions from MBI
                    # Hand off vertical mechanics to a separate integrator.

                    # Dimensionless contact time should be about 5; integrate longer
                    # to be conservative.
                    τ_start = ω_D * t_
                    τ_end = τ_start + 50  # todo tweak this val
                    contact_τ = np.linspace(τ_start, τ_end, VERTICAL_PRECISON)
                    contact_t[j] = contact_τ / ω_D

                    # todo instead of just vertical speed, perhaps use total?
                    # contact_mot stays dimensionless, with contact_motion[j] is dimensioned.

                    # Integrator handoff notes: We switch integrators when the bottom of the drop
                    # touches the oscillating bath. This means bath height + drop radius == drop center.
                    # s positions always refers to drop center. The integrator starts at Z=0, which
                    # is the same, it the drop center is one radius above the bath.

                    contact_mot = vertical.log_spring(contact_τ, vz)

                    contact_motion[j, :] = contact_mot
                    contact_motion[j, :, 0] *= R_0  # Re-dimensionalize sz.


                    # todo not sure if we should use the airborne integrator, or contact
                    # todo one for this tick. Let's try airborne. Seems to work better,
                    # todo avoiding an odd spike
                    soln[i+1, j] = rk4(rhs_airborne, drop_conditions, t_, h, args=())  # no bounce
                    # Commented-out contact-version below.
                    # soln[i+1, j] = 0, 0, contact_motion[j, 0, 0], 0, 0, contact_motion[j, 0, 1]


                    # Add an impact for processing the wave field.
                    v = sqrt(vx ** 2 + vy ** 2 + vz ** 2)  # todo net v, or vz only??
                    impacts_.append(Impact(t_, sx, sy, v))


                    # grad_x, grad_y = surface_height_gradient(t_, impacts_, sx, sy)
                    # # This bounce velocity change overrides the default, of last step's accel.
                    #
                    # vx_bounce, vy_bounce, vz_bounce = bounce_v(grad_x, grad_y,
                    #                                            vx, vy, vz)
                    #
                    # # todo here, or in bounce_v, goes the force imparted on the drop:
                    # # todo −mg∇h(x_p, t) horizontal force??


                else:
                    # The drop's airborne and continues to be so; model using a simple airborne
                    # kinematics integrator.

                    # Integrate the drop's motion through the air.
                    soln[i+1, j] = rk4(rhs_airborne, drop_conditions, t_, h, args=())  # no bounce

    return soln, impacts_


def plot_trajectories(soln: np.ndarray):
    n_drops = soln.shape[1]

    for drop_i in range(n_drops):
        plt.plot(soln[:, drop_i, 0], soln[:, drop_i, 1])

    plt.show()


def quickplot(t: np.ndarray):
    soln, impacts = ode_standalone(t)
    plot_trajectories(soln)
