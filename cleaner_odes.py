# Holding code for cleaner ODEs than the one currently used; the one in drops.py
# is messy and specific, but the only way I've been able to make it work.


from scikits.odes import dae
from scikits.odes.sundials import ida
from scipy.optimize import fsolve

from scikits.odes import dae
from scikits.odes.sundials import ida
from scipy.optimize import fsolve

import PyDSTool



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

