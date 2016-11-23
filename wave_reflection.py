from functools import partial
from typing import Iterator

import numba
import numpy as np
from numpy import sqrt, sin, cos, tan, arctan2
import scipy.optimize

from drops import D, τ

jit = numba.jit(nopython=True)

# Use arrays instead of Point objects, to make working with numba easiser.
# todo sort out Points vs arrays; numba is sensitive! Looks like arrays result in
# todo MUCH faster performance compared to a tuple or namedtuple
# todo just using x and y as separate vars also offers big speed boosts, and the cost of readability.


@jit
def point_to_line(linept0: np.ndarray, linept1: np.ndarray, point: np.ndarray) -> float:
    """Calculate the distance between a line, defined by two points, and a point."""
    dx, dy = linept1[0] - linept0[0], linept1[1] - linept0[1]
    num = abs(dy * point[0] - dx * point[1] + linept1[0] * linept0[1] - linept1[1] * linept0[0])
    denom = sqrt(dy**2 + dx**2)
    return num / denom


@jit
def simple_collision(impact_prime: np.ndarray, θiw: float) -> np.ndarray:
    """Calculate where on the wall of a circular corral a line coming from an impact
    will hit, given an initial angle from the impact."""
    # impact_prime is in a cartesian coordinate system centered on the corral's center.
    # slope and y-intercept of the line connecting the impact to the wall
    m_ = tan(θiw)
    y_int = impact_prime[1] - m_ * impact_prime[0]  # Could also use point_prime here; same result.

    # # D is corral radius, as used in the Molacek and Bush paper.
    # # solve the system of equations:
    # # y = m_ * x + b, x**2 + y**2 = D**2
    # # Result: x**2 *(1+m_**2) + x*(2*b*m_) - (D**2 + b**2) = 0

    # # Solve with quadratic formula:
    # # Note: b here is for quadratic formula; use y_int above ie y = mx + b
    a, b, c = 1 + m_**2, 2*y_int*m_, - D**2 + y_int**2

    # If the impact's to the left of the point, use the positive root; else negative

    # todo if they're equal, you get a zero div error; fix later.
    root_sign = -1 if τ/4 <= θiw < 3*τ/4 else 1

    x_wall = (-b + root_sign * sqrt(b**2 - 4 * a * c)) / (2 * a)
    y_wall = m_ * x_wall + y_int

    # return (x_wall, y_wall)
    return np.array([x_wall, y_wall])


@jit
def cast_ray(impact: np.ndarray, sample_pt: np.ndarray, center: np.ndarray, θiw: float) -> float:
    """Calculate the distance between a point, and a ray cast from the impact,
    bounced off one wall, in a circular corral."""
    # θiw is the angle of the impact to the wall we're trying.'
    # Circular corral only for now.
    # Create a grid system centered on the corral center; find impact and the point
    # we're examining in that grid.

    # primes are the coordinates in the coral's coord system.
    impact_prime = np.array([impact[0] - center[0], impact[1] - center[1]])
    sample_prime = np.array([sample_pt[0] - center[0], sample_pt[1] - center[1]])
    collision_pt = simple_collision(impact_prime, θiw)

    # print(collision_pt, 'colpt')
    θw = arctan2(collision_pt[1], collision_pt[0]) % τ  # normal angle to the wall
    θw = (θw + τ/2) % τ

    unit_normal = np.array([cos(θw), sin(θw)])
    v = np.array([cos(θiw), sin(θiw)])
    reflection = v - 2*(v @ unit_normal) * unit_normal

    reflection_pt = np.array([collision_pt[0] + reflection[0], collision_pt[1] + reflection[1]])
    return point_to_line(collision_pt, reflection_pt, sample_prime)


@jit
def cast_ray_fsolvable(impact: np.ndarray, sample_pt: np.ndarray, center: np.ndarray,
                       θiw: np.ndarray) -> float:
    """Workaround; modified cast_ray, since fsolve passes arrays of size (1,),
    rather than floats. Can't do checks or argument-based-selection with numba."""
    return cast_ray(impact, sample_pt, center, θiw[0])


def find_wall_collision(impact: np.ndarray, sample_pt: np.ndarray, center: np.ndarray) -> Iterator[np.ndarray]:
    """Calculate where a wave would hit the nearest wall. Cast rays, and guess"""
    # Note: using  @jit on all functions this calls yields a dramatic performance increase,
    # but unable to @jit this function due to use of fsolve.
    # Circular corral only for now.
    # Create a grid system centered on the corral center; find impact and the point
    # we're examining in that grid.

    # Take this many sample points between 0 and τ for rough root estimates.
    ROUGH_SAMPLE_PTS = 1000
    SAMPLE_THRESHOLD = .4  # Check distances below this value for roots with fsolve.

    cast = partial(cast_ray, impact, sample_pt, center)
    cast_fsolvable = partial(cast_ray_fsolvable, impact, sample_pt, center)

    # The code below is very slow, but should give accurate resolves.
    # Using fsolve precludes numba; sometimes it throws in an arroy instead of
    # a value, so need to work around with isinstance; numba can't do that.
    # there has to be a better way.

    # Guess roots by finding where the low points are.
    θ = np.linspace(0, τ, ROUGH_SAMPLE_PTS)
    dist_rough = np.array(list(map(cast, θ)))
    low_dists = θ[dist_rough < SAMPLE_THRESHOLD]

    # Find precise roots; point fsolve to our guesses in low_dists for starting points.
    roots = np.array([scipy.optimize.fsolve(cast_fsolvable, d)[0] for d in low_dists])
    # fsolve will give slightly different answers for the same roots; round
    # so set eliminates them properly as duplicates.
    roots = set(round(r, 8) for r in roots)
    return map(partial(simple_collision, impact), roots)


def find_reflection_points(impact: np.ndarray, sample_pt: np.ndarray) -> np.ndarray:
    """Find which points outside the coral are reflected back to the sample point."""
    coral_center = np.array([0, 0])  # todo this only works with a circle!

    # height = 0 # The height to add based on reflected points outside the circular corral.
    wall_collision_pts = find_wall_collision(impact, sample_pt, coral_center)
    for wall_pt in wall_collision_pts:
        dx_wall_sample, dy_wall_sample = sample_pt[0] - wall_pt[0], sample_pt[1] - wall_pt[1]
        dx_impact_wall, dy_impact_wall = wall_pt[0] - impact[0], wall_pt[1] - impact[1]

        dist_wall_sample = sqrt(dx_wall_sample ** 2 + dy_wall_sample ** 2)
        dist_impact_wall = sqrt(dx_impact_wall ** 2 + dy_impact_wall ** 2)
        scale_factor = dist_wall_sample / dist_impact_wall

        yield np.array([dx_impact_wall, dy_impact_wall]) * scale_factor
