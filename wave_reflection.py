# This file contains functions used to calculate wave reflections off a wall.

from functools import partial

from scipy import optimize

from constants import *


# D is duplicated from constants; re-define here to avoid a circular import.
D = 76  # Cylindrical bath container diameter, mm  # todo should be x 10**-3 I think.


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
def simple_collision(impact: np.ndarray, center: np.ndarray, θiw: float) -> np.ndarray:
    """Calculate where on the wall of a circular corral a line coming from an impact
    Attempting with vectors, since my previous mx+b and circle equation attempt
    created singularities. Oddly enough, this is almost 2x slower, but worth it
    to avoid singularities.
    From http://stackoverflow.com/questions/1073336/circle-line-segment-collision-detection-algorithm"""
    # L is the end point of a unit vector starting at impact, and traveling along θiw
    L = impact + np.array([cos(θiw), sin(θiw)])

    d = L - impact
    f = impact - center
    r = D/2

    # Solve the quadratic formula
    a, b, c = d @ d, 2 * (f @ d), f @ f - r**2
    root_sign = 1  # It looks like we always use the positive root.
    t = (-b + root_sign * sqrt(b ** 2 - 4 * a * c)) / (2 * a)

    Px = impact[0] + t * d[0]
    Py = impact[1] + t * d[1]
    return np.array([Px, Py])


@jit
def cast_ray(impact: np.ndarray, sample_pt: np.ndarray, center: np.ndarray, θiw: float) -> float:
    """Calculate the distance between a point, and a ray cast from the impact,
    bounced off one wall, in a circular corral."""
    # θiw is the angle of the impact to the wall we're trying.'
    # Circular corral only for now.
    # Create a grid system centered on the corral center; find impact and the point
    # we're examining in that grid.

    # primes are the coordinates in the coral's coord system.
    sample_prime = np.array([sample_pt[0] - center[0], sample_pt[1] - center[1]])
    collision_pt = simple_collision(impact, center, θiw)

    # print(collision_pt, 'colpt')
    θw = arctan2(collision_pt[1], collision_pt[0]) % (2*π)  # normal angle to the wall
    θw = (θw + π) % (2*π)

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


@jit
def find_low_zones(low_dist, low_θ) -> List[Tuple[float, float]]:
    dist_diff = np.diff(low_dist)
    zones = []  # θ start and end poitns, where its corresponding dist is below the thresh
    in_zone = False

    if dist_diff[0] < 0:  # We're starting in zone, despite not getting the normal cue.
        zone_start = low_θ[0]
        in_zone = True

    # Don't check first value since we already checked; don't check last since
    # we can't check its diff.
    for i, dist in enumerate(low_dist[1:-1]):
        if dist_diff[i] <= 0 and not in_zone:  # Starting a zone
            in_zone = True
            zone_start = low_θ[i]

        elif dist_diff[i] <= 0 and in_zone:  # Already in a zone; keep going.
            pass

        elif dist_diff[i] > 0 and not in_zone:  # Not in a zone; keep going
            pass

        else:  # diff is > 0, and in zone. We're on our way out of a zone.
            in_zone = False
            # Plus one to make sure you actually capture the root; not stop right before.
            zone_end = low_θ[i+2]
            zones.append((zone_start, zone_end))

    return zones


# @jit
def find_wall_collisions(impact: np.ndarray, sample_pt: np.ndarray, center: np.ndarray) -> \
        Iterator[np.ndarray]:
    """Calculate where a wave would hit the nearest wall. Cast rays, and guess"""
    # Note: using  @jit on all functions this calls yields a dramatic performance increase,
    # but unable to @jit this function due to use of fsolve.
    # Circular corral only for now.
    # Create a grid system centered on the corral center; find impact and the point
    # we're examining in that grid.
    # Take this many sample points between 0 and 2*pi for rough root estimates.
    ROUGH_SAMPLE_PTS = 100  # Higher is more accurate, but slower
    # Check distances below this value for roots with fsolve.
    THRESH_SCALER = .4  # scaler; scale by this amount each time
    # todo numbafy.

    cast = partial(cast_ray, impact, sample_pt, center)
    cast_fsolvable = partial(cast_ray_fsolvable, impact, sample_pt, center)

    zones = [(0, 2*π)]  # Initial zone; all angles.

    # Fsolve will smooth over the result, so n need not be high.
    n = 3  # Number of iterations to go through, each narrowing down the possible roots.
    thresh = D * THRESH_SCALER
    for i in range(n):
        sub_zones = []
        for zone in zones:
            θ = np.linspace(zone[0], zone[1], ROUGH_SAMPLE_PTS)

            # dist_rough = np.array([cast_ray(impact, sample_pt, θ_) for θ_ in θ])

            dist_rough = np.array(list(map(cast, θ)))
            # plt.plot(θ, dist_rough)
            # plt.show()
            # Roots should fall near one of these points.
            low_θ = θ[dist_rough < thresh]
            low_dist = dist_rough[dist_rough < thresh]
            sub_zones.extend(find_low_zones(low_dist, low_θ))

        zones = sub_zones
        # We're subdividing each time, so use a lower and lower thresh to
        # narrow down on the possible roots.
        thresh *= THRESH_SCALER

    θ_guesses = ((zone[1] + zone[1]) / 2 for zone in zones)

    # Find precise roots; the guesses should be pretty good, but let fsolve refine.
    # Using fsolve doesn't appreciably add to solve time.
    θ_roots = (optimize.fsolve(cast_fsolvable, guess)[0] for guess in θ_guesses)
    # θ_roots = list(θ_roots)
    # print(θ_roots, 'roots')
    return map(partial(simple_collision, impact, center), θ_roots)


def find_reflection_points(impact: np.ndarray, sample_pt: np.ndarray) -> np.ndarray:
    """Find which points outside the coral are reflected back to the sample point."""
    coral_center = np.array([0, 0])  # todo this only works with a circle!

    # height = 0 # The height to add based on reflected points outside the circular corral.
    wall_collision_pts = find_wall_collisions(impact, sample_pt, coral_center)
    wall_collision_pts = list(wall_collision_pts)

    for wall_pt in wall_collision_pts:
        dx_wall_sample, dy_wall_sample = sample_pt[0] - wall_pt[0], sample_pt[1] - wall_pt[1]
        dx_impact_wall, dy_impact_wall = wall_pt[0] - impact[0], wall_pt[1] - impact[1]

        dist_wall_sample = sqrt(dx_wall_sample ** 2 + dy_wall_sample ** 2)
        dist_impact_wall = sqrt(dx_impact_wall ** 2 + dy_impact_wall ** 2)
        scale_factor = dist_wall_sample / dist_impact_wall

        # A vector, (0-centered) representing where to take the point outside the circle,
        # that we might reflect onto the sample point. scale factor adjusts how far
        # to modify the impact-wall line, with  1 being its original length.
        vector = np.array([dx_impact_wall, dy_impact_wall]) * (1 + scale_factor)
        yield vector + impact  # vector is centered at 0, 0; move to to the impact's ref.
