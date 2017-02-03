from typing import Iterable, Tuple
from itertools import product

import numba
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from numpy import sin, cos

import drops, wave_reflection
from drops import D, τ


def wave_field(t: float, impacts: Iterable[drops.Impact], resolution: int=1,
               plot=True, corral=False) -> np.ndarray:
    """Calculate a wave's effect on a 2d field."""
    # Be careful about resolution; computation time is proportional to its square.
    D = 76  # todo temp
    grid_x = (-int(D/2), int(D/2))
    grid_y = (-int(D/2), int(D/2))

    array_width = (grid_x[1] - grid_x[0]) * resolution
    array_height = (grid_y[1] - grid_y[0]) * resolution

    h = np.zeros([array_height, array_width])

    scaled_x = (grid_x[0] * resolution, grid_x[1] * resolution)
    scaled_y = (grid_y[0] * resolution, grid_y[1] * resolution)

    print(scaled_x, scaled_y)
    range_x = range(*scaled_y)
    range_y = range(*scaled_x)

    for i, j in product(range_x, range_y):
        sx = j / resolution
        sy = i / resolution

        # Arrays index up-down, then left-right. Cartesian coordinates index
        # left-right, then down-up, hence the y-axis sign swap.
        index_x = int(array_width / 2 + sx * resolution)
        index_y = int(array_height / 2 - sy * resolution) - 1

        h[index_y, index_x] = drops.net_surface_height(t, impacts, np.array([sx, sy]),
                                                       corral=corral)

    if plot:
        fig, ax = plt.subplots()
        cax = ax.imshow(h, extent=[*grid_x, *grid_y])
        fig.colorbar(cax)
        # plot_surface(h)

    return h


def reflection_field(impact: np.ndarray, center: np.ndarray, θiw: float) -> np.ndarray:
    """Plot the distances to a reflected ray in a circular corral, for every
    point in a circle.  Useful for seeing a ray's reflected path."""
    result = np.zeros([D + 5, D + 5])

    range_θ = np.linspace(0, τ, 1000)
    range_r = np.linspace(0, D/2, 100)

    for θ, r in product(range_θ, range_r):
        point = np.array([r * cos(θ), r * sin(θ)])  # Convert from polar to cartesian
        d = wave_reflection.cast_ray(impact, point, center, θiw)

        index_x = int(result.shape[1] / 2 + point[0])
        index_y = int(result.shape[0] / 2 - point[1]) - 1

        result[index_y, index_x] = d

    plt.imshow(result, extent=[-D/2, D/2, -D/2, D/2])
    plt.colorbar()


def plot_field(f, scale0: Tuple, scale1: Tuple, resolution0: float, resolution1: float,
               polar=False, args: Tuple=(), plot=True) -> np.ndarray:
    """Generic version of wave_field/reflection_field"""
    # Be careful about resolution; computation time is proportional to its square.

    array_width = (scale0[1] - scale0[0]) * resolution0
    array_height = (scale1[1] - scale1[0]) * resolution1

    result = np.zeros([array_height, array_width])

    scaled_x = (grid_x[0] * resolution, grid_x[1] * resolution)
    scaled_y = (grid_y[0] * resolution, grid_y[1] * resolution)

    for i, j in product(range(*scaled_y), range(*scaled_x)):
        if polar:
            point = np.array([j * cos(i), j * sin(i)])  # Convert from polar to cartesian

        s0 = j / resolution0
        s1 = i / resolution1

        # Arrays index up-down, then left-right. Cartesian coordinates index
        # left-right, then down-up, hence the y-axis sign swap.
        index_x = int(array_width / 2 + s0 * resolution0)
        index_y = int(array_height / 2 - s1 * resolution1) - 1

        result[index_y, index_x] = f(*args, drops.Point(s0, s1))

    if plot:
        plt.imshow(result, extent=[*scale0, *scale1])
        plt.colorbar()
        # plot_surface(h)

    return result


def plot_path(soln: np.ndarray, plot=True) -> np.ndarray:
    """Plot the drop's x and y positions, parameterized for time."""
    _, num_drops, t = soln.shape
    for i in range(num_drops):
        x, y = soln[:, i, 0], soln[:, i, 1]
        plt.plot(x, y)

    if plot:
        plt.show()

    # waves = wave_field()


"""
impacts = [drops.Impact(10, 100, 100, 1), drops.Impact(10, 100, 105, 1), drops.Impact(10, 100, 95, 1), drops.Impact(10, 105, 100, 1), drops.Impact(10, 105, 105, 1), drops.Impact(10, 105, 95, 1), drops.Impact(10, 95, 100, 1), drops.Impact(10, 95, 105, 1), drops.Impact(10, 95, 95, 1)]"""

def plot_surface(z: np.ndarray) -> None:
    """Make a surface plot from a 2d array."""
    from matplotlib import cm

    y, x = np.mgrid[:z.shape[0], :z.shape[1]]

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(x, y, z, cmap=cm.viridis, linewidth=.2, cstride=6, rstride=6)

    plt.tight_layout(rect=(0, 0, 1, 1))
    plt.show()