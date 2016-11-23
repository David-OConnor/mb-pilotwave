from typing import Iterable
from itertools import product

import numba
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation

import drops


def wave_field(t: float, impacts: Iterable[drops.Impact], resolution: int=1, plot=True) -> np.ndarray:
    """Calculate a wave's effect on a 2d field."""
    # Be careful about resolution; computation time is proportional to its square.
    grid_x = (-20, 20)
    grid_y = (-20, 20)

    array_width = (grid_x[1] - grid_x[0]) * resolution
    array_height = (grid_y[1] - grid_y[0]) * resolution

    h = np.zeros([array_height, array_width])

    scaled_x = (grid_x[0] * resolution, grid_x[1] * resolution)
    scaled_y = (grid_y[0] * resolution, grid_y[1] * resolution)

    for i, j in product(range(*scaled_y), range(*scaled_x)):
        sx = j / resolution
        sy = i / resolution

        index_x = int(int(array_width / 2) + sx * resolution)
        index_y = int(int(array_height / 2) + sy * resolution)

        h[index_y, index_x] = drops.net_surface_height(t, impacts, drops.Point(sx, sy))

    if plot:
        plt.imshow(h, extent=[*grid_x, *grid_y])
        # plot_surface(h)

    return h


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