from typing import Iterable

import numba
import numpy as np
from matplotlib import pyplot as plt

import drops


def wave_field(t: float, impacts: Iterable[drops.Impact], resolution: int=1, plot=True) -> np.ndarray:
    """Calculate a wave's effect on a 2d field."""
    # Be careful about resolution; computation time is proportional to its square, I think.
    grid_width = 200
    grid_height = 200
    array_width = grid_width * resolution
    array_height = grid_height * resolution

    h = np.zeros([array_height, array_width])

    for i in range(array_height):
        for j in range(array_width):
            h[i, j] = drops.net_surface_height(t, j/resolution, i/resolution, impacts)

    if plot:
        plt.imshow(h)
        # plot_surface(h)

    return h


"""
impacts = [drops.Impact(10, 100, 100, 1), drops.Impact(10, 100, 105, 1), drops.Impact(10, 100, 95, 1), drops.Impact(10, 105, 100, 1), drops.Impact(10, 105, 105, 1), drops.Impact(10, 105, 95, 1), drops.Impact(10, 95, 100, 1), drops.Impact(10, 95, 105, 1), drops.Impact(10, 95, 95, 1)]"""


def plot_path(result: np.ndarray) -> None:
    """Plot the drop's x and y positions, parameterized for time."""
    x, y = result[:, 0], result[:, 1]
    plt.plot(x[::1000], y[::1000])
    print(x[::1000])
    print(y[::1000])
    plt.show()


def plot_surface(z: np.ndarray) -> None:
    """Make a surface plot from a 2d array."""
    from matplotlib import cm

    y, x = np.mgrid[:z.shape[0], :z.shape[1]]

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(x, y, z, cmap=cm.viridis, linewidth=.2, cstride=6, rstride=6)

    plt.tight_layout(rect=(0, 0, 1, 1))
    plt.show()