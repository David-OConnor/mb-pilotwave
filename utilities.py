from typing import Iterable

import numba
import numpy as np
from matplotlib import pyplot as plt

import quantum


def wave_field(t: float, impacts: Iterable[quantum.Impact], plot=True) -> np.ndarray:
    """Calculate a wave's effect on a 2d field."""
    h = np.zeros([500, 500])

    for i in range(500):
        for j in range(500):
            h[i, j] = quantum.net_surface_height(t, j, i, impacts)

    if plot:
        plt.imshow(h)
        # plot_surface(h)

    return h


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