from collections import namedtuple
from typing import Tuple

import numpy as np


# Assume we're not modeling the up and down motions; each simulation tick
# represents the particle impacting the grid

# todo Do I want a discrete grid as an array, or should I take a more precise route?
# todo What about modeling the particle's verticle motion: analytic, or numerical?
# todo ... I think it has to be numerical, for now.

# does particle height represent energy??

# 2D grid, simulatinga a vibrating silicon/oil field. Positive values indicate
# a height above neutral; negative below.
GRID_SIZE = (10000, 10000)
VIBRATION_FREQ = 1/100  # vibration cycles per tick?
RUN_TIME = 500  # in ticks
G = 9.81  # Gravity, in m/s^s

# pos in x, y, height above neutral; respective velocities
PARTICLE_START = (5000, 5000, 10., 0, 0, 0, 0, 0, 0)

# todo include decay due to friction/drag??


# todo do I need accel? Constant down due to g, and bounce accel could be applied instantly??

class Particle:
    # The particle's position (s) values are grid indices.
    def __init__(self, sx, sy, sz, vy, vx, vz, ax, ay, az):
        self.sx = sx
        self.sy = sy
        self.sz = sz
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.ax = ax
        self.ay = ay
        self.az = az

    def step(self) -> None:
        self.sy += self.vy
        self.vz += self.az
        self.sz += self.vz

        self.vx += self.ax
        self.sx += self.vx
        self.vy += self.ay

    def bounce(self, grid: np.ndarray) -> None:
        pass


class Grid:
    def __init__(self, size: Tuple[int, int]):
        # grid is a n x n x 3 array containing a 2d grid of fluid height, vertical
        # velocity, and vertical acceleration.
        self.grid = np.zeros(*size, 3)

    def step(self) -> None:
        self.grid[:, :, 0] += self.grid[:, :, 1]  # Add velocity to position
        self.grid[:, :, 1] += self.grid[:, :, 2]  # Add acceleration to velocity

    def bounce(self, impact_point: Tuple[int, int], velocity: float=1.) -> None:
        # Impact point corresponds to a point on grid's first two dimensions.
        pass


def main():
    grid = Grid(GRID_SIZE)
    p = Particle(*PARTICLE_START)

    for t in RUN_TIME:
        p.step()
        grid.step()

        # todo interpolate, instead of taking the nearest one!!
        nearest_grid_point = int(round(p.sx)), int(round(p.sy))

        if p.sz <= grid.grid[nearest_grid_point[0], nearest_grid_point[1], 0]:
            grid.bounce(nearest_grid_point)
            p.bounce(p, grid)
