"""Lightweight Stable Fluid solver for interactive demos."""
from __future__ import annotations

from dataclasses import dataclass
from math import floor

import numpy as np
from numba import njit


@dataclass
class FluidParams:
    n: int = 96
    dt: float = 0.015
    diffusion: float = 1e-4
    viscosity: float = 1e-4
    dissipation: float = 0.996


@njit(cache=True, fastmath=True)
def _set_bnd_numba(b: int, x: np.ndarray, n: int) -> None:
    for i in range(1, n + 1):
        if b == 2:
            x[0, i] = -x[1, i]
            x[n + 1, i] = -x[n, i]
        else:
            x[0, i] = x[1, i]
            x[n + 1, i] = x[n, i]

        if b == 1:
            x[i, 0] = -x[i, 1]
            x[i, n + 1] = -x[i, n]
        else:
            x[i, 0] = x[i, 1]
            x[i, n + 1] = x[i, n]

    x[0, 0] = 0.5 * (x[1, 0] + x[0, 1])
    x[0, n + 1] = 0.5 * (x[1, n + 1] + x[0, n])
    x[n + 1, 0] = 0.5 * (x[n, 0] + x[n + 1, 1])
    x[n + 1, n + 1] = 0.5 * (x[n, n + 1] + x[n + 1, n])


@njit(cache=True, fastmath=True)
def _lin_solve_numba(b: int, x: np.ndarray, x0: np.ndarray, a: float, c: float, n: int) -> None:
    c_recip = 1.0 / c
    for _ in range(20):
        for j in range(1, n + 1):
            for i in range(1, n + 1):
                x[j, i] = (
                    x0[j, i]
                    + a * (
                        x[j, i - 1]
                        + x[j, i + 1]
                        + x[j - 1, i]
                        + x[j + 1, i]
                    )
                ) * c_recip
        _set_bnd_numba(b, x, n)


@njit(cache=True, fastmath=True)
def _diffuse_numba(b: int, x: np.ndarray, x0: np.ndarray, diff: float, dt: float, n: int) -> None:
    a = dt * diff * n * n
    _lin_solve_numba(b, x, x0, a, 1.0 + 4.0 * a, n)


@njit(cache=True, fastmath=True)
def _advect_numba(
    b: int,
    d: np.ndarray,
    d0: np.ndarray,
    veloc_x: np.ndarray,
    veloc_y: np.ndarray,
    dt: float,
    n: int,
) -> None:
    dt0 = dt * n
    for j in range(1, n + 1):
        for i in range(1, n + 1):
            x = i - dt0 * veloc_x[j, i]
            y = j - dt0 * veloc_y[j, i]
            if x < 0.5:
                x = 0.5
            elif x > n + 0.5:
                x = n + 0.5
            if y < 0.5:
                y = 0.5
            elif y > n + 0.5:
                y = n + 0.5
            i0 = int(floor(x))
            i1 = i0 + 1
            j0 = int(floor(y))
            j1 = j0 + 1
            s1 = x - i0
            s0 = 1.0 - s1
            t1 = y - j0
            t0 = 1.0 - t1
            d[j, i] = (
                s0 * (t0 * d0[j0, i0] + t1 * d0[j1, i0])
                + s1 * (t0 * d0[j0, i1] + t1 * d0[j1, i1])
            )
    _set_bnd_numba(b, d, n)


@njit(cache=True, fastmath=True)
def _project_numba(
    veloc_x: np.ndarray,
    veloc_y: np.ndarray,
    p: np.ndarray,
    div: np.ndarray,
    n: int,
) -> None:
    for j in range(1, n + 1):
        for i in range(1, n + 1):
            div[j, i] = -0.5 * (
                veloc_x[j, i + 1]
                - veloc_x[j, i - 1]
                + veloc_y[j + 1, i]
                - veloc_y[j - 1, i]
            ) / n
            p[j, i] = 0.0
    _set_bnd_numba(0, div, n)
    _set_bnd_numba(0, p, n)
    _lin_solve_numba(0, p, div, 1.0, 4.0, n)

    for j in range(1, n + 1):
        for i in range(1, n + 1):
            veloc_x[j, i] -= 0.5 * n * (p[j, i + 1] - p[j, i - 1])
            veloc_y[j, i] -= 0.5 * n * (p[j + 1, i] - p[j - 1, i])
    _set_bnd_numba(1, veloc_x, n)
    _set_bnd_numba(2, veloc_y, n)


class StableFluid:
    """Jos Stam style grid-based solver with density visualization."""

    def __init__(self, params: FluidParams | None = None) -> None:
        self.params = params or FluidParams()
        n = self.params.n
        shape = (n + 2, n + 2)
        self.density = np.zeros(shape, dtype=np.float32)
        self.prev_density = np.zeros(shape, dtype=np.float32)
        self.vel_x = np.zeros(shape, dtype=np.float32)
        self.vel_y = np.zeros(shape, dtype=np.float32)
        self.prev_vel_x = np.zeros(shape, dtype=np.float32)
        self.prev_vel_y = np.zeros(shape, dtype=np.float32)

    def warmup(self, iterations: int = 2) -> None:
        """Prime Numba kernels so the first live frame is smooth."""
        for _ in range(max(1, iterations)):
            self.step()
        self.clear()

    def clear(self) -> None:
        for field in (
            self.density,
            self.prev_density,
            self.vel_x,
            self.vel_y,
            self.prev_vel_x,
            self.prev_vel_y,
        ):
            field.fill(0.0)

    def add_density(self, x: int, y: int, amount: float, radius: int = 2) -> None:
        self._paint(self.density, x, y, amount, radius)

    def add_velocity(self, x: int, y: int, amount_x: float, amount_y: float, radius: int = 2) -> None:
        self._paint(self.vel_x, x, y, amount_x, radius)
        self._paint(self.vel_y, x, y, amount_y, radius)

    def _paint(self, field: np.ndarray, x: int, y: int, value: float, radius: int) -> None:
        n = self.params.n
        x = int(np.clip(x, 1, n))
        y = int(np.clip(y, 1, n))
        rr = max(1, radius)
        x0 = max(1, x - rr)
        x1 = min(n, x + rr)
        y0 = max(1, y - rr)
        y1 = min(n, y + rr)
        field[y0 : y1 + 1, x0 : x1 + 1] += value

    def step(self) -> None:
        p = self.params
        _diffuse_numba(1, self.prev_vel_x, self.vel_x, p.viscosity, p.dt, p.n)
        _diffuse_numba(2, self.prev_vel_y, self.vel_y, p.viscosity, p.dt, p.n)

        _project_numba(self.prev_vel_x, self.prev_vel_y, self.vel_x, self.vel_y, p.n)

        _advect_numba(1, self.vel_x, self.prev_vel_x, self.prev_vel_x, self.prev_vel_y, p.dt, p.n)
        _advect_numba(2, self.vel_y, self.prev_vel_y, self.prev_vel_x, self.prev_vel_y, p.dt, p.n)

        _project_numba(self.vel_x, self.vel_y, self.prev_vel_x, self.prev_vel_y, p.n)

        _diffuse_numba(0, self.prev_density, self.density, p.diffusion, p.dt, p.n)
        _advect_numba(0, self.density, self.prev_density, self.vel_x, self.vel_y, p.dt, p.n)
        self.density *= p.dissipation
