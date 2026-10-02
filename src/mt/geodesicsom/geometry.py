# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
Distances between the neurons of a lattice, computed on demand on the compute device.

A SOM needs d(i, j) -- the distance between neurons i and j in neighbour spacings -- to build
its neighbourhood function.  Storing the whole (n, n) matrix limits the map size (frequency 64,
40 962 neurons, would need 6.7 GB), so a NeuronDistance computes rows of it when they are
needed, from the lattice geometry, on the GPU when there is one.  When the matrix is small
enough (Backend.cache_bytes) it is built once and kept on the device instead.

    SphereDistance   great-circle distance on a geodesic dome (GeodesicSOM)
    PlaneDistance    Euclidean distance on a plane or a torus (PlaneSOM)
    DenseDistance    a precomputed (n, n) matrix (any other lattice)
"""
from __future__ import annotations

import numpy as np
from mt.geodesicdome.backend import Backend
from numpy import ndarray


class NeuronDistance:
    """Distances between n neurons in neighbour spacings; see the module docstring."""

    n: int

    def __init__(self):
        self._device: dict[str, dict] = {}        # per backend: inputs (and maybe the full matrix) on the device
        self._max: float | None = None

    # ------------------------------------------------ to be provided by subclasses
    def _inputs(self, b: Backend) -> dict:
        """The arrays `_rows` needs, moved to the device."""
        raise NotImplementedError

    def _rows(self, b: Backend, inputs: dict, index):
        """Rows `index` (a slice or a device index array) of the distance matrix, on the device."""
        raise NotImplementedError

    # --------------------------------------------------------------------- public
    def _state(self, b: Backend) -> dict:
        state = self._device.get(b.spec)
        if state is None:
            state = {'inputs': self._inputs(b)}
            if b.fits_cache(self.n, self.n):           # small enough: build the matrix once
                full = [self._rows(b, state['inputs'], s) for s in b.blocks(self.n, b.block_rows(self.n, 2))]
                state['full'] = b.cat(full, axis=0)
            self._device[b.spec] = state
        return state

    def rows(self, b: Backend, index):
        """Rows `index` of the (n, n) distance matrix on the device (index: slice or device indices)."""
        state = self._state(b)
        full = state.get('full')
        return full[index] if full is not None else self._rows(b, state['inputs'], index)

    def gaussian_matmul(self, b: Backend, sigma: float, m):
        """
        H @ m on the device, where H[i, j] = exp(-d(i, j)^2 / (2 sigma^2)) is the neighbourhood
        matrix and m an (n, k) device array.  H is formed a block of rows at a time.
        """
        c = -1.0 / (2.0 * float(sigma) ** 2)

        def block(s):
            d = self.rows(b, s)
            return b.exp(d * d * c) @ m

        rows = b.block_rows(self.n, arrays=2, total=self.n)
        return b.cat(b.map_blocks(block, self.n, rows), axis=0)

    def full(self, b: Backend) -> ndarray:
        """The whole (n, n) matrix as a float32 numpy array."""
        parts = [b.to_numpy(self.rows(b, s)) for s in b.blocks(self.n, b.block_rows(self.n, 2))]
        return np.concatenate(parts, axis=0).astype(np.float32, copy=False)

    def max(self, b: Backend) -> float:
        """The largest distance between two neurons."""
        if self._max is None:
            parts = b.map_blocks(lambda s: float(b.to_numpy(b.max(self.rows(b, s)))), self.n,
                                 b.block_rows(self.n, 2, total=self.n))
            self._max = max(parts)
        return self._max

    def release(self, b: Backend | None = None) -> None:
        """Frees the device copies (all of them, or those of backend `b`)."""
        if b is None:
            self._device.clear()
        else:
            self._device.pop(b.spec, None)

    def __getstate__(self):                       # device arrays are not pickled
        state = self.__dict__.copy()
        state['_device'] = {}
        return state


class DenseDistance(NeuronDistance):
    """A precomputed (n, n) distance matrix."""

    def __init__(self, matrix: ndarray):
        super().__init__()
        self.matrix = np.asarray(matrix, dtype=np.float32)
        self.n = len(self.matrix)

    def _inputs(self, b):
        return {'matrix': b.asarray(self.matrix)}

    def _state(self, b):                          # the matrix *is* the cache
        state = self._device.get(b.spec)
        if state is None:
            state = {'inputs': self._inputs(b)}
            state['full'] = state['inputs']['matrix']
            self._device[b.spec] = state
        return state

    def _rows(self, b, inputs, index):
        return inputs['matrix'][index]

    def full(self, b=None) -> ndarray:
        return self.matrix

    def max(self, b=None) -> float:
        return float(self.matrix.max()) if self.matrix.size else 0.0


class SphereDistance(NeuronDistance):
    """Great-circle distances between unit vectors `points` (n, 3), in units of `ring_length` radians."""

    def __init__(self, points: ndarray, ring_length: float):
        super().__init__()
        self.points = np.asarray(points, dtype=float)
        self.ring_length = float(ring_length)
        self.n = len(self.points)

    def _inputs(self, b):
        p = b.asarray(self.points)
        return {'p': p, 'pt': p.T}

    def _rows(self, b, inputs, index):
        cos = inputs['p'][index] @ inputs['pt']
        return b.arccos(b.clip(cos, -1.0, 1.0)) * (1.0 / self.ring_length)

    def max(self, b=None) -> float:
        # a geodesic dome is centrally symmetric: every vertex has its antipode on the dome
        return float(np.pi / self.ring_length)


class PlaneDistance(NeuronDistance):
    """
    Euclidean distances between `positions` (n, 2); with `torus=True` the plane wraps around
    with periods (width, height).
    """

    def __init__(self, positions: ndarray, width: float, height: float, torus: bool = False):
        super().__init__()
        self.positions = np.asarray(positions, dtype=float)
        self.width, self.height, self.torus = float(width), float(height), bool(torus)
        self.n = len(self.positions)

    def _inputs(self, b):
        pos = b.asarray(self.positions)
        return {'x': pos[:, 0], 'y': pos[:, 1]}

    def _rows(self, b, inputs, index):
        x, y = inputs['x'], inputs['y']
        dx = b.abs(x[index][:, None] - x[None, :])
        dy = b.abs(y[index][:, None] - y[None, :])
        if self.torus:
            dx = b.minimum(dx, self.width - dx)
            dy = b.minimum(dy, self.height - dy)
        return b.sqrt(dx * dx + dy * dy)
