# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
LatticeSOM: the training and measurements shared by GeodesicSOM (sphere) and PlaneSOM (plane or torus).

A subclass describes its lattice once, with `_set_lattice(...)`; everything else -- initialisation,
batch/online training, best matching units, hits, U-matrix, the distance between neighbouring
neurons (per edge, per face, per attribute), quantisation and topographic error -- lives here.

Every neuron has a weight vector with one value per attribute.  Distances between neurons are
measured in neighbour spacings (1 = the mean distance between neighbouring neurons), so `sigma`
means the same thing on any lattice.  Missing values (NaN, see mt.geodesicsom.util.MISSING) are
ignored when finding the best matching unit and when updating the weights.

The heavy work -- distances from samples to neurons, best matching units, and the neighbourhood
products of batch training -- runs on a GPU when one is available and otherwise on every CPU
core (see mt.geodesicdome.backend).  Choose the device with the `backend` argument
('auto', 'cuda', 'mps', 'cupy', 'cpu', 'numpy'), with `som.to(...)`, or with the environment
variable MTGEODESIC_BACKEND.  Results always come back as numpy arrays.
"""
from collections.abc import Sequence

import numpy as np
from mt.geodesicdome.backend import Backend, get_backend
from numpy import ndarray

from mt.geodesicsom.geometry import DenseDistance, NeuronDistance
from mt.geodesicsom.som import SOM, InitializationType


class LatticeSOM(SOM):
    """
    Base class of GeodesicSOM and PlaneSOM (not used directly).

    :param grid: the lattice (a Manifold)
    :param seed: seed or numpy Generator for random initialisation and online training
    :param backend: compute device: None (the default backend, normally 'auto': GPU if available,
                    else all CPU cores), a spec such as 'cuda', 'mps', 'cpu', 'numpy', or a Backend

    Attributes set by `_set_lattice`
        n_nodes         number of neurons
        faces           (F, k) neuron indices of every face (triangles, or squares on a rectilinear plane)
        edges           (E, 2) pairs of neighbouring neurons
        neighbours      list of arrays: the neighbours of every neuron
        node_distance   (n, n) float32 distance between neurons, in neighbour spacings (built on first use;
                        training does not need it, so large maps never have to hold the whole matrix)
        distance        the NeuronDistance that computes node_distance a block at a time on the device
        index_map       (vertices,) neuron index of every stored lattice vertex, row = vertex.id
        init_coords     (n, k) lattice coordinates in about [-1, 1], for linear initialisation
        default_sigma   starting neighbourhood width used when train() is not given one
    """

    def __init__(self, grid, seed=None, backend: str | Backend | None = None):
        super().__init__(grid)
        self.rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        self.weights: ndarray | None = None                              # (n_nodes, dim)
        self.dim: int = 0
        self.history: list[float] = []                                      # quantisation error per epoch
        self._backend_spec = backend
        self._node_distance: ndarray | None = None

    # ------------------------------------------------------------------ device
    @property
    def backend(self) -> Backend:
        """The compute backend (resolved when first needed)."""
        return get_backend(self._backend_spec)

    def to(self, backend: str | Backend | None):
        """Moves the computation to another backend ('cuda', 'mps', 'cpu', 'numpy', ...); returns self."""
        old = self.backend
        self._backend_spec = backend
        if getattr(self, 'distance', None) is not None and self.backend.spec != old.spec:
            self.distance.release(old)
        return self

    def _set_lattice(self, faces: ndarray, node_distance, index_map: ndarray, init_coords: ndarray,
                     edges: ndarray | None = None):
        """
        :param node_distance: an (n, n) array, or a NeuronDistance that computes it on demand
        """
        self.faces = np.asarray(faces, dtype=int)
        if isinstance(node_distance, NeuronDistance):
            self.distance: NeuronDistance = node_distance
            self._node_distance = None
        else:
            self.distance = DenseDistance(node_distance)
            self._node_distance = self.distance.matrix
        self.n_nodes = int(self.distance.n)
        if edges is None:
            k = self.faces.shape[1]
            edges = np.vstack([self.faces[:, [i, (i + 1) % k]] for i in range(k)])
        edges = np.sort(np.asarray(edges, dtype=int), axis=1)
        self.edges: ndarray = np.unique(edges[edges[:, 0] != edges[:, 1]], axis=0)
        neighbours: list[list] = [[] for _ in range(self.n_nodes)]
        for a, b in self.edges:
            neighbours[a].append(b)
            neighbours[b].append(a)
        self.neighbours: list[ndarray] = [np.array(n, dtype=int) for n in neighbours]
        self._edge_keys = np.union1d(self.edges[:, 0] * self.n_nodes + self.edges[:, 1],
                                     self.edges[:, 1] * self.n_nodes + self.edges[:, 0])
        self.index_map: ndarray = np.asarray(index_map, dtype=int)
        self.init_coords: ndarray = np.asarray(init_coords, dtype=float)
        self.default_sigma: float = float(self.distance.max(self.backend)) / 4.0

    @property
    def node_distance(self) -> ndarray:
        """(n, n) float32 distances between neurons in neighbour spacings (computed on first use)."""
        if self._node_distance is None:
            self._node_distance = self.distance.full(self.backend)
        return self._node_distance

    @node_distance.setter
    def node_distance(self, matrix: ndarray):
        self.distance = DenseDistance(matrix)
        self._node_distance = self.distance.matrix

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_node_distance'] = None if not isinstance(self.distance, DenseDistance) else self._node_distance
        return state

    # ----------------------------------------------------------- initialisation
    def initialise(self, type: InitializationType = InitializationType.Linear, dataset: ndarray = None,
                   dim: int | None = None):
        """
        Linear: weights spread over the first principal components of `dataset` (three on a sphere,
        two on a plane), placed by each neuron's position on the lattice (needs a dataset).  Random: uniform inside the
        per-attribute [min, max] of `dataset`, or in [0, 1) with `dim` attributes and no dataset.
        """
        if dataset is None:
            dim = dim or self.dim
            if not dim:
                raise ValueError('initialise needs a dataset, or dim for random initialisation')
            self._set_weights(self.rng.random((self.n_nodes, dim)))
            return
        x = _as_2d(dataset)
        if type == InitializationType.Linear:
            filled = np.where(np.isnan(x), np.nanmean(x, axis=0), x)
            mean = filled.mean(axis=0)
            if len(filled) > 1:
                eigval, eigvec = np.linalg.eigh(np.atleast_2d(np.cov(filled, rowvar=False)))
            else:
                eigval, eigvec = np.zeros(x.shape[1]), np.eye(x.shape[1])
            order = np.argsort(eigval)[::-1][:3]
            axes = eigvec[:, order] * np.sqrt(np.maximum(eigval[order], 0.0))    # (dim, k)
            coords = self.init_coords[:, :axes.shape[1]]
            self._set_weights(mean + coords @ axes[:, :coords.shape[1]].T)
        else:
            lo, hi = np.nanmin(x, axis=0), np.nanmax(x, axis=0)
            self._set_weights(lo + self.rng.random((self.n_nodes, x.shape[1])) * (hi - lo))

    # ------------------------------------------------------------------ training
    def train(self, dataset: ndarray, epochs: int = 20, sigma: float | None = None, sigma_end: float = 1.0,
              mode: str = 'batch', learning_rate: Sequence[float] = (0.5, 0.02), verbose: bool = False):
        """
        Trains the map on the compute backend (GPU if available, else all CPU cores).

        :param dataset: (samples, attributes); NaN marks a missing value
        :param epochs: passes over the data
        :param sigma: starting neighbourhood width in neighbour spacings (default: a quarter of the
                      largest distance between two neurons)
        :param sigma_end: final neighbourhood width in rings
        :param mode: 'batch' (fast, deterministic) or 'online' (classic sample-by-sample updates)
        :param learning_rate: (start, end) for online mode
        :return: self
        """
        if mode not in ('batch', 'online'):
            raise ValueError("mode must be 'batch' or 'online'")
        x = _as_2d(dataset)
        if self.weights is None:
            self.initialise(InitializationType.Linear, x)
        if x.shape[1] != self.dim:
            raise ValueError(f'dataset has {x.shape[1]} attributes; the map has {self.dim}')
        if sigma is None:
            sigma = self.default_sigma

        b = self.backend
        data = _DeviceData(b, x, centre=self.weights.mean(axis=0), rows=b.block_rows(self.n_nodes, 3, len(x)))
        w = b.asarray(self.weights - data.centre)                          # a device copy, centred
        pending_qe, s_prev = False, sigma                                  # batch: QE of the last update still owed

        for epoch in range(epochs):
            t = epoch / max(epochs - 1, 1)
            s = sigma * (sigma_end / sigma) ** t
            if mode == 'batch':
                # one pass finds the BMUs for this update and the QE of the previous one
                bmu, mind = self._match(data, w, want_min=pending_qe)
                if pending_qe:
                    self._record(mind, epoch - 1, epochs, s_prev, verbose)
                sums = b.zeros((self.n_nodes, self.dim))
                counts = b.zeros((self.n_nodes, self.dim if data.has_missing else 1))
                for blk, idx in zip(data.slices, bmu, strict=True):
                    xb, mb = data.get(blk)
                    b.index_add(sums, idx, xb)
                    b.index_add(counts, idx, mb if data.has_missing else data.ones(blk))
                nd = self.distance.gaussian_matmul(b, s, b.cat([sums, counts], axis=1))
                num, den = nd[:, :self.dim], nd[:, self.dim:]
                w = b.where(den > 1e-12, num / b.maximum(den, 1e-12), w)
                pending_qe, s_prev = True, s
            else:
                w = self._online_epoch(data, w, s, epoch, epochs, learning_rate)
                self._record(self._match(data, w)[1], epoch, epochs, s, verbose)
        if pending_qe:
            self._record(self._match(data, w)[1], epochs - 1, epochs, s_prev, verbose)

        self.weights = b.to_numpy(w).astype(float) + data.centre
        self._store_on_vertices()
        return self

    def _record(self, mind, epoch, epochs, s, verbose):
        mind = np.concatenate(mind) if mind else np.zeros(0)
        qe = float(np.sqrt(np.maximum(mind, 0.0)).mean()) if len(mind) else float('nan')
        self.history.append(qe)
        if verbose:
            print(f'epoch {epoch + 1:3d}/{epochs}  sigma {s:5.2f}  QE {qe:.4f}')

    def _online_epoch(self, data, w, s, epoch, epochs, learning_rate):
        b = self.backend
        lr0, lr1 = learning_rate
        c = -1.0 / (2.0 * s * s)
        n = data.n
        for k, i in enumerate(self.rng.permutation(n)):
            lr = lr0 * (lr1 / lr0) ** ((epoch + k / n) / epochs)
            xi, mi = data.get(slice(i, i + 1))                               # (1, dim) each
            delta = (xi - w) * mi if data.has_missing else xi - w            # 0 where the value is missing
            win = b.argmin(b.sum(delta * delta, axis=1), axis=0)
            d = self.distance.rows(b, win.reshape(1))[0]
            w = w + (b.exp(d * d * c) * lr)[:, None] * delta
        return w

    def _match(self, data, w, want_min: bool = True):
        """Best matching unit of every sample, per data block, on the device (and each squared distance)."""
        b = self.backend
        ww = _WeightTerms(b, w)

        def block(blk):
            d = ww.sq_distances(*data.get(blk))
            idx = b.argmin(d, axis=1)
            return idx, (b.to_numpy(b.min(d, axis=1)) if want_min else None)

        out = b.map_blocks(block, data.n, data.rows)
        return [o[0] for o in out], [o[1] for o in out]

    def _neighbourhood(self, sigma: float) -> ndarray:
        return np.exp(-(self.node_distance ** 2) / np.float32(2.0 * sigma * sigma))

    # ------------------------------------------------------------------ queries
    def _device_scan(self, dataset: ndarray, fn):
        """Runs fn(d_block) on the (samples, neurons) squared distances, block by block; returns numpy parts."""
        x = _as_2d(dataset)
        b = self.backend
        data = _DeviceData(b, x, centre=self.weights.mean(axis=0), rows=b.block_rows(self.n_nodes, 3, len(x)))
        ww = _WeightTerms(b, b.asarray(self.weights - data.centre))
        return b.map_blocks(lambda blk: b.to_numpy(fn(ww.sq_distances(*data.get(blk)))), data.n, data.rows)

    def distances(self, dataset: ndarray) -> ndarray:
        """(samples, neurons) squared distances, ignoring missing values."""
        parts = self._device_scan(dataset, lambda d: d)
        return np.concatenate(parts).astype(float, copy=False) if parts else np.zeros((0, self.n_nodes))

    def bmu(self, dataset: ndarray, second: bool = False) -> ndarray:
        """Best matching unit of every sample (neuron index); with second=True, (samples, 2) best two."""
        b = self.backend
        k = min(2, self.n_nodes)
        parts = self._device_scan(dataset, (lambda d: b.smallest(d, k)) if second else (lambda d: b.argmin(d, 1)))
        if not parts:
            return np.zeros((0, 2) if second else 0, dtype=int)
        return np.concatenate(parts).astype(int, copy=False)

    def hits(self, dataset: ndarray) -> ndarray:
        """Number of samples mapped to each neuron."""
        return np.bincount(self.bmu(dataset), minlength=self.n_nodes)

    def u_matrix(self) -> ndarray:
        """Mean distance from each neuron's weight vector to its neighbours' (high = cluster border)."""
        d = self.edge_distance()
        total = np.zeros(self.n_nodes)
        np.add.at(total, self.edges[:, 0], d)
        np.add.at(total, self.edges[:, 1], d)
        return total / np.array([len(n) for n in self.neighbours])

    def edge_distance(self) -> ndarray:
        """Euclidean distance between the weight vectors of every pair of neighbouring neurons (per edge)."""
        return np.linalg.norm(self.weights[self.edges[:, 0]] - self.weights[self.edges[:, 1]], axis=1)

    def face_distance(self) -> ndarray:
        """
        For every face between neighbouring neurons (a triangle, or a square on a rectilinear plane),
        the mean Euclidean distance between the weight vectors along its edges: a U-matrix drawn
        *between* the neurons.
        """
        w = self.weights[self.faces]                                          # (F, k, dim)
        return np.linalg.norm(w - np.roll(w, -1, axis=1), axis=2).mean(axis=1)

    def face_component_difference(self) -> ndarray:
        """
        (faces, attributes): for every face between neighbouring neurons and every attribute k, the
        mean absolute difference |w_a[k] - w_b[k]| along its edges -- the neighbour distance of
        face_distance(), computed one attribute at a time.
        """
        w = self.weights[self.faces]                                          # (F, k, dim)
        return np.abs(w - np.roll(w, -1, axis=1)).mean(axis=1)

    def face_component_value(self) -> ndarray:
        """(faces, attributes): the mean weight of each attribute over every face's neurons."""
        return self.weights[self.faces].mean(axis=1)

    def quantisation_error(self, dataset: ndarray) -> float:
        """Mean distance between each sample and its best matching unit."""
        b = self.backend
        parts = self._device_scan(dataset, lambda d: b.min(d, axis=1))
        mind = np.concatenate(parts) if parts else np.zeros(0)
        return float(np.sqrt(np.maximum(mind, 0.0)).mean())

    def topographic_error(self, dataset: ndarray) -> float:
        """Fraction of samples whose best and second-best units are not neighbours."""
        best2 = self.bmu(dataset, second=True)
        adjacent = np.isin(best2[:, 0] * self.n_nodes + best2[:, 1], self._edge_keys)
        return float(1.0 - adjacent.mean())

    def node_labels(self, dataset: ndarray, labels: Sequence) -> list[dict]:
        """For every neuron, {label: count} of the samples mapped to it (empty dict if none)."""
        result: list[dict] = [{} for _ in range(self.n_nodes)]
        for b, lab in zip(self.bmu(dataset), labels, strict=True):
            result[b][lab] = result[b].get(lab, 0) + 1
        return result

    def bmu_labels(self, dataset: ndarray, labels: Sequence, mode: str = 'majority',
                   max_per_node: int = 3) -> list[str | None]:
        """
        A text label for every neuron from the labels of the samples it is the best matching unit of
        (None for neurons that win no sample).

        :param labels: one label (or name) per sample, e.g. class labels or sample names
        :param mode: 'majority' -- the most common label ('A');
                     'all'      -- every distinct label, most common first ('A/B');
                     'counts'   -- labels with their counts ('A×3 B×1');
                     'first'    -- the first sample's label only (useful for unique sample names)
        :param max_per_node: at most this many labels in 'all' and 'counts' ('…' marks the rest)
        """
        if mode not in ('majority', 'all', 'counts', 'first'):
            raise ValueError("mode must be 'majority', 'all', 'counts' or 'first'")
        labels = list(labels)
        if len(labels) != len(dataset):
            raise ValueError(f'{len(labels)} labels for {len(dataset)} samples')
        won: list[list] = [[] for _ in range(self.n_nodes)]
        for b, lab in zip(self.bmu(dataset), labels, strict=True):
            won[b].append(lab)
        text: list[str | None] = []
        for labs in won:
            if not labs:
                text.append(None)
                continue
            if mode == 'first':
                text.append(str(labs[0]))
                continue
            values, counts = np.unique(np.asarray(labs, dtype=str), return_counts=True)
            order = np.lexsort((values, -counts))                          # most common first, then by name
            values, counts = values[order], counts[order]
            if mode == 'majority':
                text.append(str(values[0]))
                continue
            shown = [f'{v}×{c}' if mode == 'counts' else str(v)
                     for v, c in zip(values[:max_per_node], counts[:max_per_node], strict=True)]
            more = '…' if len(values) > max_per_node else ''
            text.append(('/' if mode == 'all' else ' ').join(shown) + more)
        return text

    def vertex_weights(self) -> ndarray:
        """Weights for every stored lattice vertex (on a dome, seam copies repeated), row = vertex.id."""
        return self.weights[self.index_map]

    # ------------------------------------------------------------------ helpers
    def _set_weights(self, w: ndarray):
        self.weights = np.array(w, dtype=float)
        self.dim = self.weights.shape[1]
        self._store_on_vertices()

    def _store_on_vertices(self):
        for v in self.grid.get_all_vertices():
            v.set_data(self.weights[self.index_map[v.id]])


def _as_2d(dataset) -> ndarray:
    x = np.asarray(dataset, dtype=float)
    return x[:, None] if x.ndim == 1 else x


class _DeviceData:
    """
    A dataset prepared for the device: centred (distances do not change, float32 stays accurate),
    missing values zeroed with a 0/1 mask beside them, cut into row blocks, and uploaded once when
    it fits the backend's cache (otherwise each block is uploaded when it is used).
    """

    def __init__(self, b: Backend, x: ndarray, centre: ndarray, rows: int):
        self.b = b
        self.n = len(x)
        mask = ~np.isnan(x)
        self.has_missing = not bool(mask.all())
        self.centre = np.nan_to_num(np.asarray(centre, dtype=float))
        self.x0 = np.where(mask, x - self.centre, 0.0)
        self.mask = mask.astype(float) if self.has_missing else None
        cols = x.shape[1] if x.ndim > 1 else 1
        self.rows = int(rows)
        self.slices = b.blocks(self.n, self.rows)
        self.resident = b.fits_cache(self.n, cols * (2 if self.has_missing else 1))
        if self.resident:
            self.x_dev = b.asarray(self.x0)
            self.m_dev = b.asarray(self.mask) if self.has_missing else None
        self._ones = None

    def get(self, blk: slice):
        """(x block, mask block or None) on the device."""
        if self.resident:
            return self.x_dev[blk], (self.m_dev[blk] if self.has_missing else None)
        return self.b.asarray(self.x0[blk]), (self.b.asarray(self.mask[blk]) if self.has_missing else None)

    def ones(self, blk: slice):
        n = blk.stop - blk.start
        if self._ones is None or self._ones.shape[0] < n:
            self._ones = self.b.zeros((max(n, self.rows), 1)) + 1.0
        return self._ones[:n]


class _WeightTerms:
    """The parts of |x - w|^2 = |x|^2 - 2 x.w + |w|^2 that depend only on the weights (device arrays)."""

    def __init__(self, b: Backend, w):
        self.b = b
        self.wt = w.T
        self.w2 = b.sum(w * w, axis=1)                    # (n,)
        self.w2t = (w * w).T                              # (dim, n), for samples with missing values

    def sq_distances(self, xb, mb):
        b = self.b
        cross = xb @ self.wt
        known = (mb @ self.w2t) if mb is not None else self.w2[None, :]
        d = b.sum(xb * xb, axis=1)[:, None] - 2.0 * cross + known
        return b.maximum(d, 0.0)
