# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""The SOMs give the same answers on every compute backend (GPU backends are tested where present)."""
import pickle

import numpy as np
import pytest
from mt.geodesicdome import backend as bk
from mt.geodesicdome.backend import NumpyBackend, _ArrayModuleBackend, get_backend
from mt.geodesicdome.grid.plane import Topology

from mt.geodesicsom.GeodesicSOM import GeodesicSOM
from mt.geodesicsom.PlaneSOM import PlaneSOM


def _backends():
    found = [('numpy-1', NumpyBackend(threads=1), 1e-9), ('numpy-4', NumpyBackend(threads=4), 1e-9),
             # a numpy-interface backend (as CuPy) with tiny memory: data streamed, distances never cached
             ('streamed', _ArrayModuleBackend(np, 'cpu', np.float64, False, 1, 1 << 14, 0), 1e-9)]
    for spec in bk.available_backends():
        if spec != 'numpy':
            b = get_backend(spec)
            found.append((spec, b, 1e-9 if b.dtype == np.float64 else 2e-3))
    if 'torch:cpu' in bk.available_backends():
        found.append(('torch-float32', get_backend('torch:cpu', dtype='float32'), 2e-3))
    return found


BACKENDS = _backends()


def data(missing: bool):
    rng = np.random.default_rng(0)
    x = np.vstack([c + 0.5 * rng.standard_normal((80, 5)) for c in rng.normal(scale=4, size=(4, 5))]) + 50.0
    if missing:
        x[rng.random(x.shape) < 0.1] = np.nan
    return x


def make(kind, b):
    if kind == 'sphere':
        return GeodesicSOM(5, seed=2, backend=b)
    return PlaneSOM(7, 9, topology=Topology.Donut, seed=2, backend=b)


@pytest.mark.parametrize('kind', ['sphere', 'torus'])
@pytest.mark.parametrize('missing', [False, True])
@pytest.mark.parametrize('mode', ['batch', 'online'])
@pytest.mark.parametrize('name,b,tol', BACKENDS, ids=[n for n, _, _ in BACKENDS])
def test_same_result_on_every_backend(name, b, tol, mode, missing, kind):
    x = data(missing)
    epochs = 5 if mode == 'batch' else 2
    ref = make(kind, NumpyBackend(threads=1)).train(x, epochs=epochs, mode=mode)
    som = make(kind, b).train(x, epochs=epochs, mode=mode)
    scale = np.nanmax(np.abs(x - np.nanmean(x, axis=0)))
    assert np.allclose(som.weights, ref.weights, atol=tol * scale * (50 if mode == 'online' else 5))
    assert np.allclose(som.history, ref.history, rtol=tol * 10)
    if tol < 1e-6:
        assert np.array_equal(som.bmu(x), ref.bmu(x))
        assert np.array_equal(som.bmu(x, second=True), ref.bmu(x, second=True))
    else:
        assert (som.bmu(x) == ref.bmu(x)).mean() > 0.97
    assert np.allclose(som.distances(x[:7]), ref.distances(x[:7]), rtol=tol * 10, atol=tol * scale ** 2)
    assert som.quantisation_error(x) == pytest.approx(ref.quantisation_error(x), rel=tol * 10)
    assert abs(som.topographic_error(x) - ref.topographic_error(x)) < (1e-12 if tol < 1e-6 else 0.03)


@pytest.mark.parametrize('name,b,tol', BACKENDS, ids=[n for n, _, _ in BACKENDS])
def test_node_distance_is_built_on_demand(name, b, tol):
    som = GeodesicSOM(4, backend=b)
    assert som._node_distance is None                               # not needed to build the map
    p = som.points
    expected = np.arccos(np.clip(p @ p.T, -1, 1)) / som.ring_length
    assert np.allclose(som.node_distance, expected, atol=max(tol, 1e-5) * 10)
    assert som.default_sigma == pytest.approx(expected.max() / 4, rel=1e-4)
    plane = PlaneSOM(4, 6, topology=Topology.Donut, backend=b)
    d = plane.node_distance
    assert d.shape == (24, 24) and np.allclose(d, d.T) and np.allclose(np.diag(d), 0)
    assert np.allclose(d[plane.edges[:, 0], plane.edges[:, 1]], 1.0, atol=1e-5)


def test_switching_backend_and_pickling():
    x = data(False)
    som = GeodesicSOM(4, seed=1, backend='numpy').train(x, epochs=3)
    assert som.backend.name == 'numpy'
    q = som.quantisation_error(x)
    assert som.to('cpu').quantisation_error(x) == pytest.approx(q)
    clone = pickle.loads(pickle.dumps(som))
    assert np.array_equal(clone.bmu(x), som.bmu(x))


def test_dense_distance_lattices_still_work():
    """A subclass may still hand _set_lattice a full matrix (as mtDeepGeodesicSOM's LineSOM does)."""
    som = PlaneSOM(3, 4)
    som._set_lattice(som.faces, som.node_distance, som.index_map, som.init_coords, som.edges)
    som.train(data(False), epochs=3)
    assert som.weights.shape == (12, 5)
    old_max = float(som.node_distance.max())
    som.node_distance = som.node_distance * 2.0
    assert som.distance.max() == pytest.approx(2.0 * old_max)


def test_empty_dataset_queries():
    som = GeodesicSOM(2, backend='numpy')
    som.initialise(dim=3)
    assert som.bmu(np.zeros((0, 3))).shape == (0,)
    assert som.bmu(np.zeros((0, 3)), second=True).shape == (0, 2)
    assert som.distances(np.zeros((0, 3))).shape == (0, som.n_nodes)
