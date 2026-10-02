# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
import numpy as np
import pytest

from mt.geodesicsom.GeodesicSOM import GeodesicSOM
from mt.geodesicsom.som import InitializationType


def clusters(seed=0, k=4, dim=6, n=60):
    rng = np.random.default_rng(seed)
    centres = rng.normal(scale=4, size=(k, dim))
    x = np.vstack([c + 0.3 * rng.standard_normal((n, dim)) for c in centres])
    return x, np.repeat(np.arange(k), n)


def test_structure():
    som = GeodesicSOM(4)
    assert som.n_nodes == 162 and len(som.faces) == 320 and len(som.edges) == 480
    assert sorted({len(n) for n in som.neighbours}) == [5, 6]
    assert np.allclose(som.node_distance[som.edges[:, 0], som.edges[:, 1]], 1.0, atol=0.25)


@pytest.mark.parametrize("mode", ["batch", "online"])
def test_training_reduces_error(mode):
    x, _ = clusters()
    som = GeodesicSOM(6, seed=1)
    som.initialise(InitializationType.Random, x)
    before = som.quantisation_error(x)
    som.train(x, epochs=10 if mode == "batch" else 3, mode=mode)
    assert som.weights.shape == (som.n_nodes, x.shape[1])
    assert som.quantisation_error(x) < 0.5 * before
    assert som.topographic_error(x) < 0.1
    assert len(som.history) == (10 if mode == "batch" else 3)


def test_weights_stored_on_every_vertex_including_seams():
    x, _ = clusters()
    som = GeodesicSOM(4).train(x, epochs=5)
    for v in som.dome.get_all_vertices():
        assert np.array_equal(v.data, som.weights[som.index_map[v.id]])
    assert som.vertex_weights().shape == (len(som.dome.get_all_vertices()), x.shape[1])


def test_distances_between_neurons():
    x, _ = clusters()
    som = GeodesicSOM(4).train(x, epochs=5)
    w = som.weights
    a, b = som.edges[0]
    assert som.edge_distance()[0] == pytest.approx(np.linalg.norm(w[a] - w[b]))
    f = som.faces[0]
    expected = (np.linalg.norm(w[f[0]] - w[f[1]]) + np.linalg.norm(w[f[1]] - w[f[2]])
                + np.linalg.norm(w[f[2]] - w[f[0]])) / 3
    assert som.face_distance()[0] == pytest.approx(expected)
    assert som.face_distance().shape == (len(som.faces),)
    assert som.u_matrix().shape == (som.n_nodes,)


def test_hits_bmu_and_labels():
    x, y = clusters()
    som = GeodesicSOM(4).train(x, epochs=5)
    assert som.hits(x).sum() == len(x)
    best2 = som.bmu(x, second=True)
    assert np.array_equal(best2[:, 0], som.bmu(x))
    labels = som.node_labels(x, y)
    assert sum(sum(d.values()) for d in labels) == len(x)


def test_missing_values_are_ignored():
    x, _ = clusters()
    xm = x.copy()
    xm[::4, 1] = np.nan
    som = GeodesicSOM(4).train(xm, epochs=5)
    assert not np.isnan(som.weights).any()
    w = som.weights[0]
    sample = np.array([np.nan] + list(w[1:] + 1.0))
    assert som.distances(sample[None])[0, 0] == pytest.approx((x.shape[1] - 1) * 1.0)


def test_random_initialise_without_data_needs_dim():
    som = GeodesicSOM(2)
    with pytest.raises(ValueError):
        som.initialise(InitializationType.Random)
    som.initialise(InitializationType.Random, dim=5)
    assert som.weights.shape == (42, 5)


def test_bmu_labels():
    x, y = clusters()
    som = GeodesicSOM(4, seed=0).train(x, epochs=5)
    names = np.array([f"s{i}" for i in range(len(x))])
    majority = som.bmu_labels(x, y)
    counts = som.bmu_labels(x, y, "counts")
    hits = som.hits(x)
    assert len(majority) == som.n_nodes
    assert all((t is None) == (h == 0) for t, h in zip(majority, hits, strict=True))
    busiest = int(np.argmax(hits))
    won = y[som.bmu(x) == busiest]
    values, n = np.unique(won, return_counts=True)
    assert majority[busiest] == str(values[np.argmax(n)])
    assert counts[busiest].startswith(f"{values[np.argmax(n)]}×{n.max()}")
    first = som.bmu_labels(x, names, "first")
    assert first[busiest] in set(names[som.bmu(x) == busiest])
    with pytest.raises(ValueError):
        som.bmu_labels(x, y, "nope")
    with pytest.raises(ValueError):
        som.bmu_labels(x, y[:5])
