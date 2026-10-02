# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
import numpy as np
import pytest
from mt.geodesicdome.grid.geodesicdome import GeodesicDome
from mt.geodesicdome.grid.plane import Lattice, Topology

from mt.geodesicsom.GeodesicSOM import GeodesicSOM
from mt.geodesicsom.neuron import Neuron
from mt.geodesicsom.PlaneSOM import PlaneSOM
from mt.geodesicsom.som import SOM, InitializationType
from mt.geodesicsom.util import MISSING


def test_som_is_abstract():
    with pytest.raises(TypeError):
        SOM(GeodesicDome(1))
    assert isinstance(GeodesicSOM(GeodesicDome(1)), SOM)


@pytest.mark.parametrize("lattice", [Lattice.Hexagonal, Lattice.Rectilinear])
@pytest.mark.parametrize("topology", [Topology.Plane, Topology.Donut])
def test_plane_som_grid(lattice, topology):
    som = PlaneSOM(row=6, col=8, dim=3, lattice=lattice, topology=topology)
    assert (som.row, som.col, som.dim) == (6, 8, 3)
    assert len(som.grid.get_all_vertices()) == 6 * 8
    assert som.grid.lattice is lattice and som.grid.topology is topology


def test_plane_som_random_initialise_without_data():
    som = PlaneSOM(4, 5, 3, seed=0)
    som.initialise(InitializationType.Random)
    w = np.array([v.data for v in som.grid.get_all_vertices()])
    assert w.shape == (20, 3)
    assert np.all((w >= 0) & (w <= 1))


def test_plane_som_random_initialise_within_data_range():
    data = np.random.default_rng(1).uniform([-5, 10, 0], [-1, 20, 0.5], size=(100, 3))
    som = PlaneSOM(4, 5, 3, seed=0)
    som.initialise(InitializationType.Random, data)
    w = np.array([v.data for v in som.grid.get_all_vertices()])
    assert np.all(w >= data.min(0)) and np.all(w <= data.max(0))


def test_plane_som_linear_initialise_spans_the_principal_plane():
    rng = np.random.default_rng(2)
    data = rng.normal(size=(300, 2)) @ np.array([[3.0, 0.0, 1.0], [0.0, 1.0, 0.0]])   # a 2-D sheet in 3-D
    som = PlaneSOM(6, 9, seed=0)
    som.initialise(InitializationType.Linear, data)
    assert som.weights.shape == (54, 3)
    centred = som.weights - som.weights.mean(0)
    assert np.linalg.svd(centred, compute_uv=False)[2] < 1e-9      # the initial map is flat, like the data


@pytest.mark.parametrize("lattice, degree", [(Lattice.Hexagonal, 6), (Lattice.Rectilinear, 4)])
def test_plane_som_neighbours(lattice, degree):
    flat = PlaneSOM(6, 8, lattice=lattice)
    torus = PlaneSOM(6, 8, lattice=lattice, topology=Topology.Donut)
    assert max(len(n) for n in flat.neighbours) == degree and min(len(n) for n in flat.neighbours) < degree
    assert {len(n) for n in torus.neighbours} == {degree}              # no border on a torus
    for som in (flat, torus):
        assert np.allclose(som.node_distance[som.edges[:, 0], som.edges[:, 1]], 1.0)
    assert torus.node_distance.max() < flat.node_distance.max()
    assert flat.faces.shape[1] == (3 if lattice == Lattice.Hexagonal else 4)


@pytest.mark.parametrize("topology", [Topology.Plane, Topology.Donut])
def test_plane_som_training(topology):
    rng = np.random.default_rng(0)
    centres = rng.normal(scale=4, size=(4, 5))
    x = np.vstack([c + 0.3 * rng.standard_normal((50, 5)) for c in centres])
    som = PlaneSOM(8, 10, topology=topology, seed=0)
    som.initialise(InitializationType.Random, x)
    before = som.quantisation_error(x)
    som.train(x, epochs=15)
    assert som.quantisation_error(x) < 0.5 * before and som.topographic_error(x) < 0.1
    assert som.face_distance().shape == (len(som.faces),)
    assert som.face_component_difference().shape == (len(som.faces), 5)
    for v in som.grid.get_all_vertices():
        assert np.array_equal(v.data, som.weights[v.id])


def test_neuron_construction():
    v = GeodesicDome(1).get_all_vertices()[0]
    assert Neuron(v).weights is None and Neuron(v).dimension == 0
    n = Neuron(v, dimension=4)
    assert n.dimension == 4 and np.array_equal(n.weights, np.zeros(4))
    n = Neuron(v, weights=np.array([1.0, 2.0]))
    assert n.dimension == 2 and n.vertex is v


def test_neuron_distance_skips_missing():
    v = GeodesicDome(1).get_all_vertices()[0]
    a = Neuron(v, weights=np.array([0.0, 0.0, 0.0]))
    b = Neuron(v, weights=np.array([3.0, MISSING, 4.0]))
    assert a.distance(b) == pytest.approx(5.0)
