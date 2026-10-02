# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
import numpy as np
import pytest

from mt.geodesicsom import datasets

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from mt.geodesicsom.GeodesicSOM import GeodesicSOM  # noqa: E402
from mt.geodesicsom.gui.explorer import explore, main, train_and_explore  # noqa: E402


def test_datasets(tmp_path):
    ds = datasets.clusters(n_clusters=3, dim=4, per_cluster=10)
    assert ds.data.shape == (30, 4) and len(ds.labels) == 30 and len(ds.feature_names) == 4
    assert datasets.colours(n=20).data.shape == (20, 3)
    z = datasets.standardise(ds.data)
    assert np.allclose(z.mean(0), 0) and np.allclose(z.std(0), 1)

    path = tmp_path / "d.csv"
    path.write_text("a,b,name,kind\n1,2,x,p\n3,,y,q\n5,6,z,p\n")
    d = datasets.read_csv(str(path), label_column="kind", verbose=False)
    assert d.feature_names == ["a", "b"] and list(d.labels) == ["p", "q", "p"]
    assert np.isnan(d.data[1, 1]) and d.data[2, 0] == 5
    with pytest.raises(ValueError):
        datasets.read_csv(str(path), label_column="missing")


def test_explore_opens_linked_views():
    ds = datasets.clusters(n_clusters=3, dim=4, per_cluster=20)
    x = datasets.standardise(ds.data)
    som = GeodesicSOM(5, seed=0).train(x, epochs=5)
    views = explore(som, x, ds.labels, ds.feature_names, select=3, show=False)
    assert views.map.selected == 3 and views.matrix.selected == 3
    views.map.rotate(d_lon=25)
    assert np.allclose(views.matrix.rotation, views.map.rotation)
    views.matrix.rotate(d_lat=10)
    assert np.allclose(views.matrix.rotation, views.map.rotation)
    only = explore(som, x, single=False, show=False)
    assert only.map is None and only.matrix is not None


def test_train_and_explore():
    ds = datasets.clusters(n_clusters=3, dim=4, per_cluster=20)
    som, views = train_and_explore(ds.data, ds.labels, ds.feature_names, frequency=4, epochs=5,
                                   verbose=False, show=False)
    assert som.weights.shape == (162, 4) and views.map is not None and views.matrix is not None


def test_command_line(tmp_path):
    prefix = tmp_path / "out" / "demo"
    views = main(["--freq", "4", "--epochs", "3", "--save", str(prefix), "--no-gui"])
    assert (tmp_path / "out" / "demo_map.png").stat().st_size > 0
    assert (tmp_path / "out" / "demo_matrix.png").stat().st_size > 0
    assert views.map.selected is not None
    views = main(["--data", "colours", "--freq", "3", "--epochs", "2", "--only", "matrix", "--no-gui"])
    assert views.map is None and len(views.matrix.panels) == 4


@pytest.mark.parametrize("name, shape, n_labels", [("animals", (16, 13), 16), ("clusters", (560, 12), 7),
                                                   ("iris", (150, 4), 3),
                                                   ("penguins", (342, 4), 3), ("wine", (178, 13), 3)])
def test_sample_data(name, shape, n_labels):
    ds = datasets.load_sample(name)
    assert ds.data.shape == shape and len(set(ds.labels)) == n_labels and len(ds.feature_names) == shape[1]
    assert not np.isnan(ds.data).all(axis=1).any()
    assert name in datasets.samples()
    assert datasets.sample_path(name).endswith(".csv")
    assert datasets.load(name).data.shape == shape


def test_kohonen_animals():
    ds = datasets.animals()
    assert set(np.unique(ds.data)) == {0.0, 1.0}
    row = dict(zip(ds.labels, ds.data, strict=True))
    names = list(ds.feature_names)
    assert row["duck"][names.index("swim")] == 1 and row["cow"][names.index("hooves")] == 1
    assert (ds.data[:, :3].sum(axis=1) == 1).all()                    # exactly one of small / medium / big
    assert np.array_equal(row["owl"], row["hawk"]) and np.array_equal(row["horse"], row["zebra"])

    sem = datasets.animals(symbols=True, symbol_weight=0.2)
    assert sem.data.shape == (16, 29) and sem.feature_names[0] == "is dove" and sem.feature_names[16] == "small"
    assert np.allclose(sem.data[:, :16], 0.2 * np.eye(16))
    assert np.allclose(np.linalg.norm(sem.data[:, 16:], axis=1), 1.0)
    assert len({tuple(r) for r in sem.data}) == 16                     # the symbol part tells every animal apart


def test_kohonen_animals_example(tmp_path):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parents[1] / "examples" / "05_kohonen_animals.py"
    spec = importlib.util.spec_from_file_location("kohonen_animals", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    views = module.main(["--freq", "3", "--epochs", "5", "--no-gui", "--save", str(tmp_path / "animals")])
    assert (tmp_path / "animals_map.png").stat().st_size > 0 and len(views.matrix.panels) == 14
    views = module.main(["--symbols", "--lattice", "hexagonal", "--epochs", "5", "--no-gui"])
    assert views.matrix is None and views.map.node_labels_on


def test_sample_clusters_match_the_generator():
    assert np.allclose(datasets.load_sample("clusters").data, datasets.clusters().data, rtol=1e-5, atol=1e-4)
    with pytest.raises(ValueError):
        datasets.load_sample("nope")


def test_command_line_with_a_sample(tmp_path):
    views = main(["--data", "wine", "--freq", "3", "--epochs", "2", "--only", "matrix", "--no-gui"])
    assert len(views.matrix.panels) == 13 + 1 and views.matrix.panels[1]["name"] == "alcohol"
