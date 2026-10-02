# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""PlaneSOMViewer and PlaneComponentMatrixViewer, driven headlessly (Agg)."""
import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
from matplotlib.backend_bases import KeyEvent, MouseEvent  # noqa: E402
from mt.geodesicdome.grid.plane import Lattice, Topology  # noqa: E402

from mt.geodesicsom.GeodesicSOM import GeodesicSOM  # noqa: E402
from mt.geodesicsom.gui import (  # noqa: E402
    PlaneComponentMatrixViewer,
    PlaneSOMViewer,
    SOMViewer,
    explore,
    link_views,
)
from mt.geodesicsom.PlaneSOM import PlaneSOM  # noqa: E402


@pytest.fixture(scope="module", params=[(Lattice.Hexagonal, Topology.Plane), (Lattice.Rectilinear, Topology.Donut)])
def trained(request):
    lattice, topology = request.param
    rng = np.random.default_rng(0)
    centres = rng.normal(scale=3, size=(3, 4))
    x = np.vstack([c + 0.4 * rng.standard_normal((40, 4)) for c in centres])
    y = np.repeat(["a", "b", "c"], 40)
    return PlaneSOM(8, 12, lattice=lattice, topology=topology, seed=0).train(x, epochs=8), x, y


def _click(viewer, ax, xy, dblclick=False):
    px, py = ax.transData.transform(xy)
    for name in ("button_press_event", "button_release_event"):
        viewer.fig.canvas.callbacks.process(name, MouseEvent(name, viewer.fig.canvas, px, py, button=1,
                                                             dblclick=dblclick and name == "button_press_event"))


def test_plane_viewer_layers_and_click(trained):
    som, x, y = trained
    v = PlaneSOMViewer(som, x, labels=y, feature_names=list("abcd"))
    assert "Classes" in v.layers and "Component difference" in v.layers
    for layer in v.layers:
        v.set_layer(layer)
        assert len(v._faces.get_facecolor()) == len(som.faces)
    v.set_layer("Neighbour distance")
    assert v.layer_norm.vmax == pytest.approx(som.face_distance().max())
    _click(v, v.ax, som.positions[17])
    assert v.selected == 17 and "column 5" in v._detail_text.get_text()
    canvas = v.fig.canvas
    canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", canvas, "]"))
    assert v.layer == "Component" and v.component == 1


def test_plane_matrix_click_modes_and_double_click(trained):
    som, x, y = trained
    m = PlaneComponentMatrixViewer(som, list("abcd"), data=x, labels=y)
    assert len(m.panels) == som.dim + 1
    _click(m, m.panels[2]["ax"], som.positions[30])
    assert m.selected == 30 and all(p["marker"].get_visible() for p in m.panels)
    assert m.panels[2]["title"].get_text().startswith("b  = ")
    canvas = m.fig.canvas
    canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", canvas, "v"))
    assert m.mode == "value"
    assert m.panels[1]["colourbar"].mappable.norm.vmax == pytest.approx(som.face_component_value()[:, 0].max())
    px, py = m.panels[3]["ax"].transData.transform(som.positions[0])
    canvas.callbacks.process("button_press_event",
                             MouseEvent("button_press_event", canvas, px, py, button=1, dblclick=True))
    (child,) = m.children
    assert child.layer == "Component difference" and child.component == 2
    child.select_node(5)                                    # selection is linked both ways
    assert m.selected == 5
    m.select_node(9)
    assert child.selected == 9


def test_explore_plane_som(trained):
    som, x, y = trained
    views = explore(som, x, y, show=False, select=4)
    assert isinstance(views.map, PlaneSOMViewer) and isinstance(views.matrix, PlaneComponentMatrixViewer)
    assert views.map.selected == 4 and views.matrix.selected == 4
    views.matrix.select_node(11)
    assert views.map.selected == 11


def test_wrong_som_type_is_rejected(trained):
    som, _, _ = trained
    with pytest.raises(TypeError):
        SOMViewer(som)
    sphere = GeodesicSOM(2).train(np.random.default_rng(0).random((30, 3)), epochs=2)
    with pytest.raises(TypeError):
        PlaneSOMViewer(sphere)


def test_selection_links_sphere_views():
    x = np.random.default_rng(1).random((60, 3))
    som = GeodesicSOM(4).train(x, epochs=3)
    a, b = SOMViewer(som, x), SOMViewer(som, x)
    link_views(a, b)
    a.select_node(12)
    assert b.selected == 12


def test_node_labels_on_both_viewers(trained):
    som, x, y = trained
    v = PlaneSOMViewer(som, x, labels=y, node_labels="counts")
    expected = [t for t in som.bmu_labels(x, y, "counts") if t is not None]
    assert sorted(t.get_text() for t in v._label_texts) == sorted(expected)
    i = int(v._label_nodes[0])
    assert np.allclose(v._label_texts[0].get_position(), som.positions[i])
    v.show_node_labels(False)                              # off, but the style is kept
    assert v._label_texts == [] and not v.node_labels_on and v.node_label_mode == "counts"
    v.toggle_node_labels()
    assert v.node_labels_on and v._label_texts[0].get_text() in expected
    v.set_node_labels(None)                                # None also turns them off
    assert not v.node_labels_on
    names = [f"n{k}" for k in range(len(x))]
    v.set_node_labels("first", sample_names=names)
    assert all(t.get_text().startswith("n") for t in v._label_texts)
    with pytest.raises(ValueError):
        v.set_node_labels("nope")
    with pytest.raises(ValueError):
        PlaneSOMViewer(som, node_labels="majority")        # no data


def _key(viewer, key):
    viewer.fig.canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", viewer.fig.canvas, key))


def _click_widget(viewer, ax, xy_axes):
    px, py = ax.transAxes.transform(xy_axes)
    for name in ("button_press_event", "button_release_event"):
        viewer.fig.canvas.callbacks.process(name, MouseEvent(name, viewer.fig.canvas, px, py, button=1))


def test_label_and_sample_switches(trained):
    som, x, y = trained
    v = PlaneSOMViewer(som, x, labels=y)
    checks = dict(zip([t.get_text() for t in v._checks.labels], v._checks.get_status(), strict=True))
    assert checks == {"sample dots": True, "neuron labels": False}
    _key(v, "l")                                           # l: on/off
    assert v.node_labels_on and v.node_label_mode == "majority"
    assert v._checks.get_status() == [True, True]          # the check box follows the key
    _key(v, "L")                                           # L: next style
    assert v.node_label_mode == "all" and v._style_button.label.get_text() == "all · 7pt"
    _key(v, "l")
    assert not v.node_labels_on and v._checks.get_status() == [True, False]
    _key(v, "s")
    assert not v.show_samples and not v._samples.get_visible() and v._checks.get_status() == [False, False]
    # the widgets themselves
    v._checks.set_active(1)                                # tick 'neuron labels'
    assert v.node_labels_on and len(v._label_texts) > 0
    v._checks.set_active(0)                                # tick 'sample dots'
    assert v.show_samples
    _click_widget(v, v._style_button.ax, (0.5, 0.5))
    assert v.node_label_mode == "counts"


def test_switches_without_labels(trained):
    som, x, _ = trained
    v = PlaneSOMViewer(som, x)                            # data but no labels: only the sample-dots switch
    assert [t.get_text() for t in v._checks.labels] == ["sample dots"] and v._style_button is None
    _key(v, "l")                                           # nothing to label: ignored
    assert not v.node_labels_on
    assert PlaneSOMViewer(som)._checks is None             # no data: no switches


def test_node_labels_rotate_with_the_sphere():
    x = np.random.default_rng(2).random((80, 3))
    y = np.where(x[:, 0] > 0.5, "big", "small")
    som = GeodesicSOM(4, seed=0).train(x, epochs=4)
    v = SOMViewer(som, x, labels=y, node_labels="majority")
    assert len(v._label_texts) == int((som.hits(x) > 0).sum())
    v.rotate(d_lon=40, d_lat=10)
    assert np.allclose(v._label_texts[0].get_position(), v._project(som.points[v._label_nodes[:1]])[0])
    v.show_node_labels(False)
    assert v._label_texts == []
    v.rotate(d_lon=10)                                     # rotating with labels off is fine
    views = explore(som, x, y, node_labels="all", show=False)
    assert views.map.node_label_mode == "all" and views.map.node_labels_on
    sv = SOMViewer(som, x, sample_names=[f"id{k}" for k in range(len(x))])
    assert sv.node_labels_on and sv.node_label_mode == "first"
    _key(sv, "L")                                          # with sample names, 'first' is part of the cycle
    assert sv.node_label_mode == "majority"


def test_label_size(trained):
    som, x, y = trained
    v = PlaneSOMViewer(som, x, labels=y, node_labels="majority", label_size=9)
    assert v.label_size == 9 and all(t.get_fontsize() == 9 for t in v._label_texts)
    v.set_label_size(12)                                   # resized in place
    assert all(t.get_fontsize() == 12 for t in v._label_texts)
    assert v._style_button.label.get_text() == "majority · 12pt"
    _key(v, "-")
    assert v.label_size == 10.0
    _key(v, "=")                                           # '=' works as '+' without shift
    assert v.label_size == 12.0
    _click_widget(v, v._size_buttons[1].ax, (0.5, 0.5))    # A+
    assert v.label_size == 14.4 and v._label_texts[0].get_fontsize() == 14.4
    v.set_label_size(100)
    assert v.label_size == 30.0                            # kept between 3 and 30
    v.show_node_labels(False)
    v.set_label_size(5)                                    # applies when they come back
    v.show_node_labels(True)
    assert all(t.get_fontsize() == 5 for t in v._label_texts)
    views = explore(som, x, y, node_labels="counts", label_size=6, show=False)
    assert views.map.label_size == 6
