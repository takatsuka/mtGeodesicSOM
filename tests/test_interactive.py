# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""SOMViewer, driven headlessly (Agg) with synthetic mouse and key events."""
import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
from matplotlib.backend_bases import KeyEvent, MouseEvent  # noqa: E402

from mt.geodesicsom.GeodesicSOM import GeodesicSOM  # noqa: E402
from mt.geodesicsom.gui import SOMViewer  # noqa: E402


@pytest.fixture(scope="module")
def trained():
    rng = np.random.default_rng(0)
    centres = rng.normal(scale=3, size=(3, 5))
    x = np.vstack([c + 0.4 * rng.standard_normal((40, 5)) for c in centres])
    y = np.repeat(["a", "b", "c"], 40)
    return GeodesicSOM(5, seed=0).train(x, epochs=8), x, y


def _mouse(viewer, name, x, y):
    viewer.fig.canvas.callbacks.process(name, MouseEvent(name, viewer.fig.canvas, x, y, button=1))


def test_all_layers(trained, tmp_path):
    som, x, y = trained
    v = SOMViewer(som, x, labels=y, feature_names=list("vwxyz"))
    assert v.layers == ["Neighbour distance", "U-matrix (neurons)", "Component", "Component difference",
                        "PCA colour", "Hits", "Classes"]
    for layer in v.layers:
        v.set_layer(layer)
        assert v.face_rgba.shape == (len(som.faces), 4)
    v.set_layer("Neighbour distance")
    assert v.layer_norm.vmin == pytest.approx(som.face_distance().min())
    v.save(tmp_path / "view.png")
    assert (tmp_path / "view.png").stat().st_size > 0


def test_weights_only(trained):
    som, _, _ = trained
    v = SOMViewer(som)
    assert v.layers == ["Neighbour distance", "U-matrix (neurons)", "Component", "Component difference", "PCA colour"]


def test_click_selects_the_neuron_under_the_pointer(trained):
    som, x, y = trained
    v = SOMViewer(som, x, labels=y)
    target = 17
    xy = v._project(som.points[[target]])[0]
    px, py = v.ax.transData.transform(xy)
    _mouse(v, "button_press_event", px, py)
    _mouse(v, "button_release_event", px, py)
    assert v.selected == target


def test_drag_rotates_without_selecting(trained):
    som, x, y = trained
    v = SOMViewer(som, x, labels=y)
    px, py = v.ax.transData.transform([0.0, 0.0])
    _mouse(v, "button_press_event", px, py)
    _mouse(v, "motion_notify_event", px + 60, py)
    _mouse(v, "button_release_event", px + 60, py)
    assert v.selected is None
    assert abs(v.centre[1]) > 5


def test_component_keys(trained):
    som, x, _ = trained
    v = SOMViewer(som, x)
    canvas = v.fig.canvas
    canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", canvas, "]"))
    assert v.layer == "Component" and v.component == 1
    canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", canvas, "["))
    canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", canvas, "["))
    assert v.component == som.dim - 1


def test_needs_trained_som():
    with pytest.raises(ValueError):
        SOMViewer(GeodesicSOM(2))


# ------------------------------------------------------------------ ComponentMatrixViewer / link_views
from mt.geodesicsom.gui import ComponentMatrixViewer, link_views  # noqa: E402


def test_face_component_difference(trained):
    som, _, _ = trained
    d = som.face_component_difference()
    assert d.shape == (len(som.faces), som.dim)
    f, k = som.faces[3], 2
    w = som.weights[:, k]
    assert d[3, k] == pytest.approx((abs(w[f[0]] - w[f[1]]) + abs(w[f[1]] - w[f[2]]) + abs(w[f[2]] - w[f[0]])) / 3)


def test_matrix_panels_and_colours(trained):
    som, x, y = trained
    m = ComponentMatrixViewer(som, list("vwxyz"), data=x, labels=y)
    assert len(m.panels) == som.dim + 1 and m.panels[0]["attribute"] is None
    diff = som.face_component_difference()
    p = m.panels[3]                                        # attribute 2
    assert p["colourbar"].mappable.norm.vmax == pytest.approx(diff[:, 2].max())
    m.set_mode("value")
    assert p["colourbar"].mappable.norm.vmax == pytest.approx(som.face_component_value()[:, 2].max())
    m.set_shared_scale(True)
    assert m.panels[1]["colourbar"].mappable.norm.vmax == m.panels[5]["colourbar"].mappable.norm.vmax
    assert len(ComponentMatrixViewer(som, include_total=False).panels) == som.dim


def test_matrix_drag_in_any_panel_rotates_all(trained):
    som, _, _ = trained
    m = ComponentMatrixViewer(som)
    ax = m.panels[4]["ax"]
    px, py = ax.transData.transform([0.0, 0.0])
    _mouse(m, "button_press_event", px, py)
    for k in range(1, 6):
        _mouse(m, "motion_notify_event", px + 10 * k, py)
    _mouse(m, "button_release_event", px + 50, py)
    assert m.selected is None and abs(m.centre[1]) > 5
    # every panel was given the same, rotated polygons
    paths = [p["faces"].get_paths() for p in m.panels]
    assert all(len(ps) == len(paths[0]) for ps in paths)
    assert np.allclose(paths[-1][0].vertices, paths[0][0].vertices)


def test_matrix_click_selects_on_every_panel(trained):
    som, _, _ = trained
    m = ComponentMatrixViewer(som, list("vwxyz"))
    target = 40
    xy = m._project(som.points[[target]])[0]
    px, py = m.panels[2]["ax"].transData.transform(xy)
    _mouse(m, "button_press_event", px, py)
    _mouse(m, "button_release_event", px, py)
    assert m.selected == target
    assert m.panels[2]["title"].get_text().startswith("w  = ")
    assert all(p["marker"].get_visible() for p in m.panels)


def test_matrix_keys(trained):
    som, _, _ = trained
    m = ComponentMatrixViewer(som)
    canvas = m.fig.canvas
    for key, check in [("v", lambda: m.mode == "value"), ("c", lambda: m.shared_scale),
                       ("p", lambda: m.projection_name == "Kavrayskiy VII"), ("right", lambda: m.centre[1] != 0)]:
        canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", canvas, key))
        assert check(), key


def test_double_click_opens_linked_single_view(trained):
    som, x, y = trained
    m = ComponentMatrixViewer(som, data=x, labels=y)
    ax = m.panels[3]["ax"]
    px, py = ax.transData.transform([0.0, 0.0])
    m.fig.canvas.callbacks.process("button_press_event",
                                   MouseEvent("button_press_event", m.fig.canvas, px, py, button=1, dblclick=True))
    (v,) = m.children
    assert v.layer == "Component difference" and v.component == 2
    v.rotate(d_lon=30)
    assert np.allclose(v.rotation, m.rotation)
    m.rotate(d_lat=-20)
    assert np.allclose(v.rotation, m.rotation)


def test_link_views_between_separate_viewers(trained):
    som, x, _ = trained
    a, b = SOMViewer(som, x), SOMViewer(som, x, layer="Component difference")
    c = ComponentMatrixViewer(som)
    link_views(a, b, c)
    a.set_view(20, 45)
    assert np.allclose(b.rotation, a.rotation) and np.allclose(c.rotation, a.rotation)
