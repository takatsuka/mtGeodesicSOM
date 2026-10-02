# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
Interactive views of a trained PlaneSOM (flat hexagonal or rectilinear grid, or a torus).

    from mt.geodesicsom.PlaneSOM import PlaneSOM
    from mt.geodesicsom.gui import PlaneSOMViewer, PlaneComponentMatrixViewer

    som = PlaneSOM(20, 30).train(data)
    PlaneSOMViewer(som, data, labels=labels, feature_names=names).show()
    PlaneComponentMatrixViewer(som, feature_names=names).show()

PlaneSOMViewer has the same layers and neuron inspector as SOMViewer (see
mt.geodesicsom.gui.inspector): the faces between neighbouring neurons (triangles on a hexagonal
grid, squares on a rectilinear one) are coloured by the Euclidean distance between their attribute
vectors, and so on.  PlaneComponentMatrixViewer shows every attribute at once, one map each.

Mouse and keyboard
    click                inspect / mark the neuron under the pointer (on every map of the matrix)
    double-click         (matrix) open that attribute in a PlaneSOMViewer, selection linked
    [ / ]                previous / next attribute    s   show or hide the samples
    v / c                (matrix) |difference| <-> value / one colour scale for all maps <-> one each
    e                    show or hide the face edges
"""
import math
from collections.abc import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from matplotlib.widgets import Button
from mt.geodesicdome.grid.plane import Topology

from mt.geodesicsom.gui.inspector import LAYER_DISTANCE, SOMInspector
from mt.geodesicsom.PlaneSOM import PlaneSOM

MODES = ('difference', 'value')
_CMAP = {'difference': 'magma', 'value': 'coolwarm'}


def _limits(som: PlaneSOM, ax):
    lo, hi = som.positions.min(axis=0), som.positions.max(axis=0)
    ax.set_xlim(lo[0] - 0.8, hi[0] + 0.8)
    ax.set_ylim(hi[1] + 0.8, lo[1] - 0.8)                    # row 0 at the top, like a matrix
    ax.set_aspect('equal')
    ax.set_axis_off()


def _topology_text(som: PlaneSOM):
    return 'torus: the edges wrap around' if som.topology == Topology.Donut else 'plane with borders'


class PlaneSOMViewer(SOMInspector):
    """
    Interactive map of a trained PlaneSOM, with the same layers and inspector as SOMViewer.

    :param som: a trained PlaneSOM
    :param data, labels, feature_names, layer, component, show_samples, node_labels, sample_names,
           label_size: as for SOMViewer
    :param edges: draw the face edges (default: when there are at most 1 500 neurons)
    :param figsize, title: display options
    """

    def __init__(self, som: PlaneSOM, data=None, labels: Sequence | None = None,
                 feature_names: Sequence[str] | None = None, *, layer: str = LAYER_DISTANCE,
                 component: int = 0, show_samples: bool | None = None, node_labels: str | None = None,
                 sample_names: Sequence | None = None, label_size: float | None = None, edges: bool | None = None,
                 figsize=(16, 8), title: str | None = None):
        if not isinstance(som, PlaneSOM):
            raise TypeError('PlaneSOMViewer shows a PlaneSOM; use SOMViewer for a GeodesicSOM')
        self._init_inspector(som, data, labels, feature_names, component, show_samples, sample_names,
                             label_size)
        self.show_edges = som.n_nodes <= 1500 if edges is None else bool(edges)

        self.fig = plt.figure(figsize=figsize)
        manager = self.fig.canvas.manager
        if manager is not None:
            manager.set_window_title('mt.geodesicsom PlaneSOMViewer')
        self.fig.suptitle(title or f'PlaneSOM: {som.row} x {som.col} {som.lattice.name.lower()} grid '
                                   f'({_topology_text(som)}), {som.dim} attributes each', fontsize=13)
        self.ax = self.fig.add_axes([0.15, 0.08, 0.56, 0.84])
        _limits(som, self.ax)
        self._faces = PolyCollection(som.positions[som.faces], zorder=1)
        self.ax.add_collection(self._faces)
        self._build_inspector()

        if self.data is not None:
            jitter = np.random.default_rng(0).normal(scale=0.2, size=(len(self.data), 2))
            xy = np.clip(som.positions[self.bmu] + jitter, som.positions.min(axis=0),
                         som.positions.max(axis=0))            # keep the dots on the map
            self._samples = self.ax.scatter(xy[:, 0], xy[:, 1], s=7, c=self._sample_colours(),
                                            edgecolors='white', linewidths=0.3, zorder=4)
        else:
            self._samples = None
        self.fig.canvas.mpl_connect('key_press_event', self._on_plane_key)
        self._ready = True
        self.set_layer(layer)
        if node_labels is not None or sample_names is not None:
            self.set_node_labels(node_labels or self.node_label_mode)

    # ================================================================ public API
    def show(self):
        plt.show()

    def save(self, path, **kwargs):
        self.fig.savefig(path, **kwargs)

    def node_at(self, x: float, y: float) -> int:
        """The neuron nearest to map coordinates (x, y)."""
        return int(np.argmin(((self.som.positions - [x, y]) ** 2).sum(axis=1)))

    # ============================================================ SOMInspector
    def _show_layer_colours(self, values, cmap):
        rgba, norm = self._face_rgba(values, cmap)
        self._faces.set_facecolor(rgba)
        self._faces.set_edgecolor((1, 1, 1, 0.5) if self.show_edges else rgba)
        self._faces.set_linewidth(0.3 if self.show_edges else 0.25)
        self._rgba = rgba
        self.fig.canvas.draw_idle()
        return norm

    def _node_xy(self, nodes):
        return self.som.positions[nodes]

    def _node_position_text(self, i):
        return f'column {i % self.som.col}  row {i // self.som.col}'

    def _update_overlays(self):
        if not self._ready:
            return
        if self._samples is not None:
            self._samples.set_visible(self.show_samples)
        self._position_node_labels()
        if self.selected is not None:
            x, y = self.som.positions[self.selected]
            for m in (self._marker, self._marker_dot):
                m.set_data([x], [y])
                m.set_visible(True)
        else:
            self._marker.set_visible(False)
            self._marker_dot.set_visible(False)

    def _on_plane_key(self, event):
        if event.key == 'e':
            self.show_edges = not self.show_edges
            self._faces.set_edgecolor((1, 1, 1, 0.5) if self.show_edges else self._rgba)
            self.fig.canvas.draw_idle()


class PlaneComponentMatrixViewer:
    """
    One map per attribute of a trained PlaneSOM, in a grid.  In 'difference' mode each face between
    neighbouring neurons is coloured by the mean |w_a[k] - w_b[k]| of that attribute along its edges;
    'value' shows the attribute itself.  Clicking marks a neuron on every map.

    :param som: a trained PlaneSOM
    :param feature_names, mode, include_total, shared_scale, ncols, data, labels, edges, figsize, title:
           as for ComponentMatrixViewer
    """

    def __init__(self, som: PlaneSOM, feature_names: Sequence[str] | None = None, *, mode: str = 'difference',
                 include_total: bool = True, shared_scale: bool = False, ncols: int | None = None,
                 data=None, labels=None, edges: bool = False, figsize=None, title: str | None = None):
        if not isinstance(som, PlaneSOM):
            raise TypeError('PlaneComponentMatrixViewer shows a PlaneSOM; use ComponentMatrixViewer for a GeodesicSOM')
        if som.weights is None:
            raise ValueError('the PlaneSOM has no weights yet: initialise() and train() it first')
        if mode not in MODES:
            raise ValueError(f'mode must be one of {MODES}')
        self.som = som
        self.feature_names = list(feature_names) if feature_names is not None else \
            [f'attr {i}' for i in range(som.dim)]
        if len(self.feature_names) != som.dim:
            raise ValueError(f'{len(self.feature_names)} feature names for {som.dim} attributes')
        self.data, self.labels = data, labels
        self.mode = mode
        self.include_total = bool(include_total)
        self.shared_scale = bool(shared_scale)
        self.show_edges = bool(edges)
        self.selected: int | None = None
        self.children: list = []
        self._select_callbacks: list[Callable] = []
        self._values = {'difference': som.face_component_difference(), 'value': som.face_component_value()}
        self._total = som.face_distance()

        n_panels = som.dim + (1 if self.include_total else 0)
        aspect = (np.ptp(som.positions[:, 0]) + 1.6) / (np.ptp(som.positions[:, 1]) + 1.6)
        self.ncols = ncols or max(1, math.ceil(math.sqrt(n_panels * aspect * 0.9)))
        self.nrows = math.ceil(n_panels / self.ncols)
        panel_w = 3.6
        panel_h = panel_w / aspect * 0.86 + 0.35
        figsize = figsize or (panel_w * self.ncols, panel_h * self.nrows + 0.9)
        self.fig = plt.figure(figsize=figsize)
        manager = self.fig.canvas.manager
        if manager is not None:
            manager.set_window_title('mt.geodesicsom PlaneComponentMatrixViewer')
        self.fig.suptitle(title or f'PlaneSOM {som.row} x {som.col} ({_topology_text(som)}): one map per attribute',
                          fontsize=12, y=1.0 - 0.12 / figsize[1], va='top')

        top, bottom = 1.0 - 0.75 / figsize[1], 0.25 / figsize[1]
        cell_w, cell_h = 1.0 / self.ncols, (top - bottom) / self.nrows
        polygons = som.positions[som.faces]
        self.panels: list[dict] = []
        names = (['all attributes'] if self.include_total else []) + self.feature_names
        for i, name in enumerate(names):
            row, col = divmod(i, self.ncols)
            x0, y0 = col * cell_w, top - (row + 1) * cell_h
            ax = self.fig.add_axes([x0 + 0.005, y0 + 0.02 * cell_h, cell_w * 0.86, cell_h * 0.84])
            _limits(som, ax)
            cax = self.fig.add_axes([x0 + cell_w * 0.875, y0 + 0.14 * cell_h, cell_w * 0.025, cell_h * 0.6])
            faces = PolyCollection(polygons, zorder=1)
            ax.add_collection(faces)
            (marker,) = ax.plot([], [], marker='o', ms=8, mfc='none', mec='#00e5ff', mew=1.6, zorder=5)
            attribute = None if (self.include_total and i == 0) else i - (1 if self.include_total else 0)
            title_artist = ax.set_title(name, fontsize=8.5, pad=2)
            self.panels.append(dict(ax=ax, cax=cax, faces=faces, marker=marker, attribute=attribute, name=name,
                                    title=title_artist, rgba=None, colourbar=None))

        h = 0.34 / figsize[1]
        self._mode_button = Button(self.fig.add_axes([0.005, 1 - 0.5 / figsize[1], 0.11, h]), '')
        self._mode_button.label.set_fontsize(8)
        self._mode_button.on_clicked(lambda _e: self.set_mode('value' if self.mode == 'difference' else 'difference'))
        self._info = self.fig.text(0.5, 0.004, '', ha='center', va='bottom', fontsize=8, color='#444444')
        canvas = self.fig.canvas
        canvas.mpl_connect('button_press_event', self._on_press)
        canvas.mpl_connect('key_press_event', self._on_key)
        self._apply_colours()

    # ================================================================ public API
    def show(self):
        plt.show()

    def save(self, path, **kwargs):
        self.fig.savefig(path, **kwargs)

    def set_mode(self, mode: str):
        if mode not in MODES:
            raise ValueError(f'mode must be one of {MODES}')
        self.mode = mode
        self._apply_colours()

    def set_shared_scale(self, shared: bool):
        self.shared_scale = bool(shared)
        self._apply_colours()

    def select_node(self, node: int | None):
        """Marks a neuron on every map and shows its value of each attribute in the titles."""
        node = None if node is None else int(node)
        changed = node != self.selected
        self.selected = node
        for p in self.panels:
            if node is None:
                p['marker'].set_visible(False)
            else:
                x, y = self.som.positions[node]
                p['marker'].set_data([x], [y])
                p['marker'].set_visible(True)
        self._update_titles()
        self.fig.canvas.draw_idle()
        if changed:
            for callback in list(self._select_callbacks):
                callback(self)

    def on_select(self, callback: Callable):
        """Registers callback(viewer), called when the selected neuron changes."""
        self._select_callbacks.append(callback)

    def node_at(self, x: float, y: float) -> int:
        return int(np.argmin(((self.som.positions - [x, y]) ** 2).sum(axis=1)))

    def open_single(self, attribute: int | None):
        """Opens a PlaneSOMViewer on one attribute ('Component difference'), its selection linked to this grid."""
        from mt.geodesicsom.gui.inspector import LAYER_COMPONENT_DIFF
        from mt.geodesicsom.gui.link import link_views
        layer = LAYER_DISTANCE if attribute is None else LAYER_COMPONENT_DIFF
        viewer = PlaneSOMViewer(self.som, self.data, labels=self.labels, feature_names=self.feature_names,
                                layer=layer, component=attribute or 0)
        if self.selected is not None:
            viewer.select_node(self.selected)
        link_views(self, viewer)
        self.children.append(viewer)
        viewer.fig.show()
        return viewer

    # ================================================================== drawing
    def _apply_colours(self):
        values = self._values[self.mode]
        cmap = plt.get_cmap(_CMAP[self.mode])
        shared = (values.min(), values.max()) if self.shared_scale else None
        for p in self.panels:
            if p['attribute'] is None:
                v, pmap = self._total, plt.get_cmap('magma')
                lo, hi = v.min(), v.max()
            else:
                v, pmap = values[:, p['attribute']], cmap
                lo, hi = shared or (v.min(), v.max())
            norm = Normalize(lo, hi if hi > lo else lo + 1e-12)
            p['rgba'] = pmap(norm(v))
            p['faces'].set_facecolor(p['rgba'])
            p['faces'].set_edgecolor((1, 1, 1, 0.5) if self.show_edges else p['rgba'])
            p['faces'].set_linewidth(0.3 if self.show_edges else 0.25)
            p['cax'].clear()
            p['colourbar'] = self.fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=pmap), cax=p['cax'])
            p['cax'].tick_params(labelsize=6, length=2, pad=1)
        label = '|difference| between neighbours' if self.mode == 'difference' else 'attribute value'
        self._mode_button.label.set_text(f'colour: {label.split()[0]}')
        self._info.set_text(f'colour = {label} · click selects · double-click opens · v mode · c scale · e edges')
        self._update_titles()
        self.fig.canvas.draw_idle()

    def _update_titles(self):
        for p in self.panels:
            text = p['name']
            if self.selected is not None:
                w = self.som.weights[self.selected]
                text += f'  (neuron {self.selected})' if p['attribute'] is None else f'  = {w[p["attribute"]]:.4g}'
            p['title'].set_text(text)

    # ========================================================== event handlers
    def _on_press(self, event):
        for p in self.panels:
            if event.inaxes is p['ax'] and event.button == 1 and event.xdata is not None:
                if event.dblclick:
                    self.open_single(p['attribute'])
                else:
                    self.select_node(self.node_at(event.xdata, event.ydata))
                return

    def _on_key(self, event):
        if event.key == 'v':
            self.set_mode('value' if self.mode == 'difference' else 'difference')
        elif event.key == 'c':
            self.set_shared_scale(not self.shared_scale)
        elif event.key == 'e':
            self.show_edges = not self.show_edges
            self._apply_colours()
