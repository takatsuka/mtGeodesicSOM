# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
ComponentMatrixViewer: one rotatable geodesic-dome map per attribute of a trained GeodesicSOM, in a
grid, all rotating together (needs matplotlib).

    from mt.geodesicsom.gui import ComponentMatrixViewer

    ComponentMatrixViewer(som, feature_names=names).show()

Every panel shows the same sphere in the same orientation.  In the default 'difference' mode,
each triangle between three neighbouring neurons is coloured by how much *that one attribute*
changes between them: the mean |w_a[k] - w_b[k]| along the triangle's three edges.  Bright
lines are where attribute k changes sharply; comparing panels shows which attributes separate
which clusters.  The optional first panel ('all attributes') is the Euclidean neighbour
distance over the whole attribute vector, as in SOMViewer.

Mouse and keyboard (in any panel -- every panel follows)
    drag                 rotate the sphere           click     select the neuron under the pointer
    double-click         open that attribute in a full SOMViewer, rotation and selection linked
    arrows / shift       rotate by 5 / 1 degrees     , .       roll
    v                    'difference' <-> 'value' (the attribute itself, a component plane)
    c                    colour scale per panel <-> one scale shared by all panels
    p  projection        r  reset view               g  graticule      e  triangle edges
"""
import math
from collections.abc import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon
from matplotlib.path import Path
from matplotlib.widgets import Button
from mt.geodesicdome.interactive import rotation as rot
from mt.geodesicdome.interactive.sphere_map import SphereMap
from mt.geodesicdome.interactive.viewer import DEFAULT_PROJECTIONS

from mt.geodesicsom.GeodesicSOM import GeodesicSOM

MODES = ('difference', 'value')
_CMAP = {'difference': 'magma', 'value': 'coolwarm'}


class ComponentMatrixViewer:
    """
    Grid of synchronised, rotatable maps: one per attribute of a trained GeodesicSOM.

    :param som: a trained GeodesicSOM
    :param feature_names: optional name per attribute
    :param mode: 'difference' (default) -- colour between neurons = |difference of the attribute|;
                 'value' -- the attribute's value (component plane)
    :param include_total: add a first panel with the Euclidean distance over all attributes
    :param shared_scale: one colour scale for all attribute panels (default: one per panel)
    :param projection: 'Equal Earth', 'Kavrayskiy VII', 'Wagner VI' or 'Wagner III'
    :param view: initial (lat, lon) in degrees at the centre of every panel
    :param ncols: panels per row (default: chosen from the number of panels)
    :param data, labels: optional; passed on to the SOMViewer opened by a double-click
    :param edges, graticule, figsize, title: display options
    """

    def __init__(self, som: GeodesicSOM, feature_names: Sequence[str] | None = None, *, mode: str = 'difference',
                 include_total: bool = True, shared_scale: bool = False, projection: str = 'Equal Earth',
                 view: Sequence[float] | None = None, ncols: int | None = None, data=None,
                 labels=None, edges: bool = False, graticule: bool = True, figsize=None,
                 title: str | None = None):
        if not isinstance(som, GeodesicSOM):
            raise TypeError('ComponentMatrixViewer shows a GeodesicSOM; use PlaneComponentMatrixViewer for a PlaneSOM')
        if som.weights is None:
            raise ValueError('the GeodesicSOM has no weights yet: initialise() and train() it first')
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
        self.show_graticule = bool(graticule)
        self.projections = dict(DEFAULT_PROJECTIONS)
        if projection not in self.projections:
            raise ValueError(f'unknown projection {projection!r}; choose from {list(self.projections)}')
        self.projection_name = projection
        self.map = SphereMap.from_manifold(som.dome, self.projections[projection])
        if not np.array_equal(self.map.faces, som.faces):
            raise RuntimeError('the map and the SOM disagree about the dome faces')

        self.rotation = np.eye(3) if view is None else rot.view_rotation(*np.radians(view))
        self._home = self.rotation.copy()
        self.selected: int | None = None
        self._callbacks: list[Callable[[ComponentMatrixViewer], None]] = []
        self._select_callbacks: list[Callable[[ComponentMatrixViewer], None]] = []
        self._drag = None                     # (panel index, x, y) while the mouse button is down
        self._moved = 0.0
        self._background = None
        self._paths: list[Path] = []
        self._buffer = None
        self.children: list = []              # SOMViewers opened by double-click

        self._values = {'difference': som.face_component_difference(), 'value': som.face_component_value()}
        self._total = som.face_distance()
        n_panels = som.dim + (1 if self.include_total else 0)
        self.ncols = ncols or max(1, math.ceil(math.sqrt(n_panels * 1.6)))
        self.nrows = math.ceil(n_panels / self.ncols)
        self._build_figure(figsize, title)
        self._apply_colours()
        self._redraw_outline()
        self.update()

    # ================================================================ public API
    def show(self):
        plt.show()

    def save(self, path, **kwargs):
        self.fig.savefig(path, **kwargs)

    @property
    def centre(self):
        """(lat, lon) in degrees of the original-sphere point at the centre of every panel."""
        return tuple(float(a) for a in np.degrees(rot.centre_of_view(self.rotation)))

    def rotate(self, d_lon: float = 0.0, d_lat: float = 0.0, roll: float = 0.0):
        r = rot.rot_x(np.radians(roll)) @ rot.drag_rotation(np.radians(d_lon), np.radians(d_lat))
        self.set_rotation(r @ self.rotation)

    def set_view(self, lat: float, lon: float, roll: float = 0.0):
        self.set_rotation(rot.view_rotation(*np.radians([lat, lon, roll])))

    def set_rotation(self, matrix):
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape != (3, 3) or np.linalg.det(matrix) <= 0:
            raise ValueError('rotation must be a 3x3 rotation matrix')
        self.rotation = rot.orthonormalise(matrix)
        self.update()

    def reset(self):
        self.set_rotation(self._home)

    def set_projection(self, name: str):
        if name not in self.projections:
            raise ValueError(f'unknown projection {name!r}; choose from {list(self.projections)}')
        self.projection_name = name
        self.map.projection = self.projections[name]
        self._redraw_outline()
        self.update()

    def set_mode(self, mode: str):
        """'difference' (|change of the attribute| between neighbouring neurons) or 'value'."""
        if mode not in MODES:
            raise ValueError(f'mode must be one of {MODES}')
        self.mode = mode
        self._apply_colours()
        self.update()

    def set_shared_scale(self, shared: bool):
        self.shared_scale = bool(shared)
        self._apply_colours()
        self.update()

    def select_node(self, node: int | None):
        """Marks a neuron on every panel and shows its value of each attribute in the titles."""
        node = None if node is None else int(node)
        changed = node != self.selected
        self.selected = node
        self._update_titles()
        self.update()
        if changed:
            for callback in list(self._select_callbacks):
                callback(self)

    def on_select(self, callback: Callable[['ComponentMatrixViewer'], None]):
        """Registers callback(viewer), called when the selected neuron changes."""
        self._select_callbacks.append(callback)

    def on_rotate(self, callback: Callable[['ComponentMatrixViewer'], None]):
        """Registers callback(viewer), called after every change of the view."""
        self._callbacks.append(callback)

    def open_single(self, attribute: int | None):
        """
        Opens a SOMViewer for one attribute ('Component difference', or 'Neighbour distance' for None)
        whose rotation and selection are linked to this grid.
        """
        from mt.geodesicsom.gui.inspector import LAYER_COMPONENT_DIFF, LAYER_DISTANCE
        from mt.geodesicsom.gui.link import link_views
        from mt.geodesicsom.gui.som_viewer import SOMViewer
        layer = LAYER_DISTANCE if attribute is None else LAYER_COMPONENT_DIFF
        viewer = SOMViewer(self.som, self.data, labels=self.labels, feature_names=self.feature_names,
                           layer=layer, component=attribute or 0, projection=self.projection_name)
        viewer.set_rotation(self.rotation)
        if self.selected is not None:
            viewer.select_node(self.selected)
        link_views(self, viewer)
        self.children.append(viewer)
        viewer.fig.show()
        return viewer

    # ============================================================ figure set-up
    def _build_figure(self, figsize, title):
        panel_w, panel_h = 4.2, 2.55
        figsize = figsize or (panel_w * self.ncols, panel_h * self.nrows + 0.9)
        self.fig = plt.figure(figsize=figsize)
        manager = self.fig.canvas.manager
        if manager is not None:
            manager.set_window_title('mt.geodesicsom ComponentMatrixViewer')
        top = 1.0 - 0.75 / figsize[1]
        bottom = 0.25 / figsize[1]
        self.fig.suptitle(title or f'GeodesicSOM {self.som.n_nodes} neurons: one map per attribute '
                                   '(drag any map -- all follow)', fontsize=12, y=1.0 - 0.12 / figsize[1], va='top')
        cell_w, cell_h = 1.0 / self.ncols, (top - bottom) / self.nrows

        self.panels: list[dict] = []
        names = (['all attributes'] if self.include_total else []) + self.feature_names
        for i, name in enumerate(names):
            row, col = divmod(i, self.ncols)
            x0, y0 = col * cell_w, top - (row + 1) * cell_h
            ax = self.fig.add_axes([x0 + 0.005, y0 + 0.02 * cell_h, cell_w * 0.86, cell_h * 0.84])
            ax.set_aspect('equal')
            ax.set_axis_off()
            cax = self.fig.add_axes([x0 + cell_w * 0.875, y0 + 0.14 * cell_h, cell_w * 0.025, cell_h * 0.6])
            cax.tick_params(labelsize=6, length=2, pad=1)
            outline = Polygon(np.zeros((3, 2)), closed=True, facecolor='none', edgecolor='#333333',
                              linewidth=0.8, zorder=3)
            ax.add_patch(outline)
            faces = PolyCollection([], antialiased=True, zorder=1)
            faces.set_clip_path(outline)
            ax.add_collection(faces)
            (grat,) = ax.plot([], [], color='black', lw=0.4, alpha=0.4, zorder=2)
            grat.set_clip_path(outline)
            (marker,) = ax.plot([], [], marker='o', ms=8, mfc='none', mec='#00e5ff', mew=1.6, zorder=5)
            attribute = None if (self.include_total and i == 0) else i - (1 if self.include_total else 0)
            title = ax.set_title(name, fontsize=8.5, pad=2)
            self.panels.append(dict(ax=ax, cax=cax, outline=outline, faces=faces, graticule=grat, marker=marker,
                                    attribute=attribute, name=name, title=title, rgba=None, colourbar=None))

        # controls
        h = 0.34 / figsize[1]
        self._mode_button = Button(self.fig.add_axes([0.005, 1 - 0.5 / figsize[1], 0.11, h]), '')
        self._mode_button.label.set_fontsize(8)
        self._mode_button.on_clicked(lambda _e: self.set_mode('value' if self.mode == 'difference' else 'difference'))
        self._proj_button = Button(self.fig.add_axes([0.12, 1 - 0.5 / figsize[1], 0.11, h]), '')
        self._proj_button.label.set_fontsize(8)
        self._proj_button.on_clicked(lambda _e: self._next_projection())
        self._reset_button = Button(self.fig.add_axes([0.235, 1 - 0.5 / figsize[1], 0.06, h]), 'reset')
        self._reset_button.label.set_fontsize(8)
        self._reset_button.on_clicked(lambda _e: self.reset())
        self._info = self.fig.text(0.5, 0.004, '', ha='center', va='bottom', fontsize=8, color='#444444')

        canvas = self.fig.canvas
        canvas.mpl_connect('button_press_event', self._on_press)
        canvas.mpl_connect('motion_notify_event', self._on_motion)
        canvas.mpl_connect('button_release_event', self._on_release)
        canvas.mpl_connect('key_press_event', self._on_key)

    def _redraw_outline(self):
        outline = self.map.outline()
        pad = 0.03 * np.ptp(outline[:, 0])
        for p in self.panels:
            p['outline'].set_xy(outline)
            p['ax'].set_xlim(outline[:, 0].min() - pad, outline[:, 0].max() + pad)
            p['ax'].set_ylim(outline[:, 1].min() - pad, outline[:, 1].max() + pad)
        self._map_size = (np.ptp(outline[:, 0]), np.ptp(outline[:, 1]))
        self._proj_button.label.set_text(self.projection_name)

    def _apply_colours(self):
        """Per-face RGBA and colour bar for every panel, for the current mode and scale."""
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
            p['cax'].clear()
            p['colourbar'] = self.fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=pmap), cax=p['cax'])
            p['cax'].tick_params(labelsize=6, length=2, pad=1)
        label = '|difference| between neighbours' if self.mode == 'difference' else 'attribute value'
        self._mode_button.label.set_text(f'colour: {label.split()[0]}')
        self._mode_text = label
        self._update_titles()

    def _update_titles(self):
        for p in self.panels:
            text = p['name']
            if self.selected is not None:
                w = self.som.weights[self.selected]
                if p['attribute'] is None:
                    text += f'  (neuron {self.selected})'
                else:
                    text += f'  = {w[p["attribute"]]:.4g}'
            p['title'].set_text(text)

    # ================================================================== drawing
    def _set_polygons(self, polygons):
        """All panels show identical polygons, so one pool of Paths is shared by every collection."""
        n, width = polygons.shape[0], polygons.shape[1] + 1
        if self._buffer is None or len(self._buffer) < n:
            capacity = int(n * 1.25) + 16
            self._buffer = np.zeros((capacity, width, 2))
            codes = np.full(width, Path.LINETO, dtype=Path.code_type)
            codes[0], codes[-1] = Path.MOVETO, Path.CLOSEPOLY
            self._paths = [Path(self._buffer[i], codes) for i in range(capacity)]
            if not np.shares_memory(self._paths[0].vertices, self._buffer):
                self._buffer = None
        if self._buffer is None:
            for p in self.panels:
                p['faces'].set_verts(polygons)
            return
        self._buffer[:n, :-1] = polygons
        self._buffer[:n, -1] = polygons[:, 0]
        for p in self.panels:
            p['faces']._paths = self._paths[:n]
            p['faces'].stale = True

    def _project(self, xyz):
        q = xyz @ self.rotation.T
        return self.map.project(np.arcsin(np.clip(q[:, 2], -1, 1)), np.arctan2(q[:, 1], q[:, 0]))

    def update(self):
        polygons, face_index = self.map.polygons(self.rotation)
        self._set_polygons(polygons)
        dragging = self._drag is not None and self._moved > 0
        grat = self.map.graticule(self.rotation) if self.show_graticule else None
        mark = self._project(self.som.points[[self.selected]]) if self.selected is not None else None
        for p in self.panels:
            colours = p['rgba'][face_index]
            f = p['faces']
            f.set_facecolor(colours)
            if dragging:
                f.set_antialiased(False)
                f.set_edgecolor('none')
                f.set_linewidth(0)
            else:
                f.set_antialiased(True)
                f.set_edgecolor((1, 1, 1, 0.5) if self.show_edges else colours)
                f.set_linewidth(0.3 if self.show_edges else 0.25)
            if grat is not None:
                p['graticule'].set_data(grat[:, 0], grat[:, 1])
            p['graticule'].set_visible(self.show_graticule)
            if mark is not None:
                p['marker'].set_data(mark[:, 0], mark[:, 1])
            p['marker'].set_visible(mark is not None)

        if dragging and self._background is not None:
            canvas = self.fig.canvas
            canvas.restore_region(self._background)
            for p in self.panels:
                for a in self._dynamic(p):
                    p['ax'].draw_artist(a)
            canvas.blit(self.fig.bbox)
        else:
            lat, lon = self.centre
            self._info.set_text(f'{self.projection_name} · colour = {self._mode_text} · centre lat {lat:+.1f}° '
                                f'lon {lon:+.1f}° · drag rotates all · click selects · double-click opens · '
                                'v mode · c scale · p projection · r reset')
            self.fig.canvas.draw_idle()
        for callback in list(self._callbacks):
            callback(self)

    @staticmethod
    def _dynamic(panel):
        return (panel['faces'], panel['graticule'], panel['outline'], panel['marker'])

    def _grab_background(self):
        canvas = self.fig.canvas
        if not getattr(canvas, 'supports_blit', False):
            return None
        artists = [a for p in self.panels for a in self._dynamic(p)]
        visible = [a.get_visible() for a in artists]
        for a in artists:
            a.set_visible(False)
        try:
            canvas.draw()
            return canvas.copy_from_bbox(self.fig.bbox)
        except Exception:                    # backend without blitting
            return None
        finally:
            for a, v in zip(artists, visible, strict=True):
                a.set_visible(v)

    # ========================================================== event handlers
    def _panel_of(self, event):
        for i, p in enumerate(self.panels):
            if event.inaxes is p['ax']:
                return i
        return None

    def _on_press(self, event):
        i = self._panel_of(event)
        if i is None or event.button != 1:
            return
        if event.dblclick:
            self._drag = None
            self.open_single(self.panels[i]['attribute'])
            return
        self._drag = (i, event.x, event.y)
        self._moved = 0.0

    def _on_motion(self, event):
        if self._drag is None or event.x is None:
            return
        i, x, y = self._drag
        dx, dy = event.x - x, event.y - y
        if self._moved == 0.0 and np.hypot(dx, dy) < 3:
            return                                   # not a drag yet
        if self._moved == 0.0:
            self._moved = 1e-9
            self._background = self._grab_background()
        self._moved += np.hypot(dx, dy)
        ax = self.panels[i]['ax']
        (x0, y0), (x1, y1) = ax.transData.transform([[0.0, 0.0], list(self._map_size)])
        px_lon, px_lat = abs(x1 - x0) / (2 * np.pi), abs(y1 - y0) / np.pi
        self._drag = (i, event.x, event.y)
        self.set_rotation(rot.drag_rotation(dx / px_lon, dy / px_lat) @ self.rotation)

    def _on_release(self, event):
        if self._drag is None:
            return
        i, _, _ = self._drag
        was_drag = self._moved > 0
        self._drag = None
        self._background = None
        if was_drag:
            self.update()                            # full quality again
        elif event.inaxes is self.panels[i]['ax'] and event.xdata is not None:
            xy = self._project(self.som.points)
            self.select_node(int(np.argmin(((xy - [event.xdata, event.ydata]) ** 2).sum(axis=1))))

    def _next_projection(self):
        names = list(self.projections)
        self.set_projection(names[(names.index(self.projection_name) + 1) % len(names)])

    def _on_key(self, event):
        key = event.key or ''
        step = 1.0 if key.startswith('shift+') else 5.0
        key = key.replace('shift+', '')
        moves = {'left': (-step, 0, 0), 'right': (step, 0, 0), 'up': (0, step, 0),
                 'down': (0, -step, 0), ',': (0, 0, -step), '.': (0, 0, step)}
        if key in moves:
            self.rotate(*moves[key])
        elif key == 'r':
            self.reset()
        elif key == 'p':
            self._next_projection()
        elif key == 'v':
            self.set_mode('value' if self.mode == 'difference' else 'difference')
        elif key == 'c':
            self.set_shared_scale(not self.shared_scale)
        elif key == 'g':
            self.show_graticule = not self.show_graticule
            self.update()
        elif key == 'e':
            self.show_edges = not self.show_edges
            self.update()
