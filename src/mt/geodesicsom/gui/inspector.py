# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
SOMInspector: the layers and the neuron inspector shared by SOMViewer (sphere) and PlaneSOMViewer
(flat grid).  Not used on its own.

Layers
    Neighbour distance   colour *between* the neurons: every face is coloured by the mean Euclidean
                         distance between the attribute vectors of its neurons.
                         Dark valleys = clusters, bright ridges = borders between clusters.
    U-matrix (neurons)   the same idea per neuron: mean distance to its neighbours' attribute vectors
    Component            one attribute of the weight vectors (component plane)
    Component difference colour between neurons from ONE attribute: each face is coloured by the
                         mean |difference| of that attribute between its neurons
    PCA colour           the whole attribute vector as a colour (first three principal components -> RGB)
    Hits                 how many samples each neuron wins (needs data)
    Classes              majority label of the samples each neuron wins (needs labels)

Neuron labels: `set_node_labels(style)` writes on every neuron that is the best matching unit of some
samples the label(s) of those samples -- 'majority', 'all', 'counts' or 'first' (see
LatticeSOM.bmu_labels) -- from the class labels or from `sample_names` (any text per sample, e.g. names).
`show_node_labels(True/False)` / `toggle_node_labels()` turn them on and off, keeping the style.

On-screen controls (below the colour bar): check boxes 'sample dots' and 'neuron labels', a
'labels: <style>' button that switches the label style, and 'A-' / 'A+' buttons for the label size.

Keys: [ / ] previous / next attribute (Component layers), s sample dots on/off,
l neuron labels on/off, L next label style (majority -> all -> counts), - / + smaller / larger labels.
Click a neuron to see its attribute vector in the side panel; click a bar there to show that attribute.
"""
from collections.abc import Callable, Sequence

import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.widgets import Button, CheckButtons, RadioButtons

LAYER_DISTANCE = 'Neighbour distance'
LAYER_UMATRIX = 'U-matrix (neurons)'
LAYER_COMPONENT = 'Component'
LAYER_COMPONENT_DIFF = 'Component difference'
LAYER_PCA = 'PCA colour'
LAYER_HITS = 'Hits'
LAYER_CLASSES = 'Classes'

LABEL_STYLES = ('majority', 'all', 'counts', 'first')
_CHECK_SAMPLES = 'sample dots'
_CHECK_LABELS = 'neuron labels'
_LEGEND_ROWS = 6                                     # class legend entries per column
LABEL_SIZE_RANGE = (3.0, 30.0)                         # smallest and largest label font size, in points

LAYER_CMAPS = {LAYER_DISTANCE: 'magma', LAYER_UMATRIX: 'magma', LAYER_COMPONENT: 'coolwarm',
               LAYER_COMPONENT_DIFF: 'magma', LAYER_HITS: 'Blues'}
COMPONENT_LAYERS = (LAYER_COMPONENT, LAYER_COMPONENT_DIFF)


class SOMInspector:
    """
    Mixin.  A subclass calls `_init_inspector(...)` before building its figure and
    `_build_inspector()` after, and provides:
        self.fig, self.ax                   the figure and the map axes
        _show_layer_colours(values, cmap)   draw the layer (values: per face or per neuron, scalars or RGBA)
                                            and return the Normalize used, or None for colours
        _update_overlays()                  redraw the sample dots and the selection marker
        _node_position_text(i)              a short description of where neuron i is
        _node_xy(nodes)                     map coordinates of neurons (for the neuron labels)
    """

    # ================================================================ set-up
    def _init_inspector(self, som, data, labels, feature_names, component, show_samples, sample_names=None,
                        label_size=None):
        if som.weights is None:
            raise ValueError('the SOM has no weights yet: initialise() and train() it first')
        self.som = som
        self.data = None if data is None else np.asarray(data, dtype=float)
        self.labels = None if labels is None else np.asarray(labels)
        if self.labels is not None and (self.data is None or len(self.labels) != len(self.data)):
            raise ValueError('labels need data of the same length')
        self.feature_names = list(feature_names) if feature_names is not None else \
            [f'attr {i}' for i in range(som.dim)]
        if len(self.feature_names) != som.dim:
            raise ValueError(f'{len(self.feature_names)} feature names for {som.dim} attributes')

        # everything the layers need, computed once
        self.face_distance = som.face_distance()
        self.u_matrix = som.u_matrix()
        self.bmu = som.bmu(self.data) if self.data is not None else None
        self.hits = np.bincount(self.bmu, minlength=som.n_nodes) if self.bmu is not None else None
        self.classes, self.class_colours, self.node_class = self._classes()
        self._weight_mean = som.weights.mean(axis=0)
        self._weight_std = som.weights.std(axis=0) + 1e-12
        self._component_difference = None                   # (faces, attributes), computed when first needed

        self.layers = [LAYER_DISTANCE, LAYER_UMATRIX, LAYER_COMPONENT, LAYER_COMPONENT_DIFF, LAYER_PCA]
        if self.hits is not None:
            self.layers.append(LAYER_HITS)
        if self.classes is not None:
            self.layers.append(LAYER_CLASSES)
        self.layer = LAYER_DISTANCE
        self.component = int(component) % som.dim
        self.selected: int | None = None
        self.layer_norm = None
        self.show_samples = self.data is not None if show_samples is None else bool(show_samples)
        self._select_callbacks: list[Callable] = []
        self._press_px = None
        self._ready = False
        self.node_label_mode: str = 'majority'               # the label style used when labels are on
        self.node_labels_on: bool = False
        self.sample_names = None                             # text per sample for the neuron labels (default: labels)
        if sample_names is not None:
            if self.data is None or len(sample_names) != len(self.data):
                raise ValueError('sample_names needs data of the same length')
            self.sample_names = np.asarray(sample_names)
            self.node_label_mode = 'first'
        self._syncing = False
        self._checks = self._style_button = None
        self._label_texts: list = []
        self._label_nodes = np.zeros(0, dtype=int)
        self._label_options = dict(max_per_node=3, fontsize=7.0)
        if label_size is not None:
            self._label_options['fontsize'] = self._clamp_label_size(label_size)

    def _build_inspector(self, map_rect=(0.15, 0.08, 0.56, 0.84)):
        fig = self.fig
        self.ax.set_position(list(map_rect))
        self._cax = fig.add_axes([map_rect[0] + map_rect[2] + 0.01, 0.22, 0.01, 0.56])
        self._colourbar = None

        rax = fig.add_axes([0.01, 0.08, 0.13, 0.045 * len(self.layers) + 0.03], frameon=False)
        rax.set_title('layer', fontsize=9, loc='left')
        self._layer_radio = RadioButtons(rax, self.layers, active=0)
        self._layer_radio.on_clicked(self._on_layer)

        self._detail = fig.add_axes([0.80, 0.46, 0.18, 0.42])
        self._detail_text = fig.text(0.785, 0.40, '', va='top', ha='left', fontsize=8.5, family='monospace')
        self._layer_text = fig.text(map_rect[0], 0.935, '', ha='left', fontsize=10, color='#222222')
        fig.text(map_rect[0], 0.905, 'click a neuron to inspect it  ·  [ ] change attribute  ·  s sample dots  ·  '
                                     'l neuron labels  ·  L label style  ·  - / + label size',
                 ha='left', fontsize=8.5, color='#666666')
        if self.classes is not None and len(self.classes) <= 3 * _LEGEND_ROWS:
            # columns of at most _LEGEND_ROWS entries, so the legend stays above the controls below it
            handles = [plt.Line2D([], [], ls='', marker='o', color=self.class_colours[i], label=str(c))
                       for i, c in enumerate(self.classes)]
            ncol = -(-len(handles) // _LEGEND_ROWS)
            fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.005, 0.99), fontsize=8 if ncol == 1 else 7,
                       ncol=ncol, columnspacing=0.8, handletextpad=0.2, title='classes', title_fontsize=8,
                       frameon=False)
        self._build_display_controls(map_rect)
        (self._marker,) = self.ax.plot([], [], marker='o', ms=13, mfc='none', mec='#00e5ff', mew=2.2, zorder=5)
        (self._marker_dot,) = self.ax.plot([], [], marker='o', ms=3, color='#00e5ff', zorder=5)

        canvas = fig.canvas
        canvas.mpl_connect('button_press_event', self._on_inspector_press)
        canvas.mpl_connect('button_release_event', self._on_inspector_release)
        canvas.mpl_connect('key_press_event', self._on_inspector_key)
        canvas.mpl_connect('pick_event', self._on_pick)

    def _sample_colours(self):
        if self.classes is None:
            return 'black'
        return self.class_colours[[self.classes.index(c) for c in self.labels.tolist()]]

    # ================================================================ public API
    def set_layer(self, layer: str, component: int | None = None):
        """Shows one of `self.layers`; `component` picks the attribute for the Component layers."""
        if layer not in self.layers:
            raise ValueError(f'unknown layer {layer!r}; choose from {self.layers}')
        if component is not None:
            self.component = int(component) % self.som.dim
        self.layer = layer
        if self._layer_radio.value_selected != layer:
            self._layer_radio.set_active(self.layers.index(layer))    # comes back through _on_layer
            return
        values, cmap = self._layer_values()
        norm = self._show_layer_colours(values, cmap)
        self.layer_norm = norm                              # the colour scale of the current layer (None: colours)
        self._update_colourbar(norm, cmap)
        self._update_panel()

    def select_node(self, node: int | None):
        """Selects a neuron (index into som.weights) and shows its attributes; None clears."""
        node = None if node is None else int(node)
        changed = node != self.selected
        self.selected = node
        self._update_panel()
        self._update_overlays()
        self.fig.canvas.draw_idle()
        if changed:
            for callback in list(self._select_callbacks):
                callback(self)

    def _build_display_controls(self, map_rect):
        """Check boxes for the sample dots and the neuron labels, and the label-style button."""
        if self.data is None:
            return
        options = [_CHECK_SAMPLES]
        if self.labels is not None or self.sample_names is not None:
            options.append(_CHECK_LABELS)
        x0 = map_rect[0] + map_rect[2] + 0.003
        cax = self.fig.add_axes([x0, 0.085, 0.07, 0.035 * len(options) + 0.01], frameon=False)
        self._checks = CheckButtons(cax, options, [self.show_samples, self.node_labels_on][:len(options)])
        for text in self._checks.labels:
            text.set_fontsize(8)
        self._checks.on_clicked(self._on_check)
        if _CHECK_LABELS in options:
            bax = self.fig.add_axes([x0, 0.035, 0.044, 0.04])
            self._style_button = Button(bax, '')
            self._style_button.label.set_fontsize(7)
            self._style_button.on_clicked(lambda _event: self.next_label_style())
            self._size_buttons = []
            for k, (text, factor) in enumerate((('A−', 1 / 1.2), ('A+', 1.2))):
                sax = self.fig.add_axes([x0 + 0.046 + 0.013 * k, 0.035, 0.012, 0.04])
                button = Button(sax, text)
                button.label.set_fontsize(7)
                button.on_clicked(lambda _event, f=factor: self.set_label_size(self.label_size * f))
                self._size_buttons.append(button)
        self._sync_controls()

    def _sync_controls(self):
        """Makes the check boxes and the style button match the current state (without firing callbacks)."""
        if self._checks is None:
            return
        self._syncing = True
        try:
            wanted = {_CHECK_SAMPLES: bool(self.show_samples), _CHECK_LABELS: bool(self.node_labels_on)}
            for i, (text, status) in enumerate(zip(self._checks.labels, self._checks.get_status(), strict=True)):
                if status != wanted[text.get_text()]:
                    self._checks.set_active(i)
            if self._style_button is not None:
                self._style_button.label.set_text(f'{self.node_label_mode} · {self.label_size:g}pt')
        finally:
            self._syncing = False

    def _on_check(self, label):
        if self._syncing:
            return
        status = dict(zip([t.get_text() for t in self._checks.labels], self._checks.get_status(), strict=True))
        if label == _CHECK_SAMPLES:
            self.set_show_samples(status[label])
        elif label == _CHECK_LABELS:
            self.show_node_labels(status[label])

    def set_show_samples(self, on: bool = True):
        """Shows or hides the sample dots (drawn at each sample's best matching unit)."""
        self.show_samples = bool(on)
        self._update_overlays()
        self._sync_controls()
        self.fig.canvas.draw_idle()

    def show_node_labels(self, on: bool = True):
        """Turns the neuron labels on or off, keeping the label style (see set_node_labels)."""
        if on:
            self.set_node_labels(self.node_label_mode)
            return
        for t in self._label_texts:
            t.remove()
        self._label_texts, self._label_nodes = [], np.zeros(0, dtype=int)
        self.node_labels_on = False
        self._update_overlays()
        self._sync_controls()
        self.fig.canvas.draw_idle()

    def toggle_node_labels(self):
        """Neuron labels on <-> off."""
        self.show_node_labels(not self.node_labels_on)

    @property
    def label_size(self) -> float:
        """The font size of the neuron labels, in points."""
        return self._label_options['fontsize']

    @staticmethod
    def _clamp_label_size(size) -> float:
        return float(round(min(max(float(size), LABEL_SIZE_RANGE[0]), LABEL_SIZE_RANGE[1]), 1))

    def set_label_size(self, size: float):
        """
        Sets the font size of the neuron labels (points, kept between 3 and 30).  Labels already on the map are
        resized in place; the size also applies when they are turned on later.
        """
        self._label_options['fontsize'] = self._clamp_label_size(size)
        for t in self._label_texts:
            t.set_fontsize(self.label_size)
        self._sync_controls()
        self.fig.canvas.draw_idle()

    def next_label_style(self):
        """Switches to the next label style (majority -> all -> counts, plus 'first' with sample_names) and turns
        the labels on."""
        styles = list(LABEL_STYLES if self.sample_names is not None else LABEL_STYLES[:3])
        current = styles.index(self.node_label_mode) if self.node_label_mode in styles else -1
        self.set_node_labels(styles[(current + 1) % len(styles)])

    def set_node_labels(self, mode: str | None = 'majority', sample_names: Sequence | None = None,
                        max_per_node: int | None = None, fontsize: float | None = None):
        """
        Writes a label on every neuron that is the best matching unit of at least one sample: the label(s)
        of the samples it wins.  Needs `data`, and `labels` or `sample_names`.

        :param mode: the label style: 'majority' (most common label), 'all' (every distinct label, 'A/B'),
                     'counts' ('A×3 B×1'), 'first' (the first sample's text, e.g. for unique names);
                     None turns the labels off (as show_node_labels(False))
        :param sample_names: optional text per sample to use instead of the class labels (e.g. sample names);
                             kept for later calls
        :param max_per_node: at most this many labels per neuron in 'all' / 'counts' (default 3)
        :param fontsize: label font size in points (default 7; see also set_label_size)
        """
        if mode is None:
            self.show_node_labels(False)
            return
        if mode not in LABEL_STYLES:
            raise ValueError(f'label style must be one of {LABEL_STYLES}')
        if sample_names is not None:
            if self.data is None or len(sample_names) != len(self.data):
                raise ValueError('sample_names needs data of the same length')
            self.sample_names = np.asarray(sample_names)
        if max_per_node is not None:
            self._label_options['max_per_node'] = int(max_per_node)
        if fontsize is not None:
            self._label_options['fontsize'] = self._clamp_label_size(fontsize)
        for t in self._label_texts:
            t.remove()
        self._label_texts, self._label_nodes = [], np.zeros(0, dtype=int)
        names = self.sample_names if self.sample_names is not None else self.labels
        if self.data is None or names is None:
            self.node_labels_on = False
            raise ValueError('neuron labels need data, and labels or sample_names')
        self.node_label_mode = mode
        self.node_labels_on = True
        text = self.som.bmu_labels(self.data, names, mode, self._label_options['max_per_node'])
        self._label_nodes = np.array([i for i, t in enumerate(text) if t is not None], dtype=int)
        clip = getattr(self, '_outline', None)
        outline = [patheffects.withStroke(linewidth=2.2, foreground='white')]
        for i in self._label_nodes:
            colour = self.class_colours[self.node_class[i]] if self.classes is not None \
                and self.node_class[i] >= 0 else '#111111'
            t = self.ax.text(0, 0, text[i], fontsize=self._label_options['fontsize'], ha='center', va='center',
                             color=colour, zorder=6, path_effects=outline, clip_on=True)
            if clip is not None:
                t.set_clip_path(clip)
            self._label_texts.append(t)
        self._update_overlays()
        self._sync_controls()
        self.fig.canvas.draw_idle()

    def _position_node_labels(self):
        if self._label_texts:
            xy = self._node_xy(self._label_nodes)
            for t, (x, y) in zip(self._label_texts, xy, strict=True):
                t.set_position((x, y))

    def on_select(self, callback: Callable):
        """Registers callback(viewer), called when the selected neuron changes."""
        self._select_callbacks.append(callback)

    # ================================================================== layers
    def _layer_values(self):
        """(values, cmap name): values per face or per neuron (scalars), or per-face RGBA."""
        som = self.som
        if self.layer == LAYER_DISTANCE:
            return self.face_distance, LAYER_CMAPS[LAYER_DISTANCE]
        if self.layer == LAYER_UMATRIX:
            return self.u_matrix, LAYER_CMAPS[LAYER_UMATRIX]
        if self.layer == LAYER_COMPONENT_DIFF:
            if self._component_difference is None:
                self._component_difference = som.face_component_difference()
            return self._component_difference[:, self.component], LAYER_CMAPS[LAYER_COMPONENT_DIFF]
        if self.layer == LAYER_COMPONENT:
            return som.weights[:, self.component], LAYER_CMAPS[LAYER_COMPONENT]
        if self.layer == LAYER_HITS:
            return self.hits.astype(float), LAYER_CMAPS[LAYER_HITS]
        if self.layer == LAYER_PCA:
            w = som.weights - self._weight_mean
            _, _, vt = np.linalg.svd(w, full_matrices=False)
            p = w @ vt[:3].T
            if p.shape[1] < 3:
                p = np.column_stack([p, np.zeros((len(p), 3 - p.shape[1]))])
            lo, hi = np.percentile(p, 1, axis=0), np.percentile(p, 99, axis=0)
            rgb = np.clip((p - lo) / np.maximum(hi - lo, 1e-12), 0, 1)[som.faces].mean(axis=1)
            return np.column_stack([rgb, np.ones(len(rgb))]), None
        if self.layer == LAYER_CLASSES:
            corner = self.node_class[som.faces]                         # (F, k), -1 = no samples
            face_class = np.full(len(corner), -1)
            for i, row in enumerate(corner):
                known = row[row >= 0]
                if len(known):
                    face_class[i] = np.bincount(known).argmax()
            rgba = np.tile([0.88, 0.88, 0.88, 1.0], (len(corner), 1))
            rgba[face_class >= 0] = self.class_colours[face_class[face_class >= 0]]
            return rgba, None
        raise ValueError(self.layer)

    def _face_rgba(self, values, cmap):
        """Per-face RGBA from per-face or per-neuron values, and the Normalize used (None for colours)."""
        values = np.asarray(values, dtype=float)
        faces = self.som.faces
        if values.ndim == 2:
            return values, None
        per_face = values if len(values) == len(faces) else values[faces].mean(axis=1)
        norm = Normalize(per_face.min(), per_face.max() if per_face.max() > per_face.min() else per_face.min() + 1e-12)
        return plt.get_cmap(cmap)(norm(per_face)), norm

    def _classes(self):
        if self.labels is None:
            return None, None, None
        classes = list(dict.fromkeys(self.labels.tolist()))
        index = {c: i for i, c in enumerate(classes)}
        cmap = plt.get_cmap('tab10' if len(classes) <= 10 else 'tab20')
        colours = np.array([cmap(i % cmap.N) for i in range(len(classes))])
        counts = np.zeros((self.som.n_nodes, len(classes)), dtype=int)
        np.add.at(counts, (self.bmu, [index[c] for c in self.labels.tolist()]), 1)
        node_class = np.where(counts.sum(1) > 0, counts.argmax(1), -1)
        return classes, colours, node_class

    # ============================================================ side panel
    def _update_colourbar(self, norm, cmap):
        self._cax.clear()
        if norm is None:
            self._cax.set_visible(False)
            return
        self._cax.set_visible(True)
        self._colourbar = self.fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(norm.vmin, norm.vmax),
                                                                  cmap=plt.get_cmap(cmap)), cax=self._cax)
        self._colourbar.ax.tick_params(labelsize=8)

    def _update_panel(self):
        name = self.layer
        if self.layer == LAYER_COMPONENT_DIFF:
            name += (f': {self.feature_names[self.component]} -- mean |difference| of this attribute between '
                     'neighbouring neurons   ([ and ] or click a bar)')
        elif self.layer == LAYER_COMPONENT:
            name += f': {self.feature_names[self.component]}   ([ and ] or click a bar to change)'
        elif self.layer == LAYER_DISTANCE:
            name += ': mean Euclidean distance between the attribute vectors of neighbouring neurons'
        elif self.layer == LAYER_PCA:
            name += ': attribute vectors -> first 3 principal components -> RGB'
        self._layer_text.set_text(name)

        ax = self._detail
        ax.clear()
        ax.tick_params(labelsize=7)
        if self.selected is None:
            ax.set_axis_off()
            q = f'QE {self.som.quantisation_error(self.data):.3f}  TE {self.som.topographic_error(self.data):.3f}' \
                if self.data is not None else ''
            self._detail_text.set_text('click a neuron on the map\nto see its attribute vector\n\n' + q)
            return
        ax.set_axis_on()
        i = self.selected
        w = self.som.weights[i]
        z = (w - self._weight_mean) / self._weight_std
        y = np.arange(len(z))
        bars = ax.barh(y, z, color=np.where(z >= 0, '#d6604d', '#4393c3'), picker=True)
        for k, bar in enumerate(bars):
            bar.set_gid(str(k))
            if self.layer in COMPONENT_LAYERS and k == self.component:
                bar.set_edgecolor('black')
                bar.set_linewidth(1.2)
        ax.set_yticks(y, self.feature_names if len(z) <= 30 else [''] * len(z))
        ax.invert_yaxis()
        ax.axvline(0, color='#555555', lw=0.6)
        ax.set_xlabel('standardised weight (vs all neurons)', fontsize=7)
        ax.set_title(f'neuron {i}', fontsize=9)

        nb = self.som.neighbours[i]
        lines = [self._node_position_text(i),
                 f'neighbour distance {np.linalg.norm(self.som.weights[nb] - w, axis=1).mean():.3f}']
        if self.hits is not None:
            lines.append(f'hits {self.hits[i]}')
        if self.classes is not None and self.hits[i]:
            won = self.labels[self.bmu == i]
            values, counts = np.unique(won, return_counts=True)
            order = np.argsort(counts)[::-1][:4]
            lines.append('labels ' + ', '.join(f'{values[k]}:{counts[k]}' for k in order))
        lines.append('')
        width = max(len(n) for n in self.feature_names[:16])
        lines += [f'{n[:width]:<{width}} {v:10.4g}' for n, v in zip(self.feature_names[:16], w[:16], strict=True)]
        if len(w) > 16:
            lines.append(f'... {len(w) - 16} more')
        self._detail_text.set_text('\n'.join(lines))

    # ========================================================== event handlers
    def _on_layer(self, label):
        self.layer = label
        self.set_layer(label)

    def _on_inspector_press(self, event):
        self._press_px = (event.x, event.y) if event.inaxes is self.ax and event.button == 1 else None

    def _on_inspector_release(self, event):
        if self._press_px is None or event.inaxes is not self.ax or event.xdata is None:
            return
        moved = np.hypot(event.x - self._press_px[0], event.y - self._press_px[1])
        self._press_px = None
        if moved < 4:                                        # a click, not a drag
            self.select_node(self.node_at(event.xdata, event.ydata))

    def _on_pick(self, event):
        gid = event.artist.get_gid() if event.artist is not None else None
        if event.mouseevent.inaxes is self._detail and gid is not None:
            self.set_layer(self.layer if self.layer in COMPONENT_LAYERS else LAYER_COMPONENT, int(gid))

    def _on_inspector_key(self, event):
        key = event.key or ''
        if key in ('[', ']'):
            step = 1 if key == ']' else -1
            layer = self.layer if self.layer in COMPONENT_LAYERS else LAYER_COMPONENT
            self.set_layer(layer, (self.component + step) % self.som.dim)
        elif key == 's' and getattr(self, '_samples', None) is not None:
            self.set_show_samples(not self.show_samples)
        elif key in ('l', 'L') and self.data is not None and (self.labels is not None or self.sample_names is not None):
            if key == 'l':
                self.toggle_node_labels()
            else:
                self.next_label_style()
        elif key in ('+', '=', '-'):
            self.set_label_size(self.label_size * (1.2 if key in ('+', '=') else 1 / 1.2))


def standard_layers() -> Sequence[str]:
    return (LAYER_DISTANCE, LAYER_UMATRIX, LAYER_COMPONENT, LAYER_COMPONENT_DIFF, LAYER_PCA, LAYER_HITS, LAYER_CLASSES)
