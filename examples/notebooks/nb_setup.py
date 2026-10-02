# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
Shared set-up for the example notebooks in this folder.

    import nb_setup
    WIDGET = nb_setup.setup()        # True when figures are live (ipympl), False when static

`setup()` makes the package importable from a source checkout (no `pip install` needed) and picks
the matplotlib backend:

* with `ipympl` installed  -> `%matplotlib widget`: the viewers are live -- drag to rotate a sphere,
  click a neuron, use the radio buttons and keys exactly as in a window;
* otherwise                -> `%matplotlib inline`: static images; the ipywidgets controls below still
  work and redraw them.

`controls(viewer, ...)` builds ipywidgets controls for a SOMViewer, PlaneSOMViewer or
ComponentMatrixViewer (layer, attribute, selected neuron, view centre, ...), and `LiveFigure` is a
figure that controls redraw in place under either backend.
"""
import os
import sys

try:
    import numpy  # noqa: F401
except ImportError:
    raise ImportError(
        f'numpy is not installed for the Python running this notebook:\n    {sys.executable}\n'
        "Run ./setup_env.sh in the repository once, then choose the project's environment "
        '(~/.venvs/<project>/bin/python) as the interpreter (PyCharm: Settings > Project > Python Interpreter; '
        'VS Code / Jupyter: the kernel picker).'
    ) from None

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src')
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)                          # run from a checkout without installing

WIDGET = False


def setup(interactive: bool = True) -> bool:
    """Chooses the matplotlib backend; returns True when figures are live (ipympl)."""
    global WIDGET
    try:
        from IPython import get_ipython
        ipython = get_ipython()
    except ImportError:
        ipython = None
    if ipython is None:                               # plain python, not a notebook
        return False

    WIDGET = False
    if interactive and not os.environ.get('MTG_NOTEBOOK_STATIC'):
        try:
            import ipympl  # noqa: F401
            ipython.run_line_magic('matplotlib', 'widget')
            WIDGET = True
        except ImportError:
            pass
    if not WIDGET:
        ipython.run_line_magic('matplotlib', 'inline')
    print('figures: live (ipympl) -- drag the maps, click neurons, use the buttons on the figures' if WIDGET else
          'figures: static (inline) -- `pip install ipympl` for live, rotatable maps')
    return WIDGET


def _display(*figures):
    from IPython.display import display
    for fig in figures:
        display(fig)


def building():
    """
    Context manager for building viewers inside a widget callback, where they are not shown by
    themselves:  `with nb_setup.building(): views = explore(..., show=False)`, then `show(...)`.
    """
    import matplotlib.pyplot as plt
    return plt.ioff()


def show(*viewers):
    """Displays viewers (or figures) here: live canvases with ipympl, static images otherwise."""
    import matplotlib.pyplot as plt
    from IPython.display import display
    for item in viewers:
        if item is None:
            continue
        fig = getattr(item, 'fig', item)
        if WIDGET:
            display(fig.canvas)
        else:
            display(fig)
            plt.close(fig)


class LiveFigure:
    """
    A figure that interactive controls redraw in place.

        live = LiveFigure(figsize=(8, 6))

        @interact(n=(1, 10))
        def draw(n):
            fig = live.clear()
            ...
            live.refresh()

        live.show()            # put the (live) canvas under the controls
    """

    def __init__(self, fig=None, **figure_kwargs):
        import matplotlib.pyplot as plt
        if fig is None:
            with plt.ioff():
                fig = plt.figure(**figure_kwargs)
        self.fig = fig
        if not WIDGET:
            plt.close(fig)                            # inline: shown only through refresh()

    def clear(self):
        self.fig.clear()
        return self.fig

    def refresh(self):
        if WIDGET:
            self.fig.canvas.draw_idle()
        else:
            _display(self.fig)

    def show(self):
        if WIDGET:
            from IPython.display import display
            display(self.fig.canvas)


def controls(viewer, *also, neuron: int | None = None):
    """
    ipywidgets controls for a trained-SOM viewer, displayed with an output area.

    :param viewer: SOMViewer, PlaneSOMViewer or ComponentMatrixViewer (anything with the same methods)
    :param also: viewers linked to it (e.g. the matrix of `explore()`); with static figures they are
                 redrawn too, so you see the linked selection and rotation
    :param neuron: neuron to select first (default: keep the current selection)

    With live figures the controls change the figures above in place, and clicking a neuron on a
    map updates the neuron box.  With static figures every change draws fresh images below.
    """
    import ipywidgets as w
    import matplotlib.pyplot as plt
    import numpy as np
    from IPython.display import clear_output, display

    rows = []
    n_nodes = viewer.som.n_nodes
    names = list(getattr(viewer, 'feature_names', None) or [f'attribute {k}' for k in range(viewer.som.dim)])
    out = w.Output()
    figures = [v.fig for v in (viewer, *also)]
    if not WIDGET:
        for fig in figures:
            plt.close(fig)                            # from now on drawn only by the controls

    def redraw():
        if WIDGET:
            return
        with out:
            clear_output(wait=True)
            _display(*figures)

    # ------------------------------------------------------------- layer (SOMViewer, PlaneSOMViewer)
    if hasattr(viewer, 'layers'):
        layer = w.Dropdown(options=viewer.layers, value=viewer.layer, description='layer')
        attribute = w.Dropdown(options=[(n, k) for k, n in enumerate(names)], value=viewer.component,
                               description='attribute')

        def on_layer(_=None):
            viewer.set_layer(layer.value, attribute.value)
            attribute.disabled = 'Component' not in layer.value
            redraw()

        layer.observe(on_layer, names='value')
        attribute.observe(on_layer, names='value')
        attribute.disabled = 'Component' not in layer.value
        rows.append(w.HBox([layer, attribute]))

    # ------------------------------------------------------------- mode (ComponentMatrixViewer)
    if hasattr(viewer, 'set_mode'):
        mode = w.ToggleButtons(options=['difference', 'value'], value=viewer.mode, description='colour by')
        shared = w.Checkbox(value=viewer.shared_scale, description='one colour scale')

        def on_mode(_=None):
            viewer.set_mode(mode.value)
            viewer.set_shared_scale(shared.value)
            redraw()

        mode.observe(on_mode, names='value')
        shared.observe(on_mode, names='value')
        rows.append(w.HBox([mode, shared]))

    # ------------------------------------------------------------- selected neuron
    current = viewer.selected if getattr(viewer, 'selected', None) is not None else -1
    node = w.BoundedIntText(value=current if neuron is None else neuron, min=-1, max=n_nodes - 1,
                            description='neuron', layout=w.Layout(width='190px'))
    busiest = w.Button(description='busiest neuron', disabled=getattr(viewer, 'hits', None) is None)
    syncing = []

    def on_node(_=None):
        if syncing:
            return
        viewer.select_node(None if node.value < 0 else node.value)
        redraw()

    def on_busiest(_):
        node.value = int(np.argmax(viewer.hits))

    def follow_click(v):                              # live figures: a click on the map updates the box
        syncing.append(1)
        node.value = -1 if v.selected is None else int(v.selected)
        syncing.pop()

    node.observe(on_node, names='value')
    busiest.on_click(on_busiest)
    if hasattr(viewer, 'on_select'):
        viewer.on_select(follow_click)
    node_row = [node, busiest]

    # ------------------------------------------------------------- neuron labels
    if hasattr(viewer, 'set_node_labels') and getattr(viewer, 'labels', None) is not None:
        labels = w.Dropdown(options=['off', 'majority', 'all', 'counts'],
                            value=viewer.node_label_mode if viewer.node_labels_on else 'off',
                            description='labels', layout=w.Layout(width='220px'))

        def on_labels(_=None):
            if labels.value == 'off':
                viewer.show_node_labels(False)
            else:
                viewer.set_node_labels(labels.value)
            redraw()

        labels.observe(on_labels, names='value')
        node_row.append(labels)
    rows.append(w.HBox(node_row))

    # ------------------------------------------------------------- spheres: where to look
    if hasattr(viewer, 'set_view'):
        lat0, lon0 = viewer.centre
        lat = w.FloatSlider(value=round(lat0), min=-90, max=90, step=1, description='centre lat',
                            continuous_update=False)
        lon = w.FloatSlider(value=round(lon0), min=-180, max=180, step=1, description='centre lon',
                            continuous_update=False)
        projection = w.Dropdown(options=list(viewer.projections), value=viewer.projection_name,
                                description='projection')

        def on_view(_=None):
            if projection.value != viewer.projection_name:
                viewer.set_projection(projection.value)
            viewer.set_view(lat.value, lon.value)
            redraw()

        for c in (lat, lon, projection):
            c.observe(on_view, names='value')
        rows.append(w.HBox([lat, lon, projection]))

    if neuron is not None:
        on_node()
    box = w.VBox(rows)
    display(box, out)
    redraw()
    controls.last = (box, out)                        # for tests and for re-displaying the controls


def thumbnails(views: dict, ncols: int = 3, dpi: int = 60, width: float = 4.5, title: str | None = None):
    """
    A static overview: a small image of the map of every viewer in {name: viewer}, in a grid (both
    backends). Viewers are rendered off-screen, so this does not disturb live figures.
    """
    import io
    import math

    import matplotlib.pyplot as plt

    images = []
    for name, view in views.items():
        buf = io.BytesIO()
        fig = view.fig
        box = 'tight'
        if hasattr(view, 'ax'):                       # a single map: just the map, without the side panels
            box = view.ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).expanded(1.02, 1.06)
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches=box)
        buf.seek(0)
        images.append((name, plt.imread(buf)))
    nrows = math.ceil(len(images) / ncols)
    aspect = images[0][1].shape[0] / images[0][1].shape[1] if images else 0.5
    with plt.ioff():
        fig, axes = plt.subplots(nrows, ncols, figsize=(width * ncols, width * aspect * nrows + 0.6), squeeze=False)
    for ax in axes.flat:
        ax.set_axis_off()
    for ax, (name, image) in zip(axes.flat, images, strict=False):
        ax.imshow(image)
        ax.set_title(name, fontsize=11)
    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    from IPython.display import display
    if WIDGET:
        display(fig.canvas)
    else:
        display(fig)
        plt.close(fig)


def pick(views: dict, first: str | None = None):
    """
    A drop-down to choose one of {name: viewer}, shown below with its controls. The viewers stay
    linked, so with live figures a neuron selected in one view is selected in the others as well.
    """
    import ipywidgets as w
    from IPython.display import clear_output, display

    choice = w.Dropdown(options=list(views), value=first or next(iter(views)), description='view',
                        layout=w.Layout(width='420px'))
    out = w.Output()

    def on_choice(_=None):
        with out:
            clear_output(wait=True)
            show(views[choice.value])
            controls(views[choice.value])

    choice.observe(on_choice, names='value')
    display(choice, out)
    on_choice()
