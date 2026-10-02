# Example notebooks

The example scripts in `examples/` as Jupyter notebooks. Each one walks through its script in small, explained
steps, puts ipywidgets controls under every viewer, and ends with a **Try it** section to train again with other
settings.

| Notebook | Shows | Try it |
|---|---|---|
| `01_plane_som.ipynb` | a flat `PlaneSOM`: train, measure (QE, TE), query, and the two linked views | data, lattice, torus, size, epochs |
| `02_geodesicsom_interactive.ipynb` | a spherical `GeodesicSOM` in the `SOMViewer`, by controls and by code | data, frequency, epochs, batch/online, seed |
| `03_component_matrix.ipynb` | one map per attribute, which attributes make the borders, one attribute on its own | data, frequency, difference/value, colour scale |
| `04_explore_all.ipynb` | `explore()` and `train_and_explore()` | data, sphere or flat map |
| `05_kohonen_animals.ipynb` | Kohonen's animals: the semantic map, labelled with the animals | the paper's symbol part, flat map, size, seed |

## Running them

```bash
pip install -e ".[notebooks]"        # or ./setup_env.sh, which includes it
jupyter lab examples/notebooks
```

In PyCharm or VS Code, choose `~/.venvs/mtGeodesicSOM/bin/python` as the interpreter / kernel first.

They work straight from a checkout: the first cell imports `nb_setup.py` (in this folder), which puts `src/` on
the path and chooses the matplotlib backend.

* **With `ipympl`** (in the `notebooks` extra) the viewers are live, exactly as in a window: drag a sphere to
  rotate it, click a neuron, and use the radio buttons, check boxes and keys on the figure.
* **Without it** the viewers are static images; the controls under them still work and redraw them. Set the
  environment variable `MTG_NOTEBOOK_STATIC=1` to force this.

`nb_setup` also has the helpers the notebooks use, handy in your own: `controls(viewer, *linked)` (ipywidgets for
a `SOMViewer`, `PlaneSOMViewer` or `ComponentMatrixViewer`), `show(viewer)`, `thumbnails({name: viewer})` and
`pick({name: viewer})`.

The notebooks are stored without outputs.
