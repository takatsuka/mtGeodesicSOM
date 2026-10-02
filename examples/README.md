# Examples

Run from the repository root, e.g. `python examples/01_plane_som.py`
(after `./setup_env.sh`, or `pip install -e .`).

| Script | Shows |
|---|---|
| `01_plane_som.py` | **a flat SOM:** train a `PlaneSOM` (hexagonal/rectilinear, plane/torus), query it, and open its two linked views (`--lattice`, `--topology`, `--rows`, `--cols`, `--data file.csv`, `--save`) |
| `02_geodesicsom_interactive.py` | **interactive:** train a `GeodesicSOM` and explore it in `SOMViewer` (`--data clusters\|colours\|file.csv`, `--label-column`, `--freq`, `--epochs`, `--mode`, `--save`) |
| `03_component_matrix.py` | **interactive:** one synchronised map per attribute, coloured by that attribute's difference between neighbouring neurons (`--mode difference\|value`, `--shared-scale`, `--ncols`, `--save`) |
| `04_explore_all.py [file.csv [label column]]` | **everything at once:** trains a `GeodesicSOM` and opens both views, linked, with `explore()` (same as `python -m mt.geodesicsom`) |
| `05_kohonen_animals.py` | **Kohonen's animals:** the classic semantic map (Ritter & Kohonen 1989) — 16 animals, 13 yes/no attributes, each neuron labelled with the animals it wins (`--symbols`, `--lattice hexagonal`, `--save`) |

`notebooks/` has the same examples as Jupyter notebooks, with interactive controls
(`pip install -e ".[notebooks]"`, then `jupyter lab examples/notebooks`).
