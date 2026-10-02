# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
All the visualisations of a trained SOM in one call, and the `python -m mt.geodesicsom` command.

    from mt.geodesicsom.gui import explore, train_and_explore

    # a GeodesicSOM or PlaneSOM you trained yourself
    views = explore(som, data, labels=labels, feature_names=names)

    # or train one and open everything
    som, views = train_and_explore(data, labels, names, frequency=10)                  # sphere
    som, views = train_and_explore(data, labels, names, lattice='hexagonal', rows=20, cols=30)   # plane

Two windows open, linked: selecting a neuron in one selects it in the other, and on a sphere
they also rotate together (drag either one).
    views.map      the whole attribute vectors: neighbour distance between neurons (Euclidean),
                   U-matrix, component planes, per-attribute differences, PCA colour, hits, classes;
                   click a neuron to see its attribute vector
                   (SOMViewer for a GeodesicSOM, PlaneSOMViewer for a PlaneSOM)
    views.matrix   one map per attribute, coloured by how much that attribute changes between
                   neighbouring neurons; double-click a map to open it on its own
                   (ComponentMatrixViewer / PlaneComponentMatrixViewer)

From a terminal (needs matplotlib):
    python -m mt.geodesicsom                                   # sphere; demo: 7 labelled clusters, 12 attributes
    python -m mt.geodesicsom --lattice hexagonal               # flat hexagonal PlaneSOM (20 x 30)
    python -m mt.geodesicsom --lattice rectilinear --topology torus --rows 24 --cols 36
    python -m mt.geodesicsom --data penguins                   # a sample shipped with the package (iris, wine, ...)
    python -m mt.geodesicsom --data mydata.csv --label-column species
    python -m mt.geodesicsom --save out/som --no-gui           # writes out/som_map.png, out/som_matrix.png
    python -m mt.geodesicsom --help
"""
import argparse
import os
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

from mt.geodesicsom import datasets
from mt.geodesicsom.GeodesicSOM import GeodesicSOM
from mt.geodesicsom.PlaneSOM import PlaneSOM


class Views(NamedTuple):
    map: object          # SOMViewer / PlaneSOMViewer, or None
    matrix: object       # ComponentMatrixViewer / PlaneComponentMatrixViewer, or None


def explore(som, data=None, labels: Sequence | None = None, feature_names: Sequence[str] | None = None, *,
            single: bool = True, matrix: bool = True, link: bool = True, layer: str = 'Neighbour distance',
            matrix_mode: str = 'difference', projection: str = 'Equal Earth',
            view: Sequence[float] | None = None, select: int | None = None,
            node_labels: str | None = None, sample_names: Sequence | None = None,
            label_size: float | None = None, show: bool = True) -> Views:
    """
    Opens the interactive views of a trained GeodesicSOM or PlaneSOM.

    :param som: a trained GeodesicSOM or PlaneSOM
    :param data, labels, feature_names: optional; data enables hits and sample markers, labels the class map
    :param single: open the single map (all layers, click-to-inspect)
    :param matrix: open the matrix (one map per attribute)
    :param link: keep the two views in step (selection; rotation on a sphere)
    :param layer: the layer the single map shows first
    :param matrix_mode: 'difference' or 'value' for the matrix
    :param projection, view: (sphere only) map projection and initial (lat, lon) centre in degrees
    :param select: neuron to select at the start (default: none)
    :param node_labels, sample_names: write the labels of the samples each neuron wins on the single map
                                      ('majority', 'all', 'counts', 'first'); see SOMViewer.set_node_labels
    :param label_size: font size of those labels in points (default 7)
    :param show: open the windows and wait until they are closed (False: just build them, e.g. to save)
    :return: Views(map, matrix)
    """
    from mt.geodesicsom.gui.link import link_views

    single_view = matrix_view = None
    if isinstance(som, GeodesicSOM):
        from mt.geodesicsom.gui.matrix_viewer import ComponentMatrixViewer
        from mt.geodesicsom.gui.som_viewer import SOMViewer
        if single:
            single_view = SOMViewer(som, data, labels=labels, feature_names=feature_names, layer=layer,
                                    projection=projection, view=view, node_labels=node_labels,
                                    sample_names=sample_names, label_size=label_size)
        if matrix:
            matrix_view = ComponentMatrixViewer(som, feature_names, mode=matrix_mode, projection=projection,
                                                view=view, data=data, labels=labels)
    elif isinstance(som, PlaneSOM):
        from mt.geodesicsom.gui.plane_viewer import PlaneComponentMatrixViewer, PlaneSOMViewer
        if single:
            single_view = PlaneSOMViewer(som, data, labels=labels, feature_names=feature_names, layer=layer,
                                         node_labels=node_labels, sample_names=sample_names,
                                         label_size=label_size)
        if matrix:
            matrix_view = PlaneComponentMatrixViewer(som, feature_names, mode=matrix_mode, data=data, labels=labels)
    else:
        raise TypeError(f'explore() shows a GeodesicSOM or a PlaneSOM, not {type(som).__name__}')

    if link and single_view is not None and matrix_view is not None:
        link_views(single_view, matrix_view)
    if select is not None:
        for v in (single_view, matrix_view):
            if v is not None:
                v.select_node(select)
    if show:
        import matplotlib.pyplot as plt
        plt.show()
    return Views(single_view, matrix_view)


def make_som(lattice: str = 'sphere', *, frequency: int = 10, rows: int = 20, cols: int = 30,
             topology: str = 'plane', seed=0):
    """
    A new, untrained SOM: lattice 'sphere' (GeodesicSOM of the given frequency), or 'hexagonal' /
    'rectilinear' (PlaneSOM of rows x cols, topology 'plane' or 'torus').
    """
    from mt.geodesicdome.grid.plane import Lattice, Topology
    if lattice == 'sphere':
        return GeodesicSOM(frequency, seed=seed)
    lattices = {'hexagonal': Lattice.Hexagonal, 'rectilinear': Lattice.Rectilinear}
    topologies = {'plane': Topology.Plane, 'torus': Topology.Donut}
    if lattice not in lattices:
        raise ValueError("lattice must be 'sphere', 'hexagonal' or 'rectilinear'")
    if topology not in topologies:
        raise ValueError("topology must be 'plane' or 'torus'")
    return PlaneSOM(rows, cols, lattice=lattices[lattice], topology=topologies[topology], seed=seed)


def train_and_explore(data, labels: Sequence | None = None, feature_names: Sequence[str] | None = None, *,
                      lattice: str = 'sphere', frequency: int = 10, rows: int = 20, cols: int = 30,
                      topology: str = 'plane', epochs: int = 30, mode: str = 'batch', standardise: bool = True,
                      seed=0, verbose: bool = True, **explore_options):
    """
    Trains a SOM on `data` (standardised first, unless standardise=False) and opens explore().

    :param lattice: 'sphere' (GeodesicSOM, `frequency`), or 'hexagonal' / 'rectilinear'
                    (PlaneSOM, `rows` x `cols`, `topology` 'plane' or 'torus')
    :return: (som, views); the SOM is trained on the standardised data if standardise=True
    """
    x = datasets.standardise(data) if standardise else np.asarray(data, dtype=float)
    som = make_som(lattice, frequency=frequency, rows=rows, cols=cols, topology=topology, seed=seed)
    som.initialise(dataset=x)
    som.train(x, epochs=epochs, mode=mode)
    if verbose:
        print(f'{som}: {len(x)} samples, quantisation error {som.quantisation_error(x):.3f}, '
              f'topographic error {som.topographic_error(x):.3f}')
    return som, explore(som, x, labels, feature_names, **explore_options)


# ------------------------------------------------------------------------------------ command line
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='python -m mt.geodesicsom',
        description='Train a SOM on a sphere (GeodesicSOM) or a flat grid (PlaneSOM) and explore it: a map of '
                    'the whole attribute vectors and one map per attribute.')
    parser.add_argument('--data', default='clusters',
                        help="a sample shipped with the package: 'clusters' (default), 'animals', 'iris', 'penguins', "
                             "'wine'; 'colours' (random RGB); or a CSV file with a header row")
    parser.add_argument('--label-column', help='CSV column holding class labels')
    parser.add_argument('--lattice', default='sphere', choices=['sphere', 'hexagonal', 'rectilinear'],
                        help='sphere (GeodesicSOM, default) or a flat grid (PlaneSOM)')
    parser.add_argument('--freq', type=int, default=10, help='sphere: dome frequency (default 10: 1002 neurons)')
    parser.add_argument('--rows', type=int, default=20, help='plane: rows (default 20)')
    parser.add_argument('--cols', type=int, default=30, help='plane: columns (default 30)')
    parser.add_argument('--topology', default='plane', choices=['plane', 'torus'],
                        help='plane: with borders (default) or wrapping around like a torus')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--mode', default='batch', choices=['batch', 'online'], help='training mode')
    parser.add_argument('--no-standardise', action='store_true', help='train on the raw attribute values')
    parser.add_argument('--projection', default='Equal Earth',
                        choices=['Equal Earth', 'Kavrayskiy VII', 'Wagner VI', 'Wagner III'],
                        help='sphere: map projection')
    parser.add_argument('--only', choices=['map', 'matrix'], help='open only one of the two views')
    parser.add_argument('--matrix-mode', default='difference', choices=['difference', 'value'])
    parser.add_argument('--node-labels', choices=['majority', 'all', 'counts'],
                        help='start with neuron labels on, in this style (the window has an on/off check box; key l)')
    parser.add_argument('--label-size', type=float, metavar='POINTS',
                        help='font size of the neuron labels (default 7; A-/A+ buttons or - / + keys in the window)')
    parser.add_argument('--no-link', action='store_true', help='do not keep the two views in step')
    parser.add_argument('--save', metavar='PREFIX', help='save PREFIX_map.png and PREFIX_matrix.png')
    parser.add_argument('--no-gui', action='store_true', help='do not open windows (use with --save)')
    parser.add_argument('--seed', type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> Views:
    args = build_parser().parse_args(argv)
    if args.no_gui:
        import matplotlib
        matplotlib.use('Agg')

    ds = datasets.load(args.data, args.label_column, args.seed)
    # colours are already in [0, 1] and the animals' attributes are 0/1: standardising would only distort them
    standardise = not args.no_standardise and args.data not in ('colours', 'animals')
    som, views = train_and_explore(ds.data, ds.labels, ds.feature_names, lattice=args.lattice,
                                   frequency=args.freq, rows=args.rows, cols=args.cols, topology=args.topology,
                                   epochs=args.epochs, mode=args.mode, standardise=standardise, seed=args.seed,
                                   single=args.only != 'matrix', matrix=args.only != 'map',
                                   link=not args.no_link, matrix_mode=args.matrix_mode, node_labels=args.node_labels,
                                   label_size=args.label_size,
                                   projection=args.projection, show=False)
    if args.save:
        folder = os.path.dirname(args.save)
        if folder:
            os.makedirs(folder, exist_ok=True)
        x = datasets.standardise(ds.data) if standardise else ds.data
        busiest = int(np.argmax(som.hits(x)))
        for suffix, v, dpi in (('map', views.map, 110), ('matrix', views.matrix, 90)):
            if v is not None:
                v.select_node(busiest)
                v.save(f'{args.save}_{suffix}.png', dpi=dpi)
                print(f'saved {args.save}_{suffix}.png')
    if not args.no_gui:
        import matplotlib.pyplot as plt
        plt.show()
    return views


if __name__ == '__main__':
    main()
