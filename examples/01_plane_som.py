# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
01 -- A Self-Organising Map on a flat grid (PlaneSOM): train it, query it, explore it.

PlaneSOM puts its neurons on mtGeodesicDome's Plane: a hexagonal grid (6 neighbours per neuron)
or a rectilinear one (4), either with borders or wrapping around like a torus.  Every neuron
holds a weight vector with one value per attribute of the data.

The script
  1. loads data (7 labelled clusters in 12 dimensions, or your CSV file) and standardises it,
  2. trains a PlaneSOM and reports how well it fits,
  3. shows a few queries: best matching units, hits, the neuron that wins most samples,
  4. opens two linked windows (see mt.geodesicsom.gui):
       * the map of the whole attribute vectors: each triangle (or square) between neighbouring
         neurons is coloured by the Euclidean distance between their attribute vectors -- dark
         basins are clusters, bright ridges are the borders.  Switch layers on the left; click a
         neuron to see its attribute vector.
       * one map per attribute, coloured by how much that attribute changes between neighbouring
         neurons.  Click marks a neuron on every map; double-click opens one attribute on its own.

Run:
    python examples/01_plane_som.py                                    # 20 x 30 hexagonal, with borders
    python examples/01_plane_som.py --topology torus                   # no borders
    python examples/01_plane_som.py --lattice rectilinear --rows 16 --cols 24
    python examples/01_plane_som.py --data wine   # a sample shipped with the package (iris, penguins, wine, clusters)
    python examples/01_plane_som.py --data mydata.csv --label-column species
    python examples/01_plane_som.py --save examples/output/01_plane_som --no-gui
"""
import argparse

import numpy as np
from mt.geodesicdome.grid.plane import Lattice, Topology

from mt.geodesicsom import datasets
from mt.geodesicsom.PlaneSOM import PlaneSOM
from mt.geodesicsom.som import InitializationType


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument('--data', default='clusters',
                        help="sample: 'clusters', 'animals', 'iris', 'penguins', 'wine'; 'colours'; or a CSV file")
    parser.add_argument('--label-column', help='CSV column holding class labels')
    parser.add_argument('--lattice', default='hexagonal', choices=['hexagonal', 'rectilinear'])
    parser.add_argument('--topology', default='plane', choices=['plane', 'torus'])
    parser.add_argument('--rows', type=int, default=20)
    parser.add_argument('--cols', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--save', metavar='PREFIX', help='save PREFIX_map.png and PREFIX_matrix.png')
    parser.add_argument('--no-gui', action='store_true')
    args = parser.parse_args()

    # 1. data -------------------------------------------------------------------------------------
    x, labels, names = datasets.load(args.data, args.label_column)   # a sample name or a CSV path
    if args.data != 'colours':
        x = datasets.standardise(x)          # so that no attribute dominates the Euclidean distance

    # 2. train ------------------------------------------------------------------------------------
    lattice = Lattice.Hexagonal if args.lattice == 'hexagonal' else Lattice.Rectilinear
    topology = Topology.Donut if args.topology == 'torus' else Topology.Plane
    som = PlaneSOM(args.rows, args.cols, lattice=lattice, topology=topology, seed=0)
    som.initialise(InitializationType.Linear, x)       # spread over the first two principal components
    print(f'{som}')
    print(f'  before training: quantisation error {som.quantisation_error(x):.3f}')
    som.train(x, epochs=args.epochs)                   # batch training; mode='online' also works
    print(f'  after {args.epochs} epochs: quantisation error {som.quantisation_error(x):.3f}, '
          f'topographic error {som.topographic_error(x):.3f}')

    # 3. query ------------------------------------------------------------------------------------
    bmu = som.bmu(x)                                   # best matching neuron of every sample
    hits = som.hits(x)                                 # samples won by every neuron
    busiest = int(np.argmax(hits))
    print(f'  {np.count_nonzero(hits)} of {som.n_nodes} neurons win at least one sample')
    print(f'  neuron {busiest} (column {busiest % som.col}, row {busiest // som.col}) wins {hits[busiest]} samples'
          + (f": {', '.join(sorted({str(v) for v in labels[bmu == busiest]}))}" if labels is not None else ''))
    print(f'  largest distance between neighbouring neurons: {som.edge_distance().max():.3f}')

    # 4. explore ----------------------------------------------------------------------------------
    if args.no_gui:
        import matplotlib
        matplotlib.use('Agg')
    from mt.geodesicsom.gui import explore

    views = explore(som, x, labels, names, select=busiest, show=False)
    if args.save:
        views.map.save(f'{args.save}_map.png', dpi=110)
        views.matrix.save(f'{args.save}_matrix.png', dpi=90)
        print(f'saved {args.save}_map.png and {args.save}_matrix.png')
    if not args.no_gui:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == '__main__':
    main()
