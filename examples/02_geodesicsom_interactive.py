# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
02 -- Train a spherical SOM (GeodesicSOM) and explore it interactively.

Every neuron holds an attribute vector with as many values as the data.  The map opens on the
"Neighbour distance" layer: the triangles between neurons are coloured by the Euclidean distance
between the attribute vectors of their neurons, so clusters show as dark basins separated by
bright ridges.  Drag to rotate the sphere, click a neuron to see its attribute vector, and switch
layers on the left (U-matrix, component planes, PCA colour, hits, classes).

Run:
    python examples/02_geodesicsom_interactive.py                      # 7 labelled clusters, 12 attributes
    python examples/02_geodesicsom_interactive.py --data wine   # a sample shipped with the package
    python examples/02_geodesicsom_interactive.py --data colours       # random RGB colours (3 attributes)
    python examples/02_geodesicsom_interactive.py --data mydata.csv --label-column species
    python examples/02_geodesicsom_interactive.py --save examples/output/02_geodesicsom.png --no-gui

A CSV needs a header row; its numeric columns become the attributes (empty cells = missing),
and --label-column names an optional column of class labels.
"""
import argparse

import numpy as np

from mt.geodesicsom import datasets
from mt.geodesicsom.GeodesicSOM import GeodesicSOM


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument('--data', default='clusters',
                        help="sample: 'clusters', 'animals', 'iris', 'penguins', 'wine'; 'colours'; or a CSV file")
    parser.add_argument('--label-column', help='CSV column holding class labels')
    parser.add_argument('--freq', type=int, default=10, help='dome frequency (10 -> 1002 neurons)')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--mode', default='batch', choices=['batch', 'online'])
    parser.add_argument('--no-standardise', action='store_true', help='train on the raw attribute values')
    parser.add_argument('--projection', default='Equal Earth')
    parser.add_argument('--save', help='save the view to this image file')
    parser.add_argument('--no-gui', action='store_true')
    parser.add_argument('--seed', type=int, default=3)
    args = parser.parse_args()

    x, y, names = datasets.load(args.data, args.label_column)   # a sample name or a CSV path

    # attributes on different scales would otherwise dominate the Euclidean distance
    if not args.no_standardise and args.data != 'colours':
        x = datasets.standardise(x)

    som = GeodesicSOM(args.freq, seed=args.seed)
    som.initialise(dataset=x)
    som.train(x, epochs=args.epochs, mode=args.mode)
    print(f'{som}: {len(x)} samples, QE {som.quantisation_error(x):.3f}, TE {som.topographic_error(x):.3f}')

    if args.no_gui:
        import matplotlib
        matplotlib.use('Agg')
    from mt.geodesicsom.gui import SOMViewer

    viewer = SOMViewer(som, x, labels=y, feature_names=names, projection=args.projection)
    if args.save:
        busiest = int(np.argmax(viewer.hits))
        viewer.select_node(busiest)
        viewer.save(args.save, dpi=110)
        print(f'saved {args.save}')
    if not args.no_gui:
        viewer.show()


if __name__ == '__main__':
    main()
