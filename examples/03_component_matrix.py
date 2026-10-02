# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
03 -- One synchronised geodesic-dome map per attribute of a trained GeodesicSOM.

Every neuron holds a multi-attribute weight vector.  This shows all attributes at once, one
map each, in a grid.  In each map, the triangle between three neighbouring neurons is coloured
by how much THAT attribute changes between them (mean |w_a[k] - w_b[k]|), so you can see which
attributes form which cluster borders.  The first map is the Euclidean distance over all
attributes, for reference.

Drag any map and all of them rotate together.  Click to mark a neuron on every map (its value
of each attribute appears in the titles); double-click a map to open that attribute in a full
SOMViewer whose rotation stays linked to the grid.  v switches between |difference| and the
attribute value; c toggles one colour scale for all maps.

Run:
    python examples/03_component_matrix.py                    # 7 labelled clusters, 12 attributes
    python examples/03_component_matrix.py --data wine   # a sample shipped with the package
    python examples/03_component_matrix.py --data mydata.csv --label-column species
    python examples/03_component_matrix.py --save examples/output/03_component_matrix.png --no-gui
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
    parser.add_argument('--mode', default='difference', choices=['difference', 'value'])
    parser.add_argument('--shared-scale', action='store_true', help='one colour scale for all attribute maps')
    parser.add_argument('--no-total', action='store_true', help="leave out the 'all attributes' map")
    parser.add_argument('--ncols', type=int, help='maps per row')
    parser.add_argument('--projection', default='Equal Earth')
    parser.add_argument('--save', help='save the grid to this image file')
    parser.add_argument('--no-gui', action='store_true')
    parser.add_argument('--seed', type=int, default=3)
    args = parser.parse_args()

    x, y, names = datasets.load(args.data, args.label_column)   # a sample name or a CSV path
    if args.data != 'colours':
        x = datasets.standardise(x)

    som = GeodesicSOM(args.freq, seed=args.seed)
    som.initialise(dataset=x)
    som.train(x, epochs=args.epochs)
    print(f'{som}: {len(x)} samples, QE {som.quantisation_error(x):.3f}, TE {som.topographic_error(x):.3f}')

    if args.no_gui:
        import matplotlib
        matplotlib.use('Agg')
    from mt.geodesicsom.gui import ComponentMatrixViewer

    viewer = ComponentMatrixViewer(som, names, mode=args.mode, shared_scale=args.shared_scale,
                                   include_total=not args.no_total, ncols=args.ncols,
                                   projection=args.projection, data=x, labels=y)
    if args.save:
        viewer.select_node(int(np.argmax(som.hits(x))))
        viewer.save(args.save, dpi=90)
        print(f'saved {args.save}')
    if not args.no_gui:
        viewer.show()


if __name__ == '__main__':
    main()
