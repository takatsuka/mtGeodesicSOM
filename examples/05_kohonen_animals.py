# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
05 -- Kohonen's animals: the classic "semantic map" (Ritter & Kohonen 1989) on a sphere.

Sixteen animals (dove, hen, duck, goose, owl, hawk, eagle, fox, dog, wolf, cat, tiger, lion, horse,
zebra, cow) are described by 13 yes/no attributes: small, medium, big; two legs, four legs, hair,
hooves, mane, feathers; likes to hunt, run, fly, swim.  Nothing tells the SOM which animals are
birds, hunters or grazers, yet the trained map puts them in those groups.  Every neuron is labelled
with the animals it wins, so the groups can be read straight off the map.

Owl and hawk have the same attributes, and so do horse and zebra.  With --symbols the data is set up
as in the paper: each attribute vector is scaled to unit length and a small "symbol" part (0.2 for
the animal itself, 0 for the others) is added, so every animal is a different input.

Run:
    python examples/05_kohonen_animals.py                     # the 13 attributes; map + one map per attribute
    python examples/05_kohonen_animals.py --symbols           # the paper's semantic map (map only)
    python examples/05_kohonen_animals.py --lattice hexagonal # a flat 10 x 15 map, as in the paper
    python examples/05_kohonen_animals.py --save examples/output/05_animals --no-gui
"""
import argparse

import numpy as np

from mt.geodesicsom import datasets
from mt.geodesicsom.gui.explorer import make_som


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument('--symbols', action='store_true',
                        help="add the paper's symbol part (0.2 x one-hot animal code; attributes scaled to length 1)")
    parser.add_argument('--lattice', default='sphere', choices=['sphere', 'hexagonal', 'rectilinear'])
    parser.add_argument('--freq', type=int, default=4, help='sphere: dome frequency (default 4: 162 neurons)')
    parser.add_argument('--rows', type=int, default=10, help='plane: rows (default 10)')
    parser.add_argument('--cols', type=int, default=15, help='plane: columns (default 15)')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--label-size', type=float, default=9, metavar='POINTS')
    parser.add_argument('--save', metavar='PREFIX', help='save PREFIX_map.png (and PREFIX_matrix.png)')
    parser.add_argument('--no-gui', action='store_true')
    parser.add_argument('--seed', type=int, default=3)
    args = parser.parse_args(argv)
    if args.no_gui:
        import matplotlib
        matplotlib.use('Agg')
    from mt.geodesicsom.gui import explore

    # 16 x 13 binary attributes (or 16 x 29 with the symbol part); labels are the animals' names.
    # The attributes are all 0/1 already, so they are not standardised.
    x, animals, names = datasets.animals(symbols=args.symbols)

    som = make_som(args.lattice, frequency=args.freq, rows=args.rows, cols=args.cols, seed=args.seed)
    som.initialise(dataset=x)
    som.train(x, epochs=args.epochs)
    print(f'{som}: QE {som.quantisation_error(x):.3f}, TE {som.topographic_error(x):.3f}')
    for node, winners in enumerate(som.bmu_labels(x, animals, mode='all')):
        if winners:
            print(f'  neuron {node:4d}: {winners}')

    # label every neuron with all the animals it wins ('owl, hawk'); with the symbol part the matrix
    # would have 29 panels, 16 of them just the animal codes, so only the map is opened then
    view = None
    if args.lattice == 'sphere':                  # turn the sphere so the animals are in the middle of the map
        centre = som.points[som.bmu(x)].mean(axis=0)
        view = (np.degrees(np.arcsin(centre[2] / np.linalg.norm(centre))), np.degrees(np.arctan2(centre[1], centre[0])))
    views = explore(som, x, labels=animals, feature_names=names, node_labels='all', label_size=args.label_size,
                    matrix=not args.symbols, view=view, show=False)
    if args.save:
        views.map.save(f'{args.save}_map.png', dpi=110)
        print(f'saved {args.save}_map.png')
        if views.matrix is not None:
            views.matrix.save(f'{args.save}_matrix.png', dpi=90)
            print(f'saved {args.save}_matrix.png')
    if not args.no_gui:
        import matplotlib.pyplot as plt
        plt.show()
    return views


if __name__ == '__main__':
    main()
