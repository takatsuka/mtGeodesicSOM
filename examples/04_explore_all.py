# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
04 -- Everything at once: train a GeodesicSOM and open all its interactive views, linked.

Two windows open and rotate together -- drag in either and both follow:

  * SOMViewer: the sphere coloured by the Euclidean distance between the attribute vectors of
    neighbouring neurons, with layers for the U-matrix, each attribute (value or difference),
    PCA colour, hits and classes.  Click a neuron to see its whole attribute vector.
  * ComponentMatrixViewer: one map per attribute, each coloured by how much that attribute
    changes between neighbouring neurons.  Double-click a map to open it on its own (also linked).

This is the same as `python -m mt.geodesicsom`; the code below shows the two library calls.

Run:
    python examples/04_explore_all.py
    python examples/04_explore_all.py wine                         # a sample shipped with the package
    python examples/04_explore_all.py path/to/data.csv species     # CSV file and (optional) label column
"""
import sys

from mt.geodesicsom import datasets
from mt.geodesicsom.GeodesicSOM import GeodesicSOM
from mt.geodesicsom.gui import explore

# 1. data: a sample name ('clusters', 'animals', 'iris', 'penguins', 'wine') or a CSV file from the command line;
#    by default the demo clusters (7 labelled clusters, 12 attributes)
name = sys.argv[1] if len(sys.argv) > 1 else 'clusters'
data, labels, names = datasets.load(name, label_column=sys.argv[2] if len(sys.argv) > 2 else None)
x = datasets.standardise(data)              # so that no attribute dominates the Euclidean distance

# 2. train a spherical SOM: 1002 neurons, each with one weight per attribute
som = GeodesicSOM(10, seed=0)
som.initialise(dataset=x)                   # linear (PCA) initialisation
som.train(x, epochs=30)
print(f'{som}: QE {som.quantisation_error(x):.3f}, TE {som.topographic_error(x):.3f}')

# 3. open every view, rotating together
views = explore(som, x, labels=labels, feature_names=names)

# (to build them without windows, e.g. for saving images:
#  views = explore(som, x, labels, names, show=False); views.matrix.save('matrix.png'))
