# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
PlaneSOM: a Self-Organising Map on a flat hexagonal or rectilinear grid, with borders or as a torus.

    import numpy as np
    from mt.geodesicdome.grid.plane import Lattice, Topology
    from mt.geodesicsom.PlaneSOM import PlaneSOM

    data = np.random.rand(500, 6)                          # 500 samples, 6 attributes
    som = PlaneSOM(row=20, col=30)                         # 600 neurons, hexagonal, with borders
    som.initialise(dataset=data)                           # linear (PCA) initialisation
    som.train(data, epochs=20)                             # batch training
    som.weights                                            # (600, 6): one weight vector per neuron

The lattice is mt.geodesicdome.grid.plane.Plane.  On a hexagonal lattice every neuron has 6
neighbours (odd rows are shifted by half a spacing), on a rectilinear one 4.  With
Topology.Donut the map wraps around at its edges like a torus (use an even number of rows
with a hexagonal torus), so, like GeodesicSOM, it has no border effects.

Training and all measurements (hits, U-matrix, the distances between neighbouring neurons per
edge/face/attribute, quantisation and topographic error) are shared with GeodesicSOM through
mt.geodesicsom.lattice_som.LatticeSOM.  Faces are triangles on a hexagonal grid and squares
on a rectilinear one; no faces are drawn across the wrap of a torus.
"""

import numpy as np
from mt.geodesicdome.backend import Backend
from mt.geodesicdome.grid.plane import Lattice, Plane, Topology
from numpy import ndarray

from mt.geodesicsom.geometry import PlaneDistance
from mt.geodesicsom.lattice_som import LatticeSOM

ROW_HEIGHT = {Lattice.Hexagonal: np.sqrt(3.0) / 2.0, Lattice.Rectilinear: 1.0}


class PlaneSOM(LatticeSOM):
    """
    SOM on a flat grid of `row` x `col` neurons.

    :param row, col: number of rows and columns
    :param dim: number of attributes (optional; taken from the data at initialisation)
    :param lattice: Lattice.Hexagonal (default, 6 neighbours) or Lattice.Rectilinear (4 neighbours)
    :param topology: Topology.Plane (default, with borders) or Topology.Donut (a torus)
    :param seed: seed or numpy Generator for random initialisation and online training
    :param backend: compute device -- None/'auto' (GPU if available, else every CPU core), 'cuda',
                    'mps', 'cupy', 'cpu', 'numpy' or a mt.geodesicdome.backend.Backend

    Neuron i sits at column i % col, row i // col.  Extra attributes: `row`, `col`,
    `positions` ((n, 2) map coordinates with neighbour spacing 1), `width`, `height` (map size).
    """

    def __init__(self, row: int, col: int, dim: int | None = None, lattice: Lattice = Lattice.Hexagonal,
                 topology: Topology = Topology.Plane, seed=None, backend: str | Backend | None = None):
        if row < 2 or col < 2:
            raise ValueError('a PlaneSOM needs at least 2 rows and 2 columns')
        super().__init__(Plane(col, row, lattice, topology), seed, backend)
        self.row, self.col = int(row), int(col)
        self.lattice, self.topology = lattice, topology
        self.dim = int(dim or 0)

        n = row * col
        xs, ys = np.arange(n) % col, np.arange(n) // col
        h = ROW_HEIGHT[lattice]
        shift = 0.5 * (ys % 2) if lattice == Lattice.Hexagonal else 0.0
        self.positions: ndarray = np.column_stack([xs + shift, ys * h])
        self.width, self.height = float(col), float(row * h)          # periods of the torus

        node_distance = PlaneDistance(self.positions, self.width, self.height, torus=topology == Topology.Donut)

        plane: Plane = self.grid
        faces = plane.get_all_triangles().reshape(-1, plane.get_number_of_vertices_per_face())
        edges = []
        for v in plane.get_all_vertices():                             # neighbours incl. across the wrap
            plane.unmark_vertices()
            edges += [(v.id, u.id) for u in plane.get_neighbours(v, False)]
        plane.unmark_vertices()

        span = np.maximum(self.positions.max(axis=0) - self.positions.min(axis=0), 1e-12)
        init_coords = 2.0 * (self.positions - self.positions.min(axis=0)) / span - 1.0
        self._set_lattice(faces, node_distance, index_map=np.arange(n), init_coords=init_coords,
                          edges=np.array(edges))

    def __repr__(self) -> str:
        return (f'PlaneSOM(row={self.row}, col={self.col}, lattice={self.lattice.name}, '
                f'topology={self.topology.name}, dim={self.dim})')
