# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
GeodesicSOM: a spherical Self-Organising Map whose neurons are the nodes of a geodesic dome.

    import numpy as np
    from mt.geodesicsom.GeodesicSOM import GeodesicSOM

    data = np.random.rand(500, 6)                  # 500 samples, 6 attributes
    som = GeodesicSOM(8)                           # 642 neurons on GeodesicDome(8)
    som.initialise(dataset=data)                   # linear (PCA) initialisation
    som.train(data, epochs=20)                     # batch training
    som.weights                                    # (642, 6): one weight vector per neuron

Every neuron has a weight vector with as many attributes as the data.  A sphere has no
border, so every neuron has the same neighbourhood (6 neighbours, 5 at the 12 icosahedron
corners) and there are no edge effects.  The indexed geodesic data structure is described in
Y. Wu and M. Takatsuka, Neural Networks 19(6-7):900-910, 2006. doi:10.1016/j.neunet.2006.05.021

Neurons are the *unique* points of the dome.  mtGeodesicDome stores points on the seams of its
unfolded net more than once; after training, every stored copy gets the same weight vector
through `vertex.set_data`, so mt.geodesicdome tools (e.g. ProjectionViewer(colors='data')) work.

Training, best matching units, U-matrix and the distances between neurons come from
mt.geodesicsom.lattice_som.LatticeSOM, shared with PlaneSOM.
"""

import numpy as np
from mt.geodesicdome.backend import Backend
from mt.geodesicdome.compute import DomeArrays
from mt.geodesicdome.grid.geodesicdome import GeodesicDome

from mt.geodesicsom.geometry import SphereDistance
from mt.geodesicsom.lattice_som import LatticeSOM


class GeodesicSOM(LatticeSOM):
    """
    Spherical SOM on a geodesic dome.

    :param grid: a GeodesicDome, or its frequency (default 8: 642 neurons)
    :param seed: seed or numpy Generator for random initialisation and online training
    :param backend: compute device -- None/'auto' (GPU if available, else every CPU core), 'cuda',
                    'mps', 'cupy', 'cpu', 'numpy' or a mt.geodesicdome.backend.Backend

    Distances between neurons are great-circle distances measured in *rings* (one ring = the
    mean distance between neighbouring neurons), so `sigma` means the same thing at any frequency.
    They are computed on the device from the neuron positions a block at a time, so the
    (neurons x neurons) matrix is only kept whole when it fits the device comfortably:
    frequency 64 (40 962 neurons) trains without ever holding its 6.7 GB matrix.

    Extra attributes: `dome` (the GeodesicDome), `points` ((n, 3) unit vectors of the neurons),
    `ring_length` (radians between neighbouring neurons).
    """

    def __init__(self, grid: GeodesicDome | int = 8, seed=None, backend: str | Backend | None = None):
        dome = GeodesicDome(grid) if isinstance(grid, (int, np.integer)) else grid
        super().__init__(dome, seed, backend)
        self.dome: GeodesicDome = dome
        mesh = DomeArrays.from_dome(dome)
        self.points = mesh.points
        self.ring_length: float = mesh.ring_length                          # radians per ring
        self._set_lattice(mesh.faces, SphereDistance(mesh.points, mesh.ring_length), mesh.index_map,
                          init_coords=self.points, edges=mesh.edges)

    def __repr__(self) -> str:
        return f'GeodesicSOM(frequency={self.dome.frequency}, neurons={self.n_nodes}, dim={self.dim})'
