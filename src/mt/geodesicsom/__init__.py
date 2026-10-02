# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
mt.geodesicsom -- Self-Organising Maps on geodesic domes and flat grids.

Modules
    mt.geodesicsom.GeodesicSOM   GeodesicSOM: SOM on a geodesic dome (sphere) -- no borders
    mt.geodesicsom.PlaneSOM      PlaneSOM: SOM on a flat hexagonal/rectilinear grid, with borders or as a torus
    mt.geodesicsom.lattice_som   LatticeSOM: training and measurements shared by both (batch/online, U-matrix, ...)
    mt.geodesicsom.datasets      demo data, CSV loading, standardisation
    mt.geodesicsom.som           SOM abstract base class and InitializationType
    mt.geodesicsom.neuron        Neuron: a weight vector attached to a lattice vertex
    mt.geodesicsom.util          MISSING (the value used for missing data)
    mt.geodesicsom.gui           interactive visualisation (needs matplotlib): SOMViewer, ComponentMatrixViewer,
                                 PlaneSOMViewer, PlaneComponentMatrixViewer, link_views, explore();
                                 `python -m mt.geodesicsom`

Everything outside mt.geodesicsom.gui needs only numpy and mtgeodesicdome.  Training and queries
run on a GPU when torch (CUDA or Apple MPS) or cupy is installed, and otherwise on every CPU core
(see mt.geodesicdome.backend; choose with backend='cuda'/'mps'/'cpu' or MTGEODESIC_BACKEND).

The lattices come from mtgeodesicdome (mt.geodesicdome).  The spherical SOM
and its indexed geodesic data structure are described in
Y. Wu and M. Takatsuka, "Spherical self-organizing map using efficient indexed
geodesic data structure", Neural Networks 19(6-7):900-910, 2006.
doi:10.1016/j.neunet.2006.05.021

Note: `mt` itself is a namespace package (no __init__.py), so this
distribution installs alongside mtgeodesicdome and other mt.* libraries.
"""

__version__ = '1.1.2'
