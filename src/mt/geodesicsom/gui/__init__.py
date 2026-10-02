# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
mt.geodesicsom.gui -- interactive visualisation of trained SOMs (needs matplotlib:
`pip install "mtgeodesicsom[interactive]"`).

    from mt.geodesicsom.gui import explore, SOMViewer, ComponentMatrixViewer, link_views

Spherical maps (GeodesicSOM), rotatable with the mouse
    SOMViewer                     one map: neighbour distance between neurons, U-matrix, component
                                  planes, per-attribute differences, PCA colour, hits, classes;
                                  click a neuron to see its attribute vector
    ComponentMatrixViewer         one synchronised map per attribute, in a grid

Flat maps (PlaneSOM: hexagonal or rectilinear, plane or torus)
    PlaneSOMViewer                the same layers and inspector as SOMViewer
    PlaneComponentMatrixViewer    one map per attribute, in a grid

Both kinds
    explore(som, data, ...)       opens the right pair of views for the SOM, linked
    train_and_explore(data, ...)  trains a GeodesicSOM or PlaneSOM first
    link_views(a, b, ...)         keeps viewers rotating (spheres) and selecting together

Modules: som_viewer, matrix_viewer, plane_viewer, inspector (shared layers and side panel),
link, explorer (also `python -m mt.geodesicsom`).
"""
_EXPORTS = {
    'SOMViewer': 'mt.geodesicsom.gui.som_viewer',
    'ComponentMatrixViewer': 'mt.geodesicsom.gui.matrix_viewer',
    'PlaneSOMViewer': 'mt.geodesicsom.gui.plane_viewer',
    'PlaneComponentMatrixViewer': 'mt.geodesicsom.gui.plane_viewer',
    'link_views': 'mt.geodesicsom.gui.link',
    'explore': 'mt.geodesicsom.gui.explorer',
    'train_and_explore': 'mt.geodesicsom.gui.explorer',
    'Views': 'mt.geodesicsom.gui.explorer',
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    # imported on first use, so `import mt.geodesicsom.gui` itself does not load matplotlib
    if name in _EXPORTS:
        import importlib
        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
