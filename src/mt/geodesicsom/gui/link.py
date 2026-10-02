# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""link_views: keep several viewers in step (rotation and selected neuron)."""
import numpy as np


def link_views(*views, rotation: bool = True, selection: bool = True):
    """
    Keeps viewers in step, in the same window or in separate ones:

    * rotation  -- turning any one (by mouse, keys or code) turns the others to the same orientation;
                   for viewers with `rotation`, `set_rotation()` and `on_rotate()`: SOMViewer,
                   ComponentMatrixViewer and mt.geodesicdome's ProjectionViewer
    * selection -- selecting a neuron in one selects it in the others; for viewers with `selected`,
                   `select_node()` and `on_select()`: SOMViewer, ComponentMatrixViewer, PlaneSOMViewer,
                   PlaneComponentMatrixViewer

    Viewers without a capability are simply left out of it.  Returns the views.
    """
    busy = [False]

    def follow_rotation(source):
        if busy[0]:
            return
        busy[0] = True
        try:
            for view in views:
                if view is not source and hasattr(view, 'set_rotation') \
                        and not np.allclose(view.rotation, source.rotation):
                    view.set_rotation(source.rotation)
        finally:
            busy[0] = False

    def follow_selection(source):
        if busy[0]:
            return
        busy[0] = True
        try:
            for view in views:
                if view is not source and hasattr(view, 'select_node') and view.selected != source.selected:
                    view.select_node(source.selected)
        finally:
            busy[0] = False

    for view in views:
        if rotation and hasattr(view, 'on_rotate'):
            view.on_rotate(follow_rotation)
        if selection and hasattr(view, 'on_select'):
            view.on_select(follow_selection)
    return views
