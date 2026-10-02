# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
Quick demo for running from PyCharm (the green Run button) or `python main.py`.

Trains a GeodesicSOM and opens all its interactive views, linked so that they rotate together:
the map of the whole attribute vectors (SOMViewer) and one map per attribute
(ComponentMatrixViewer).  Same options as `python -m mt.geodesicsom`, for example

    python main.py --data colours
    python main.py --lattice hexagonal                  # a flat PlaneSOM instead of the sphere
    python main.py --lattice rectilinear --topology torus
    python main.py --data mydata.csv --label-column species
    python main.py --only matrix --matrix-mode value
    python main.py --save output/demo --no-gui

This file is for development only; it is not part of the installed package.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from mt.geodesicsom import __version__  # noqa: E402
from mt.geodesicsom.gui.explorer import main  # noqa: E402

if __name__ == "__main__":
    print(f"mt.geodesicsom {__version__}")
    main()
