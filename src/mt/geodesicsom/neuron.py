# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.

import numpy as np
from mt.geodesicdome.vertex import Vertex
from numpy import array


class INeuron:
    pass


class Neuron(INeuron):
    def __init__(self, vertex: Vertex, weights: array = None, dimension: int = 0):
        self.vertex = vertex
        if weights is not None:
            self.weights: array = weights
            self.dimension: int = len(self.weights)
        elif dimension > 0:
            self.dimension: int = dimension
            self.weights: array = np.zeros(dimension)
        else:
            self.weights = None
            self.dimension = 0

        self.labels: list[str] = []

    def distance(self, neuron: INeuron) -> float:
        distance: float = 0
        for i in range(len(self.weights)):
            if np.isnan(neuron.weights[i]):  # missing data
                continue

            dd: float = neuron.weights[i] - self.weights[i]
            distance += dd * dd

        return np.sqrt(distance)
