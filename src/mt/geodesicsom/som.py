# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
from abc import ABCMeta, abstractmethod
from enum import Enum

from mt.geodesicdome.manifold import Manifold


class InitializationType(Enum):
    Random = 0
    Linear = 1


class SOM(metaclass=ABCMeta):
    def __init__(self, grid: Manifold):
        self.set_grid(grid)

    def set_grid(self, grid: Manifold):
        self.grid = grid

    @abstractmethod
    def initialise(self, type: InitializationType = InitializationType.Linear):
        print(f'SOM initialised with {type}')

    @abstractmethod
    def train(self):
        print('SOM trained')
