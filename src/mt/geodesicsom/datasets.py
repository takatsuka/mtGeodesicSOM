# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
Sample data and loaders for trying out a SOM (numpy only).

    from mt.geodesicsom import datasets

    ds = datasets.load_sample('penguins')        # sample data shipped with the package (see samples())
    ds = datasets.read_csv('data.csv', label_column='species')   # your own CSV file
    ds = datasets.load('iris')                   # a sample name, 'clusters', 'colours' or a CSV path
    ds = datasets.animals(symbols=True)          # Kohonen's animals, set up as a "semantic map"
    x = datasets.standardise(ds.data)            # zero mean, unit variance per attribute

Sample data (CSV files in mt/geodesicsom/data; sources and licences in data/README.md)
    animals     16 x 13,  binary attributes, one label per animal (Ritter & Kohonen 1989)
    clusters    560 x 12, 7 labelled Gaussian clusters (synthetic; the same as clusters())
    iris        150 x 4,  3 species (Fisher 1936)
    penguins    342 x 4,  3 species (Palmer penguins, CC0; 2 rows without measurements are skipped)
    wine        178 x 13, 3 cultivars (UCI Wine, CC BY 4.0)

Generated on the fly: clusters(...) with any number of clusters and attributes, colours() (random RGB),
animals(symbols=True) (the animals with Ritter & Kohonen's symbol part added).

Every loader returns a Dataset(data, labels, feature_names): data is (samples, attributes) with
NaN for missing values; labels is an array or None.
"""
import csv
from collections.abc import Sequence
from importlib import resources
from typing import NamedTuple

import numpy as np


class Dataset(NamedTuple):
    data: np.ndarray
    labels: np.ndarray | None
    feature_names: Sequence[str]


# name -> (file, label column, description)
SAMPLES: dict[str, tuple] = {
    'animals': ('animals.csv', 'animal', '16 animals x 13 binary attributes, one label each (Ritter & Kohonen 1989)'),
    'clusters': ('clusters.csv', 'cluster', '560 samples x 12 attributes, 7 labelled Gaussian clusters (synthetic)'),
    'iris': ('iris.csv', 'species', '150 flowers x 4 measurements, 3 species (Fisher 1936)'),
    'penguins': ('penguins.csv', 'species', '342 penguins x 4 measurements, 3 species (Palmer penguins, CC0)'),
    'wine': ('wine.csv', 'cultivar', '178 wines x 13 chemical measurements, 3 cultivars (UCI, CC BY 4.0)'),
}


def samples() -> dict[str, str]:
    """The sample data sets shipped with the package: {name: description}."""
    return {name: info[2] for name, info in SAMPLES.items()}


def sample_path(name: str) -> str:
    """The path of a sample's CSV file (e.g. to open it elsewhere, or for `--data`)."""
    if name not in SAMPLES:
        raise ValueError(f'unknown sample {name!r}; choose from {list(SAMPLES)}')
    return str(resources.files('mt.geodesicsom').joinpath('data', SAMPLES[name][0]))


def load_sample(name: str) -> Dataset:
    """Loads a sample shipped with the package ('animals', 'clusters', 'iris', 'penguins' or 'wine')."""
    path = sample_path(name)                                  # raises for an unknown name
    return read_csv(path, label_column=SAMPLES[name][1], verbose=False)


def load(name: str, label_column: str | None = None, seed=3) -> Dataset:
    """
    One entry point for scripts: a sample name ('animals', 'clusters', 'iris', 'penguins', 'wine'), 'colours'
    (random RGB), or the path of a CSV file (with `label_column` naming its label column, if any).
    """
    if name == 'colours':
        return colours(seed=seed)
    if name == 'clusters' and seed != 3:
        return clusters(seed=seed)
    if name in SAMPLES:
        return load_sample(name)
    return read_csv(name, label_column)


def clusters(n_clusters: int = 7, dim: int = 12, per_cluster: int = 80, seed=3) -> Dataset:
    """Labelled Gaussian clusters ('cluster A', 'cluster B', ...) with random centres and spreads."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(scale=3.0, size=(n_clusters, dim))
    spread = rng.uniform(0.4, 1.0, size=n_clusters)
    x = np.vstack([c + s * rng.standard_normal((per_cluster, dim)) for c, s in zip(centres, spread, strict=True)])
    names = [f'cluster {chr(65 + k)}' if k < 26 else f'cluster {k + 1}' for k in range(n_clusters)]
    return Dataset(x, np.repeat(names, per_cluster), [f'feature {i + 1}' for i in range(dim)])


def animals(symbols: bool = False, symbol_weight: float = 0.2) -> Dataset:
    """
    Kohonen's animal data: 16 animals described by 13 binary attributes (size, legs, hair, hooves, mane,
    feathers; likes to hunt, run, fly, swim), from Ritter & Kohonen, "Self-organizing semantic maps" (1989).
    The labels are the animals' names, so each sample has its own label.

    With `symbols=True` the data is set up as in the paper's semantic map: each attribute vector is scaled to
    unit length, and a 16-value "symbol" part is put in front, which is `symbol_weight` (0.2 in the paper) for
    the animal itself and 0 otherwise.  The symbol part tells apart animals whose attributes are the same
    (owl and hawk, horse and zebra); the attributes, being the larger part, decide the layout of the map.
    Train on this data as it is (do not standardise it).
    """
    ds = load_sample('animals')
    if not symbols:
        return ds
    attributes = ds.data / np.linalg.norm(ds.data, axis=1, keepdims=True)
    symbol = symbol_weight * np.eye(len(ds.data))
    return Dataset(np.hstack([symbol, attributes]), ds.labels,
                   [f'is {name}' for name in ds.labels] + list(ds.feature_names))


def colours(n: int = 1500, seed=3) -> Dataset:
    """Random RGB colours in [0, 1): the classic SOM demo."""
    return Dataset(np.random.default_rng(seed).random((n, 3)), None, ['red', 'green', 'blue'])


def read_csv(path: str, label_column: str | None = None, verbose: bool = True) -> Dataset:
    """
    Reads a CSV file with a header row.  Numeric columns become attributes (empty cells and NA are
    missing values); other columns are skipped, except `label_column`, which gives the labels.
    Rows with no numeric value at all are left out.
    """
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f'{path} has no data rows')
    if label_column is not None and label_column not in rows[0]:
        raise ValueError(f'{path} has no column {label_column!r}; columns: {list(rows[0])}')
    labels = np.array([r[label_column] for r in rows]) if label_column else None

    def number(text):
        text = (text or '').strip()
        return float(text) if text and text.upper() not in ('NA', 'NAN', 'N/A') else np.nan

    names, columns = [], []
    for name in rows[0]:
        if name == label_column:
            continue
        try:
            columns.append([number(r[name]) for r in rows])
            names.append(name)
        except ValueError:
            if verbose:
                print(f'skipping non-numeric column {name!r}')
    if not names:
        raise ValueError(f'{path} has no numeric columns')
    data = np.array(columns).T
    keep = ~np.isnan(data).all(axis=1)
    if not keep.all() and verbose:
        print(f'skipping {int((~keep).sum())} rows with no numeric values')
    return Dataset(data[keep], labels[keep] if labels is not None else None, names)


def standardise(x: np.ndarray) -> np.ndarray:
    """Zero mean and unit variance per attribute (NaN-aware), so no attribute dominates the Euclidean distance."""
    x = np.asarray(x, dtype=float)
    return (x - np.nanmean(x, axis=0)) / (np.nanstd(x, axis=0) + 1e-12)
