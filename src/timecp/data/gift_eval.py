"""GiftEval-format dataset wrapper.

Adapts the ``Salesforce/GiftEval`` and ``Datadog/BOOM`` datasets (columns:
``item_id``, ``start``, ``freq``, ``target``) into GluonTS-compatible
training / validation / test splits, following the logic in
``extern/gift-eval/src/gift_eval/data.py``.

Unlike the original, the HuggingFace dataset object is passed in directly
(obtained via :func:`~timecp.data.loader.load_dataset`) rather than being
resolved through an environment variable.
"""

# Copyright (c) 2023, Salesforce, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Original source: extern/gift-eval/src/gift_eval/data.py
# Modifications: accept pre-loaded hf_dataset instead of env-var path.

from __future__ import annotations

import math
from enum import Enum
from functools import cached_property
from typing import Iterable, Iterator

import datasets as hf_datasets
import pyarrow.compute as pc
from gluonts.dataset import DataEntry
from gluonts.dataset.common import ProcessDataEntry
from gluonts.dataset.split import TestData, TrainingDataset, split
from gluonts.itertools import Map
from gluonts.time_feature import norm_freq_str
from gluonts.transform import Transformation
from pandas.tseries.frequencies import to_offset
from toolz import compose

# ---------------------------------------------------------------------------
# Constants (mirrors gift-eval)
# ---------------------------------------------------------------------------

TEST_SPLIT = 0.1
MAX_WINDOW = 20

M4_PRED_LENGTH_MAP = {
    'A': 6,
    'Q': 8,
    'M': 18,
    'W': 13,
    'D': 14,
    'H': 48,
}

PRED_LENGTH_MAP = {
    'M': 12,
    'W': 8,
    'D': 30,
    'H': 48,
    'T': 48,
    'S': 60,
}

TFB_PRED_LENGTH_MAP = {
    'A': 6,
    'H': 48,
    'Q': 8,
    'D': 14,
    'M': 18,
    'W': 13,
    'U': 8,
    'T': 8,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class Term(Enum):
    """Forecast horizon multiplier."""

    SHORT = 'short'
    MEDIUM = 'medium'
    LONG = 'long'

    @property
    def multiplier(self) -> int:
        if self == Term.SHORT:
            return 1
        elif self == Term.MEDIUM:
            return 10
        else:  # LONG
            return 15


def _itemize_start(data_entry: DataEntry) -> DataEntry:
    data_entry['start'] = data_entry['start'].item()
    return data_entry


def _maybe_reconvert_freq(freq: str) -> str:
    """Map newer pandas frequency aliases back to the legacy short forms."""
    deprecated_map = {
        'Y': 'A',
        'YE': 'A',
        'QE': 'Q',
        'ME': 'M',
        'h': 'H',
        'min': 'T',
        's': 'S',
        'us': 'U',
    }
    return deprecated_map.get(freq, freq)


class _MultivariateToUnivariate(Transformation):
    """Split a multivariate target into separate univariate entries."""

    def __init__(self, field: str):
        self.field = field

    def __call__(
        self,
        data_it: Iterable[DataEntry],
        is_train: bool = False,
    ) -> Iterator:
        for data_entry in data_it:
            item_id = data_entry['item_id']
            val_ls = list(data_entry[self.field])
            for dim_id, val in enumerate(val_ls):
                univariate_entry = data_entry.copy()
                univariate_entry[self.field] = val
                univariate_entry['item_id'] = item_id + '_dim' + str(dim_id)
                yield univariate_entry


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class GiftEvalDataset:
    """Wraps a GiftEval-format HuggingFace dataset with GluonTS integration.

    GiftEval-format columns: ``item_id`` (str), ``start`` (timestamp),
    ``freq`` (str), ``target`` (list[float]).

    This class is compatible with ``Salesforce/GiftEval`` and datasets of the
    same schema such as ``Datadog/BOOM``.

    Parameters
    ----------
    hf_dataset:
        A ``datasets.Dataset`` already loaded in GiftEval format (use
        :func:`~timecp.data.loader.load_dataset` to obtain one).
    name:
        Human-readable name used when computing prediction lengths for
        M4-style datasets.  Pass the subset name, e.g. ``"m4_hourly"``.
    term:
        Forecast horizon multiplier — ``"short"`` (1×), ``"medium"`` (10×),
        or ``"long"`` (15×).
    to_univariate:
        If *True*, multivariate series are split into individual univariate
        series (one per dimension).
    """

    def __init__(
        self,
        hf_dataset: hf_datasets.Dataset,
        name: str = '',
        term: Term | str = Term.SHORT,
        to_univariate: bool = False,
    ):
        self.hf_dataset = hf_dataset.with_format('numpy')
        self.name = name
        self.term = Term(term)

        process = ProcessDataEntry(
            self.freq,
            one_dim_target=(self.target_dim == 1),
        )
        self.gluonts_dataset = Map(compose(process, _itemize_start), self.hf_dataset)

        if to_univariate:
            self.gluonts_dataset = _MultivariateToUnivariate('target').apply(
                self.gluonts_dataset
            )

    # ------------------------------------------------------------------
    # Cached properties
    # ------------------------------------------------------------------

    @cached_property
    def freq(self) -> str:
        """Pandas frequency string for this dataset."""
        return self.hf_dataset[0]['freq']

    @cached_property
    def target_dim(self) -> int:
        """Number of target dimensions (1 for univariate)."""
        target = self.hf_dataset[0]['target']
        return target.shape[0] if len(target.shape) > 1 else 1

    @cached_property
    def past_feat_dynamic_real_dim(self) -> int:
        """Number of past dynamic real features (0 if absent)."""
        if 'past_feat_dynamic_real' not in self.hf_dataset[0]:
            return 0
        feat = self.hf_dataset[0]['past_feat_dynamic_real']
        return feat.shape[0] if len(feat.shape) > 1 else 1

    @cached_property
    def prediction_length(self) -> int:
        """Default prediction length based on frequency and term."""
        freq = norm_freq_str(to_offset(self.freq).name)
        freq = _maybe_reconvert_freq(freq)
        base = (
            M4_PRED_LENGTH_MAP[freq]
            if 'm4' in self.name.lower()
            else PRED_LENGTH_MAP[freq]
        )
        return self.term.multiplier * base

    @cached_property
    def windows(self) -> int:
        """Number of rolling test windows."""
        if 'm4' in self.name.lower():
            return 1
        w = math.ceil(TEST_SPLIT * self._min_series_length / self.prediction_length)
        return min(max(1, w), MAX_WINDOW)

    @cached_property
    def _min_series_length(self) -> int:
        if self.hf_dataset[0]['target'].ndim > 1:
            lengths = pc.list_value_length(
                pc.list_flatten(
                    pc.list_slice(self.hf_dataset.data.column('target'), 0, 1)
                )
            )
        else:
            lengths = pc.list_value_length(self.hf_dataset.data.column('target'))
        return min(lengths.to_numpy())

    @cached_property
    def sum_series_length(self) -> int:
        """Total number of time-step observations across all series."""
        if self.hf_dataset[0]['target'].ndim > 1:
            lengths = pc.list_value_length(
                pc.list_flatten(self.hf_dataset.data.column('target'))
            )
        else:
            lengths = pc.list_value_length(self.hf_dataset.data.column('target'))
        return sum(lengths.to_numpy())

    # ------------------------------------------------------------------
    # GluonTS split properties
    # ------------------------------------------------------------------

    @property
    def training_dataset(self) -> TrainingDataset:
        """GluonTS training split (all data before the first test window)."""
        training_dataset, _ = split(
            self.gluonts_dataset,
            offset=-self.prediction_length * (self.windows + 1),
        )
        return training_dataset

    @property
    def validation_dataset(self) -> TrainingDataset:
        """GluonTS validation split."""
        validation_dataset, _ = split(
            self.gluonts_dataset,
            offset=-self.prediction_length * self.windows,
        )
        return validation_dataset

    @property
    def test_data(self) -> TestData:
        """GluonTS rolling-window test split."""
        _, test_template = split(
            self.gluonts_dataset,
            offset=-self.prediction_length * self.windows,
        )
        return test_template.generate_instances(
            prediction_length=self.prediction_length,
            windows=self.windows,
            distance=self.prediction_length,
        )
