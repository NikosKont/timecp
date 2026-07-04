"""Tests for timecp.data — dataset loading, GiftEvalDataset, and to_fev conversion.

Runs entirely from local data already present in data/GiftEval/.
No network access is required for the core tests.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from unittest.mock import patch

import datasets as hf_datasets
import pandas as pd
import pytest

# Project root relative to this file (tests/ → project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_GIFT_EVAL_DIR = _PROJECT_ROOT / 'data' / 'GiftEval'
_HAS_LOCAL_GIFT_EVAL = _GIFT_EVAL_DIR.exists()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gift_eval_dataset(
    n_series: int = 3, n_steps: int = 100
) -> hf_datasets.Dataset:
    """Build a minimal in-memory GiftEval-format dataset."""
    start = datetime.datetime(2020, 1, 1)
    rows = []
    for i in range(n_series):
        rows.append(
            {
                'item_id': f'series_{i}',
                'start': start,
                'freq': 'H',
                'target': [float(j + i * 100) for j in range(n_steps)],
            }
        )
    return hf_datasets.Dataset.from_list(rows)


def _make_fev_dataset(n_series: int = 3, n_steps: int = 100) -> hf_datasets.Dataset:
    """Build a minimal in-memory FEV-format dataset."""
    start = pd.Timestamp('2020-01-01')
    timestamps = pd.date_range(start=start, periods=n_steps, freq='h').tolist()
    rows = []
    for i in range(n_series):
        rows.append(
            {
                'id': f'series_{i}',
                'timestamp': timestamps,
                'target': [float(j + i * 100) for j in range(n_steps)],
            }
        )
    return hf_datasets.Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# loader.py tests
# ---------------------------------------------------------------------------


class TestLoadDataset:
    def test_loads_from_local_disk(self, tmp_path: Path):
        """load_dataset should use local disk when the path exists."""
        from timecp.data import load_dataset

        # The loader constructs: cache_dir / DatasetName
        dataset_dir = tmp_path / 'Dataset'
        ds = _make_gift_eval_dataset()
        ds.save_to_disk(str(dataset_dir))

        loaded = load_dataset('Fake/Dataset', cache_dir=tmp_path)
        assert len(loaded) == len(ds)
        assert loaded.column_names == ds.column_names

    def test_falls_back_to_huggingface_when_no_local(self, tmp_path: Path):
        """load_dataset should call hf_datasets.load_dataset when no local copy."""
        from timecp.data import load_dataset

        fake_hf_dataset = _make_gift_eval_dataset()

        with patch(
            'timecp.data.loader.hf_datasets.load_dataset', return_value=fake_hf_dataset
        ) as mock_load:
            result = load_dataset(
                'Fake/Dataset',
                cache_dir=tmp_path / 'nonexistent',
                cache=False,
            )
        mock_load.assert_called_once()
        assert len(result) == len(fake_hf_dataset)

    def test_saves_to_cache_after_download(self, tmp_path: Path):
        """load_dataset should persist the dataset to disk when cache=True."""
        from timecp.data import load_dataset

        fake_hf_dataset = _make_gift_eval_dataset()
        # Loader constructs: cache_dir / DatasetName
        expected_save_path = tmp_path / 'Dataset'

        with patch(
            'timecp.data.loader.hf_datasets.load_dataset', return_value=fake_hf_dataset
        ):
            load_dataset('Fake/Dataset', cache_dir=tmp_path, cache=True)

        assert expected_save_path.exists()
        reloaded = hf_datasets.load_from_disk(str(expected_save_path))
        assert len(reloaded) == len(fake_hf_dataset)

    def test_subset_passed_to_huggingface(self, tmp_path: Path):
        """subset= should be forwarded as the 'name' kwarg to load_dataset."""
        from timecp.data import load_dataset

        fake_hf_dataset = _make_fev_dataset()

        with patch(
            'timecp.data.loader.hf_datasets.load_dataset', return_value=fake_hf_dataset
        ) as mock_load:
            load_dataset(
                'autogluon/chronos_datasets',
                subset='m4_hourly',
                cache_dir=tmp_path / 'nonexistent',
                cache=False,
            )

        call_kwargs = mock_load.call_args.kwargs
        assert call_kwargs.get('name') == 'm4_hourly'

    def test_default_cache_path_includes_dataset_name(self, tmp_path: Path):
        """Default local path should include the DatasetName component."""
        from timecp.data import load_dataset

        fake_hf_dataset = _make_gift_eval_dataset()
        expected_save_path = tmp_path / 'GiftEval'

        with patch(
            'timecp.data.loader.hf_datasets.load_dataset', return_value=fake_hf_dataset
        ):
            load_dataset('Salesforce/GiftEval', cache_dir=tmp_path, cache=True)

        assert expected_save_path.exists()

    def test_default_cache_path_includes_subset(self, tmp_path: Path):
        """When a subset is given the path should include it."""
        from timecp.data import load_dataset

        fake_hf_dataset = _make_fev_dataset()
        # Loader constructs: cache_dir / DatasetName / subset
        expected_save_path = tmp_path / 'chronos_datasets' / 'm4_hourly'

        with patch(
            'timecp.data.loader.hf_datasets.load_dataset', return_value=fake_hf_dataset
        ):
            load_dataset(
                'autogluon/chronos_datasets',
                subset='m4_hourly',
                cache_dir=tmp_path,
                cache=True,
            )

        assert expected_save_path.exists()

    @pytest.mark.skipif(
        not _HAS_LOCAL_GIFT_EVAL, reason='Local GiftEval data not available'
    )
    def test_loads_local_gift_eval_electricity_h(self):
        """Integration: load electricity/H from the local GiftEval copy."""
        from timecp.data import load_dataset

        ds = load_dataset(
            'Salesforce/GiftEval',
            subset='electricity/H',
        )
        assert len(ds) > 0
        assert 'target' in ds.column_names


# ---------------------------------------------------------------------------
# gift_eval.py tests
# ---------------------------------------------------------------------------


class TestGiftEvalDataset:
    def test_basic_properties(self):
        from timecp.data import GiftEvalDataset

        hf_ds = _make_gift_eval_dataset(n_series=5, n_steps=200)
        ds = GiftEvalDataset(hf_ds, name='test')

        assert ds.freq == 'H'
        assert ds.target_dim == 1
        assert ds.prediction_length > 0
        assert ds.windows >= 1

    def test_training_dataset_iterable(self):
        from timecp.data import GiftEvalDataset

        hf_ds = _make_gift_eval_dataset(n_series=3, n_steps=300)
        ds = GiftEvalDataset(hf_ds, name='test')

        train_items = list(ds.training_dataset)
        assert len(train_items) == 3
        assert 'target' in train_items[0]

    def test_term_multiplies_prediction_length(self):
        from timecp.data import GiftEvalDataset, Term

        hf_ds = _make_gift_eval_dataset(n_series=2, n_steps=500)
        ds_short = GiftEvalDataset(hf_ds, name='test', term=Term.SHORT)
        ds_medium = GiftEvalDataset(hf_ds, name='test', term=Term.MEDIUM)

        assert ds_medium.prediction_length == 10 * ds_short.prediction_length

    def test_m4_prediction_length(self):
        """M4 datasets should use the M4 prediction length map."""
        from timecp.data import GiftEvalDataset

        hf_ds = _make_gift_eval_dataset(n_series=2, n_steps=200)
        ds_m4 = GiftEvalDataset(hf_ds, name='m4_hourly')

        # M4 hourly prediction length should be positive and use M4 map
        assert ds_m4.prediction_length > 0

    def test_to_univariate_splits_dimensions(self):
        """to_univariate=True should split multivariate entries."""
        from timecp.data import GiftEvalDataset

        # Build 2D target dataset
        rows = [
            {
                'item_id': f's{i}',
                'start': datetime.datetime(2020, 1, 1),
                'freq': 'H',
                'target': [
                    [float(j) for j in range(100)],
                    [float(j + 1000) for j in range(100)],
                ],
            }
            for i in range(3)
        ]
        hf_ds = hf_datasets.Dataset.from_list(rows)
        ds = GiftEvalDataset(hf_ds, name='test', to_univariate=True)

        train_items = list(ds.training_dataset)
        # 3 series × 2 dims = 6 univariate series
        assert len(train_items) == 6

    @pytest.mark.skipif(
        not _HAS_LOCAL_GIFT_EVAL, reason='Local GiftEval data not available'
    )
    def test_gift_eval_dataset_from_local_electricity(self):
        """Integration: wrap the real local electricity/H dataset."""
        from timecp.data import GiftEvalDataset, load_dataset

        hf_ds = load_dataset('Salesforce/GiftEval', subset='electricity/H')
        ds = GiftEvalDataset(hf_ds, name='electricity/H')

        assert ds.target_dim == 1
        assert ds.prediction_length == 48  # H → 48, short term × 1
        assert ds.windows >= 1


# ---------------------------------------------------------------------------
# convert.py tests
# ---------------------------------------------------------------------------


class TestToFev:
    def test_column_rename(self):
        from timecp.data import to_fev

        hf_ds = _make_gift_eval_dataset(n_series=2, n_steps=50)
        fev_ds = to_fev(hf_ds)

        assert fev_ds.variate_names is None  # type: ignore[attr-defined]
        assert fev_ds.freq == 'H'  # type: ignore[attr-defined]
        assert 'id' in fev_ds.column_names
        assert 'timestamp' in fev_ds.column_names
        assert 'target' in fev_ds.column_names
        assert 'item_id' not in fev_ds.column_names
        assert 'start' not in fev_ds.column_names
        assert 'freq' not in fev_ds.column_names

    def test_timestamp_length_matches_target(self):
        from timecp.data import to_fev

        n_steps = 72
        hf_ds = _make_gift_eval_dataset(n_series=2, n_steps=n_steps)
        fev_ds = to_fev(hf_ds)

        for row in fev_ds:
            assert len(row['timestamp']) == n_steps
            assert len(row['target']) == n_steps

    def test_timestamps_are_correct(self):
        """Generated timestamps should start at 'start' with the given frequency."""
        from timecp.data import to_fev

        start = datetime.datetime(2021, 6, 1)
        ds = hf_datasets.Dataset.from_list(
            [{'item_id': 's0', 'start': start, 'freq': 'D', 'target': [1.0, 2.0, 3.0]}]
        )
        fev_ds = to_fev(ds)
        timestamps = fev_ds[0]['timestamp']

        assert len(timestamps) == 3
        expected = [
            pd.Timestamp('2021-06-01'),
            pd.Timestamp('2021-06-02'),
            pd.Timestamp('2021-06-03'),
        ]
        for t, e in zip(timestamps, expected):
            assert pd.Timestamp(t) == e

    def test_id_values_preserved(self):
        from timecp.data import to_fev

        hf_ds = _make_gift_eval_dataset(n_series=3, n_steps=20)
        fev_ds = to_fev(hf_ds)

        ids = fev_ds['id']
        assert set(ids) == {'series_0', 'series_1', 'series_2'}

    def test_raises_on_missing_columns(self):
        from timecp.data import to_fev

        bad_ds = hf_datasets.Dataset.from_list([{'id': 'x', 'target': [1.0]}])
        with pytest.raises(ValueError, match='missing required GiftEval columns'):
            to_fev(bad_ds)

    def test_custom_column_names(self):
        from timecp.data import to_fev

        hf_ds = _make_gift_eval_dataset(n_series=1, n_steps=10)
        fev_ds = to_fev(hf_ds, id_column='series_id', timestamp_column='time')

        assert 'series_id' in fev_ds.column_names
        assert 'time' in fev_ds.column_names

    def test_extra_columns_kept_by_default(self):
        """Extra columns beyond the core four should be preserved."""
        from timecp.data import to_fev

        rows = [
            {
                'item_id': 's0',
                'start': datetime.datetime(2020, 1, 1),
                'freq': 'H',
                'target': [1.0, 2.0],
                'category': 'energy',
            }
        ]
        hf_ds = hf_datasets.Dataset.from_list(rows)
        fev_ds = to_fev(hf_ds)

        assert 'category' in fev_ds.column_names
        assert fev_ds[0]['category'] == 'energy'

    def test_extra_columns_dropped_when_disabled(self):
        from timecp.data import to_fev

        rows = [
            {
                'item_id': 's0',
                'start': datetime.datetime(2020, 1, 1),
                'freq': 'H',
                'target': [1.0, 2.0],
                'category': 'energy',
            }
        ]
        hf_ds = hf_datasets.Dataset.from_list(rows)
        fev_ds = to_fev(hf_ds, keep_extra_columns=False)

        assert 'category' not in fev_ds.column_names

    def test_multivariate_splits_into_variate_columns(self):
        """Multivariate target [n_dims, n_steps] should become separate named columns."""
        from timecp.data import to_fev

        n_dims, n_steps = 3, 20
        rows = [
            {
                'item_id': 's0',
                'start': datetime.datetime(2020, 1, 1),
                'freq': 'D',
                'target': [
                    [float(d * 100 + t) for t in range(n_steps)] for d in range(n_dims)
                ],
                'variate_names': ['alpha', 'beta', 'gamma'],
            }
        ]
        hf_ds = hf_datasets.Dataset.from_list(rows)
        fev_ds = to_fev(hf_ds)

        assert fev_ds.variate_names == ['alpha', 'beta', 'gamma']  # type: ignore[attr-defined]
        assert fev_ds.freq == 'D'  # type: ignore[attr-defined]
        assert 'target' not in fev_ds.column_names
        assert 'variate_names' not in fev_ds.column_names
        for col in ['alpha', 'beta', 'gamma']:
            assert col in fev_ds.column_names
            assert len(fev_ds[0][col]) == n_steps

    def test_multivariate_fallback_column_names(self):
        """Multivariate without variate_names uses target_0, target_1, ..."""
        from timecp.data import to_fev

        rows = [
            {
                'item_id': 's0',
                'start': datetime.datetime(2020, 1, 1),
                'freq': 'D',
                'target': [[1.0, 2.0], [3.0, 4.0]],
            }
        ]
        hf_ds = hf_datasets.Dataset.from_list(rows)
        fev_ds = to_fev(hf_ds)

        assert fev_ds.variate_names == ['target_0', 'target_1']  # type: ignore[attr-defined]
        assert 'target_0' in fev_ds.column_names
        assert 'target_1' in fev_ds.column_names

    def test_meta_survives_save_load(self, tmp_path: Path):
        """freq and variate_names must round-trip through save_to_disk/load_from_disk."""
        from timecp.data import attach_fev_meta, to_fev

        rows = [
            {
                'item_id': 's0',
                'start': datetime.datetime(2020, 1, 1),
                'freq': 'D',
                'target': [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                'variate_names': ['x', 'y'],
            }
        ]
        hf_ds = hf_datasets.Dataset.from_list(rows)
        fev_ds = to_fev(hf_ds)

        p = tmp_path / 'ds'
        fev_ds.save_to_disk(str(p))
        reloaded = hf_datasets.load_from_disk(str(p))
        attach_fev_meta(reloaded)
        assert reloaded.freq == 'D'  # type: ignore[attr-defined]
        assert reloaded.variate_names == ['x', 'y']  # type: ignore[attr-defined]

    def test_nested_list_column_transposed(self):
        """Nested-list covariates [n_dims, n_steps] are transposed to [n_steps, n_dims].

        The column name is preserved (not split into _0, _1, ...).
        The outer length after transposition equals len(target), so the FEV
        validator's length check passes.
        """
        from timecp.data import to_fev

        n_dims, n_steps = 3, 10
        covariate = [
            [float(d * 100 + t) for t in range(n_steps)] for d in range(n_dims)
        ]
        rows = [
            {
                'item_id': 's0',
                'start': datetime.datetime(2020, 1, 1),
                'freq': 'H',
                'target': [float(t) for t in range(n_steps)],
                'past_feat_dynamic_real': covariate,
            }
        ]
        hf_ds = hf_datasets.Dataset.from_list(rows)
        fev_ds = to_fev(hf_ds)

        # Column name preserved — no splitting into _0, _1, ...
        assert 'past_feat_dynamic_real' in fev_ds.column_names
        assert 'past_feat_dynamic_real_0' not in fev_ds.column_names

        transposed = fev_ds[0]['past_feat_dynamic_real']

        # Outer dim is now n_steps
        assert len(transposed) == n_steps
        # Inner dim is n_dims
        assert len(transposed[0]) == n_dims

        # Check a specific value: original[dim][t] == transposed[t][dim]
        for d in range(n_dims):
            for t in range(n_steps):
                assert transposed[t][d] == pytest.approx(covariate[d][t])


class TestFormatDetection:
    def test_gift_eval_detected(self):
        from timecp.data import is_gift_eval_format

        hf_ds = _make_gift_eval_dataset()
        assert is_gift_eval_format(hf_ds)

    def test_fev_detected(self):
        from timecp.data import is_fev_format

        hf_ds = _make_fev_dataset()
        assert is_fev_format(hf_ds)

    def test_gift_eval_not_detected_as_fev(self):
        from timecp.data import is_fev_format

        hf_ds = _make_gift_eval_dataset()
        assert not is_fev_format(hf_ds)

    def test_fev_not_detected_as_gift_eval(self):
        from timecp.data import is_gift_eval_format

        hf_ds = _make_fev_dataset()
        assert not is_gift_eval_format(hf_ds)
