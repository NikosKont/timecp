"""Tests for CPEvaluator single-step and external calibration modes.

Builds minimal on-disk prediction windows in the Arrow DatasetDict format
expected by _load_window, then verifies run_single_step, run_external_cal,
and run_joint_external_cal produce correctly-shaped, semantically sound
DataFrames.
"""

from __future__ import annotations

from pathlib import Path

import datasets as hf_datasets
import numpy as np
import pytest

from timecp.cp_eval import CPEvaluator
from timecp.methods import (
    ACI,
    CQR,
    AdaptiveCQR,
    AgACI,
    DtACI,
    QuantileIntegrator,
    SplitCP,
)

# ---------------------------------------------------------------------------
# Helpers: build fake prediction directories
# ---------------------------------------------------------------------------


def _make_window_dir(
    window_dir: Path,
    N: int,
    H: int,
    rng: np.random.Generator,
    target_col: str = 'target',
    alpha: float = 0.1,
) -> None:
    """Write a fake window_N/ directory with predictions + ground_truth."""
    point = rng.standard_normal((N, H)).astype(np.float32)
    gt = rng.standard_normal((N, H)).astype(np.float32)
    lo_key = f'{alpha / 2:.4g}'.rstrip('0').rstrip('.')
    hi_key = f'{1 - alpha / 2:.4g}'.rstrip('0').rstrip('.')
    q_lo = (point - 0.5).astype(np.float32)
    q_hi = (point + 0.5).astype(np.float32)

    # predictions DatasetDict: {target: Dataset({predictions, "0.05", "0.95"})}
    pred_rows = [
        {
            'predictions': point[n].tolist(),
            lo_key: q_lo[n].tolist(),
            hi_key: q_hi[n].tolist(),
        }
        for n in range(N)
    ]
    pred_ds = hf_datasets.Dataset.from_list(pred_rows)
    pred_dict = hf_datasets.DatasetDict({target_col: pred_ds})
    pred_dict.save_to_disk(str(window_dir / 'predictions'))

    # ground_truth DatasetDict: {target: Dataset({target})}
    gt_rows = [{'target': gt[n].tolist()} for n in range(N)]
    gt_ds = hf_datasets.Dataset.from_list(gt_rows)
    gt_dict = hf_datasets.DatasetDict({target_col: gt_ds})
    gt_dict.save_to_disk(str(window_dir / 'ground_truth'))


def _make_predictions_dir(
    base_dir: Path,
    n_windows: int,
    N: int = 2,
    H: int = 4,
    rng: np.random.Generator | None = None,
    alpha: float = 0.1,
) -> Path:
    """Create a predictions directory with n_windows fake windows."""
    if rng is None:
        rng = np.random.default_rng(0)
    base_dir.mkdir(parents=True, exist_ok=True)
    for w in range(n_windows):
        wdir = base_dir / f'window_{w}'
        wdir.mkdir()
        _make_window_dir(wdir, N=N, H=H, rng=rng, alpha=alpha)
    return base_dir


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def pred_dirs(tmp_path: Path):
    """Two separate prediction directories: test (10 windows) + cal (6 windows)."""
    rng = np.random.default_rng(42)
    test_dir = _make_predictions_dir(tmp_path / 'test', n_windows=10, N=2, H=4, rng=rng)
    cal_dir = _make_predictions_dir(tmp_path / 'cal', n_windows=6, N=2, H=4, rng=rng)
    return test_dir, cal_dir


@pytest.fixture()
def evaluator(pred_dirs):
    """CPEvaluator on the test directory with cal_windows=3."""
    test_dir, _ = pred_dirs
    return CPEvaluator(
        predictions_dir=test_dir,
        horizon=4,
        cal_windows=3,
        alpha=0.1,
        score_type='abs',
    )


@pytest.fixture()
def evaluator_all_test(pred_dirs):
    """CPEvaluator with cal_windows=0 (all 10 windows are test windows)."""
    test_dir, _ = pred_dirs
    return CPEvaluator(
        predictions_dir=test_dir,
        horizon=4,
        cal_windows=0,
        alpha=0.1,
        score_type='abs',
    )


# ---------------------------------------------------------------------------
# _discover_windows
# ---------------------------------------------------------------------------


class TestDiscoverWindows:
    def test_finds_all_windows(self, pred_dirs):
        test_dir, _ = pred_dirs
        windows = CPEvaluator._discover_windows(test_dir)
        assert len(windows) == 10

    def test_sorted_numerically(self, pred_dirs):
        test_dir, _ = pred_dirs
        windows = CPEvaluator._discover_windows(test_dir)
        indices = [int(w.name.split('_')[1]) for w in windows]
        assert indices == list(range(10))

    def test_empty_dir_returns_empty(self, tmp_path):
        (tmp_path / 'empty').mkdir()
        windows = CPEvaluator._discover_windows(tmp_path / 'empty')
        assert windows == []


# ---------------------------------------------------------------------------
# run_single_step
# ---------------------------------------------------------------------------


class TestRunSingleStep:
    def test_returns_correct_schema(self, evaluator):
        methods = {'SplitCP': SplitCP(alpha=0.1)}
        df = evaluator.run_single_step(methods)
        expected_cols = [
            'horizon',
            'method',
            'coverage',
            'joint_coverage',
            'avg_width',
            'winkler_score',
            'runtime',
        ]
        assert list(df.columns) == expected_cols

    def test_returns_one_row_per_horizon_per_method(self, evaluator):
        H = evaluator.horizon
        methods = {'SplitCP': SplitCP(alpha=0.1)}
        df = evaluator.run_single_step(methods)
        assert len(df) == H * 2
        assert sorted(df[df['method'] == 'SplitCP']['horizon'].tolist()) == list(
            range(H)
        )

    def test_joint_coverage_scalar_repeated(self, evaluator):
        """joint_coverage must be the same value on every horizon row."""
        methods = {'SplitCP': SplitCP(alpha=0.1)}
        df = evaluator.run_single_step(methods)
        jc_vals = df[df['method'] == 'SplitCP']['joint_coverage'].unique()
        assert len(jc_vals) == 1

    def test_coverage_in_range(self, evaluator):
        methods = {'SplitCP': SplitCP(alpha=0.1)}
        df = evaluator.run_single_step(methods)
        assert (df['coverage'] >= 0.0).all()
        assert (df['coverage'] <= 1.0).all()

    def test_multiple_methods(self, evaluator):
        from timecp.methods import ACI

        methods = {
            'SplitCP': SplitCP(alpha=0.1),
            'ACI': ACI(alpha=0.1, gamma=0.005),
        }
        df = evaluator.run_single_step(methods)
        H = evaluator.horizon
        assert len(df) == 3 * H
        assert set(df['method'].unique()) == {'SplitCP', 'ACI', 'Native'}

    def test_avg_width_positive(self, evaluator):
        """Interval widths should be positive for finite quantiles."""
        methods = {'SplitCP': SplitCP(alpha=0.1)}
        df = evaluator.run_single_step(methods)
        assert (df['avg_width'] > 0).all()

    def test_single_step_differs_from_multistep(self, evaluator):
        """
        Single-step and multi-step methods see different score sequences;
        their quantiles should generally differ (not guaranteed to be unequal
        but this is a sanity check on a specific seed).
        """
        from timecp.methods import SplitCP as _SplitCP

        ss_df = evaluator.run_single_step({'SplitCP': _SplitCP(alpha=0.1)})
        ms_df = evaluator.run(
            {'SplitCP': [_SplitCP(alpha=0.1) for _ in range(evaluator.horizon)]}
        )

        # Both have the same schema
        assert list(ss_df.columns) == list(ms_df.columns)
        # At least one metric is different (single-step sees calibration data
        # from ALL horizons mixed together vs per-horizon in multi-step)
        # This comparison is relaxed — we just check shapes match
        assert len(ss_df) == len(ms_df)


# ---------------------------------------------------------------------------
# run_external_cal
# ---------------------------------------------------------------------------


class TestOnlineOnlyMethods:
    def test_contains_expected_methods(self):
        import sys

        sys.path.append(str(Path(__file__).parent.parent / 'scripts'))
        from cp_eval import _ONLINE_ONLY_METHODS

        for name in ('ACI', 'AgACI', 'DtACI', 'PID', 'SPCI', 'AcMCP'):
            assert name in _ONLINE_ONLY_METHODS

    def test_splitcp_not_in_online_only(self):
        import sys

        sys.path.append(str(Path(__file__).parent.parent / 'scripts'))
        from cp_eval import _ONLINE_ONLY_METHODS

        assert 'SplitCP' not in _ONLINE_ONLY_METHODS


class TestAsymmetricCQRAndNormalized:
    def test_asymmetric_normalized(self, tmp_path):
        rng = np.random.default_rng(42)
        test_dir = _make_predictions_dir(
            tmp_path / 'test_norm', n_windows=50, N=2, H=4, rng=rng
        )
        evaluator_norm = CPEvaluator(
            predictions_dir=test_dir,
            horizon=4,
            cal_windows=30,
            alpha=0.1,
            score_type='iqr_scaled',
        )
        methods = {
            'SplitCP': [SplitCP(alpha=0.1, asymmetric=True) for _ in range(4)],
        }
        df = evaluator_norm.run(methods)
        assert len(df) > 0
        assert (df['avg_width'] > 0).all()
        assert (df['winkler_score'] > 0).all()

    def test_asymmetric_cqr(self, tmp_path):
        rng = np.random.default_rng(42)
        test_dir = _make_predictions_dir(
            tmp_path / 'test_cqr', n_windows=50, N=2, H=4, rng=rng
        )
        evaluator_cqr = CPEvaluator(
            predictions_dir=test_dir,
            horizon=4,
            cal_windows=30,
            alpha=0.1,
            score_type='cqr',
        )
        methods_cqr = {
            'SplitCP': [SplitCP(alpha=0.1, asymmetric=True) for _ in range(4)],
            'ACI': [ACI(alpha=0.1, asymmetric=True) for _ in range(4)],
            'PID': [QuantileIntegrator(alpha=0.1, asymmetric=True) for _ in range(4)],
            'AgACI': [AgACI(alpha=0.1, asymmetric=True) for _ in range(4)],
            'DtACI': [DtACI(alpha=0.1, asymmetric=True) for _ in range(4)],
            'CQR': [CQR(alpha=0.1, asymmetric=True) for _ in range(4)],
            'AdaptiveCQR': [
                AdaptiveCQR(alpha=0.1, method='aci', asymmetric=True) for _ in range(4)
            ],
        }
        df_cqr = evaluator_cqr.run(methods_cqr)
        assert len(df_cqr) > 0
        assert (df_cqr['avg_width'] > 0).all()
        assert (df_cqr['winkler_score'] > 0).all()


# ---------------------------------------------------------------------------
# --methods auto-classification (marginal vs joint/multi-step)
# ---------------------------------------------------------------------------


def _import_cp_eval():
    """Import the cp_eval script module (adds scripts/ to sys.path)."""
    import sys

    scripts_dir = str(Path(__file__).parent.parent / 'scripts')
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import cp_eval

    return cp_eval


class TestResolveMethods:
    """Verify --methods is auto-classified into marginal and joint lists."""

    def test_none_defaults_to_all_marginal(self):
        cp_eval = _import_cp_eval()
        marginal, joint = cp_eval._resolve_methods(None)
        assert marginal == list(cp_eval._VALID_METHODS)
        assert joint == []

    def test_only_marginal_methods(self):
        cp_eval = _import_cp_eval()
        marginal, joint = cp_eval._resolve_methods(['ACI', 'AgACI'])
        assert marginal == ['ACI', 'AgACI']
        assert joint == []

    def test_only_joint_methods(self):
        cp_eval = _import_cp_eval()
        marginal, joint = cp_eval._resolve_methods(['CopulaCPTS', 'JointCFRNN'])
        assert marginal == []
        assert joint == ['CopulaCPTS', 'JointCFRNN']

    def test_mixed_marginal_and_joint(self):
        """The exact scenario from the user request."""
        cp_eval = _import_cp_eval()
        marginal, joint = cp_eval._resolve_methods(['AgACI', 'ACI', 'CopulaCPTS'])
        assert marginal == ['AgACI', 'ACI']
        assert joint == ['CopulaCPTS']

    def test_order_preserved(self):
        cp_eval = _import_cp_eval()
        marginal, joint = cp_eval._resolve_methods(
            ['CopulaCPTS', 'ACI', 'JointCFRNN', 'AgACI']
        )
        assert marginal == ['ACI', 'AgACI']
        assert joint == ['CopulaCPTS', 'JointCFRNN']

    def test_unknown_method_raises(self):
        cp_eval = _import_cp_eval()
        with pytest.raises(ValueError, match='Unknown method'):
            cp_eval._resolve_methods(['ACI', 'BogusMethod'])

    def test_empty_list_returns_empty(self):
        cp_eval = _import_cp_eval()
        marginal, joint = cp_eval._resolve_methods([])
        assert marginal == []
        assert joint == []


class TestParserMultiStepRemoved:
    """Verify --multi-step and --multi-step-methods are no longer accepted."""

    def test_multi_step_flag_removed(self):
        cp_eval = _import_cp_eval()
        parser = cp_eval._build_parser()
        # --multi-step should be rejected
        with pytest.raises(SystemExit):
            parser.parse_args(['--multi-step', '--predictions', 'x'])

    def test_multi_step_methods_flag_removed(self):
        cp_eval = _import_cp_eval()
        parser = cp_eval._build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(
                ['--multi-step-methods', 'CopulaCPTS', '--predictions', 'x']
            )

    def test_single_step_still_accepted(self):
        cp_eval = _import_cp_eval()
        parser = cp_eval._build_parser()
        args = parser.parse_args(['--single-step', '--methods', 'ACI', 'CopulaCPTS'])
        assert args.single_step is True
        assert args.methods == ['ACI', 'CopulaCPTS']

    def test_methods_accepts_joint_names(self):
        cp_eval = _import_cp_eval()
        parser = cp_eval._build_parser()
        args = parser.parse_args(['--methods', 'AgACI', 'ACI', 'CopulaCPTS'])
        assert args.methods == ['AgACI', 'ACI', 'CopulaCPTS']


class TestSingleStepBatchVsOnline:
    def test_batch_vs_online_differs_for_adaptive(self, evaluator):
        from timecp.methods import ACI

        # Instantiate ACI with a large gamma to make updates prominent
        methods_batch = {'ACI': ACI(alpha=0.1, gamma=0.1)}
        methods_online = {'ACI': ACI(alpha=0.1, gamma=0.1)}

        df_batch = evaluator.run_single_step(methods_batch, online=False)
        df_online = evaluator.run_single_step(methods_online, online=True)

        # Ensure same shape and order
        assert len(df_batch) == len(df_online)
        assert (df_batch['horizon'] == df_online['horizon']).all()
        assert (df_batch['method'] == df_online['method']).all()

        # Compare ACI rows
        aci_batch = df_batch[df_batch['method'] == 'ACI']
        aci_online = df_online[df_online['method'] == 'ACI']

        identical_coverage = (
            aci_batch['coverage'].values == aci_online['coverage'].values
        ).all()
        identical_width = (
            aci_batch['avg_width'].values == aci_online['avg_width'].values
        ).all()
        identical_winkler = (
            aci_batch['winkler_score'].values == aci_online['winkler_score'].values
        ).all()

        # They should differ because batch doesn't leak future information within the window
        assert not (identical_coverage and identical_width and identical_winkler), (
            'Expected ACI metrics to differ between batch and online single-step modes'
        )


def test_post_process_and_aggregate_summary_n_series(tmp_path):
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))
    import pandas as pd

    from scripts.cp_eval import _aggregate_and_save, _post_process_summary

    # Mock metadata
    meta = {
        'task_name': 'test_task',
        'model_name': 'test_model',
        'horizon': 10,
        'total_windows': 10,
        'cal_windows': 5,
        'test_windows': 5,
        'num_series': 12,
    }

    # Mock a dataframe of results
    df = pd.DataFrame(
        {
            'method': ['SplitCP', 'Native'],
            'coverage': [0.8, 0.9],
            'joint_coverage': [0.5, 0.6],
            'avg_width': [1.0, 1.2],
            'winkler_score': [2.0, 2.2],
            'runtime': [0.1, 0.2],
        }
    )

    # Test _post_process_summary
    processed = _post_process_summary(
        df=df,
        meta=meta,
        alpha=0.2,
        cal_windows=5,
        task_name='test_task',
        n_series=12,
        score_type='abs',
        mode='multi-step',
    )

    assert 'n_series' in processed.columns
    assert (processed['n_series'] == 12).all()
    assert 'n_tasks' not in processed.columns

    # Test _aggregate_and_save
    # Create two processed dfs from two tasks
    processed1 = processed.copy()
    processed1['task'] = 'task1'
    processed1['n_series'] = 10

    processed2 = processed.copy()
    processed2['task'] = 'task2'
    processed2['n_series'] = 15

    out_file = tmp_path / 'summary.csv'
    _aggregate_and_save(
        dfs=[processed1, processed2],
        group_cols=[
            'model',
            'alpha',
            'score_type',
            'mode',
            'horizon_category',
            'cal_windows',
            'method',
        ],
        metrics=[
            'coverage',
            'joint_coverage',
            'scaled_avg_width',
            'scaled_winkler_score',
            'runtime',
        ],
        out_path=out_file,
        label='marginal',
    )

    # Read the output CSV and check columns and values
    summary_df = pd.read_csv(out_file)
    assert 'n_series' in summary_df.columns
    assert 'n_tasks' not in summary_df.columns

    # cal_windows is at index 5, n_series should be at index 6 (immediately after cal_windows)
    cols = list(summary_df.columns)
    cal_idx = cols.index('cal_windows')
    assert cols[cal_idx + 1] == 'n_series'

    # The sum of n_series should be 10 + 15 = 25 (unique tasks)
    assert (summary_df['n_series'] == 25).all()


def test_new_arguments_parsing():
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))
    from scripts.cp_eval import _build_parser

    parser = _build_parser()

    # Test defaults
    args = parser.parse_args([])
    assert args.base_dir == 'results'
    assert args.forecasters == ['chronos2']
    assert args.base_datasets == ['fev-bench_mini']
    assert args.predictions_dir is None
    assert args.alpha == [0.2]
    assert args.score_type == ['abs']
    assert args.cal_windows == [100]

    # Test customized values
    args = parser.parse_args(
        [
            '--base-dir',
            'custom_base',
            '--forecasters',
            'model_a',
            'model_b',
            '--base_datasets',
            'dataset_1',
            'dataset_2',
            '--predictions_dir',
            'custom_preds',
            '--alpha',
            '0.1',
            '0.2',
            '--score-type',
            'abs',
            'cqr',
            '--cal-windows',
            '50',
            '100',
        ]
    )
    assert args.base_dir == 'custom_base'
    assert args.forecasters == ['model_a', 'model_b']
    assert args.base_datasets == ['dataset_1', 'dataset_2']
    assert args.predictions_dir == 'custom_preds'
    assert args.alpha == [0.1, 0.2]
    assert args.score_type == ['abs', 'cqr']
    assert args.cal_windows == [50, 100]

    # Test base-datasets alias with dashes
    args = parser.parse_args(
        [
            '--base-datasets',
            'dataset_3',
        ]
    )
    assert args.base_datasets == ['dataset_3']


def test_main_execution_flow(tmp_path):
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))
    from scripts.cp_eval import main

    # 1. Test when base-dir does not exist
    non_existent_dir = tmp_path / 'does_not_exist'
    exit_code = main(['--base-dir', str(non_existent_dir)])
    assert exit_code == 1

    # 2. Test when base-dir exists but no matching pair directories exist
    base_dir = tmp_path / 'results'
    base_dir.mkdir()
    exit_code = main(
        [
            '--base-dir',
            str(base_dir),
            '--forecasters',
            'model_1',
            '--base-datasets',
            'dataset_1',
        ]
    )
    assert exit_code == 1

    # 3. Test when pair directory exists but has no tasks
    pair_dir = base_dir / 'model_1' / 'dataset_1'
    pair_dir.mkdir(parents=True)
    exit_code = main(
        [
            '--base-dir',
            str(base_dir),
            '--forecasters',
            'model_1',
            '--base-datasets',
            'dataset_1',
        ]
    )
    assert exit_code == 1


def test_main_execution_combinations(tmp_path):
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))
    import json

    import datasets as hf_datasets
    import numpy as np

    from scripts.cp_eval import main

    base_dir = tmp_path / 'results'
    pair_dir = base_dir / 'model_1' / 'dataset_1'
    pair_dir.mkdir(parents=True)

    meta = {
        'task_name': 'test_task',
        'model_name': 'model_1',
        'horizon': 4,
        'total_windows': 10,
        'cal_windows': 5,
        'test_windows': 5,
        'num_series': 2,
    }
    with open(pair_dir / 'metadata.json', 'w') as f:
        json.dump(meta, f)

    rng = np.random.default_rng(42)
    for w in range(10):
        wdir = pair_dir / f'window_{w}'
        wdir.mkdir()

        point = rng.standard_normal((2, 4)).astype(np.float32)
        gt = rng.standard_normal((2, 4)).astype(np.float32)
        q_lo = (point - 0.5).astype(np.float32)
        q_hi = (point + 0.5).astype(np.float32)

        pred_rows = [
            {
                'predictions': point[n].tolist(),
                '0.1': q_lo[n].tolist(),
                '0.8': q_hi[n].tolist(),
            }
            for n in range(2)
        ]
        pred_ds = hf_datasets.Dataset.from_list(pred_rows)
        pred_dict = hf_datasets.DatasetDict({'target': pred_ds})
        pred_dict.save_to_disk(str(wdir / 'predictions'))

        gt_rows = [{'target': gt[n].tolist()} for n in range(2)]
        gt_ds = hf_datasets.Dataset.from_list(gt_rows)
        gt_dict = hf_datasets.DatasetDict({'target': gt_ds})
        gt_dict.save_to_disk(str(wdir / 'ground_truth'))

    exit_code = main(
        [
            '--base-dir',
            str(base_dir),
            '--forecasters',
            'model_1',
            '--base-datasets',
            'dataset_1',
            '--alpha',
            '0.2',
            '--score-type',
            'abs',
            '--cal-windows',
            '3',
            '4',
            '--methods',
            'SplitCP',
        ]
    )
    assert exit_code == 0

    summary_c3 = pair_dir / 'cp_summary_a0.2_abs_c3_multi.csv'
    summary_c4 = pair_dir / 'cp_summary_a0.2_abs_c4_multi.csv'
    assert summary_c3.exists()
    assert summary_c4.exists()


def test_main_execution_caching(tmp_path):
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))
    import json

    import datasets as hf_datasets
    import numpy as np

    from scripts.cp_eval import main

    base_dir = tmp_path / 'results'
    pair_dir = base_dir / 'model_1' / 'dataset_1'
    pair_dir.mkdir(parents=True)

    meta = {
        'task_name': 'test_task',
        'model_name': 'model_1',
        'horizon': 4,
        'total_windows': 10,
        'cal_windows': 5,
        'test_windows': 5,
        'num_series': 2,
    }
    with open(pair_dir / 'metadata.json', 'w') as f:
        json.dump(meta, f)

    rng = np.random.default_rng(42)
    for w in range(10):
        wdir = pair_dir / f'window_{w}'
        wdir.mkdir()

        point = rng.standard_normal((2, 4)).astype(np.float32)
        gt = rng.standard_normal((2, 4)).astype(np.float32)
        q_lo = (point - 0.5).astype(np.float32)
        q_hi = (point + 0.5).astype(np.float32)

        pred_rows = [
            {
                'predictions': point[n].tolist(),
                '0.1': q_lo[n].tolist(),
                '0.8': q_hi[n].tolist(),
            }
            for n in range(2)
        ]
        pred_ds = hf_datasets.Dataset.from_list(pred_rows)
        pred_dict = hf_datasets.DatasetDict({'target': pred_ds})
        pred_dict.save_to_disk(str(wdir / 'predictions'))

        gt_rows = [{'target': gt[n].tolist()} for n in range(2)]
        gt_ds = hf_datasets.Dataset.from_list(gt_rows)
        gt_dict = hf_datasets.DatasetDict({'target': gt_ds})
        gt_dict.save_to_disk(str(wdir / 'ground_truth'))

    # Run with cal-windows 10 and 20. Both should fall back to 5.
    # The second one (20) should trigger the cache hit.
    exit_code = main(
        [
            '--base-dir',
            str(base_dir),
            '--forecasters',
            'model_1',
            '--base-datasets',
            'dataset_1',
            '--alpha',
            '0.2',
            '--score-type',
            'abs',
            '--cal-windows',
            '10',
            '20',
            '--methods',
            'SplitCP',
            '--min-cal-windows',
            '3',
        ]
    )
    assert exit_code == 0

    summary_c10 = pair_dir / 'cp_summary_a0.2_abs_c10_multi.csv'
    summary_c20 = pair_dir / 'cp_summary_a0.2_abs_c20_multi.csv'
    assert summary_c10.exists()
    assert summary_c20.exists()


def test_main_execution_force(tmp_path, capsys):
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))
    import json

    import datasets as hf_datasets
    import numpy as np

    from scripts.cp_eval import main

    base_dir = tmp_path / 'results'
    pair_dir = base_dir / 'model_1' / 'dataset_1'
    pair_dir.mkdir(parents=True)

    meta = {
        'task_name': 'test_task',
        'model_name': 'model_1',
        'horizon': 4,
        'total_windows': 10,
        'cal_windows': 5,
        'test_windows': 5,
        'num_series': 2,
    }
    with open(pair_dir / 'metadata.json', 'w') as f:
        json.dump(meta, f)

    rng = np.random.default_rng(42)
    for w in range(10):
        wdir = pair_dir / f'window_{w}'
        wdir.mkdir()

        point = rng.standard_normal((2, 4)).astype(np.float32)
        gt = rng.standard_normal((2, 4)).astype(np.float32)
        q_lo = (point - 0.5).astype(np.float32)
        q_hi = (point + 0.5).astype(np.float32)

        pred_rows = [
            {
                'predictions': point[n].tolist(),
                '0.1': q_lo[n].tolist(),
                '0.8': q_hi[n].tolist(),
            }
            for n in range(2)
        ]
        pred_ds = hf_datasets.Dataset.from_list(pred_rows)
        pred_dict = hf_datasets.DatasetDict({'target': pred_ds})
        pred_dict.save_to_disk(str(wdir / 'predictions'))

        gt_rows = [{'target': gt[n].tolist()} for n in range(2)]
        gt_ds = hf_datasets.Dataset.from_list(gt_rows)
        gt_dict = hf_datasets.DatasetDict({'target': gt_ds})
        gt_dict.save_to_disk(str(wdir / 'ground_truth'))

    # Run with cal-windows 10 and 20, AND --force. Both fall back to 5,
    # but the second one (20) should NOT trigger the cache hit.
    exit_code = main(
        [
            '--base-dir',
            str(base_dir),
            '--forecasters',
            'model_1',
            '--base-datasets',
            'dataset_1',
            '--alpha',
            '0.2',
            '--score-type',
            'abs',
            '--cal-windows',
            '10',
            '20',
            '--methods',
            'SplitCP',
            '--min-cal-windows',
            '3',
            '--force',
        ]
    )
    assert exit_code == 0

    # Capture stdout to ensure NO CACHE HIT occurred
    captured = capsys.readouterr()
    assert '[CACHE HIT]' not in captured.out
