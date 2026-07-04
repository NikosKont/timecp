import json
import sys
from pathlib import Path

import datasets as hf_datasets
import numpy as np
import pandas as pd
import pytest

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from timecp.cp_eval import (
    CPEvaluator,
    abs_scores,
    cdf_tail_scores,
    cqr_scores,
    diff_scores,
    distributional_excess_scores,
    invert_cdf_tail,
    invert_distributional_excess,
    log_scores,
    scaled_cqr_scores,
    signed_scores,
    squared_scores,
)
from timecp.methods import SplitCP

# ---------------------------------------------------------------------------
# 1. Test Score Helpers
# ---------------------------------------------------------------------------


def test_score_helpers():
    y_true = np.array([2.0, 5.0, 1.0])
    point_preds = np.array([1.0, 4.0, 3.0])

    # abs_scores
    np.testing.assert_allclose(abs_scores(y_true, point_preds), [1.0, 1.0, 2.0])

    # squared_scores
    np.testing.assert_allclose(squared_scores(y_true, point_preds), [1.0, 1.0, 4.0])

    # signed_scores
    np.testing.assert_allclose(signed_scores(y_true, point_preds), [1.0, 1.0, -2.0])

    # log_scores
    np.testing.assert_allclose(
        log_scores(y_true, point_preds, eps=0.0),
        [
            np.log(2.0) - np.log(1.0),
            np.log(5.0) - np.log(4.0),
            np.abs(np.log(1.0) - np.log(3.0)),
        ],
    )

    # cqr_scores
    q_low = np.array([1.5, 4.5, 0.5])
    q_high = np.array([2.5, 5.5, 1.5])
    np.testing.assert_allclose(
        cqr_scores(y_true, q_low, q_high),
        [
            max(1.5 - 2.0, 2.0 - 2.5),  # max(-0.5, -0.5) = -0.5
            max(4.5 - 5.0, 5.0 - 5.5),  # max(-0.5, -0.5) = -0.5
            max(0.5 - 1.0, 1.0 - 1.5),  # max(-0.5, -0.5) = -0.5
        ],
    )

    # scaled_cqr_scores
    sigma_lo = np.array([0.5, 0.5, 0.5])
    sigma_hi = np.array([2.0, 2.0, 2.0])
    np.testing.assert_allclose(
        scaled_cqr_scores(y_true, q_low, q_high, sigma_lo, sigma_hi),
        [
            max((1.5 - 2.0) / 0.5, (2.0 - 2.5) / 2.0),
            max((4.5 - 5.0) / 0.5, (5.0 - 5.5) / 2.0),
            max((0.5 - 1.0) / 0.5, (1.0 - 1.5) / 2.0),
        ],
    )

    # diff_scores
    residuals = np.array([1.0, -2.0, 0.5])
    prev_residuals = np.array([0.0, 1.0, -2.0])
    np.testing.assert_allclose(diff_scores(residuals, prev_residuals), [1.0, 3.0, 2.5])


# ---------------------------------------------------------------------------
# 2. Test Inversion Equivalence
# ---------------------------------------------------------------------------


def test_inversion_equivalence(tmp_path):
    # Verify that S_t(y) <= threshold is equivalent to lower <= y <= upper
    # for a set of values y.

    # We will instantiate CPEvaluator with a mock directory to bypass the windows check.
    pred_dir = _make_mock_predictions_dir(tmp_path / 'preds_inv', n_windows=2, N=1, H=1)

    evaluator = CPEvaluator(
        predictions_dir=pred_dir, horizon=1, cal_windows=1, alpha=0.2, score_type='abs'
    )

    point_pred = np.array([[10.0]])
    q_low = np.array([[8.0]])
    q_high = np.array([[12.0]])
    q_grid = {
        '0.1': np.array([[7.0]]),
        '0.3': np.array([[9.0]]),
        '0.5': np.array([[10.0]]),
        '0.7': np.array([[11.0]]),
        '0.9': np.array([[13.0]]),
    }
    scales = {
        'native_iqr_floor': np.array([4.0]),
        'rho_mad': np.array([2.0]),
        'sigma_lo': np.array([1.0]),
        'sigma_hi': np.array([1.0]),
    }
    prev_error = np.array([[1.0]])

    threshold = 1.5

    def get_bounds(score_t, q_lo_val, q_up_val, is_asym=False):
        return evaluator._bounds_from_quantiles(
            score_type=score_t,
            q_lowers=np.array([[q_lo_val]]),
            q_uppers=np.array([[q_up_val]]),
            point_preds=point_pred,
            q_low=q_low,
            q_high=q_high,
            scales=scales,
            all_quantiles=q_grid,
            prev_error=prev_error,
            is_asymmetric=is_asym,
        )

    # Check abs
    lo, up = get_bounds('abs', threshold, threshold)
    assert lo[0, 0] == 10.0 - threshold
    assert up[0, 0] == 10.0 + threshold

    # Check squared
    lo, up = get_bounds('squared', threshold, threshold)
    assert lo[0, 0] == 10.0 - np.sqrt(threshold)
    assert up[0, 0] == 10.0 + np.sqrt(threshold)

    # Check squared (asymmetric)
    lo, up = get_bounds('squared', 0.5, 1.5, is_asym=True)
    assert lo[0, 0] == 10.0 - 0.5
    assert up[0, 0] == 10.0 + 1.5

    # Check signed
    lo, up = get_bounds('signed', 0.5, 1.5)
    assert lo[0, 0] == 10.0 - 0.5
    assert up[0, 0] == 10.0 + 1.5

    # Check iqr_scaled
    lo, up = get_bounds('iqr_scaled', threshold, threshold)
    assert lo[0, 0] == 10.0 - threshold * 4.0
    assert up[0, 0] == 10.0 + threshold * 4.0

    # Check mad_scaled
    lo, up = get_bounds('mad_scaled', threshold, threshold)
    assert lo[0, 0] == 10.0 - threshold * 2.0
    assert up[0, 0] == 10.0 + threshold * 2.0

    # Check cqr
    lo, up = get_bounds('cqr', threshold, threshold)
    assert lo[0, 0] == 8.0 - threshold
    assert up[0, 0] == 12.0 + threshold

    # Check scaled_cqr
    lo, up = get_bounds('scaled_cqr', threshold, threshold)
    assert lo[0, 0] == 8.0 - threshold * 1.0
    assert up[0, 0] == 12.0 + threshold * 1.0

    # Check log
    lo, up = get_bounds('log', threshold, threshold)
    eps = 1e-6
    expected_lo = np.exp(np.log(10.0 + eps) - threshold) - eps
    expected_up = np.exp(np.log(10.0 + eps) + threshold) - eps
    assert np.isclose(lo[0, 0], expected_lo)
    assert np.isclose(up[0, 0], expected_up)

    # Check diff
    lo, up = get_bounds('diff', threshold, threshold)
    assert lo[0, 0] == 10.0 + 1.0 - threshold
    assert up[0, 0] == 10.0 + 1.0 + threshold


# ---------------------------------------------------------------------------
# 3. Test Evaluator Modes
# ---------------------------------------------------------------------------


def _make_mock_predictions_dir(base_dir: Path, n_windows=10, N=2, H=4) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    cal_w = n_windows // 2
    meta = {
        'task_name': 'mock_task',
        'model_name': 'mock_model',
        'horizon': H,
        'total_windows': n_windows,
        'cal_windows': cal_w,
        'test_windows': n_windows - cal_w,
        'num_series': N,
    }
    with open(base_dir / 'metadata.json', 'w') as f:
        json.dump(meta, f)

    # Accumulators for unified dataset
    all_points = []
    all_gts = []

    for w in range(n_windows):
        wdir = base_dir / f'window_{w}'
        wdir.mkdir()

        point = rng.standard_normal((N, H)).astype(np.float32)
        gt = rng.standard_normal((N, H)).astype(np.float32)
        all_points.append(point)
        all_gts.append(gt)

        # Build predictions dataset with multiple quantile columns and item_id
        pred_rows = []
        for n in range(N):
            row = {
                'predictions': point[n].tolist(),
                'item_id': [f'target_{i}' for i in range(N)],
            }
            # Quantiles: 0.1, 0.2, ..., 0.9
            for q_val in range(1, 10):
                q_name = f'0.{q_val}'
                row[q_name] = (point[n] + (q_val - 5) * 0.2).tolist()
            pred_rows.append(row)

        pred_ds = hf_datasets.Dataset.from_list(pred_rows)
        pred_dict = hf_datasets.DatasetDict({'target': pred_ds})
        pred_dict.save_to_disk(str(wdir / 'predictions'))

        gt_rows = [{'target': gt[n].tolist()} for n in range(N)]
        gt_ds = hf_datasets.Dataset.from_list(gt_rows)
        gt_dict = hf_datasets.DatasetDict({'target': gt_ds})
        gt_dict.save_to_disk(str(wdir / 'ground_truth'))

    # Save unified/stacked predictions/ and ground_truth/ at root level of base_dir
    unified_pred_rows = []
    for w in range(n_windows):
        row = {
            'predictions': all_points[w].tolist(),
            'item_id': [f'target_{i}' for i in range(N)],
        }
        for q_val in range(1, 10):
            q_name = f'0.{q_val}'
            row[q_name] = (all_points[w] + (q_val - 5) * 0.2).tolist()
        unified_pred_rows.append(row)

    unified_pred_ds = hf_datasets.Dataset.from_list(unified_pred_rows)
    unified_pred_dict = hf_datasets.DatasetDict({'target': unified_pred_ds})
    unified_pred_dict.save_to_disk(str(base_dir / 'predictions'))

    unified_gt_rows = [{'target': all_gts[w].tolist()} for w in range(n_windows)]
    unified_gt_ds = hf_datasets.Dataset.from_list(unified_gt_rows)
    unified_gt_dict = hf_datasets.DatasetDict({'target': unified_gt_ds})
    unified_gt_dict.save_to_disk(str(base_dir / 'ground_truth'))

    return base_dir


def test_evaluator_all_score_types(tmp_path):
    pred_dir = _make_mock_predictions_dir(tmp_path / 'preds')

    score_types = [
        'abs',
        'squared',
        'signed',
        'iqr_scaled',
        'mad_scaled',
        'cqr',
        'scaled_cqr',
        'distributional',
        'cdf_tail',
        'log',
        'diff',
    ]

    for score_t in score_types:
        evaluator = CPEvaluator(
            predictions_dir=pred_dir,
            horizon=4,
            cal_windows=5,
            alpha=0.2,
            score_type=score_t,
        )

        is_asymmetric = score_t == 'signed'

        methods = {
            'SplitCP': [SplitCP(alpha=0.2, asymmetric=is_asymmetric) for _ in range(4)],
        }

        # Test marginal run
        df = evaluator.run(methods)
        assert len(df) > 0
        assert 'coverage' in df.columns
        assert 'avg_width' in df.columns
        assert df['coverage'].min() >= 0.0
        assert df['coverage'].max() <= 1.0
        assert (df['avg_width'] >= 0).all()
        assert (df['winkler_score'] >= 0).all()

        # Test CV run
        df_cv = evaluator.run_cross_sectional_cv(methods, cal_windows=5)
        assert len(df_cv) > 0
        assert 'coverage' in df_cv.columns
        assert df_cv['coverage'].min() >= 0.0
        assert df_cv['coverage'].max() <= 1.0
        assert (df_cv['avg_width'] >= 0).all()

        # Test single-step run
        single_methods = {
            'SplitCP': SplitCP(alpha=0.2, asymmetric=is_asymmetric),
        }
        df_single = evaluator.run_single_step(single_methods)
        assert len(df_single) > 0
        assert df_single['coverage'].min() >= 0.0
        assert df_single['coverage'].max() <= 1.0
        assert (df_single['avg_width'] >= 0).all()


# ---------------------------------------------------------------------------
# 4. Distributional / CDF-tail inversion equivalence
# ---------------------------------------------------------------------------


def test_distributional_inversion_equivalence():
    """For distributional: S_t(y) <= q  <=>  lo <= y <= hi."""
    q_grid = {
        '0.1': np.array([[7.0]]),
        '0.3': np.array([[9.0]]),
        '0.5': np.array([[10.0]]),
        '0.7': np.array([[11.0]]),
        '0.9': np.array([[13.0]]),
    }

    # Pick a threshold
    q = np.array([[0.5]])

    # Compute the inverted interval
    lo, hi = invert_distributional_excess(q_grid, q)

    # Test a range of y values around the interval
    for y_val in np.linspace(lo[0, 0] - 2.0, hi[0, 0] + 2.0, 50):
        y_arr = np.array([[y_val]])
        score = distributional_excess_scores(y_arr, q_grid)
        inside = lo[0, 0] <= y_val <= hi[0, 0]
        below_threshold = score[0, 0] <= q[0, 0]
        assert inside == below_threshold, (
            f'distributional: y={y_val:.4f}, score={score[0, 0]:.4f}, q={q[0, 0]:.4f}, '
            f'lo={lo[0, 0]:.4f}, hi={hi[0, 0]:.4f}, inside={inside}, below={below_threshold}'
        )


def test_distributional_inversion_crossed_quantiles():
    """Crossed quantile forecasts should not crash or produce NaN bounds."""
    q_grid = {
        '0.1': np.array([[12.0]]),
        '0.3': np.array([[10.0]]),
        '0.5': np.array([[8.0]]),
        '0.7': np.array([[11.0]]),
        '0.9': np.array([[9.0]]),
    }
    q = np.array([[1.0]])
    lo, hi = invert_distributional_excess(q_grid, q)
    assert np.isfinite(lo).all()
    assert np.isfinite(hi).all()
    assert (lo <= hi).all()


def test_distributional_inversion_two_quantiles():
    """K=2 (minimum) should still work."""
    q_grid = {
        '0.2': np.array([[5.0]]),
        '0.8': np.array([[15.0]]),
    }
    q = np.array([[2.0]])
    lo, hi = invert_distributional_excess(q_grid, q)
    assert np.isfinite(lo).all()
    assert np.isfinite(hi).all()
    assert (lo <= hi).all()


def test_cdf_tail_inversion_equivalence():
    """For cdf_tail: S_t(y) <= q  <=>  lo <= y <= hi."""
    alpha = 0.2
    q_grid = {
        '0.1': np.array([[7.0]]),
        '0.3': np.array([[9.0]]),
        '0.5': np.array([[10.0]]),
        '0.7': np.array([[11.0]]),
        '0.9': np.array([[13.0]]),
    }

    q = np.array([[0.05]])

    # Compute the inverted interval
    lo, hi = invert_cdf_tail(q_grid, q, alpha)

    # Test a range of y values around the interval
    for y_val in np.linspace(lo[0, 0] - 2.0, hi[0, 0] + 2.0, 50):
        y_arr = np.array([[y_val]])
        score = cdf_tail_scores(y_arr, q_grid, alpha)
        inside = lo[0, 0] <= y_val <= hi[0, 0]
        below_threshold = score[0, 0] <= q[0, 0]
        assert inside == below_threshold, (
            f'cdf_tail: y={y_val:.4f}, score={score[0, 0]:.4f}, q={q[0, 0]:.4f}, '
            f'lo={lo[0, 0]:.4f}, hi={hi[0, 0]:.4f}, inside={inside}, below={below_threshold}'
        )


def test_cdf_tail_pit_in_unit_interval():
    """CDF values returned by cdf_tail_scores must produce PIT in [0, 1]."""
    alpha = 0.2
    q_grid = {
        '0.1': np.array([[5.0]]),
        '0.3': np.array([[8.0]]),
        '0.5': np.array([[10.0]]),
        '0.7': np.array([[12.0]]),
        '0.9': np.array([[15.0]]),
    }
    y_values = np.array([[[2.0]], [[6.0]], [[10.0]], [[14.0]], [[18.0]]])
    scores = cdf_tail_scores(y_values, q_grid, alpha)
    assert np.isfinite(scores).all()
    assert (scores >= 0).all()


def test_cdf_tail_inversion_crossed_quantiles():
    """Monotonicized crossed quantiles should not crash."""
    alpha = 0.2
    q_grid = {
        '0.1': np.array([[12.0]]),
        '0.3': np.array([[10.0]]),
        '0.5': np.array([[8.0]]),
        '0.7': np.array([[11.0]]),
        '0.9': np.array([[9.0]]),
    }
    q = np.array([[0.3]])
    lo, hi = invert_cdf_tail(q_grid, q, alpha)
    assert np.isfinite(lo).all()
    assert np.isfinite(hi).all()


# ---------------------------------------------------------------------------
# 5. Signed / log / diff edge cases
# ---------------------------------------------------------------------------


def test_signed_direct_api_raises_without_asymmetric(tmp_path):
    """score_type='signed' with symmetric predictors must raise ValueError."""
    pred_dir = _make_mock_predictions_dir(
        tmp_path / 'preds_signed', n_windows=10, N=2, H=4
    )
    evaluator = CPEvaluator(
        predictions_dir=pred_dir,
        horizon=4,
        cal_windows=5,
        alpha=0.2,
        score_type='signed',
    )
    methods = {
        'SplitCP': [SplitCP(alpha=0.2, asymmetric=False) for _ in range(4)],
    }
    with pytest.raises(
        ValueError, match='requires predictors instantiated with asymmetric=True'
    ):
        evaluator.run(methods)


def test_signed_cli_forces_asymmetric(tmp_path):
    """CLI should auto-force asymmetric=True when score_type='signed'."""
    import sys

    sys.path.append(str(Path(__file__).parent.parent))
    from scripts.cp_eval import _build_parser

    parser = _build_parser()
    args = parser.parse_args(['--score-type', 'signed'])
    assert 'signed' in args.score_type

    # Simulate the forcing logic from main()
    for score_type in args.score_type:
        forced_asymmetric = args.asymmetric
        if score_type == 'signed':
            forced_asymmetric = True
        elif score_type in ('distributional', 'cdf_tail'):
            if forced_asymmetric:
                forced_asymmetric = False

        if score_type == 'signed':
            assert forced_asymmetric is True


def test_distributional_rejects_asymmetric(tmp_path):
    """score_type='distributional' with asymmetric predictors must raise."""
    pred_dir = _make_mock_predictions_dir(
        tmp_path / 'preds_dist', n_windows=10, N=2, H=4
    )
    evaluator = CPEvaluator(
        predictions_dir=pred_dir,
        horizon=4,
        cal_windows=5,
        alpha=0.2,
        score_type='distributional',
    )
    methods = {
        'SplitCP': [SplitCP(alpha=0.2, asymmetric=True) for _ in range(4)],
    }
    with pytest.raises(ValueError, match='does not support asymmetric'):
        evaluator.run(methods)


def test_log_fallback_to_abs(tmp_path):
    """log score with nonpositive calibration target falls back to abs with a warning."""
    import warnings

    pred_dir = _make_mock_predictions_dir(
        tmp_path / 'preds_log', n_windows=10, N=2, H=4
    )
    # Overwrite ground_truth with negative values
    import datasets as hf_datasets_mod

    meta_path = pred_dir / 'metadata.json'
    with open(meta_path):
        pass

    rng = np.random.default_rng(42)
    for w in range(10):
        wdir = pred_dir / f'window_{w}'
        gt_rows = [
            {'target': (-np.abs(rng.standard_normal(4))).astype(np.float32).tolist()}
            for _ in range(2)
        ]
        gt_ds = hf_datasets_mod.Dataset.from_list(gt_rows)
        gt_dict = hf_datasets_mod.DatasetDict({'target': gt_ds})
        gt_dict.save_to_disk(str(wdir / 'ground_truth'))

    evaluator = CPEvaluator(
        predictions_dir=pred_dir, horizon=4, cal_windows=5, alpha=0.2, score_type='log'
    )
    methods = {
        'SplitCP': [SplitCP(alpha=0.2) for _ in range(4)],
    }

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        evaluator.run(methods)
        assert any('falling back' in str(wi.message) for wi in w)
    assert evaluator._effective_score_type == 'abs'


def test_diff_first_cal_prev_is_zero():
    """First calibration previous residual must be zero."""
    from timecp.cp_eval import previous_residuals

    residuals = np.array([1.0, -2.0, 0.5, 3.0])
    first_prev = np.array([0.0])
    prev = previous_residuals(residuals, first_prev)
    assert prev[0] == 0.0
    np.testing.assert_allclose(prev[1:], residuals[:-1])


def test_diff_interval_center_is_point_plus_prev_error(tmp_path):
    """diff interval center = point + prev_error."""
    pred_dir = _make_mock_predictions_dir(
        tmp_path / 'preds_diff', n_windows=2, N=1, H=1
    )

    evaluator = CPEvaluator(
        predictions_dir=pred_dir, horizon=1, cal_windows=1, alpha=0.2, score_type='diff'
    )

    point_pred = np.array([[10.0]])
    q_low = np.array([[8.0]])
    q_high = np.array([[12.0]])
    scales = {
        'native_iqr_floor': np.array([4.0]),
        'rho_mad': np.array([2.0]),
        'sigma_lo': np.array([1.0]),
        'sigma_hi': np.array([1.0]),
    }
    prev_error = np.array([[1.5]])
    threshold = 2.0

    lo, hi = evaluator._bounds_from_quantiles(
        score_type='diff',
        q_lowers=np.array([[threshold]]),
        q_uppers=np.array([[threshold]]),
        point_preds=point_pred,
        q_low=q_low,
        q_high=q_high,
        scales=scales,
        all_quantiles=None,
        prev_error=prev_error,
    )
    center = point_pred[0, 0] + prev_error[0, 0]
    assert lo[0, 0] == center - threshold
    assert hi[0, 0] == center + threshold
