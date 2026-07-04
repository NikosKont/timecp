"""
Conformal prediction evaluation harness for multi-step time series forecasting.

This module bridges the output of ``scripts/forecast.py`` (per-window Arrow
prediction files) with the CP methods in ``timecp.methods``.  Three evaluation
strategies are supported:

**Multi-step (default)**
    Per-horizon independent calibration (Wang & Hyndman 2024, Szabadváry 2024).
    One CP predictor per forecast horizon h ∈ {0, …, H-1}, calibrated and
    evaluated independently.  The first ``cal_windows`` sliding windows are
    used for calibration; the remaining windows are the held-out test set.

**Single-step** (``run_single_step``)
    One predictor per method across all H horizons.  Scores are flattened in
    window-major order — (w₀h₀, w₀h₁, …, w₀h_{H-1}, w₁h₀, …) — and the
    predictor is calibrated and evaluated on this single stream.  Treats
    multi-step forecasting as a repeated single-step problem.

**External calibration** (``run_external_cal``, ``run_multi_step_external_cal``)
    The original CFRNN approach (Stankeviciute et al. 2021, Algorithm 1).
    Methods are calibrated on ALL windows of a separate calibration dataset;
    a fixed quantile is applied at test time without any online updates.
    The first ``cal_windows`` windows of the test dataset are still skipped
    (same burn-in as the default multi-step mode), so results are directly
    comparable.  Only batch methods are meaningful here (SplitCP, joint
    methods); online adaptive methods degrade to SplitCP and should be
    excluded before calling.

Supported nonconformity score types
------------------------------------
Physical scores:
  * ``"abs"``: Absolute point-forecast residual: s = |y - ŷ|.
  * ``"squared"``: Squared point-forecast residual: s = (y - ŷ)².
  * ``"signed"``: Signed point-forecast residual: s = y - ŷ (for asymmetric calibration).

Scaled scores:
  * ``"iqr_scaled"``: Residual scaled by local IQR (native model interval width): s = |y - ŷ| / max(q_high - q_low, native_iqr_floor).
  * ``"mad_scaled"``: Residual scaled by local MAD: s = |y - ŷ| / max(rho_mad, epsilon).

Quantile-based scores:
  * ``"cqr"``: CQR score: s = max(q_low - y, y - q_high).
  * ``"scaled_cqr"``: CQR score scaled by asymmetric local quantile scale: s = max((q_low - y)/sigma_low, (y - q_high)/sigma_high).
  * ``"distributional"``: Excess mean-pinball loss score: s = S(y, F̂). It is inverted using a convex root-finding algorithm to find interval bounds. It is not implemented as a native-interval additive expansion.
  * ``"cdf_tail"``: Central-band PIT violation: s = max(alpha/2 - Fhat(y), Fhat(y) - (1 - alpha/2), 0). It is inverted through the quantile grid via probability-band inversion. It differs from ``"distributional"`` because it calibrates tail probability rather than CRPS-like integrated pinball loss.

Transformed / stateful scores:
  * ``"log"``: Log-scale absolute residual: s = |log(y + eps) - log(ŷ + eps)|.
  * ``"diff"``: Absolute step-difference residual: s = |(y_t - ŷ_t) - (y_{t-1} - ŷ_{t-1})|.

Metrics
-------
Per horizon and per method:

* ``coverage``             — empirical marginal coverage over test windows
* ``joint_coverage``       — fraction of test windows where all H horizons are
  simultaneously covered (method-level scalar, repeated on every horizon row)
* ``avg_width``            — mean interval width (absolute units of the target)
* ``winkler_score``        — mean Winkler score (width + (2/α)·penalty for misses)
* ``native_coverage``      — coverage of the model's own native quantile interval
* ``native_joint_coverage``— native interval joint coverage
* ``native_avg_width``     — mean width of the native quantile interval
* ``native_winkler_score`` — Winkler score of the native quantile interval

Usage
-----
::

    from timecp.cp_eval import CPEvaluator
    from timecp.methods import ACI, AgACI, QuantileIntegrator, SplitCP

    evaluator = CPEvaluator(
        predictions_dir="results/tirex/ETTm2__T",
        horizon=96,
        cal_windows=30,
        alpha=0.1,
        score_type="abs",
    )

    # Multi-step (one predictor per horizon)
    results = evaluator.run(
        methods={
            "ACI":   [ACI(alpha=0.1, gamma=0.005) for _ in range(96)],
            "AgACI": [AgACI(alpha=0.1) for _ in range(96)],
            "PID":   [QuantileIntegrator(alpha=0.1) for _ in range(96)],
        }
    )

    # Single-step (one predictor for all horizons)
    results_ss = evaluator.run_single_step(
        methods={"SplitCP": SplitCP(alpha=0.1)}
    )

    # External calibration (CFRNN-style)
    results_ext = evaluator.run_external_cal(
        methods={"SplitCP": [SplitCP(alpha=0.1) for _ in range(96)]},
        cal_dir="results/tirex/ETTm2__T_cal",
    )
"""

from __future__ import annotations

import copy
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import datasets as hf_datasets
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from timecp.base import ConformalPredictor, JointPredictor
from timecp.methods import AcMCP

from .utils import print_warning

# ---------------------------------------------------------------------------
# Score constants and specs
# ---------------------------------------------------------------------------

_VALID_SCORE_TYPES = (
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
    'joint',
)

_USER_SCORE_TYPES = tuple(x for x in _VALID_SCORE_TYPES if x != 'joint')

_LEGACY_SCORE_RENAMES = {
    'residual': 'abs',
    'normalized': 'iqr_scaled',
}


def _score_slug(name: str) -> str:
    resolved = _LEGACY_SCORE_RENAMES.get(name, name)
    return resolved.replace('_', '-')


def _safe_scale(scale, fallback, eps=1e-8):
    scale = np.asarray(scale, dtype=float)
    fallback = np.asarray(fallback, dtype=float)
    out = np.where(np.isfinite(scale) & (scale > eps), scale, fallback)
    return np.where(np.isfinite(out) & (out > eps), out, 1.0)


# ---------------------------------------------------------------------------
# Score helpers
# ---------------------------------------------------------------------------


def abs_scores(y_true: np.ndarray, point_preds: np.ndarray) -> np.ndarray:
    return np.abs(y_true - point_preds)


def squared_scores(y_true: np.ndarray, point_preds: np.ndarray) -> np.ndarray:
    return (y_true - point_preds) ** 2


def signed_scores(y_true: np.ndarray, point_preds: np.ndarray) -> np.ndarray:
    return y_true - point_preds


def log_scores(
    y_true: np.ndarray, point_preds: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    valid = (y_true + eps > 0) & (point_preds + eps > 0)
    out = np.full_like(y_true, np.inf, dtype=float)
    out[valid] = np.abs(np.log(y_true[valid] + eps) - np.log(point_preds[valid] + eps))
    return out


def cqr_scores(
    y_true: np.ndarray,
    q_low: np.ndarray,
    q_high: np.ndarray,
) -> np.ndarray:
    return np.maximum(q_low - y_true, y_true - q_high)


def scaled_cqr_scores(
    y_true: np.ndarray,
    q_low: np.ndarray,
    q_high: np.ndarray,
    sigma_lo: np.ndarray,
    sigma_hi: np.ndarray,
) -> np.ndarray:
    return np.maximum((q_low - y_true) / sigma_lo, (y_true - q_high) / sigma_hi)


def _sorted_quantile_items(
    all_quantiles: dict[str, np.ndarray],
) -> list[tuple[float, np.ndarray]]:
    items = []
    for key, arr in all_quantiles.items():
        try:
            tau = float(key)
        except ValueError:
            continue
        if 0.0 < tau < 1.0:
            items.append((tau, arr))
    items.sort(key=lambda x: x[0])
    if not items:
        raise ValueError('distributional score requires numeric quantile columns')
    return items


def mean_pinball_at_y(
    y: np.ndarray, quantile_items: list[tuple[float, np.ndarray]]
) -> np.ndarray:
    losses = []
    for tau, q_tau in quantile_items:
        losses.append(
            tau * np.maximum(y - q_tau, 0.0) + (1.0 - tau) * np.maximum(q_tau - y, 0.0)
        )
    return np.mean(losses, axis=0)


def distributional_excess_scores(
    y_true: np.ndarray,
    all_quantiles: dict[str, np.ndarray],
) -> np.ndarray:
    quantile_items = _sorted_quantile_items(all_quantiles)
    pinball_at_y = mean_pinball_at_y(y_true, quantile_items)

    pinball_losses = []
    for _, q_tau in quantile_items:
        pinball_losses.append(mean_pinball_at_y(q_tau, quantile_items))
    min_pinball = np.min(pinball_losses, axis=0)

    return pinball_at_y - min_pinball


def cdf_values_from_quantile_grid(
    y: np.ndarray, quantile_items: list[tuple[float, np.ndarray]]
) -> np.ndarray:
    taus = np.array([x[0] for x in quantile_items])
    qs = np.stack([x[1] for x in quantile_items], axis=0)  # (K, ...)
    qs = np.maximum.accumulate(qs, axis=0)

    u = np.full_like(y, np.nan, dtype=float)

    dq_first = qs[1] - qs[0]
    safe_dq_first = np.where(dq_first > 1e-8, dq_first, 1e-8)
    slope_first = (taus[1] - taus[0]) / safe_dq_first
    u_under = taus[0] + slope_first * (y - qs[0])
    u_under = np.clip(u_under, 0.0, taus[0])
    u = np.where(y <= qs[0], u_under, u)

    dq_last = qs[-1] - qs[-2]
    safe_dq_last = np.where(dq_last > 1e-8, dq_last, 1e-8)
    slope_last = (taus[-1] - taus[-2]) / safe_dq_last
    u_over = taus[-1] + slope_last * (y - qs[-1])
    u_over = np.clip(u_over, taus[-1], 1.0)
    u = np.where(y >= qs[-1], u_over, u)

    for k in range(len(taus) - 1):
        q_lo = qs[k]
        q_hi = qs[k + 1]
        tau_lo = taus[k]
        tau_hi = taus[k + 1]

        dq = q_hi - q_lo
        safe_dq = np.where(dq > 1e-8, dq, 1e-8)

        val = tau_lo + (tau_hi - tau_lo) * (y - q_lo) / safe_dq

        mask = (y > q_lo) & (y < q_hi)
        u = np.where(mask, val, u)
        u = np.where(y == q_lo, tau_lo, u)
        u = np.where(y == q_hi, tau_hi, u)

    return np.clip(u, 0.0, 1.0)


def quantile_values_from_grid(
    probs: np.ndarray, quantile_items: list[tuple[float, np.ndarray]]
) -> np.ndarray:
    taus = np.array([x[0] for x in quantile_items])
    qs = np.stack([x[1] for x in quantile_items], axis=0)  # (K, ...)
    qs = np.maximum.accumulate(qs, axis=0)

    probs = np.asarray(probs, dtype=float)
    out = np.full_like(qs[0], np.nan, dtype=float)
    p_arr = np.broadcast_to(probs, qs[0].shape)

    dq_first = qs[1] - qs[0]
    slope_first = dq_first / (taus[1] - taus[0])
    val_under = qs[0] + slope_first * (p_arr - taus[0])
    out = np.where(p_arr <= taus[0], val_under, out)

    dq_last = qs[-1] - qs[-2]
    slope_last = dq_last / (taus[-1] - taus[-2])
    val_over = qs[-1] + slope_last * (p_arr - taus[-1])
    out = np.where(p_arr >= taus[-1], val_over, out)

    for k in range(len(taus) - 1):
        tau_lo = taus[k]
        tau_hi = taus[k + 1]
        q_lo = qs[k]
        q_hi = qs[k + 1]

        val = q_lo + (q_hi - q_lo) * (p_arr - tau_lo) / (tau_hi - tau_lo)

        mask = (p_arr > tau_lo) & (p_arr < tau_hi)
        out = np.where(mask, val, out)
        out = np.where(p_arr == tau_lo, q_lo, out)
        out = np.where(p_arr == tau_hi, q_hi, out)

    return out


def cdf_tail_scores(
    y_true: np.ndarray,
    all_quantiles: dict[str, np.ndarray],
    alpha: float,
) -> np.ndarray:
    quantile_items = _sorted_quantile_items(all_quantiles)
    u = cdf_values_from_quantile_grid(y_true, quantile_items)
    return np.maximum(np.maximum(alpha / 2.0 - u, u - (1.0 - alpha / 2.0)), 0.0)


def invert_cdf_tail(
    all_quantiles: dict[str, np.ndarray],
    q: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    quantile_items = _sorted_quantile_items(all_quantiles)
    lower_prob = np.maximum(0.0, alpha / 2.0 - q)
    upper_prob = np.minimum(1.0, 1.0 - alpha / 2.0 + q)
    lo = quantile_values_from_grid(lower_prob, quantile_items)
    hi = quantile_values_from_grid(upper_prob, quantile_items)
    return lo, hi


def invert_distributional_excess(
    all_quantiles: dict[str, np.ndarray],
    q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    quantile_items = _sorted_quantile_items(all_quantiles)
    qs = np.stack(
        [x[1] for x in quantile_items], axis=0
    )  # shape (K, W, N, H) or similar
    taus = np.array([x[0] for x in quantile_items])

    pinball_losses = []
    for _, q_tau in quantile_items:
        pinball_losses.append(mean_pinball_at_y(q_tau, quantile_items))
    pinball_losses = np.stack(pinball_losses, axis=0)

    min_idx = np.argmin(pinball_losses, axis=0)
    min_pinball = np.min(pinball_losses, axis=0)

    grid = np.ogrid[tuple(slice(0, s) for s in qs.shape[1:])]
    y_star = qs[(min_idx,) + tuple(grid)]

    threshold = min_pinball + q

    mean_tau = np.mean(taus)
    mean_one_minus_tau = np.mean(1.0 - taus)

    q_min = np.min(qs, axis=0)
    q_max = np.max(qs, axis=0)

    left = q_min - np.maximum(
        np.maximum(1.0, np.abs(q_min)), threshold / mean_one_minus_tau
    )
    right = q_max + np.maximum(np.maximum(1.0, np.abs(q_max)), threshold / mean_tau)

    for _ in range(10):
        phi_left = mean_pinball_at_y(left, quantile_items)
        need_expand = phi_left <= threshold
        if not np.any(need_expand):
            break
        left = np.where(need_expand, y_star - 2.0 * (y_star - left), left)

    for _ in range(10):
        phi_right = mean_pinball_at_y(right, quantile_items)
        need_expand = phi_right <= threshold
        if not np.any(need_expand):
            break
        right = np.where(need_expand, y_star + 2.0 * (right - y_star), right)

    lo_left = left.copy()
    lo_right = y_star.copy()
    for _ in range(40):
        mid = 0.5 * (lo_left + lo_right)
        phi_mid = mean_pinball_at_y(mid, quantile_items)
        go_right = phi_mid > threshold
        lo_left = np.where(go_right, mid, lo_left)
        lo_right = np.where(go_right, lo_right, mid)
    lower_root = 0.5 * (lo_left + lo_right)

    hi_left = y_star.copy()
    hi_right = right.copy()
    for _ in range(40):
        mid = 0.5 * (hi_left + hi_right)
        phi_mid = mean_pinball_at_y(mid, quantile_items)
        go_left = phi_mid > threshold
        hi_right = np.where(go_left, mid, hi_right)
        hi_left = np.where(go_left, hi_left, mid)
    upper_root = 0.5 * (hi_left + hi_right)

    return lower_root, upper_root


def previous_residuals(residuals: np.ndarray, first_prev: np.ndarray) -> np.ndarray:
    prev = np.empty_like(residuals)
    if first_prev.ndim > 0 and first_prev.size == 1:
        prev[0] = first_prev.item()
    else:
        prev[0] = first_prev
    prev[1:] = residuals[:-1]
    return prev


def diff_scores(residuals: np.ndarray, prev_residuals: np.ndarray) -> np.ndarray:
    return np.abs(residuals - prev_residuals)


# ---------------------------------------------------------------------------
# Native-quantile evaluation helpers
# ---------------------------------------------------------------------------


def _native_metrics(
    y_true: np.ndarray,
    q_low: np.ndarray,
    q_high: np.ndarray,
    alpha: float,
) -> dict[str, float]:
    """Coverage, width, and Winkler score for the model's native interval."""
    covered = (y_true >= q_low) & (y_true <= q_high)
    widths = q_high - q_low
    coverage = float(np.mean(covered))
    avg_width = float(np.mean(widths))

    # Winkler score per observation
    penalty = np.where(
        y_true < q_low,
        (2.0 / alpha) * (q_low - y_true),
        np.where(y_true > q_high, (2.0 / alpha) * (y_true - q_high), 0.0),
    )
    winkler = float(np.mean(widths + penalty))
    return {
        'native_coverage': coverage,
        'native_avg_width': avg_width,
        'native_winkler_score': winkler,
    }


# ---------------------------------------------------------------------------
# Window loading
# ---------------------------------------------------------------------------


def _load_window(
    window_dir: Path,
    target_col: str = 'target',
    score_type: str | None = None,
    lo_key: str | None = None,
    hi_key: str | None = None,
) -> dict:
    """Load a single window's predictions from disk.

    Returns a dict with keys:

    * ``'point'``        — ``(N, H)`` float32 array of point forecasts.
    * ``'quantiles'``    — ``dict[str, (N, H) array]`` per quantile level.
    * ``'ground_truth'`` — ``(N, H)`` float32 array or ``None``.

    The Arrow file structure written by ForecasterAdapter is::

        DatasetDict({
            target: Dataset({
                predictions: [[float, ...], ...],   # (N, H) point forecasts
                "0.1": [[float, ...], ...],          # (N, H) per quantile
                ...
            })
        })

    Ground truth is stored separately by ``scripts/forecast.py`` as a
    ``ground_truth/`` DatasetDict alongside ``predictions/``.
    """
    pred_path = window_dir / 'predictions'
    ds_dict = hf_datasets.load_from_disk(str(pred_path))
    ds = ds_dict[target_col]  # type: ignore[index]
    columns: list[str] = ds.column_names  # type: ignore[union-attr]

    point = np.array(ds['predictions'], dtype=np.float32)  # type: ignore[index]
    quantile_cols: dict[str, np.ndarray] = {}
    for c in columns:
        if c not in ('predictions', 'id', 'item_id', 'timestamp', 'start', 'freq'):
            if score_type is not None and score_type not in (
                'distributional',
                'cdf_tail',
            ):
                if lo_key is not None and hi_key is not None:
                    try:
                        c_f = float(c)
                        lo_f = float(lo_key)
                        hi_f = float(hi_key)
                        if abs(c_f - lo_f) > 1e-5 and abs(c_f - hi_f) > 1e-5:
                            continue
                    except ValueError:
                        pass
            quantile_cols[c] = np.array(ds[c], dtype=np.float32)  # type: ignore[index]

    gt_path = window_dir / 'ground_truth'
    ground_truth: np.ndarray | None = None
    if gt_path.exists():
        gt_ds = hf_datasets.load_from_disk(str(gt_path))
        ground_truth = np.array(  # type: ignore[index]
            gt_ds[target_col]['target'],  # type: ignore[index]
            dtype=np.float32,
        )

    return {'point': point, 'quantiles': quantile_cols, 'ground_truth': ground_truth}


def _native_metrics_all_horizons(
    y_true: np.ndarray,  # (T, N, H)
    q_low: np.ndarray,  # (T, N, H)
    q_high: np.ndarray,  # (T, N, H)
    alpha: float,
) -> dict[str, np.ndarray]:
    """Vectorised native-quantile metrics for every horizon in one pass.

    Returns
    -------
    dict mapping key → (H,) ndarray:
        native_coverage, native_avg_width, native_winkler_score.
    """
    covered = (y_true >= q_low) & (y_true <= q_high)  # (T, N, H)
    widths = q_high - q_low  # (T, N, H)
    penalty = np.where(
        y_true < q_low,
        (2.0 / alpha) * (q_low - y_true),
        np.where(y_true > q_high, (2.0 / alpha) * (y_true - q_high), 0.0),
    )  # (T, N, H)
    winkler = widths + penalty  # (T, N, H)

    # Reduce over T and N dimensions → (H,)
    return {
        'native_coverage': np.mean(covered, axis=(0, 1)),  # (H,)
        'native_avg_width': np.mean(widths, axis=(0, 1)),  # (H,)
        'native_winkler_score': np.mean(winkler, axis=(0, 1)),  # (H,)
    }


def _slice_quantile_dict(
    qdict: dict[str, np.ndarray], slc: slice | np.ndarray
) -> dict[str, np.ndarray]:
    return {k: v[slc] for k, v in qdict.items()}


def _slice_quantile_dict_cs(
    qdict: dict[str, np.ndarray], slc: slice | np.ndarray
) -> dict[str, np.ndarray]:
    return {k: v[:, slc, :] for k, v in qdict.items()}


# ---------------------------------------------------------------------------
# Helper functions for parallel & fast evaluation
# ---------------------------------------------------------------------------


def _bounds_from_quantiles_static(
    score_type: str,
    q_lowers: np.ndarray,
    q_uppers: np.ndarray,
    point_preds: np.ndarray,
    q_low: np.ndarray,
    q_high: np.ndarray,
    scales: dict[str, np.ndarray] | None,
    all_quantiles: dict[str, np.ndarray] | None,
    prev_error: np.ndarray | None,
    alpha: float,
    is_asymmetric: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if score_type == 'abs':
        lo = point_preds - q_lowers
        hi = point_preds + q_uppers
    elif score_type == 'squared':
        if is_asymmetric:
            lo = point_preds - q_lowers
            hi = point_preds + q_uppers
        else:
            lo = point_preds - np.sqrt(np.maximum(q_lowers, 0.0))
            hi = point_preds + np.sqrt(np.maximum(q_uppers, 0.0))
    elif score_type == 'signed':
        lo = point_preds - q_lowers
        hi = point_preds + q_uppers
    elif score_type == 'iqr_scaled':
        iqr = q_high - q_low
        scale_floor = scales['native_iqr_floor'] if scales is not None else 1.0
        scale = _safe_scale(iqr, scale_floor)
        lo = point_preds - q_lowers * scale
        hi = point_preds + q_uppers * scale
    elif score_type == 'mad_scaled':
        scale = scales['rho_mad'] if scales is not None else 1.0
        lo = point_preds - q_lowers * scale
        hi = point_preds + q_uppers * scale
    elif score_type == 'cqr':
        lo = q_low - q_lowers
        hi = q_high + q_uppers
    elif score_type == 'scaled_cqr':
        sigma_lo = scales['sigma_lo'] if scales is not None else 1.0
        sigma_hi = scales['sigma_hi'] if scales is not None else 1.0
        lo = q_low - q_lowers * sigma_lo
        hi = q_high + q_uppers * sigma_hi
    elif score_type == 'distributional':
        if all_quantiles is None:
            raise ValueError('distributional inversion requires all_quantiles')
        lo, hi = invert_distributional_excess(all_quantiles, q_uppers)
    elif score_type == 'cdf_tail':
        if all_quantiles is None:
            raise ValueError('cdf_tail inversion requires all_quantiles')
        lo, hi = invert_cdf_tail(all_quantiles, q_uppers, alpha)
    elif score_type == 'log':
        eps = 1e-6
        center = np.log(point_preds + eps)
        lo = np.exp(center - q_lowers) - eps
        hi = np.exp(center + q_uppers) - eps
    elif score_type == 'diff':
        if prev_error is None:
            raise ValueError('diff inversion requires prev_error')
        center = point_preds + prev_error
        lo = center - q_lowers
        hi = center + q_uppers
    else:
        raise ValueError(f'Unknown score_type: {score_type}')

    return lo, hi


def _evaluate_one_sequence_static(
    predictor: ConformalPredictor,
    score_type: str,
    alpha: float,
    cal_scores: np.ndarray,
    test_scores: np.ndarray,
    point_preds: np.ndarray,
    y_true: np.ndarray,
    q_low: np.ndarray,
    q_high: np.ndarray,
    scales: dict[str, np.ndarray] | None,
    all_quantiles: dict[str, np.ndarray] | None,
    prev_error: np.ndarray | None,
) -> dict:
    p = predictor.clone()
    p.fit(cal_scores)

    if score_type == 'cqr' or score_type == 'scaled_cqr':
        eval_point_forecasts = np.column_stack([q_low, q_high])
    else:
        eval_point_forecasts = point_preds

    if hasattr(p, 'batch_evaluate'):
        res = p.batch_evaluate(test_scores, eval_point_forecasts, burnin=0)
    else:
        res = p.evaluate(test_scores, eval_point_forecasts, burnin=0)

    q_lowers = np.array(res.get('q_lowers', res['quantiles']))
    q_uppers = np.array(res.get('q_uppers', res['quantiles']))

    lower, upper = _bounds_from_quantiles_static(
        score_type=score_type,
        q_lowers=q_lowers,
        q_uppers=q_uppers,
        point_preds=point_preds,
        q_low=q_low,
        q_high=q_high,
        scales=scales,
        all_quantiles=all_quantiles,
        prev_error=prev_error,
        alpha=alpha,
        is_asymmetric=getattr(predictor, 'asymmetric', False),
    )

    valid = np.isfinite(lower) & np.isfinite(upper) & np.isfinite(y_true)
    covered = (y_true >= lower) & (y_true <= upper)
    widths = upper - lower

    miss = np.maximum(np.maximum(lower - y_true, y_true - upper), 0.0)
    winkler = widths + (2.0 / alpha) * miss

    coverage = float(np.mean(covered[valid])) if np.any(valid) else float('nan')
    avg_width = float(np.mean(widths[valid])) if np.any(valid) else float('nan')
    winkler_score = float(np.mean(winkler[valid])) if np.any(valid) else float('nan')

    return {
        'coverage': coverage,
        'avg_width': avg_width,
        'winkler_score': winkler_score,
        'lower': lower,
        'upper': upper,
        'covered': covered,
    }


def _evaluate_horizon_worker(
    h: int,
    predictor: ConformalPredictor,
    score_type: str,
    alpha: float,
    cal_scores_h: np.ndarray,
    test_scores_h: np.ndarray,
    test_pt_h: np.ndarray,
    test_gt_h: np.ndarray,
    test_ql_h: np.ndarray,
    test_qh_h: np.ndarray,
    scales_h: dict | None,
    test_all_q_h: dict | None,
    test_prev_error_h: np.ndarray | None,
    is_asymmetric: bool,
    cal_asym_h: np.ndarray | None,
    test_asym_h: np.ndarray | None,
) -> tuple[int, dict, np.ndarray, np.ndarray, np.ndarray]:
    N_series = test_pt_h.shape[1]
    T_test = test_pt_h.shape[0]

    covereds_list = []
    avg_widths_list = []
    winkler_scores_list = []

    lower_h = np.full((T_test, N_series), np.nan, dtype=np.float32)
    upper_h = np.full((T_test, N_series), np.nan, dtype=np.float32)
    covered_h = np.zeros((T_test, N_series), dtype=bool)

    if is_asymmetric:
        cal_scores_hn = cal_asym_h
        test_scores_hn = test_asym_h
    else:
        cal_scores_hn = cal_scores_h
        test_scores_hn = test_scores_h

    # Run the batched evaluation across all series
    res_batch = predictor.batch_evaluate_series(cal_scores_hn, test_scores_hn)
    if is_asymmetric:
        q_lowers, q_uppers = res_batch
    else:
        q_lowers = res_batch
        q_uppers = res_batch

    lower_h_val, upper_h_val = _bounds_from_quantiles_static(
        score_type=score_type,
        q_lowers=q_lowers,
        q_uppers=q_uppers,
        point_preds=test_pt_h,
        q_low=test_ql_h,
        q_high=test_qh_h,
        scales=scales_h,
        all_quantiles=test_all_q_h,
        prev_error=test_prev_error_h,
        alpha=alpha,
        is_asymmetric=is_asymmetric,
    )

    # Cast to float32 to match expected return type
    lower_h[:, :] = lower_h_val.astype(np.float32)
    upper_h[:, :] = upper_h_val.astype(np.float32)

    valid = np.isfinite(lower_h) & np.isfinite(upper_h) & np.isfinite(test_gt_h)
    covered = (test_gt_h >= lower_h) & (test_gt_h <= upper_h)
    covered_h[:, :] = covered

    width_h = upper_h - lower_h
    miss_h = np.maximum(np.maximum(lower_h - test_gt_h, test_gt_h - upper_h), 0.0)
    winkler_h = width_h + (2.0 / alpha) * miss_h

    for n in range(N_series):
        valid_n = valid[:, n]
        if valid_n.any():
            covereds_list.append(float(np.mean(covered[valid_n, n])))
            avg_widths_list.append(float(np.mean(width_h[valid_n, n])))
            winkler_scores_list.append(float(np.mean(winkler_h[valid_n, n])))
        else:
            covereds_list.append(float('nan'))
            avg_widths_list.append(float('nan'))
            winkler_scores_list.append(float('nan'))

    result = {
        'coverage': float(np.mean(covereds_list)),
        'avg_width': float(np.mean(avg_widths_list)),
        'winkler_score': float(np.mean(winkler_scores_list)),
    }

    return h, result, lower_h, upper_h, covered_h


def _evaluate_acmcp_horizon_worker(
    h_idx: int,
    predictor: AcMCP,
    cal_signed_h: np.ndarray,  # (C, N)
    test_signed_h: np.ndarray,  # (T, N)
    test_pt_h: np.ndarray,  # (T, N)
    cal_signed_history: np.ndarray | None,  # (C, N, h_idx)
) -> tuple[int, dict, np.ndarray]:
    N_series = cal_signed_h.shape[1]
    T_test = test_signed_h.shape[0]

    covereds_list = []
    avg_widths_list = []
    winkler_scores_list = []

    covered_h = np.zeros((T_test, N_series), dtype=bool)

    for n in range(N_series):
        p = predictor.clone()
        p.fit(
            cal_scores=cal_signed_h[:, n],
            signed_errors=cal_signed_history[:, n, :]
            if cal_signed_history is not None
            else None,
        )

        if hasattr(p, 'batch_evaluate'):
            res_n = p.batch_evaluate(test_signed_h[:, n], test_pt_h[:, n], burnin=0)
        else:
            res_n = p.evaluate(test_signed_h[:, n], test_pt_h[:, n], burnin=0)

        covereds_list.append(res_n['coverage'])
        avg_widths_list.append(res_n['avg_width'])
        winkler_scores_list.append(res_n['winkler_score'])

        if 'covered' in res_n:
            covered_h[:, n] = res_n['covered']

    result = {
        'coverage': float(np.mean(covereds_list)),
        'avg_width': float(np.mean(avg_widths_list)),
        'winkler_score': float(np.mean(winkler_scores_list)),
    }

    return h_idx, result, covered_h


# ---------------------------------------------------------------------------
# CPEvaluator
# ---------------------------------------------------------------------------


class CPEvaluator:
    """Evaluate CP methods against a model's native quantile predictions.

    The evaluator loads predictions produced by ``scripts/forecast.py``,
    splits them into calibration and test windows, and runs each CP method
    in the per-horizon independent mode.

    Parameters
    ----------
    predictions_dir : str or Path
        Path to the task-level output directory produced by forecast.py, e.g.
        ``results/tirex/ETTm2__T``.  Expected layout::

            <predictions_dir>/
              window_0/
                predictions/   ← DatasetDict saved to disk
                ground_truth/  ← DatasetDict with actual values
              window_1/
                ...
              metadata.json    ← written by forecast.py --cp-cal-windows

    horizon : int
        Forecast horizon H (number of steps ahead).
    cal_windows : int
        Number of leading windows to use for calibration.  The remaining
        windows are used for test evaluation.
    alpha : float
        Target miscoverage rate for both CP methods and native interval.
    score_type : str
        Which nonconformity score to use. Supported options:
        - Physical: ``"abs"`` (absolute residuals), ``"squared"`` (squared residuals),
          ``"signed"`` (signed residuals for asymmetric calibration).
        - Scaled: ``"iqr_scaled"`` (residuals scaled by model local IQR),
          ``"mad_scaled"`` (residuals scaled by model local MAD).
        - Quantile: ``"cqr"`` (CQR), ``"scaled_cqr"`` (CQR scaled by asymmetric local quantile scale),
          ``"distributional"`` (excess mean-pinball loss score), ``"cdf_tail"`` (PIT/CDF tail score).
        - Transformed/stateful: ``"log"`` (log absolute residual), ``"diff"`` (step-difference residual).
    target_col : str
        Name of the target column in the DatasetDict (default ``"target"``).
    """

    def __init__(
        self,
        predictions_dir: str | Path,
        horizon: int,
        cal_windows: int = 0,
        active_cal_windows: int | None = None,
        alpha: float = 0.1,
        score_type: str = 'abs',
        target_col: str = 'target',
        max_workers: int | None = None,
        parallel_min_horizon: int = 12,
        save_intervals: bool = False,
    ) -> None:
        self.predictions_dir = Path(predictions_dir)
        self.horizon = horizon
        self.cal_windows = cal_windows
        self.active_cal_windows = (
            active_cal_windows if active_cal_windows is not None else cal_windows
        )
        self.alpha = alpha
        self.score_type = score_type
        self._effective_score_type = score_type
        self.target_col = target_col
        self.max_workers = max_workers
        self.parallel_min_horizon = parallel_min_horizon
        self.save_intervals = save_intervals
        self._cached_data = None

        if score_type in _LEGACY_SCORE_RENAMES:
            new_name = _LEGACY_SCORE_RENAMES[score_type]
            raise ValueError(
                f'score_type={score_type!r} was renamed to {new_name!r}. '
                f'Run scripts/migrate_score_type_filenames.py for existing artifacts.'
            )
        if score_type not in _VALID_SCORE_TYPES:
            raise ValueError(
                f'score_type must be one of {_VALID_SCORE_TYPES}, got {score_type!r}'
            )

        # Discover available windows
        self._window_dirs: list[Path] = self._discover_windows(self.predictions_dir)
        self._num_windows = len(self._window_dirs)

        if cal_windows >= self._num_windows:
            raise ValueError(
                f'cal_windows ({cal_windows}) must be < total windows '
                f'({self._num_windows}).  Run forecast.py with at least '
                f'{cal_windows + 1} total windows.'
            )

    @property
    def _use_parallel(self) -> bool:
        """Whether to use threaded parallelism for horizon-level evaluation."""
        if self.max_workers == 1:
            return False
        return self.horizon >= self.parallel_min_horizon

    @property
    def _effective_max_workers(self) -> int:
        """Number of worker threads to use."""
        if self.max_workers is not None:
            return self.max_workers
        return min(os.cpu_count() or 1, self.horizon)

    # ------------------------------------------------------------------
    # Window discovery and loading
    # ------------------------------------------------------------------

    @staticmethod
    def _discover_windows(dir_path: Path) -> list[Path]:
        """Discover and sort window_N subdirectories in *dir_path*."""
        pred_dir = dir_path / 'predictions'
        if (
            pred_dir.exists()
            and pred_dir.is_dir()
            and (pred_dir / 'dataset_dict.json').exists()
        ):
            try:
                import datasets as hf_datasets

                ds_dict = hf_datasets.load_from_disk(str(pred_dir))
                if isinstance(ds_dict, hf_datasets.DatasetDict):
                    first_key = list(ds_dict.keys())[0]
                    num_windows = len(ds_dict[first_key])
                    return [dir_path / f'window_{i}' for i in range(num_windows)]
            except Exception:
                pass

        if (dir_path / 'predictions.npz').exists():
            try:
                data = np.load(str(dir_path / 'predictions.npz'))
                w_indices = set()
                for key in data.files:
                    if key.startswith('w') and '_' in key:
                        parts = key.split('_')
                        w_indices.add(int(parts[0][1:]))
                return [dir_path / f'window_{i}' for i in sorted(w_indices)]
            except Exception:
                pass

        return sorted(
            [
                d
                for d in dir_path.iterdir()
                if d.is_dir() and d.name.startswith('window_')
            ],
            key=lambda d: int(d.name.split('_')[1]),
        )

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_windows(
        self,
        base_dir: Path,
        window_dirs: list[Path],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict, np.ndarray | None]:
        """Load all windows and return stacked arrays.

        Returns
        -------
        point_preds : (W, N, H)  point forecasts (windows × series × horizon)
        q_low       : (W, N, H)  lower quantile (alpha/2)
        q_high      : (W, N, H)  upper quantile (1 - alpha/2)
        ground_truth: (W, N, H)  actual values
        all_quantiles: dict[str, (W, N, H)] for all available quantile levels
        series_ids: (N,) string array or None
        """
        import datasets as hf_datasets

        lo_key = f'{self.alpha / 2:.4g}'.rstrip('0').rstrip('.')
        hi_key = f'{1 - self.alpha / 2:.4g}'.rstrip('0').rstrip('.')
        series_ids = None

        pred_dir = base_dir / 'predictions'
        if (
            pred_dir.exists()
            and pred_dir.is_dir()
            and (pred_dir / 'dataset_dict.json').exists()
        ):
            try:
                ds_dict = hf_datasets.load_from_disk(str(pred_dir))
                if isinstance(ds_dict, hf_datasets.DatasetDict):
                    if self.target_col not in ds_dict:
                        self.target_col = list(ds_dict.keys())[0]

                    ds = ds_dict[self.target_col]

                    point_preds = np.stack(ds['predictions'], axis=0).astype(
                        np.float32
                    )  # (W, N, H)

                    all_quantiles = {}
                    for c in ds.column_names:
                        if c not in (
                            'predictions',
                            'id',
                            'item_id',
                            'timestamp',
                            'start',
                            'freq',
                        ):
                            if self.score_type not in ('distributional', 'cdf_tail'):
                                try:
                                    c_f = float(c)
                                    lo_f = float(lo_key)
                                    hi_f = float(hi_key)
                                    if (
                                        abs(c_f - lo_f) > 1e-5
                                        and abs(c_f - hi_f) > 1e-5
                                    ):
                                        continue
                                except ValueError:
                                    pass
                            all_quantiles[c] = np.stack(ds[c], axis=0).astype(
                                np.float32
                            )

                    if 'id' in ds.column_names:
                        series_ids = np.array(ds['id'][0])
                    elif 'item_id' in ds.column_names:
                        series_ids = np.array(ds['item_id'][0])

                    q_lo_arr, q_hi_arr = self._find_interval_quantiles(
                        all_quantiles, lo_key, hi_key
                    )

                    ground_truth = None
                    gt_dir = base_dir / 'ground_truth'
                    if (
                        gt_dir.exists()
                        and gt_dir.is_dir()
                        and (gt_dir / 'dataset_dict.json').exists()
                    ):
                        gt_dict = hf_datasets.load_from_disk(str(gt_dir))
                        if isinstance(gt_dict, hf_datasets.DatasetDict):
                            if self.target_col in gt_dict:
                                gt_ds = gt_dict[self.target_col]
                            else:
                                gt_ds = gt_dict[list(gt_dict.keys())[0]]

                            if 'target' in gt_ds.column_names:
                                ground_truth = np.stack(gt_ds['target'], axis=0).astype(
                                    np.float32
                                )
                            else:
                                ground_truth = np.stack(
                                    gt_ds[gt_ds.column_names[0]], axis=0
                                ).astype(np.float32)

                    if ground_truth is None:
                        raise RuntimeError(
                            'No ground_truth found in unified ground_truth DatasetDict. '
                            'Run forecast.py with --save-ground-truth.'
                        )

                    return (
                        point_preds,
                        q_lo_arr,
                        q_hi_arr,
                        ground_truth,
                        all_quantiles,
                        series_ids,
                    )
            except Exception as e:
                print(f'Failed to load as unified Arrow DatasetDict, falling back: {e}')

        # Fallback to .npz
        if (base_dir / 'predictions.npz').exists():
            try:
                preds_npz = np.load(str(base_dir / 'predictions.npz'))
                gts_npz = None
                if (base_dir / 'ground_truth.npz').exists():
                    gts_npz = np.load(str(base_dir / 'ground_truth.npz'))

                points, lows, highs, gts = [], [], [], []
                all_q: dict[str, list] = {}

                for window_dir in window_dirs:
                    w_name = window_dir.name
                    w_idx = int(w_name.split('_')[1])

                    point_key = f'w{w_idx}_{self.target_col}_predictions'
                    point = preds_npz[point_key]

                    quantile_prefix = f'w{w_idx}_{self.target_col}_'
                    quantiles = {}
                    for file_key in preds_npz.files:
                        if (
                            file_key.startswith(quantile_prefix)
                            and file_key != point_key
                        ):
                            q_name = file_key[len(quantile_prefix) :]
                            if self.score_type not in ('distributional', 'cdf_tail'):
                                try:
                                    q_f = float(q_name)
                                    lo_f = float(lo_key)
                                    hi_f = float(hi_key)
                                    if (
                                        abs(q_f - lo_f) > 1e-5
                                        and abs(q_f - hi_f) > 1e-5
                                    ):
                                        continue
                                except ValueError:
                                    pass
                            quantiles[q_name] = preds_npz[file_key]

                    ground_truth = None
                    if gts_npz is not None:
                        gt_key = f'w{w_idx}_{self.target_col}_target'
                        if gt_key in gts_npz:
                            ground_truth = gts_npz[gt_key]

                    points.append(point)
                    for k, v in quantiles.items():
                        all_q.setdefault(k, []).append(v)

                    q_lo_arr, q_hi_arr = self._find_interval_quantiles(
                        quantiles, lo_key, hi_key
                    )
                    lows.append(q_lo_arr)
                    highs.append(q_hi_arr)

                    if ground_truth is None:
                        raise RuntimeError(
                            f'No ground_truth found in ground_truth.npz for {w_name}.'
                        )
                    gts.append(ground_truth)

                point_preds = np.stack(points, axis=0)  # (W, N, H)
                q_low = np.stack(lows, axis=0)  # (W, N, H)
                q_high = np.stack(highs, axis=0)  # (W, N, H)
                ground_truth = np.stack(gts, axis=0)  # (W, N, H)
                all_quantiles = {k: np.stack(v, axis=0) for k, v in all_q.items()}

                return (
                    point_preds,
                    q_low,
                    q_high,
                    ground_truth,
                    all_quantiles,
                    series_ids,
                )
            except Exception as e:
                # Fall back to folder discovery if anything fails
                print(f'Failed to load as NPZ, falling back to window folders: {e}')
                pass

        points, lows, highs, gts = [], [], [], []
        all_q: dict[str, list] = {}

        for window_dir in window_dirs:
            data = _load_window(
                window_dir,
                self.target_col,
                score_type=self.score_type,
                lo_key=lo_key,
                hi_key=hi_key,
            )
            points.append(data['point'])

            # Collect all quantile columns
            for k, v in data['quantiles'].items():
                all_q.setdefault(k, []).append(v)

            # Identify low/high for native interval and CQR
            q_lo_arr, q_hi_arr = self._find_interval_quantiles(
                data['quantiles'], lo_key, hi_key
            )
            lows.append(q_lo_arr)
            highs.append(q_hi_arr)

            if data['ground_truth'] is None:
                raise RuntimeError(
                    f'No ground_truth found in {window_dir}.  '
                    'Run forecast.py with --save-ground-truth.'
                )
            gts.append(data['ground_truth'])

        point_preds = np.stack(points, axis=0)  # (W, N, H)
        q_low = np.stack(lows, axis=0)  # (W, N, H)
        q_high = np.stack(highs, axis=0)  # (W, N, H)
        ground_truth = np.stack(gts, axis=0)  # (W, N, H)
        all_quantiles = {k: np.stack(v, axis=0) for k, v in all_q.items()}

        return point_preds, q_low, q_high, ground_truth, all_quantiles, series_ids

    def _load_all(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict, np.ndarray | None]:
        """Load all windows of this evaluator's predictions directory."""
        if self._cached_data is None:
            self._cached_data = self._load_windows(
                self.predictions_dir, self._window_dirs
            )
        return self._cached_data

    def _get_cal_test_splits(
        self,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        dict[str, np.ndarray],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        dict[str, np.ndarray],
    ]:
        """Load and split predictions and ground truth into calibration and test sets.

        Returns:
            cal_pt, cal_gt, cal_ql, cal_qh, cal_all_q,
            test_pt, test_gt, test_ql, test_qh, test_all_q
        """
        # Old mode: split a single directory using self.cal_windows
        point_preds, q_low, q_high, ground_truth, all_quantiles, _ = self._load_all()

        cal_start = max(0, self.cal_windows - self.active_cal_windows)

        cal_pt = point_preds[cal_start : self.cal_windows]
        cal_gt = ground_truth[cal_start : self.cal_windows]
        cal_ql = q_low[cal_start : self.cal_windows]
        cal_qh = q_high[cal_start : self.cal_windows]
        cal_all_q = _slice_quantile_dict(
            all_quantiles, slice(cal_start, self.cal_windows)
        )

        test_pt = point_preds[self.cal_windows :]
        test_gt = ground_truth[self.cal_windows :]
        test_ql = q_low[self.cal_windows :]
        test_qh = q_high[self.cal_windows :]
        test_all_q = _slice_quantile_dict(all_quantiles, slice(self.cal_windows, None))
        return (
            cal_pt,
            cal_gt,
            cal_ql,
            cal_qh,
            cal_all_q,
            test_pt,
            test_gt,
            test_ql,
            test_qh,
            test_all_q,
        )

    def _find_interval_quantiles(
        self,
        quantile_cols: dict[str, np.ndarray],
        lo_key: str,
        hi_key: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Locate the lower and upper quantile arrays for the target alpha."""

        # Try exact string match first, then fuzzy float match
        def _find(target_key: str) -> np.ndarray | None:
            if target_key in quantile_cols:
                return quantile_cols[target_key]
            target_f = float(target_key)
            for k, v in quantile_cols.items():
                try:
                    if abs(float(k) - target_f) < 1e-6:
                        return v
                except ValueError:
                    pass
            return None

        lo_arr = _find(lo_key)
        hi_arr = _find(hi_key)

        if lo_arr is None or hi_arr is None:
            # Fallback: use the min and max available quantile
            keys_float = []
            for k in quantile_cols:
                try:
                    keys_float.append((float(k), k))
                except ValueError:
                    pass
            if not keys_float:
                raise RuntimeError(
                    f'No quantile columns found.  Available: {list(quantile_cols.keys())}'
                )
            keys_float.sort()
            lo_arr = quantile_cols[keys_float[0][1]]
            hi_arr = quantile_cols[keys_float[-1][1]]

        return lo_arr, hi_arr

    # ------------------------------------------------------------------
    # Score computation
    # ------------------------------------------------------------------

    def _compute_score_scales(
        self,
        cal_pt: np.ndarray,
        cal_ql: np.ndarray,
        cal_qh: np.ndarray,
        cal_gt: np.ndarray,
    ) -> dict[str, np.ndarray]:
        # Reduces all dimensions except the last one (H)
        axis = tuple(range(cal_pt.ndim - 1))

        cal_iqr = cal_qh - cal_ql
        cal_iqr_clamped = np.maximum(cal_iqr, 1e-8)
        native_iqr_floor_h = np.nanmedian(cal_iqr_clamped, axis=axis)

        residuals = cal_gt - cal_pt
        median_res = np.nanmedian(residuals, axis=axis, keepdims=True)
        rho_mad_h = 1.4826 * np.nanmedian(np.abs(residuals - median_res), axis=axis)

        lower_raw = cal_ql - cal_gt
        median_lo = np.nanmedian(lower_raw, axis=axis, keepdims=True)
        sigma_lo_h = 1.4826 * np.nanmedian(np.abs(lower_raw - median_lo), axis=axis)

        upper_raw = cal_gt - cal_qh
        median_hi = np.nanmedian(upper_raw, axis=axis, keepdims=True)
        sigma_hi_h = 1.4826 * np.nanmedian(np.abs(upper_raw - median_hi), axis=axis)

        # Apply _safe_scale
        native_iqr_floor_h = _safe_scale(native_iqr_floor_h, 1.0)
        rho_h = _safe_scale(rho_mad_h, fallback=native_iqr_floor_h)
        sigma_lo_h = _safe_scale(sigma_lo_h, fallback=rho_h)
        sigma_hi_h = _safe_scale(sigma_hi_h, fallback=rho_h)

        return {
            'native_iqr_floor': native_iqr_floor_h,
            'rho_mad': rho_h,
            'sigma_lo': sigma_lo_h,
            'sigma_hi': sigma_hi_h,
        }

    def _compute_scores(
        self,
        point_preds: np.ndarray,
        q_low: np.ndarray,
        q_high: np.ndarray,
        ground_truth: np.ndarray,
        *,
        scales: dict[str, np.ndarray] | None = None,
        all_quantiles: dict[str, np.ndarray] | None = None,
        prev_error: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute nonconformity scores.

        Returns
        -------
        scores : np.ndarray
        """
        score_t = self._effective_score_type

        if score_t == 'abs':
            return abs_scores(ground_truth, point_preds)
        elif score_t == 'squared':
            return squared_scores(ground_truth, point_preds)
        elif score_t == 'signed':
            return signed_scores(ground_truth, point_preds)
        elif score_t == 'iqr_scaled':
            iqr = q_high - q_low
            scale_floor = scales['native_iqr_floor'] if scales is not None else 1.0
            scale = _safe_scale(iqr, scale_floor)
            return abs_scores(ground_truth, point_preds) / scale
        elif score_t == 'mad_scaled':
            scale = scales['rho_mad'] if scales is not None else 1.0
            return abs_scores(ground_truth, point_preds) / scale
        elif score_t == 'cqr':
            return cqr_scores(ground_truth, q_low, q_high)
        elif score_t == 'scaled_cqr':
            sigma_lo = scales['sigma_lo'] if scales is not None else 1.0
            sigma_hi = scales['sigma_hi'] if scales is not None else 1.0
            return scaled_cqr_scores(ground_truth, q_low, q_high, sigma_lo, sigma_hi)
        elif score_t == 'distributional':
            if all_quantiles is None:
                raise ValueError('distributional score requires all_quantiles dict')
            return distributional_excess_scores(ground_truth, all_quantiles)
        elif score_t == 'cdf_tail':
            if all_quantiles is None:
                raise ValueError('cdf_tail score requires all_quantiles dict')
            return cdf_tail_scores(ground_truth, all_quantiles, self.alpha)
        elif score_t == 'log':
            return log_scores(ground_truth, point_preds)
        elif score_t == 'diff':
            if prev_error is None:
                raise ValueError('diff score requires prev_error')
            residuals = ground_truth - point_preds
            return diff_scores(residuals, prev_error)
        elif score_t == 'joint':
            # joint remains a sentinel
            return abs_scores(ground_truth, point_preds)
        else:
            raise ValueError(f'Unknown score_type: {score_t}')

    # ------------------------------------------------------------------
    # Per-horizon CP evaluation
    # ------------------------------------------------------------------

    def _compute_asymmetric_scores(
        self,
        point_preds: np.ndarray,
        q_low: np.ndarray,
        q_high: np.ndarray,
        ground_truth: np.ndarray,
        *,
        scales: dict[str, np.ndarray] | None = None,
        prev_error: np.ndarray | None = None,
    ) -> np.ndarray:
        score_t = self._effective_score_type
        if score_t == 'cqr':
            return np.stack([q_low - ground_truth, ground_truth - q_high], axis=-1)
        elif score_t == 'scaled_cqr':
            sigma_lo = scales['sigma_lo'] if scales is not None else 1.0
            sigma_hi = scales['sigma_hi'] if scales is not None else 1.0
            return np.stack(
                [(q_low - ground_truth) / sigma_lo, (ground_truth - q_high) / sigma_hi],
                axis=-1,
            )
        elif score_t in ('abs', 'squared', 'signed'):
            return ground_truth - point_preds
        elif score_t == 'iqr_scaled':
            iqr = q_high - q_low
            scale_floor = scales['native_iqr_floor'] if scales is not None else 1.0
            scale = _safe_scale(iqr, scale_floor)
            return (ground_truth - point_preds) / scale
        elif score_t == 'mad_scaled':
            scale = scales['rho_mad'] if scales is not None else 1.0
            return (ground_truth - point_preds) / scale
        elif score_t == 'log':
            eps = 1e-6
            valid = (ground_truth + eps > 0) & (point_preds + eps > 0)
            out = np.full_like(ground_truth, np.nan, dtype=float)
            out[valid] = np.log(ground_truth[valid] + eps) - np.log(
                point_preds[valid] + eps
            )
            return out
        elif score_t == 'diff':
            if prev_error is None:
                raise ValueError('diff score requires prev_error')
            residuals = ground_truth - point_preds
            return residuals - prev_error
        else:
            raise ValueError(f'Asymmetric mode not supported for score_type={score_t}')

    def _bounds_from_quantiles(
        self,
        *,
        score_type: str,
        q_lowers: np.ndarray,
        q_uppers: np.ndarray,
        point_preds: np.ndarray,
        q_low: np.ndarray,
        q_high: np.ndarray,
        scales: dict[str, np.ndarray] | None,
        all_quantiles: dict[str, np.ndarray] | None,
        prev_error: np.ndarray | None,
        is_asymmetric: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        return _bounds_from_quantiles_static(
            score_type=score_type,
            q_lowers=q_lowers,
            q_uppers=q_uppers,
            point_preds=point_preds,
            q_low=q_low,
            q_high=q_high,
            scales=scales,
            all_quantiles=all_quantiles,
            prev_error=prev_error,
            alpha=self.alpha,
            is_asymmetric=is_asymmetric,
        )

    def _evaluate_one_sequence(
        self,
        predictor: ConformalPredictor,
        cal_scores: np.ndarray,
        test_scores: np.ndarray,
        point_preds: np.ndarray,
        y_true: np.ndarray,
        q_low: np.ndarray,
        q_high: np.ndarray,
        scales: dict[str, np.ndarray] | None,
        all_quantiles: dict[str, np.ndarray] | None,
        prev_error: np.ndarray | None,
    ) -> dict:
        return _evaluate_one_sequence_static(
            predictor=predictor,
            score_type=self._effective_score_type,
            alpha=self.alpha,
            cal_scores=cal_scores,
            test_scores=test_scores,
            point_preds=point_preds,
            y_true=y_true,
            q_low=q_low,
            q_high=q_high,
            scales=scales,
            all_quantiles=all_quantiles,
            prev_error=prev_error,
        )

    # ------------------------------------------------------------------
    # Native quantile evaluation
    # ------------------------------------------------------------------

    def _evaluate_native(
        self,
        test_q_low: np.ndarray,
        test_q_high: np.ndarray,
        test_gt: np.ndarray,
    ) -> dict[str, float]:
        """Native-quantile metrics averaged over series and test windows."""
        # Flatten series and windows
        flat_gt = test_gt.reshape(-1)
        flat_lo = test_q_low.reshape(-1)
        flat_hi = test_q_high.reshape(-1)
        return _native_metrics(flat_gt, flat_lo, flat_hi, self.alpha)

    def _get_prev_errors(
        self, point_preds: np.ndarray, ground_truth: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        # point_preds, ground_truth have shape (W, N, H)
        full_residuals = ground_truth - point_preds
        full_prev_error = np.zeros_like(full_residuals)
        full_prev_error[1:] = full_residuals[:-1]

        cal_start = max(0, self.cal_windows - self.active_cal_windows)
        cal_prev_error = full_prev_error[cal_start : self.cal_windows]
        test_prev_error = full_prev_error[self.cal_windows :]
        return cal_prev_error, test_prev_error

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        methods: dict[str, list[ConformalPredictor]],
        use_cache: bool = False,
    ) -> pd.DataFrame:
        """Run full evaluation: calibrate on cal_windows, test on the rest.

        Parameters
        ----------
        methods : dict[str, list[ConformalPredictor]]
            Mapping from method name to a list of H fresh (unfitted)
            predictors — one per horizon step.

        Returns
        -------
        pd.DataFrame
            Columns: ``horizon``, ``method``, ``coverage``, ``joint_coverage``,
            ``avg_width``, ``winkler_score``, ``native_coverage``,
            ``native_joint_coverage``, ``native_avg_width``,
            ``native_winkler_score``.
        """
        for name, preds in methods.items():
            if len(preds) != self.horizon:
                raise ValueError(
                    f"Method '{name}' has {len(preds)} predictors but horizon={self.horizon}. "
                    'Provide exactly one predictor per horizon step.'
                )

        if self.score_type == 'signed':
            for name, preds in methods.items():
                for h, p in enumerate(preds):
                    if not getattr(p, 'asymmetric', False):
                        raise ValueError(
                            "score_type='signed' requires predictors instantiated with asymmetric=True."
                        )

        if self.score_type in ('distributional', 'cdf_tail'):
            for name, preds in methods.items():
                for h, p in enumerate(preds):
                    if getattr(p, 'asymmetric', False):
                        raise ValueError(
                            f'score_type={self.score_type!r} does not support asymmetric predictors.'
                        )

        # Load and split into cal / test
        (
            cal_pt,
            cal_gt,
            cal_ql,
            cal_qh,
            cal_all_q,
            test_pt,
            test_gt,
            test_ql,
            test_qh,
            test_all_q,
        ) = self._get_cal_test_splits()

        T_test = test_gt.shape[0]

        # Log score preflight
        if self.score_type == 'log':
            eps = 1e-6
            invalid_cal = np.any(cal_gt + eps <= 0) or np.any(cal_pt + eps <= 0)
            invalid_pred = np.any(test_pt + eps <= 0)
            if invalid_cal or invalid_pred:
                print_warning(
                    "log score domain preflight failed (nonpositive target/predictions); falling back to 'abs' score.",
                )
                self._effective_score_type = 'abs'

        # Compute previous errors chronologically
        cal_start = max(0, self.cal_windows - self.active_cal_windows)
        (
            point_preds_all,
            q_low_all,
            q_high_all,
            ground_truth_all,
            all_quantiles_all,
            _,
        ) = self._load_all()
        full_residuals = ground_truth_all - point_preds_all
        full_prev_error = np.zeros_like(full_residuals)
        full_prev_error[1:] = full_residuals[:-1]

        cal_prev_error = full_prev_error[cal_start : self.cal_windows]
        test_prev_error = full_prev_error[self.cal_windows :]

        # Compute scales
        scales = self._compute_score_scales(cal_pt, cal_ql, cal_qh, cal_gt)

        # Compute scores: (C, N, H) and (T_test, N, H)
        cal_scores_all = self._compute_scores(
            cal_pt,
            cal_ql,
            cal_qh,
            cal_gt,
            scales=scales,
            all_quantiles=cal_all_q,
            prev_error=cal_prev_error,
        )
        test_scores_all = self._compute_scores(
            test_pt,
            test_ql,
            test_qh,
            test_gt,
            scales=scales,
            all_quantiles=test_all_q,
            prev_error=test_prev_error,
        )

        # Pre-compute asymmetric scores for asymmetric predictors
        cal_asym_all: np.ndarray | None = None
        test_asym_all: np.ndarray | None = None

        def _get_asymmetric() -> tuple[np.ndarray, np.ndarray]:
            nonlocal cal_asym_all, test_asym_all
            if cal_asym_all is None:
                cal_asym_all = self._compute_asymmetric_scores(
                    cal_pt,
                    cal_ql,
                    cal_qh,
                    cal_gt,
                    scales=scales,
                    prev_error=cal_prev_error,
                )
                test_asym_all = self._compute_asymmetric_scores(
                    test_pt,
                    test_ql,
                    test_qh,
                    test_gt,
                    scales=scales,
                    prev_error=test_prev_error,
                )
            return cal_asym_all, test_asym_all

        # Native metrics for ALL horizons in one vectorised call: dict → (H,) arrays
        native_all = _native_metrics_all_horizons(test_gt, test_ql, test_qh, self.alpha)

        # Native joint coverage: fraction of test windows where all H steps are
        # simultaneously within the model's native quantile interval
        covered_native = (test_gt >= test_ql) & (test_gt <= test_qh)  # (T, N, H)
        native_joint_cov = float(np.mean(covered_native.all(axis=-1)))  # scalar

        method_names = list(methods.keys())
        rows: list[dict] = []
        N_series = test_pt.shape[1]

        for name in method_names:
            cache_file = (
                self.predictions_dir
                / f'intervals_{name}_a{self.alpha}_{_score_slug(self._effective_score_type)}_marginal.npz'
            )
            if use_cache and cache_file.exists():
                print(
                    f'  [Cache] Loaded marginal metrics and intervals for {name}',
                    flush=True,
                )
                loaded = np.load(cache_file, allow_pickle=True)
                h_results_list = loaded['metrics'].tolist()
                joint_cov = float(loaded['joint_cov'])
                method_time = float(loaded['runtime'])
                for item in h_results_list:
                    h = item['horizon']
                    result = item['result']
                    rows.append(
                        {
                            'horizon': h,
                            'method': name,
                            'coverage': result['coverage'],
                            'joint_coverage': joint_cov,
                            'avg_width': result['avg_width'],
                            'winkler_score': result['winkler_score'],
                            'runtime': method_time,
                        }
                    )
                continue

            t0 = time.perf_counter()
            preds = methods[name]
            # Accumulate per-window coverage indicators across horizons: (T_test, N, H)
            covered_per_horizon = np.zeros((T_test, N_series, self.horizon), dtype=bool)
            h_results: list[tuple[int, dict]] = []

            if self.save_intervals:
                method_lower = np.full(
                    (T_test, N_series, self.horizon), np.nan, dtype=np.float32
                )
                method_upper = np.full(
                    (T_test, N_series, self.horizon), np.nan, dtype=np.float32
                )
            else:
                method_lower = None
                method_upper = None

            any_asymmetric = any(getattr(p, 'asymmetric', False) for p in preds)
            if any_asymmetric:
                cal_asym_all, test_asym_all = _get_asymmetric()
            else:
                cal_asym_all, test_asym_all = None, None

            if self._use_parallel:
                tasks = []
                for h in range(self.horizon):
                    is_asym = getattr(preds[h], 'asymmetric', False)
                    if is_asym:
                        cal_asym_h = cal_asym_all[:, :, h]
                        test_asym_h = test_asym_all[:, :, h]
                    else:
                        cal_asym_h = None
                        test_asym_h = None

                    scales_h = (
                        {k: v[h] for k, v in scales.items()}
                        if scales is not None
                        else None
                    )
                    test_all_q_h = (
                        {k: v[:, :, h] for k, v in test_all_q.items()}
                        if test_all_q is not None
                        else None
                    )
                    test_prev_error_h = (
                        test_prev_error[:, :, h]
                        if test_prev_error is not None
                        else None
                    )

                    tasks.append(
                        (
                            h,
                            preds[h],
                            self._effective_score_type,
                            self.alpha,
                            cal_scores_all[:, :, h],
                            test_scores_all[:, :, h],
                            test_pt[:, :, h],
                            test_gt[:, :, h],
                            test_ql[:, :, h],
                            test_qh[:, :, h],
                            scales_h,
                            test_all_q_h,
                            test_prev_error_h,
                            is_asym,
                            cal_asym_h,
                            test_asym_h,
                        )
                    )

                h_results_unordered = []
                with ThreadPoolExecutor(
                    max_workers=self._effective_max_workers
                ) as executor:
                    futures = [
                        executor.submit(_evaluate_horizon_worker, *t) for t in tasks
                    ]
                    for fut in as_completed(futures):
                        h_results_unordered.append(fut.result())

                h_results_unordered.sort(key=lambda x: x[0])
                for h, result, lower_h, upper_h, covered_h in h_results_unordered:
                    h_results.append((h, result))
                    if (
                        self.save_intervals
                        and method_lower is not None
                        and method_upper is not None
                    ):
                        method_lower[:, :, h] = lower_h
                        method_upper[:, :, h] = upper_h
                    covered_per_horizon[:, :, h] = covered_h
            else:
                for h in range(self.horizon):
                    is_asymmetric = getattr(preds[h], 'asymmetric', False)
                    scales_h = (
                        {k: v[h] for k, v in scales.items()}
                        if scales is not None
                        else None
                    )
                    test_prev_error_h = (
                        test_prev_error[:, :, h]
                        if test_prev_error is not None
                        else None
                    )

                    covereds_list = []
                    avg_widths_list = []
                    winkler_scores_list = []

                    for n in range(N_series):
                        if is_asymmetric:
                            cal_score_slice = cal_asym_all[:, n, h]
                            test_score_slice = test_asym_all[:, n, h]
                        else:
                            cal_score_slice = cal_scores_all[:, n, h]
                            test_score_slice = test_scores_all[:, n, h]

                        res_n = _evaluate_one_sequence_static(
                            predictor=preds[h],
                            score_type=self._effective_score_type,
                            alpha=self.alpha,
                            cal_scores=cal_score_slice,
                            test_scores=test_score_slice,
                            point_preds=test_pt[:, n, h],
                            y_true=test_gt[:, n, h],
                            q_low=test_ql[:, n, h],
                            q_high=test_qh[:, n, h],
                            scales=scales_h,
                            all_quantiles={k: v[:, n, h] for k, v in test_all_q.items()}
                            if test_all_q is not None
                            else None,
                            prev_error=test_prev_error_h[:, n]
                            if test_prev_error_h is not None
                            else None,
                        )

                        covereds_list.append(res_n['coverage'])
                        avg_widths_list.append(res_n['avg_width'])
                        winkler_scores_list.append(res_n['winkler_score'])

                        if (
                            self.save_intervals
                            and method_lower is not None
                            and method_upper is not None
                        ):
                            method_lower[:, n, h] = res_n['lower']
                            method_upper[:, n, h] = res_n['upper']

                        if 'covered' in res_n:
                            covered_per_horizon[:, n, h] = res_n['covered']

                    # Combine results for horizon h
                    result = {
                        'coverage': float(np.mean(covereds_list)),
                        'avg_width': float(np.mean(avg_widths_list)),
                        'winkler_score': float(np.mean(winkler_scores_list)),
                    }

                    h_results.append((h, result))

            # Joint coverage: fraction of windows where ALL horizons are covered
            joint_cov = float(np.mean(covered_per_horizon.all(axis=-1)))
            method_time = time.perf_counter() - t0

            for h, result in h_results:
                rows.append(
                    {
                        'horizon': h,
                        'method': name,
                        'coverage': result['coverage'],
                        'joint_coverage': joint_cov,
                        'avg_width': result['avg_width'],
                        'winkler_score': result['winkler_score'],
                        'runtime': method_time,
                    }
                )

            # Save intervals
            if use_cache:
                metrics_to_save = [{'horizon': h, 'result': r} for h, r in h_results]
                save_kwargs = {
                    'metrics': np.array(metrics_to_save, dtype=object),
                    'joint_cov': joint_cov,
                    'runtime': method_time,
                }
                if (
                    self.save_intervals
                    and method_lower is not None
                    and method_upper is not None
                ):
                    save_kwargs['lower'] = method_lower
                    save_kwargs['upper'] = method_upper
                np.savez_compressed(cache_file, **save_kwargs)

        for h in range(self.horizon):
            rows.append(
                {
                    'horizon': h,
                    'method': 'Native',
                    'coverage': float(native_all['native_coverage'][h]),
                    'joint_coverage': native_joint_cov,
                    'avg_width': float(native_all['native_avg_width'][h]),
                    'winkler_score': float(native_all['native_winkler_score'][h]),
                    'runtime': float('nan'),
                }
            )

        df = pd.DataFrame(rows)
        cols = [
            'horizon',
            'method',
            'coverage',
            'joint_coverage',
            'avg_width',
            'winkler_score',
            'runtime',
        ]
        return df[cols]

    def _resolve_valid_indices(
        self, series_ids: np.ndarray, N_total: int, covariate_strategy: str
    ) -> np.ndarray:
        if covariate_strategy == 'all' or N_total == 1:
            valid_indices = np.arange(N_total)
        elif covariate_strategy == 'targets_only':
            target_idxs = [
                i for i, sid in enumerate(series_ids) if 'target' in str(sid).lower()
            ]
            if target_idxs:
                valid_indices = np.array(target_idxs)
            else:
                valid_indices = np.array([0])
        elif covariate_strategy == 'covariates_only':
            target_idxs = [
                i for i, sid in enumerate(series_ids) if 'target' in str(sid).lower()
            ]
            if target_idxs:
                valid_indices = np.array(
                    [i for i in range(N_total) if i not in target_idxs]
                )
            else:
                valid_indices = np.arange(1, N_total) if N_total > 1 else np.array([])
        else:
            valid_indices = np.arange(N_total)
        return valid_indices

    def _build_kfold(self, cal_windows: int, W: int, N_valid: int) -> KFold:
        N_cal_min = int(np.ceil(cal_windows / W))
        if N_cal_min < 1:
            N_cal_min = 1

        if N_valid <= N_cal_min:
            raise ValueError(
                f'Need at least {N_cal_min + 1} valid series for CV (N_cal_min={N_cal_min}, available={N_valid}).'
            )

        min_K = int(np.ceil(1.0 / (1.0 - N_cal_min / N_valid)))

        if min_K <= 5:
            K = min(5, N_valid)
        else:
            K = min(min_K, N_valid)

        if K < 2:
            K = 2

        return KFold(n_splits=K, shuffle=True, random_state=42)

    def _load_cached_cv_results(
        self, method_names: list[str], use_cache: bool
    ) -> tuple[list[dict], list[str]]:
        methods_to_run = []
        cached_rows = []
        for name in method_names:
            cache_file = (
                self.predictions_dir
                / f'intervals_{name}_a{self.alpha}_{_score_slug(self._effective_score_type)}_marginal_cv.npz'
            )
            if use_cache and cache_file.exists():
                print(
                    f'  [Cache] Loaded marginal CV metrics and intervals for {name}',
                    flush=True,
                )
                loaded = np.load(cache_file, allow_pickle=True)
                metrics = loaded['metrics'].tolist()
                for item in metrics:
                    h = item['horizon']
                    result = item['result']
                    cached_rows.append(
                        {
                            'horizon': h,
                            'method': name,
                            'coverage': result['coverage'],
                            'joint_coverage': float(loaded['joint_cov']),
                            'avg_width': result['avg_width'],
                            'winkler_score': result['winkler_score'],
                            'native_coverage': result['native_coverage'],
                            'native_joint_coverage': result['native_joint_coverage'],
                            'native_avg_width': result['native_avg_width'],
                            'native_winkler_score': result['native_winkler_score'],
                        }
                    )
            else:
                methods_to_run.append(name)
        return cached_rows, methods_to_run

    def run_cross_sectional_cv(
        self,
        methods: dict[str, list[ConformalPredictor]],
        cal_windows: int,
        covariate_strategy: str = 'targets_only',
        use_cache: bool = False,
    ) -> pd.DataFrame:
        """Run cross-sectional cross-validation for marginal methods."""
        point_preds, q_low, q_high, ground_truth, all_quantiles, series_ids = (
            self._load_all()
        )
        W, N_total, H = point_preds.shape

        if series_ids is None:
            raise RuntimeError(
                'No series_ids found in the dataset. Cannot perform cross-sectional CV.'
            )

        valid_indices = self._resolve_valid_indices(
            series_ids, N_total, covariate_strategy
        )
        kf = self._build_kfold(cal_windows, W, len(valid_indices))

        all_fold_dfs = []
        method_names = list(methods.keys())

        if self.score_type in _LEGACY_SCORE_RENAMES:
            new_name = _LEGACY_SCORE_RENAMES[self.score_type]
            raise ValueError(
                f'score_type={self.score_type!r} was renamed to {new_name!r}.'
            )
        if self.score_type not in _VALID_SCORE_TYPES:
            raise ValueError(
                f'score_type must be one of {_VALID_SCORE_TYPES}, got {self.score_type!r}'
            )

        if self.score_type == 'signed':
            for name, preds in methods.items():
                for h, p in enumerate(preds):
                    if not getattr(p, 'asymmetric', False):
                        raise ValueError(
                            "score_type='signed' requires asymmetric predictors."
                        )

        if self.score_type in ('distributional', 'cdf_tail'):
            for name, preds in methods.items():
                for h, p in enumerate(preds):
                    if getattr(p, 'asymmetric', False):
                        raise ValueError(
                            f'score_type={self.score_type!r} does not support asymmetric predictors.'
                        )

        # Log score preflight
        if self.score_type == 'log':
            eps = 1e-6
            invalid_cal = np.any(ground_truth + eps <= 0) or np.any(
                point_preds + eps <= 0
            )
            if invalid_cal:
                print_warning(
                    "log score domain preflight failed; falling back to 'abs' score.",
                )
                self._effective_score_type = 'abs'

        cached_rows, methods_to_run = self._load_cached_cv_results(
            method_names, use_cache
        )

        if cached_rows:
            all_fold_dfs.append(pd.DataFrame(cached_rows))

        if not methods_to_run:
            df_all = pd.concat(all_fold_dfs, ignore_index=True)
            cols_to_mean = [
                'coverage',
                'joint_coverage',
                'avg_width',
                'winkler_score',
                'native_coverage',
                'native_joint_coverage',
                'native_avg_width',
                'native_winkler_score',
            ]
            df_mean = (
                df_all.groupby(['horizon', 'method'])[cols_to_mean].mean().reset_index()
            )
            cols = [
                'horizon',
                'method',
                'coverage',
                'joint_coverage',
                'avg_width',
                'winkler_score',
                'native_coverage',
                'native_joint_coverage',
                'native_avg_width',
                'native_winkler_score',
            ]
            return df_mean[cols]

        method_intervals = {
            m: {
                'lower': np.full((W, N_total, H), np.nan, dtype=np.float32)
                if self.save_intervals
                else None,
                'upper': np.full((W, N_total, H), np.nan, dtype=np.float32)
                if self.save_intervals
                else None,
                'h_results_per_fold': [],
            }
            for m in methods_to_run
        }

        # Compute previous errors chronologically across the full tensor
        full_residuals = ground_truth - point_preds
        full_prev_error = np.zeros_like(full_residuals)
        full_prev_error[1:] = full_residuals[:-1]

        for fold_idx, (cal_rel_idx, test_rel_idx) in enumerate(kf.split(valid_indices)):
            cal_idx = valid_indices[cal_rel_idx]
            test_idx = valid_indices[test_rel_idx]

            cal_pt = point_preds[:, cal_idx, :]
            cal_gt = ground_truth[:, cal_idx, :]
            cal_ql = q_low[:, cal_idx, :]
            cal_qh = q_high[:, cal_idx, :]

            test_pt = point_preds[:, test_idx, :]
            test_gt = ground_truth[:, test_idx, :]
            test_ql = q_low[:, test_idx, :]
            test_qh = q_high[:, test_idx, :]

            cal_prev_error = full_prev_error[:, cal_idx, :]
            test_prev_error = full_prev_error[:, test_idx, :]

            cal_all_q = _slice_quantile_dict_cs(all_quantiles, cal_idx)
            test_all_q = _slice_quantile_dict_cs(all_quantiles, test_idx)

            scales = self._compute_score_scales(cal_pt, cal_ql, cal_qh, cal_gt)

            cal_scores_raw = self._compute_scores(
                cal_pt,
                cal_ql,
                cal_qh,
                cal_gt,
                scales=scales,
                all_quantiles=cal_all_q,
                prev_error=cal_prev_error,
            )
            test_scores_raw = self._compute_scores(
                test_pt,
                test_ql,
                test_qh,
                test_gt,
                scales=scales,
                all_quantiles=test_all_q,
                prev_error=test_prev_error,
            )

            cal_scores_all = cal_scores_raw.reshape(-1, H)
            test_scores_all = test_scores_raw.reshape(-1, H)

            test_gt_all = test_gt.reshape(-1, H)
            test_ql_all = test_ql.reshape(-1, H)
            test_qh_all = test_qh.reshape(-1, H)

            T_test = test_gt_all.shape[0]

            native_all = _native_metrics_all_horizons(
                test_gt.reshape(test_gt.shape[0] * test_gt.shape[1], 1, H),
                test_ql.reshape(test_ql.shape[0] * test_ql.shape[1], 1, H),
                test_qh.reshape(test_qh.shape[0] * test_qh.shape[1], 1, H),
                self.alpha,
            )

            covered_native = (test_gt_all >= test_ql_all) & (test_gt_all <= test_qh_all)
            native_joint_cov = float(np.mean(covered_native.all(axis=-1)))

            rows = []

            for name in methods_to_run:
                t0 = time.perf_counter()
                preds = methods[name]
                covered_per_horizon = np.zeros((T_test, self.horizon), dtype=bool)
                h_results = []

                cal_asym_all: np.ndarray | None = None
                test_asym_all: np.ndarray | None = None

                def _get_asymmetric() -> tuple[np.ndarray, np.ndarray]:
                    nonlocal cal_asym_all, test_asym_all
                    if cal_asym_all is None:
                        cal_asym_all = self._compute_asymmetric_scores(
                            cal_pt,
                            cal_ql,
                            cal_qh,
                            cal_gt,
                            scales=scales,
                            prev_error=cal_prev_error,
                        )
                        test_asym_all = self._compute_asymmetric_scores(
                            test_pt,
                            test_ql,
                            test_qh,
                            test_gt,
                            scales=scales,
                            prev_error=test_prev_error,
                        )
                    return cal_asym_all, test_asym_all

                any_asymmetric = any(getattr(p, 'asymmetric', False) for p in preds)
                if any_asymmetric:
                    cal_asym_all, test_asym_all = _get_asymmetric()
                else:
                    cal_asym_all, test_asym_all = None, None

                for h in range(self.horizon):
                    p = preds[h].clone()
                    is_asymmetric = getattr(p, 'asymmetric', False)
                    if is_asymmetric:
                        cal_s, test_s = cal_asym_all, test_asym_all
                        if cal_s.ndim == 4:
                            cal_score_slice = cal_s[:, :, h, :].reshape(-1, 2)
                            test_score_slice = test_s[:, :, h, :].reshape(-1, 2)
                        else:
                            cal_score_slice = cal_s[:, :, h].reshape(-1)
                            test_score_slice = test_s[:, :, h].reshape(-1)
                    else:
                        cal_score_slice = cal_scores_all[:, h]
                        test_score_slice = test_scores_all[:, h]

                    test_all_q_h = {
                        k: v[:, test_idx, h].reshape(-1)
                        for k, v in all_quantiles.items()
                    }

                    res_h = self._evaluate_one_sequence(
                        predictor=p,
                        cal_scores=cal_score_slice,
                        test_scores=test_score_slice,
                        point_preds=test_pt[:, :, h].reshape(-1),
                        y_true=test_gt[:, :, h].reshape(-1),
                        q_low=test_ql[:, :, h].reshape(-1),
                        q_high=test_qh[:, :, h].reshape(-1),
                        scales={k: v[h] for k, v in scales.items()}
                        if scales is not None
                        else None,
                        all_quantiles=test_all_q_h,
                        prev_error=test_prev_error[:, :, h].reshape(-1),
                    )

                    if (
                        self.save_intervals
                        and method_intervals[name]['lower'] is not None
                        and method_intervals[name]['upper'] is not None
                    ):
                        method_lower_2d = res_h['lower'].reshape(W, len(test_idx))
                        method_upper_2d = res_h['upper'].reshape(W, len(test_idx))
                        method_intervals[name]['lower'][:, test_idx, h] = (
                            method_lower_2d
                        )
                        method_intervals[name]['upper'][:, test_idx, h] = (
                            method_upper_2d
                        )

                    if 'covered' in res_h:
                        covered_per_horizon[:, h] = res_h['covered']

                    result = {
                        'coverage': res_h['coverage'],
                        'avg_width': res_h['avg_width'],
                        'winkler_score': res_h['winkler_score'],
                    }
                    h_results.append((h, result))

                joint_cov = float(np.mean(covered_per_horizon.all(axis=1)))
                method_time = time.perf_counter() - t0

                # Store joint_cov to average later
                method_intervals[name]['h_results_per_fold'].append(
                    {'joint_cov': joint_cov}
                )

                for h, result in h_results:
                    rows.append(
                        {
                            'horizon': h,
                            'method': name,
                            'coverage': result['coverage'],
                            'joint_coverage': joint_cov,
                            'avg_width': result['avg_width'],
                            'winkler_score': result['winkler_score'],
                            'fold': fold_idx,
                            'runtime': method_time,
                        }
                    )

            for h in range(self.horizon):
                rows.append(
                    {
                        'horizon': h,
                        'method': 'Native',
                        'coverage': float(native_all['native_coverage'][h]),
                        'joint_coverage': native_joint_cov,
                        'avg_width': float(native_all['native_avg_width'][h]),
                        'winkler_score': float(native_all['native_winkler_score'][h]),
                        'fold': fold_idx,
                        'runtime': float('nan'),
                    }
                )

            all_fold_dfs.append(pd.DataFrame(rows))

        for name in methods_to_run:
            cache_file = (
                self.predictions_dir
                / f'intervals_{name}_a{self.alpha}_{_score_slug(self._effective_score_type)}_marginal_cv.npz'
            )
            method_rows = []
            for df_fold in all_fold_dfs:
                fold_df = df_fold[df_fold['method'] == name]
                for _, row in fold_df.iterrows():
                    method_rows.append(
                        {
                            'horizon': row['horizon'],
                            'result': {
                                'coverage': row['coverage'],
                                'avg_width': row['avg_width'],
                                'winkler_score': row['winkler_score'],
                                'native_coverage': row.get(
                                    'native_coverage', float('nan')
                                ),
                                'native_joint_coverage': row.get(
                                    'native_joint_coverage', float('nan')
                                ),
                                'native_avg_width': row.get(
                                    'native_avg_width', float('nan')
                                ),
                                'native_winkler_score': row.get(
                                    'native_winkler_score', float('nan')
                                ),
                            },
                        }
                    )

            mdf = pd.DataFrame(
                [{'horizon': r['horizon'], **r['result']} for r in method_rows]
            )
            mean_metrics = (
                mdf.groupby('horizon').mean().reset_index().to_dict('records')
            )
            metrics_to_save = [
                {
                    'horizon': row['horizon'],
                    'result': {k: v for k, v in row.items() if k != 'horizon'},
                }
                for row in mean_metrics
            ]

            if use_cache:
                save_kwargs = {
                    'metrics': np.array(metrics_to_save, dtype=object),
                    'joint_cov': float(
                        np.mean(
                            [
                                x['joint_cov']
                                for x in method_intervals[name]['h_results_per_fold']
                            ]
                        )
                    ),
                }
                if (
                    self.save_intervals
                    and method_intervals[name]['lower'] is not None
                    and method_intervals[name]['upper'] is not None
                ):
                    save_kwargs['lower'] = method_intervals[name]['lower']
                    save_kwargs['upper'] = method_intervals[name]['upper']
                np.savez_compressed(cache_file, **save_kwargs)

        df_all = pd.concat(all_fold_dfs, ignore_index=True)
        cols_to_mean = [
            'coverage',
            'joint_coverage',
            'avg_width',
            'winkler_score',
            'runtime',
        ]
        df_mean = (
            df_all.groupby(['horizon', 'method'])[cols_to_mean].mean().reset_index()
        )

        cols = [
            'horizon',
            'method',
            'coverage',
            'joint_coverage',
            'avg_width',
            'winkler_score',
            'runtime',
        ]
        return df_mean[cols]

    def run_multi_step_cross_sectional_cv(
        self,
        methods: dict[str, JointPredictor],
        cal_windows: int,
        covariate_strategy: str = 'targets_only',
        use_cache: bool = False,
    ) -> pd.DataFrame:
        """Evaluate joint CP methods using N-split cross-validation."""
        point_preds, q_low, q_high, ground_truth, _, series_ids = self._load_all()
        W, N_total, H = point_preds.shape

        if series_ids is None:
            raise RuntimeError(
                'No series_ids found in dataset. Cannot perform cross-sectional CV.'
            )

        valid_indices = self._resolve_valid_indices(
            series_ids, N_total, covariate_strategy
        )
        kf = self._build_kfold(cal_windows, W, len(valid_indices))

        all_fold_dfs = []
        method_names = list(methods.keys())

        cached_rows, methods_to_run = self._load_cached_cv_results(
            method_names, use_cache
        )

        if cached_rows:
            all_fold_dfs.append(pd.DataFrame(cached_rows))

        if not methods_to_run:
            df_all = pd.concat(all_fold_dfs, ignore_index=True)
            cols_to_mean = [
                'coverage',
                'joint_coverage',
                'avg_width',
                'winkler_score',
                'native_coverage',
                'native_joint_coverage',
                'native_avg_width',
                'native_winkler_score',
            ]
            df_mean = (
                df_all.groupby(['horizon', 'method'])[cols_to_mean].mean().reset_index()
            )
            cols = [
                'horizon',
                'method',
                'coverage',
                'joint_coverage',
                'avg_width',
                'winkler_score',
                'native_coverage',
                'native_joint_coverage',
                'native_avg_width',
                'native_winkler_score',
            ]
            return df_mean[cols]

        method_intervals = {
            m: {
                'lower': np.full((W, N_total, H), np.nan, dtype=np.float32)
                if self.save_intervals
                else None,
                'upper': np.full((W, N_total, H), np.nan, dtype=np.float32)
                if self.save_intervals
                else None,
                'h_results_per_fold': [],
            }
            for m in methods_to_run
        }

        for fold_idx, (cal_rel_idx, test_rel_idx) in enumerate(kf.split(valid_indices)):
            cal_idx = valid_indices[cal_rel_idx]
            test_idx = valid_indices[test_rel_idx]

            cal_pt = point_preds[:, cal_idx, :]
            cal_gt = ground_truth[:, cal_idx, :]

            test_pt = point_preds[:, test_idx, :]
            test_gt = ground_truth[:, test_idx, :]
            test_ql = q_low[:, test_idx, :]
            test_qh = q_high[:, test_idx, :]

            # For JointPredictor, we need (W_cal * N_cal, N=1, H)
            cal_pt_flat = cal_pt.reshape(-1, 1, H)
            cal_gt_flat = cal_gt.reshape(-1, 1, H)

            test_pt_flat = test_pt.reshape(-1, 1, H)
            test_gt_flat = test_gt.reshape(-1, 1, H)
            test_ql_flat = test_ql.reshape(-1, 1, H)
            test_qh_flat = test_qh.reshape(-1, 1, H)

            covered_all = (test_gt_flat >= test_ql_flat) & (
                test_gt_flat <= test_qh_flat
            )
            native_joint = float(np.mean(covered_all.all(axis=2)))
            native_marginal = float(np.mean(covered_all))
            native_widths = float(np.mean(test_qh_flat - test_ql_flat))
            penalty = np.where(
                test_gt_flat < test_ql_flat,
                (2.0 / self.alpha) * (test_ql_flat - test_gt_flat),
                np.where(
                    test_gt_flat > test_qh_flat,
                    (2.0 / self.alpha) * (test_gt_flat - test_qh_flat),
                    0.0,
                ),
            )
            native_winkler = float(np.mean(test_qh_flat - test_ql_flat + penalty))

            rows = []
            for name, predictor in methods.items():
                t0 = time.perf_counter()
                p = copy.deepcopy(predictor)
                p.fit(cal_pt_flat, cal_gt_flat)
                result = p.evaluate(test_pt_flat, test_gt_flat)
                method_time = time.perf_counter() - t0
                rows.append(
                    {
                        'method': name,
                        'joint_coverage': result['joint_coverage'],
                        'marginal_coverage': result.get(
                            'marginal_coverage', float('nan')
                        ),
                        'avg_width': result['avg_width'],
                        'winkler_score': result['winkler_score'],
                        'fold': fold_idx,
                        'runtime': method_time,
                    }
                )

            rows.append(
                {
                    'method': 'Native',
                    'joint_coverage': native_joint,
                    'marginal_coverage': native_marginal,
                    'avg_width': native_widths,
                    'winkler_score': native_winkler,
                    'fold': fold_idx,
                    'runtime': float('nan'),
                }
            )

            all_fold_dfs.append(pd.DataFrame(rows))

        for name in methods_to_run:
            cache_file = (
                self.predictions_dir
                / f'intervals_{name}_a{self.alpha}_{_score_slug(self._effective_score_type)}_marginal_cv.npz'
            )
            # method_intervals has lower and upper of shape (W, N_total, H)
            # We want to save the final bounds.
            # We also need to aggregate the metrics. We can extract them from all_fold_dfs for this method.
            method_rows = []
            for df_fold in all_fold_dfs:
                fold_df = df_fold[df_fold['method'] == name]
                for _, row in fold_df.iterrows():
                    method_rows.append(
                        {
                            'horizon': row['horizon'],
                            'result': {
                                'coverage': row['coverage'],
                                'avg_width': row['avg_width'],
                                'winkler_score': row['winkler_score'],
                                'native_coverage': row.get(
                                    'native_coverage', float('nan')
                                ),
                                'native_joint_coverage': row.get(
                                    'native_joint_coverage', float('nan')
                                ),
                                'native_avg_width': row.get(
                                    'native_avg_width', float('nan')
                                ),
                                'native_winkler_score': row.get(
                                    'native_winkler_score', float('nan')
                                ),
                            },
                        }
                    )

            mdf = pd.DataFrame(
                [{'horizon': r['horizon'], **r['result']} for r in method_rows]
            )
            mean_metrics = (
                mdf.groupby('horizon').mean().reset_index().to_dict('records')
            )
            # Pack back into expected format
            metrics_to_save = [
                {
                    'horizon': row['horizon'],
                    'result': {k: v for k, v in row.items() if k != 'horizon'},
                }
                for row in mean_metrics
            ]

            if use_cache:
                save_kwargs = {
                    'metrics': np.array(metrics_to_save, dtype=object),
                    'joint_cov': float(
                        np.mean(
                            [
                                x['joint_cov']
                                for x in method_intervals[name]['h_results_per_fold']
                            ]
                        )
                    ),
                }
                if (
                    self.save_intervals
                    and method_intervals[name]['lower'] is not None
                    and method_intervals[name]['upper'] is not None
                ):
                    save_kwargs['lower'] = method_intervals[name]['lower']
                    save_kwargs['upper'] = method_intervals[name]['upper']
                np.savez_compressed(cache_file, **save_kwargs)

        df_all = pd.concat(all_fold_dfs, ignore_index=True)
        cols_to_mean = [
            'joint_coverage',
            'marginal_coverage',
            'avg_width',
            'winkler_score',
            'runtime',
        ]
        df_mean = df_all.groupby('method')[cols_to_mean].mean().reset_index()

        col_order = ['method'] + cols_to_mean
        return df_mean[col_order]  # type: ignore[return-value]

    def summary(
        self,
        methods: dict[str, list[ConformalPredictor]],
    ) -> pd.DataFrame:
        """Horizon-averaged summary of :meth:`run`.

        Returns
        -------
        pd.DataFrame
            One row per method with horizon-averaged metrics.
            ``joint_coverage`` is a method-level scalar so its mean equals
            itself; it is included for convenience.
        """
        df = self.run(methods)
        agg: pd.DataFrame = (
            df.groupby('method')[
                [
                    'coverage',
                    'joint_coverage',
                    'avg_width',
                    'winkler_score',
                    'native_coverage',
                    'native_joint_coverage',
                    'native_avg_width',
                    'native_winkler_score',
                    'runtime',
                ]
            ]
            .mean()
            .reset_index()
        )
        return agg

    # ------------------------------------------------------------------
    # AcMCP evaluation (requires signed errors + multi-horizon fit)
    # ------------------------------------------------------------------

    def run_acmcp(
        self,
        predictors: dict[str, list['AcMCP']],
    ) -> pd.DataFrame:
        """Evaluate AcMCP methods.

        AcMCP requires **signed** errors ``e = gt - pred`` (not absolute
        scores) and needs the full shorter-horizon error history for its OLS
        scorecaster.  This method handles that wiring automatically.

        Parameters
        ----------
        predictors : dict[str, list[AcMCP]]
            Mapping from method name to a list of H fresh (unfitted)
            ``AcMCP`` instances — one per horizon step.

        Returns
        -------
        pd.DataFrame
            Same columns as :meth:`run`: ``horizon``, ``method``,
            ``coverage``, ``joint_coverage``, ``avg_width``,
            ``winkler_score``, ``native_coverage``, ``native_joint_coverage``,
            ``native_avg_width``, ``native_winkler_score``.
        """
        for name, preds in predictors.items():
            if len(preds) != self.horizon:
                raise ValueError(
                    f"Method '{name}' has {len(preds)} predictors but "
                    f'horizon={self.horizon}.'
                )

        # Load and split
        cal_pt, cal_gt, cal_ql, cal_qh, _, test_pt, test_gt, test_ql, test_qh, _ = (
            self._get_cal_test_splits()
        )

        # Compute signed errors: shape (C, N, H) and (T_test, N, H)
        cal_signed = cal_gt - cal_pt
        test_signed = test_gt - test_pt

        T_test = test_gt.shape[0]
        N_series = test_gt.shape[1]

        # Native metrics for ALL horizons in one vectorised call
        native_all = _native_metrics_all_horizons(test_gt, test_ql, test_qh, self.alpha)

        # Native joint coverage
        covered_native = (test_gt >= test_ql) & (test_gt <= test_qh)  # (T, N, H)
        native_joint_cov = float(np.mean(covered_native.all(axis=-1)))

        method_names = list(predictors.keys())
        rows: list[dict] = []

        for name in method_names:
            t0 = time.perf_counter()
            preds = predictors[name]
            covered_per_horizon = np.zeros((T_test, N_series, self.horizon), dtype=bool)
            h_results: list[tuple[int, dict]] = []

            if self._use_parallel:
                tasks = []
                for h_idx in range(self.horizon):
                    tasks.append(
                        (
                            h_idx,
                            preds[h_idx],
                            cal_signed[:, :, h_idx],
                            test_signed[:, :, h_idx],
                            test_pt[:, :, h_idx],
                            cal_signed[:, :, :h_idx] if h_idx > 0 else None,
                        )
                    )

                h_results_unordered = []
                with ThreadPoolExecutor(
                    max_workers=self._effective_max_workers
                ) as executor:
                    futures = [
                        executor.submit(_evaluate_acmcp_horizon_worker, *t)
                        for t in tasks
                    ]
                    for fut in as_completed(futures):
                        h_results_unordered.append(fut.result())

                # Sort results by h_idx
                h_results_unordered.sort(key=lambda x: x[0])
                for h_idx, result, covered_h in h_results_unordered:
                    h_results.append((h_idx, result))
                    covered_per_horizon[:, :, h_idx] = covered_h
            else:
                for h_idx in range(self.horizon):
                    covereds_list = []
                    avg_widths_list = []
                    winkler_scores_list = []

                    for n in range(N_series):
                        cal_scores_hn = cal_signed[:, n, h_idx]
                        test_scores_hn = test_signed[:, n, h_idx]
                        test_point_hn = test_pt[:, n, h_idx]

                        p = preds[h_idx].clone()
                        p.fit(
                            cal_scores=cal_scores_hn,
                            signed_errors=cal_signed[:, n, :h_idx]
                            if h_idx > 0
                            else None,
                        )
                        if hasattr(p, 'batch_evaluate'):
                            res_n = p.batch_evaluate(
                                test_scores_hn, test_point_hn, burnin=0
                            )
                        else:
                            res_n = p.evaluate(test_scores_hn, test_point_hn, burnin=0)

                        covereds_list.append(res_n['coverage'])
                        avg_widths_list.append(res_n['avg_width'])
                        winkler_scores_list.append(res_n['winkler_score'])

                        if 'covered' in res_n:
                            covered_per_horizon[:, n, h_idx] = res_n['covered']

                    result = {
                        'coverage': float(np.mean(covereds_list)),
                        'avg_width': float(np.mean(avg_widths_list)),
                        'winkler_score': float(np.mean(winkler_scores_list)),
                    }
                    h_results.append((h_idx, result))

            # Joint coverage: fraction of windows where ALL horizons are covered
            joint_cov = float(np.mean(covered_per_horizon.all(axis=-1)))
            method_time = time.perf_counter() - t0

            for h_idx, result in h_results:
                rows.append(
                    {
                        'horizon': h_idx,
                        'method': name,
                        'coverage': result['coverage'],
                        'joint_coverage': joint_cov,
                        'avg_width': result['avg_width'],
                        'winkler_score': result['winkler_score'],
                        'runtime': method_time,
                    }
                )

        for h in range(self.horizon):
            rows.append(
                {
                    'horizon': h,
                    'method': 'Native',
                    'coverage': float(native_all['native_coverage'][h]),
                    'joint_coverage': native_joint_cov,
                    'avg_width': float(native_all['native_avg_width'][h]),
                    'winkler_score': float(native_all['native_winkler_score'][h]),
                    'runtime': float('nan'),
                }
            )

        df = pd.DataFrame(rows)
        cols = [
            'horizon',
            'method',
            'coverage',
            'joint_coverage',
            'avg_width',
            'winkler_score',
            'runtime',
        ]
        return df[cols]  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Joint coverage evaluation
    # ------------------------------------------------------------------

    def run_multi_step(
        self,
        methods: dict[str, JointPredictor],
        use_cache: bool = False,
    ) -> pd.DataFrame:
        """Evaluate joint-coverage CP methods.

        Unlike :meth:`run` (which evaluates per-horizon marginal coverage),
        this method evaluates methods that target *simultaneous* coverage:
        the probability that **all** H horizon steps are covered at once.

        Each :class:`~timecp.base.JointPredictor` is calibrated on the full
        ``(C, N, H)`` calibration block, then evaluated on each of the T test
        windows producing per-window metrics which are averaged.

        Parameters
        ----------
        methods : dict[str, JointPredictor]
            Mapping from method name to a single fresh (unfitted)
            ``JointPredictor`` instance.

        Returns
        -------
        pd.DataFrame
            Columns: ``method``, ``joint_coverage``, ``marginal_coverage``,
            ``avg_width``, ``winkler_score``, ``native_joint_coverage``,
            ``native_marginal_coverage``, ``native_avg_width``,
            ``native_winkler_score``.

            ``marginal_coverage`` is the mean per-horizon coverage averaged
            over all H horizon steps (i.e. the average fraction of test
            windows where an individual horizon step is covered).
        """
        # Load data
        cal_pt, cal_gt, cal_ql, cal_qh, test_pt, test_gt, test_ql, test_qh = (
            self._get_cal_test_splits()
        )

        # --- Native metrics ---
        covered_all = (test_gt >= test_ql) & (test_gt <= test_qh)  # (T, N, H)
        # Joint: fraction of (T*N) windows where all H steps are within native interval
        native_joint = float(np.mean(covered_all.all(axis=2)))
        # Marginal: mean per-step coverage averaged over all horizons, T, and N
        native_marginal = float(np.mean(covered_all))
        native_widths = float(np.mean(test_qh - test_ql))
        penalty = np.where(
            test_gt < test_ql,
            (2.0 / self.alpha) * (test_ql - test_gt),
            np.where(test_gt > test_qh, (2.0 / self.alpha) * (test_gt - test_qh), 0.0),
        )
        native_winkler = float(np.mean(test_qh - test_ql + penalty))

        rows: list[dict] = []

        for name, predictor in methods.items():
            cache_file = (
                self.predictions_dir
                / f'intervals_{name}_a{self.alpha}_{_score_slug(self._effective_score_type)}_multi_step.npz'
            )
            if use_cache and cache_file.exists():
                print(
                    f'  [Cache] Loaded multi-step metrics and intervals for {name}',
                    flush=True,
                )
                loaded = np.load(cache_file, allow_pickle=True)
                metrics = loaded['metrics'].item()
                rows.append(
                    {
                        'method': name,
                        'joint_coverage': float(metrics['joint_coverage']),
                        'marginal_coverage': float(
                            metrics.get('marginal_coverage', float('nan'))
                        ),
                        'avg_width': float(metrics['avg_width']),
                        'winkler_score': float(metrics['winkler_score']),
                        'runtime': float(loaded['runtime']),
                    }
                )
                continue

            t0 = time.perf_counter()
            p = copy.deepcopy(predictor)
            p.fit(cal_pt, cal_gt)
            result = p.evaluate(test_pt, test_gt)
            method_time = time.perf_counter() - t0
            radii = result.get('radii', None)

            metrics_to_save = {
                'joint_coverage': result['joint_coverage'],
                'marginal_coverage': result.get('marginal_coverage', float('nan')),
                'avg_width': result['avg_width'],
                'winkler_score': result['winkler_score'],
            }
            if use_cache:
                save_kwargs = {
                    'metrics': metrics_to_save,
                    'runtime': method_time,
                }
                if self.save_intervals and radii is not None:
                    method_lower = test_pt - radii[:, None, :]
                    method_upper = test_pt + radii[:, None, :]
                    save_kwargs['lower'] = method_lower
                    save_kwargs['upper'] = method_upper
                np.savez_compressed(cache_file, **save_kwargs)

            rows.append(
                {
                    'method': name,
                    'joint_coverage': result['joint_coverage'],
                    'marginal_coverage': result.get('marginal_coverage', float('nan')),
                    'avg_width': result['avg_width'],
                    'winkler_score': result['winkler_score'],
                    'runtime': method_time,
                }
            )

        rows.append(
            {
                'method': 'Native',
                'joint_coverage': native_joint,
                'marginal_coverage': native_marginal,
                'avg_width': native_widths,
                'winkler_score': native_winkler,
                'runtime': float('nan'),
            }
        )

        df = pd.DataFrame(rows)
        col_order = [
            'method',
            'joint_coverage',
            'marginal_coverage',
            'avg_width',
            'winkler_score',
            'runtime',
        ]
        return df[col_order]  # type: ignore[return-value]

    def summary_joint(
        self,
        methods: dict[str, JointPredictor],
    ) -> pd.DataFrame:
        """Summary wrapper around :meth:`run_multi_step` (no aggregation needed).

        Returns the same DataFrame as :meth:`run_multi_step` since joint methods
        already produce a single aggregate row per method.
        """
        return self.run_multi_step(methods)

    # ------------------------------------------------------------------
    # Single-step evaluation
    # ------------------------------------------------------------------

    def run_single_step(
        self,
        methods: dict[str, ConformalPredictor],
        use_cache: bool = False,
        online: bool = False,
    ) -> pd.DataFrame:
        """Evaluate CP methods in single-step mode."""
        for name, pred in methods.items():
            if self.score_type == 'signed' and not getattr(pred, 'asymmetric', False):
                raise ValueError(
                    "score_type='signed' requires predictor instantiated with asymmetric=True."
                )
            if self.score_type in ('distributional', 'cdf_tail') and getattr(
                pred, 'asymmetric', False
            ):
                raise ValueError(
                    f'score_type={self.score_type!r} does not support asymmetric predictor.'
                )

        # Load data and split into cal/test
        (
            cal_pt,
            cal_gt,
            cal_ql,
            cal_qh,
            cal_all_q,
            test_pt,
            test_gt,
            test_ql,
            test_qh,
            test_all_q,
        ) = self._get_cal_test_splits()

        T_test = test_gt.shape[0]
        N_series = test_gt.shape[1]
        H = self.horizon

        # Log score preflight
        if self.score_type == 'log':
            eps = 1e-6
            invalid_cal = np.any(cal_gt + eps <= 0) or np.any(cal_pt + eps <= 0)
            invalid_pred = np.any(test_pt + eps <= 0)
            if invalid_cal or invalid_pred:
                print_warning(
                    "log score domain preflight failed; falling back to 'abs' score.",
                )
                self._effective_score_type = 'abs'

        # Compute previous errors chronologically across the full tensor
        point_preds_all, _, _, ground_truth_all, _, _ = self._load_all()
        cal_prev_error, test_prev_error = self._get_prev_errors(
            point_preds_all, ground_truth_all
        )

        # Compute scales
        scales = self._compute_score_scales(cal_pt, cal_ql, cal_qh, cal_gt)

        # Compute scores: (C, N, H) and (T_test, N, H)
        cal_scores_all = self._compute_scores(
            cal_pt,
            cal_ql,
            cal_qh,
            cal_gt,
            scales=scales,
            all_quantiles=cal_all_q,
            prev_error=cal_prev_error,
        )
        test_scores_all = self._compute_scores(
            test_pt,
            test_ql,
            test_qh,
            test_gt,
            scales=scales,
            all_quantiles=test_all_q,
            prev_error=test_prev_error,
        )

        # Pre-compute asymmetric scores for asymmetric predictors
        cal_asym_all: np.ndarray | None = None
        test_asym_all: np.ndarray | None = None

        def _get_asymmetric() -> tuple[np.ndarray, np.ndarray]:
            nonlocal cal_asym_all, test_asym_all
            if cal_asym_all is None:
                cal_asym_all = self._compute_asymmetric_scores(
                    cal_pt,
                    cal_ql,
                    cal_qh,
                    cal_gt,
                    scales=scales,
                    prev_error=cal_prev_error,
                )
                test_asym_all = self._compute_asymmetric_scores(
                    test_pt,
                    test_ql,
                    test_qh,
                    test_gt,
                    scales=scales,
                    prev_error=test_prev_error,
                )
            return cal_asym_all, test_asym_all

        # Native metrics
        native_all = _native_metrics_all_horizons(test_gt, test_ql, test_qh, self.alpha)
        covered_native = (test_gt >= test_ql) & (test_gt <= test_qh)  # (T, N, H)
        native_joint_cov = float(np.mean(covered_native.all(axis=-1)))

        rows: list[dict] = []

        for name, pred in methods.items():
            t0 = time.perf_counter()
            is_asymmetric = getattr(pred, 'asymmetric', False)
            if is_asymmetric:
                cal_asym_all, test_asym_all = _get_asymmetric()
            else:
                cal_asym_all, test_asym_all = None, None

            # Accumulators for all series N
            covered_3d = np.zeros((T_test, N_series, H), dtype=bool)
            width_3d = np.zeros((T_test, N_series, H), dtype=np.float32)
            miss_3d = np.zeros((T_test, N_series, H), dtype=np.float32)

            for n in range(N_series):
                p = pred.clone()

                if is_asymmetric:
                    cal_scores_n = cal_asym_all[:, n, :]
                    test_scores_n = test_asym_all[:, n, :]
                    if cal_scores_n.ndim == 3:  # shape (C, H, 2)
                        cal_flat_n = cal_scores_n.reshape(-1, 2)
                        test_flat_n = test_scores_n.reshape(-1, 2)
                    else:
                        cal_flat_n = cal_scores_n.reshape(-1)
                        test_flat_n = test_scores_n.reshape(-1)
                else:
                    cal_scores_n = cal_scores_all[:, n, :]
                    test_scores_n = test_scores_all[:, n, :]
                    cal_flat_n = cal_scores_n.reshape(-1)
                    test_flat_n = test_scores_n.reshape(-1)

                if online:
                    if is_asymmetric:
                        p.fit(cal_flat_n)
                        quantiles_raw = p._scan_quantiles(test_flat_n)
                        q_lo_flat = quantiles_raw[:, 0]
                        q_up_flat = quantiles_raw[:, 1]
                        q_lo_2d = q_lo_flat.reshape(T_test, H)
                        q_up_2d = q_up_flat.reshape(T_test, H)
                    else:
                        p.fit(cal_flat_n)
                        quantiles_flat = p._scan_quantiles(test_flat_n)
                        q_lo_2d = quantiles_flat.reshape(T_test, H)
                        q_up_2d = quantiles_flat.reshape(T_test, H)
                else:
                    q_lo_2d = np.empty((T_test, H), dtype=np.float32)
                    q_up_2d = np.empty((T_test, H), dtype=np.float32)
                    if is_asymmetric:
                        p.fit(cal_flat_n)
                        for t in range(T_test):
                            q_lo, q_up = p.predict_quantile_pair()
                            q_lo_2d[t, :] = q_lo
                            q_up_2d[t, :] = q_up
                            for h in range(H):
                                p.update_signed(test_scores_n[t, h])
                    else:
                        p.fit(cal_flat_n)
                        for t in range(T_test):
                            q = p.predict_quantile()
                            q_lo_2d[t, :] = q
                            q_up_2d[t, :] = q
                            for h in range(H):
                                p.update(test_scores_n[t, h])

                test_all_q_n = (
                    {k: v[:, n, :] for k, v in test_all_q.items()}
                    if test_all_q is not None
                    else None
                )

                lower, upper = self._bounds_from_quantiles(
                    score_type=self._effective_score_type,
                    q_lowers=q_lo_2d,
                    q_uppers=q_up_2d,
                    point_preds=test_pt[:, n, :],
                    q_low=test_ql[:, n, :],
                    q_high=test_qh[:, n, :],
                    scales=scales,
                    all_quantiles=test_all_q_n,
                    prev_error=test_prev_error[:, n, :]
                    if test_prev_error is not None
                    else None,
                    is_asymmetric=is_asymmetric,
                )

                covered_3d[:, n, :] = (test_gt[:, n, :] >= lower) & (
                    test_gt[:, n, :] <= upper
                )
                width_3d[:, n, :] = upper - lower
                miss_3d[:, n, :] = np.maximum(
                    np.maximum(lower - test_gt[:, n, :], test_gt[:, n, :] - upper), 0.0
                )

            joint_cov = float(np.mean(covered_3d.all(axis=-1)))
            method_time = time.perf_counter() - t0

            for h in range(H):
                coverage_h = float(np.mean(covered_3d[:, :, h]))

                finite_mask = np.isfinite(width_3d[:, :, h])
                finite_widths = width_3d[:, :, h][finite_mask]
                avg_width_h = (
                    float(np.mean(finite_widths))
                    if len(finite_widths) > 0
                    else float('nan')
                )

                winkler_vals = np.where(
                    finite_mask,
                    width_3d[:, :, h] + (2.0 / self.alpha) * miss_3d[:, :, h],
                    np.inf,
                )
                finite_w = winkler_vals[np.isfinite(winkler_vals)]
                winkler_h = (
                    float(np.mean(finite_w)) if len(finite_w) > 0 else float('nan')
                )

                rows.append(
                    {
                        'horizon': h,
                        'method': name,
                        'coverage': coverage_h,
                        'joint_coverage': joint_cov,
                        'avg_width': avg_width_h,
                        'winkler_score': winkler_h,
                        'runtime': method_time,
                    }
                )

        for h in range(H):
            rows.append(
                {
                    'horizon': h,
                    'method': 'Native',
                    'coverage': float(native_all['native_coverage'][h]),
                    'joint_coverage': native_joint_cov,
                    'avg_width': float(native_all['native_avg_width'][h]),
                    'winkler_score': float(native_all['native_winkler_score'][h]),
                    'runtime': float('nan'),
                }
            )

        df = pd.DataFrame(rows)
        cols = [
            'horizon',
            'method',
            'coverage',
            'joint_coverage',
            'avg_width',
            'winkler_score',
            'runtime',
        ]
        return df[cols]
