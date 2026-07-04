"""
Evaluation harness for comparing conformal predictors.

The main entry point is :func:`compare_methods`, which runs a set of
pre-instantiated predictors on the same score / forecast sequence and
returns a tidy ``DataFrame`` with coverage, width, and Winkler score.
"""

from __future__ import annotations

import copy
import math
from typing import Mapping, Optional

import pandas as pd
from numpy.typing import NDArray

from timecp.base import ConformalPredictor, np


def compare_methods(
    methods: Mapping[str, ConformalPredictor],
    scores: NDArray,
    point_forecasts: NDArray,
    init_window: int = 100,
    burnin: Optional[int] = None,
) -> pd.DataFrame:
    """
    Evaluate multiple conformal predictors on the same time series.

    Each predictor in *methods* is **deep-copied** before evaluation, so the
    originals are not mutated and can be reused.

    For symmetric predictors *scores* should be unsigned ``|y_t − ŷ_t|``.
    For asymmetric predictors (``predictor.asymmetric=True``) *scores* should
    be signed errors ``y_t − ŷ_t``.

    Protocol for each predictor::

        predictor.fit(scores[:init_window])
        results = predictor.evaluate(
            scores[init_window:], point_forecasts[init_window:], burnin=0
        )

    Parameters
    ----------
    methods : dict[str, ConformalPredictor]
        Mapping from method name to a **fitted or unfitted** predictor.
        ``fit`` is always called internally with ``scores[:init_window]``.
    scores : NDArray of shape (T,)
        Full sequence of nonconformity scores (unsigned) or signed errors
        (for asymmetric predictors).
    point_forecasts : NDArray of shape (T,)
        Full sequence of point predictions ŷ_t.
    init_window : int
        Number of scores used for initial calibration (``fit``).
        The remaining ``T - init_window`` steps are used for evaluation.
    burnin : int or None
        Additional steps at the start of the evaluation phase to skip when
        computing metrics (the predictor still updates during burnin).
        Defaults to 0.

    Returns
    -------
    pd.DataFrame
        One row per method with columns:
        ``method``, ``coverage``, ``avg_width``, ``winkler_score``,
        ``alpha`` (target), ``init_window``.

    Examples
    --------
    >>> import numpy as np
    >>> from timecp import ACI, AgACI, TrailingWindow
    >>> from timecp.evaluation import compare_methods
    >>> rng = np.random.default_rng(0)
    >>> y = rng.standard_normal(1000)
    >>> scores = np.abs(y)                      # perfect point forecast = 0
    >>> forecasts = np.zeros(1000)
    >>> df = compare_methods(
    ...     {"ACI": ACI(0.1), "TrailingWindow": TrailingWindow(0.1)},
    ...     scores, forecasts, init_window=200,
    ... )
    >>> print(df[["method", "coverage", "avg_width"]])
    """
    scores = np.asarray(scores, dtype=float)
    point_forecasts = np.asarray(point_forecasts, dtype=float)
    T = len(scores)

    if init_window >= T:
        raise ValueError(f'init_window ({init_window}) must be smaller than T ({T})')
    if burnin is None:
        burnin = 0

    cal_scores = scores[:init_window]
    eval_scores = scores[init_window:]
    eval_forecasts = point_forecasts[init_window:]

    rows = []
    for name, predictor in methods.items():
        pred = copy.deepcopy(predictor)
        pred.fit(cal_scores)
        result = pred.evaluate(eval_scores, eval_forecasts, burnin=burnin)
        rows.append(
            {
                'method': name,
                'coverage': result['coverage'],
                'avg_width': result['avg_width'],
                'winkler_score': result['winkler_score'],
                'alpha': pred.alpha,
                'init_window': init_window,
            }
        )

    return pd.DataFrame(rows).set_index('method')


def rolling_metrics(
    predictor: ConformalPredictor,
    scores: NDArray,
    point_forecasts: NDArray,
    window: int = 50,
) -> pd.DataFrame:
    """
    Compute rolling (windowed) coverage and width over the evaluation phase.

    Useful for visualising how a predictor adapts over time.

    For asymmetric predictors (``predictor.asymmetric=True``) *scores* should
    be signed errors ``y_t − ŷ_t``.  Coverage is checked with the asymmetric
    rule ``-q_lo <= e <= q_up`` and width is ``q_lo + q_up``.

    Parameters
    ----------
    predictor : ConformalPredictor
        A **fitted** predictor (``fit`` must have been called beforehand).
    scores : NDArray of shape (T,)
        Evaluation-phase nonconformity scores (unsigned) or signed errors
        (asymmetric predictors).
    point_forecasts : NDArray of shape (T,)
        Evaluation-phase point predictions.
    window : int
        Rolling window length for computing metrics.

    Returns
    -------
    pd.DataFrame
        Columns: ``t``, ``rolling_coverage``, ``rolling_width``, ``quantile``.
    """
    scores = np.asarray(scores, dtype=float)
    point_forecasts = np.asarray(point_forecasts, dtype=float)
    T = len(scores)

    quantiles_out = np.zeros(T)
    covereds = np.zeros(T, dtype=bool)
    widths = np.zeros(T)

    is_asymmetric = getattr(predictor, 'asymmetric', False)

    for t in range(T):
        if is_asymmetric:
            q_lo, q_up = predictor.predict_quantile_pair()
            e = scores[t]
            covered = bool(-q_lo <= e <= q_up)
            width = (
                (q_lo + q_up)
                if math.isfinite(q_lo) and math.isfinite(q_up)
                else float('nan')
            )
            quantiles_out[t] = q_up if math.isfinite(q_up) else float('nan')
            covereds[t] = covered
            widths[t] = width
            predictor.update_signed(float(e))
        else:
            q = predictor.predict_quantile()
            score = scores[t]
            covered = score <= q
            width = 2.0 * q if math.isfinite(q) else float('nan')
            quantiles_out[t] = q if math.isfinite(q) else float('nan')
            covereds[t] = covered
            widths[t] = width
            predictor.update(float(score))

    rolling_cov = (
        pd.Series(covereds.astype(float))
        .rolling(window, min_periods=1)
        .mean()
        .to_numpy()
    )
    rolling_wid = pd.Series(widths).rolling(window, min_periods=1).mean().to_numpy()

    return pd.DataFrame(
        {
            't': np.arange(T),
            'rolling_coverage': rolling_cov,
            'rolling_width': rolling_wid,
            'quantile': quantiles_out,
        }
    )
