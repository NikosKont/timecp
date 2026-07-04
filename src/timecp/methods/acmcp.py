"""AcMCP — Autocorrelated Multistep-ahead Conformal Prediction.

Reference
---------
Wang, X., and Hyndman, R. J. (2024).
"Online conformal inference for multi-step time series forecasting."
arXiv:2410.13115v2.

R reference implementation: methods/cpts/conformalForecast/R/acmcp.R.

Overview
--------
AcMCP is a per-horizon asymmetric conformal predictor that extends the
conformal-PID framework with a structure-aware D (scorecasting) term.
For each horizon h it runs an online update of the form::

    q_upper[t+h] = qt_upper[t+h] + integrator_upper[t+h] + scorecaster_upper[t+h]
    qt_upper[t+h] = qt_upper[t+h-1] + lr_t * (err_upper[t] - alpha/2)

where the scorecaster forecasts the h-step-ahead error using:

* **h = 1**: the mean of recent h=1 errors.
* **h ≥ 2**: the average of two models:
    - MA(h-1) fitted to the h-step error history, h-step-ahead point forecast.
    - OLS of ``e[·, h]`` on ``e[·, 1], …, e[·, h-1]`` using the known
      shorter-horizon errors at each prediction origin as features.

Lower and upper bounds are tracked separately (asymmetric intervals).

Mapping to our evaluation pipeline
-----------------------------------
Our data is organised as *windows*: each window ``w`` holds the errors for
one calibration or test origin, averaged over N series:
``errors[w, h] = mean_n |gt[w, n, h] - pred[w, n, h]|``  (unsigned for upper)
``errors_signed[w, h] = mean_n (gt[w, n, h] - pred[w, n, h])`` (signed for asymmetric)

The R code loops over ``t`` (calibration/test origins) and horizon ``h``
independently.  We replicate that loop structure directly.

The class implements :class:`~timecp.base.ConformalPredictor` with a
per-horizon independent interface, so it plugs into ``CPEvaluator.run()``
alongside SplitCP, ACI, etc.  One ``AcMCP`` instance is created per
horizon ``h``; during ``fit`` it receives the full calibration history
of that horizon's scores and also the errors for all shorter horizons
(needed by the scorecaster).

Parameters
----------
alpha : float
    Target miscoverage rate.  Asymmetric intervals use ``alpha/2`` for
    each tail.
h : int
    Forecast horizon this instance is responsible for (1-indexed, as in the
    paper; horizon 1 = one step ahead).
ncal : int
    Burn-in length.  Scorecasting starts only after ``ncal`` observations.
    Rolling window of this length is also used for the learning-rate
    calculation.  Default 10 (minimum allowed by the R implementation).
rolling : bool
    If True, use a rolling window of length ``ncal`` for the learning rate
    and the OLS/MA fitting window.  If False, use an expanding window.
integrate : bool
    Include the integral (I) correction term.  Default True.
scorecast : bool
    Include the scorecasting (D) term.  Default True.
lr : float
    Base learning rate for quantile tracking.  Scaled by the error range
    within the rolling/expanding window.  Default 0.1.
Csat : float or None
    Saturation constant for the integrator.  If None and ``integrate=True``
    you must supply ``Tg`` and ``delta`` instead.
KI : float or None
    Integrator gain.  If None, it is set to the max absolute calibration
    error at ``fit()`` time (matching the R default).
Tg : int or None
    Target time for the absolute coverage guarantee.  Used to compute
    ``Csat`` when ``Csat`` is None.
delta : float or None
    Coverage slack for the absolute guarantee.  Used with ``Tg``.

Notes on asymmetric scoring
---------------------------
AcMCP uses **signed errors** ``e[t,h] = y[t+h] - ŷ[t+h|t]``:

* Upper tail: ``err_upper = 1`` if  ``e > q_upper``.
* Lower tail: ``err_lower = 1`` if  ``-e > q_lower``  (i.e. the prediction
  overshot the truth by more than ``q_lower``).

At test time the interval for horizon h at origin t is::

    [ŷ[t+h|t] − q_lower[t+h],  ŷ[t+h|t] + q_upper[t+h]]

Because ``predict_interval`` in the base class is symmetric, ``AcMCP``
overrides it to return the asymmetric interval directly.
"""

from __future__ import annotations

import math
from collections import deque

from numba import njit
from numpy.typing import NDArray

from timecp.base import ConformalPredictor, np


# ---------------------------------------------------------------------------
# Saturation function (matches R: KI * tan(x * log(t) / (t * Csat)))
# Note: R uses log(t) not log(t+1); we match exactly.
# ---------------------------------------------------------------------------


@njit
def _saturation_log_acmcp(x: float, t: int, Csat: float, KI: float) -> float:
    """Logarithmically saturated integrator (AcMCP formula, matches R saturation_fn_log).

    Uses ``log(t) / t`` (not ``log(t+1) / (t+1)`` as in the PID variant in
    ``_scan.py``).  This matches the R reference implementation exactly.
    """
    if KI == 0.0 or t <= 1:
        return 0.0
    arg = x * math.log(t) / (Csat * t)
    if arg >= math.pi / 2.0:
        return KI * math.inf
    if arg <= -math.pi / 2.0:
        return -KI * math.inf
    return KI * math.tan(arg)


# ---------------------------------------------------------------------------
# MA(h-1) scorecaster helper
# ---------------------------------------------------------------------------


@njit
def _ma_forecast(errors: NDArray, q: int) -> float:
    """Fit MA(q) to *errors* and return the q-step-ahead point forecast.

    Uses a simple OLS approach: the MA(q) mean is the sample mean of
    the last ``q`` residuals, which equals the unconditional mean under
    stationarity.  For the h-step forecast from an MA(h-1) model, the
    forecast at lag h equals the fitted MA coefficient at lag h convolved
    with past innovations.  We use numpy's least-squares to fit the MA
    via the innovations algorithm approximation (regress on lagged values).

    For simplicity and to avoid the innovations algorithm, we use the
    conditional expectation:  fit OLS of ``e[t]`` on ``e[t-1], …, e[t-q]``
    (treating MA coefficients as AR approximation) and predict q steps ahead
    by iterated substitution.  This matches the spirit of the R code which
    calls ``forecast::Arima(errors_subset, order=c(0,0,h-1))`` and uses
    ``forecast$mean[h]`` — the h-step-ahead forecast from the MA model.

    For an MA(q) model the h-step-ahead forecast for h > q is just the
    unconditional mean (zero for mean-zero series, or the series mean).
    We return the series mean as the h-step forecast when ``h > q``, and
    for ``h <= q`` we return the AR(q)-approximation prediction.
    """
    n = len(errors)
    if n == 0:
        return 0.0
    if q <= 0 or n < q + 1:
        return float(np.mean(errors))

    # h = q+1 (we always call with h = order+1)
    # MA(q) h-step forecast for h > q = unconditional mean
    # MA(q) h-step forecast for h <= q requires the moving-average coefficients.
    # Use OLS on lagged errors as AR(q) approximation then iterate.
    # Build design matrix X: rows = t=q..n-1, cols = e[t-1]..e[t-q]
    X = np.empty((n - q, q), dtype=np.float64)
    for j in range(q):
        X[:, j] = errors[q - j - 1 : n - j - 1]

    y = errors[q:]  # (n-q,)
    if len(y) < 2:
        return float(np.mean(errors))

    X_with_intercept = np.empty((n - q, q + 1), dtype=np.float64)
    X_with_intercept[:, 0] = 1.0
    X_with_intercept[:, 1:] = X

    try:
        coefs, _, _, _ = np.linalg.lstsq(X_with_intercept, y, rcond=-1.0)
        intercept = coefs[0]
        ar_coefs = coefs[1:]  # length q
    except Exception:
        return float(np.mean(errors))

    # 1-step forecast using the last q observations
    last_q = errors[-q:]  # (q,) most recent
    return float(intercept + ar_coefs @ last_q[::-1].copy())


# ---------------------------------------------------------------------------
# AcMCP
# ---------------------------------------------------------------------------


class AcMCP(ConformalPredictor):
    """Autocorrelated Multistep-ahead Conformal Prediction.

    A per-horizon asymmetric conformal predictor.  One instance handles
    one forecast horizon ``h``.  It must be fitted with both the calibration
    scores for horizon ``h`` **and** the signed error matrix for all horizons
    1 through h (needed by the OLS scorecaster).

    Parameters
    ----------
    alpha : float
        Target miscoverage rate.
    h : int
        Forecast horizon (1-indexed).
    ncal : int
        Burn-in / rolling-window length.  Minimum 10.
    rolling : bool
        Use a rolling window for learning rate / scorecaster fitting.
    integrate : bool
        Include the integral correction term.
    scorecast : bool
        Include the scorecasting (D) term.
    lr : float
        Base learning rate.
    Csat : float or None
        Integrator saturation constant.
    KI : float or None
        Integrator gain.  Determined from calibration data if None.
    """

    natively_asymmetric = True

    def __init__(
        self,
        alpha: float,
        h: int = 1,
        ncal: int = 10,
        rolling: bool = True,
        integrate: bool = True,
        scorecast: bool = True,
        lr: float = 0.1,
        Csat: float | None = None,
        KI: float | None = None,
    ) -> None:
        super().__init__(alpha)
        if h < 1:
            raise ValueError(f'h must be >= 1, got {h}')
        if ncal < 10:
            raise ValueError(f'ncal must be >= 10, got {ncal}')
        self.h = int(h)
        self.ncal = ncal
        self.rolling = rolling
        self.integrate = integrate
        self.scorecast = scorecast
        self.lr = lr
        self._Csat_init = Csat
        self._KI_init = KI

        # State set after fit():
        self._Csat: float = Csat if Csat is not None else 1.0
        self._KI: float = KI if KI is not None else 0.0

        # Online state (scalars, updated per test step):
        self._qt_lower: float = 0.0
        self._qt_upper: float = 0.0
        self._q_lower: float = 0.0
        self._q_upper: float = 0.0
        self._cum_err_lower: float = 0.0
        self._cum_err_upper: float = 0.0
        self._t: int = 0  # number of update steps seen

        # Rolling windows:
        self._error_window: deque[float] = deque(maxlen=ncal)  # signed errors e[t,h]
        # Per-horizon rolling windows of signed errors for the OLS scorecaster
        # Index j → errors for horizon j+1 (j = 0..h-2), rolling window length ncal
        self._shorter_error_windows: list[deque[float]] = []

        # Scorecaster state (most recent D-term predictions):
        self._last_scorecast_lower: float = 0.0
        self._last_scorecast_upper: float = 0.0

    def clone(self) -> ConformalPredictor:
        new = super().clone()
        new._error_window = deque(maxlen=self.ncal)
        new._shorter_error_windows = [
            deque(maxlen=self.ncal) for _ in range(self.h - 1)
        ]
        new._qt_lower = 0.0
        new._qt_upper = 0.0
        new._q_lower = 0.0
        new._q_upper = 0.0
        new._cum_err_lower = 0.0
        new._cum_err_upper = 0.0
        new._t = 0
        new._last_scorecast_lower = 0.0
        new._last_scorecast_upper = 0.0
        return new

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(  # type: ignore[override]
        self,
        cal_scores: NDArray,
        signed_errors: NDArray | None = None,
    ) -> 'AcMCP':
        """Calibrate AcMCP.

        Parameters
        ----------
        cal_scores : (C,) float array
            Per-window mean absolute errors for horizon ``self.h``.
            These are the symmetric nonconformity scores (|e[w, h]|).
        signed_errors : (C, H) float array or None
            Full signed error matrix, where ``signed_errors[w, j]`` is the
            mean signed error ``mean_n(gt - pred)`` for window w at horizon j
            (1-indexed: column 0 = horizon 1, …, column h-1 = horizon h).
            Required when ``scorecast=True`` and ``h >= 2``.
            If None, scorecasting is silently disabled.
        """
        cal_scores = np.asarray(cal_scores, dtype=np.float64)

        # Determine KI from calibration data if not provided
        finite_cal = cal_scores[np.isfinite(cal_scores)]
        if self._KI_init is None:
            self._KI = float(np.max(np.abs(finite_cal))) if len(finite_cal) > 0 else 1.0
        else:
            self._KI = self._KI_init
        if self._Csat_init is None:
            self._Csat = 1.0  # fallback; user should provide Tg/delta
        else:
            self._Csat = self._Csat_init

        # Fill rolling window with finite calibration errors only
        self._error_window = deque(maxlen=self.ncal)
        for e in finite_cal:
            self._error_window.append(float(e))

        # Initialise shorter-horizon windows from signed_errors columns 0..h-2
        self._shorter_error_windows = []
        if self.scorecast and self.h >= 2 and signed_errors is not None:
            signed_errors = np.asarray(signed_errors, dtype=np.float64)
            for j in range(self.h - 1):  # j = 0..h-2 → horizons 1..h-1
                dq: deque[float] = deque(maxlen=self.ncal)
                if signed_errors.ndim == 2 and signed_errors.shape[1] > j:
                    col = signed_errors[:, j]
                    for e in col:
                        if math.isfinite(e):
                            dq.append(float(e))
                self._shorter_error_windows.append(dq)
        else:
            self._shorter_error_windows = [
                deque(maxlen=self.ncal) for _ in range(self.h - 1)
            ]

        # Compute initial scorecaster prediction from calibration data
        self._last_scorecast_lower = 0.0
        self._last_scorecast_upper = 0.0
        if self.scorecast and len(finite_cal) >= self.ncal:
            sc = self._compute_scorecast(finite_cal)
            self._last_scorecast_lower = -sc
            self._last_scorecast_upper = sc

        # Initialise quantile state using the calibration quantile at level (1 - alpha/2)
        target_prob = 1.0 - self.alpha / 2.0
        if (
            signed_errors is not None
            and signed_errors.ndim == 2
            and signed_errors.shape[1] >= self.h
        ):
            h_signed = signed_errors[:, self.h - 1]
            h_signed = h_signed[np.isfinite(h_signed)]
            q0_up = (
                float(np.quantile(h_signed, target_prob, method='higher'))
                if len(h_signed) > 0
                else 0.0
            )
            q0_lo = (
                float(np.quantile(-h_signed, target_prob, method='higher'))
                if len(h_signed) > 0
                else 0.0
            )
        else:
            q0_up = (
                float(np.quantile(finite_cal, target_prob, method='higher'))
                if len(finite_cal) > 0
                else 0.0
            )
            q0_lo = q0_up

        self._qt_lower = q0_lo
        self._qt_upper = q0_up
        self._q_lower = q0_lo
        self._q_upper = q0_up
        self._cum_err_lower = 0.0
        self._cum_err_upper = 0.0
        self._t = 0
        self._last_q = None
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Scorecaster
    # ------------------------------------------------------------------

    def _compute_scorecast(self, errors_h: NDArray) -> float:
        """Compute the D-term: predicted magnitude of the h-step error.

        For h=1: mean of recent errors.
        For h>=2: average of MA(h-1) and OLS predictions.
        """
        if len(errors_h) == 0:
            return 0.0

        if self.h == 1:
            return float(np.mean(errors_h))

        # MA(h-1) fitted to the h-step error series, h-step-ahead forecast
        ma_pred = _ma_forecast(errors_h, q=self.h - 1)

        # OLS: regress e[·,h] on e[·,1], …, e[·,h-1]
        # Use aligned shorter-horizon errors from stored windows
        lr_pred = self._ols_scorecast(errors_h)

        return (ma_pred + lr_pred) / 2.0

    def _ols_scorecast(self, errors_h: NDArray) -> float:
        """OLS regression of h-step errors on shorter-horizon errors."""
        if len(self._shorter_error_windows) == 0:
            return float(np.mean(errors_h)) if len(errors_h) > 0 else 0.0

        n = len(errors_h)
        if n < 3:
            return float(np.mean(errors_h)) if n > 0 else 0.0

        # Build aligned feature matrix.  For each observation i, features are
        # the shorter-horizon errors from the same rolling window at the same
        # origin.  We align by taking the last min(n, ncal) of each window.
        k = min(n, self.ncal)
        y_arr = np.array(errors_h[-k:], dtype=np.float64)

        feature_cols = []
        for dq in self._shorter_error_windows:
            col = list(dq)[-k:] if len(dq) >= k else list(dq)
            if len(col) < k:
                col = [0.0] * (k - len(col)) + col
            feature_cols.append(col)

        X = np.column_stack(feature_cols)  # (k, h-1)
        X = np.column_stack([np.ones(k), X])  # add intercept

        if X.shape[0] <= X.shape[1]:
            return float(np.mean(y_arr))
        try:
            coefs, _, _, _ = np.linalg.lstsq(X, y_arr, rcond=None)
        except np.linalg.LinAlgError:
            return float(np.mean(y_arr))

        # Predict using most recent shorter-horizon errors
        x_new = np.array(
            [1.0]
            + [
                float(dq[-1]) if len(dq) > 0 else 0.0
                for dq in self._shorter_error_windows
            ]
        )
        return float(coefs @ x_new)

    # ------------------------------------------------------------------
    # ConformalPredictor interface
    # ------------------------------------------------------------------

    def predict_quantile(self) -> float:
        """Return the upper-tail quantile (used for symmetric evaluate)."""
        q = max(self._q_upper, 0.0)
        self._last_q = q
        return q

    def predict_interval(self, point_forecast: float) -> tuple[float, float]:
        """Return the asymmetric prediction interval.

        Parameters
        ----------
        point_forecast : float
            Model point prediction ŷ.

        Returns
        -------
        (lower, upper) where:
            lower = ŷ − q_lower
            upper = ŷ + q_upper
        """
        if not self._fitted:
            raise RuntimeError('Call fit() before predict_interval()')
        q_lo = max(self._q_lower, 0.0)
        q_up = max(self._q_upper, 0.0)
        self._last_q = q_up  # store upper for evaluate compatibility
        return float(point_forecast - q_lo), float(point_forecast + q_up)

    def update(self, new_score: float) -> None:
        """Online update using a new signed error ``e = y - ŷ``.

        If the caller passes an unsigned absolute error (as ``evaluate`` does
        via the base class), we treat it as the upper error and set the lower
        error to the same magnitude (symmetric fallback).

        For fully asymmetric use, call ``update_asymmetric(e_signed)`` directly.
        """
        # Treat as signed error (positive = actual > predicted)
        self.update_asymmetric(new_score, -new_score)

    def update_asymmetric(self, e_upper: float, e_lower: float | None = None) -> None:
        """Update with signed upper and lower errors.

        Parameters
        ----------
        e_upper : float
            Signed error for the upper tail: ``y - ŷ``.
            Coverage error occurs when ``e_upper > q_upper``.
        e_lower : float or None
            Signed error for the lower tail: ``-(y - ŷ) = ŷ - y``.
            Coverage error occurs when ``e_lower > q_lower``.
            If None, set to ``-e_upper``.
        """
        if e_lower is None:
            e_lower = -e_upper

        q_up = max(self._q_upper, 0.0)
        q_lo = max(self._q_lower, 0.0)
        t = self._t

        err_upper = 1.0 if e_upper > q_up else 0.0
        err_lower = 1.0 if e_lower > q_lo else 0.0

        # Learning rate: proportional to range of recent errors
        if len(self._error_window) <= 1:
            lr_t = self.lr
        else:
            lr_t = self.lr * (max(self._error_window) - min(self._error_window))

        # P: quantile tracking update
        self._qt_upper += lr_t * (err_upper - self.alpha / 2.0)
        self._qt_lower += lr_t * (err_lower - self.alpha / 2.0)

        # I: integrator
        integ_upper = 0.0
        integ_lower = 0.0
        if self.integrate and t > 1:
            arg_upper = self._cum_err_upper - t * self.alpha / 2.0
            arg_lower = self._cum_err_lower - t * self.alpha / 2.0
            iu = _saturation_log_acmcp(arg_upper, t, self._Csat, self._KI)
            il = _saturation_log_acmcp(arg_lower, t, self._Csat, self._KI)
            integ_upper = iu if math.isfinite(iu) else 0.0
            integ_lower = il if math.isfinite(il) else 0.0

        # D: scorecaster (update windows, recompute for next step)
        if self.scorecast and t >= self.ncal:
            err_h_arr = np.array(self._error_window, dtype=np.float64)
            sc = self._compute_scorecast(err_h_arr)
            sc = sc if math.isfinite(sc) else 0.0
            self._last_scorecast_upper = sc
            self._last_scorecast_lower = -sc
        else:
            self._last_scorecast_upper = 0.0
            self._last_scorecast_lower = 0.0

        # Update q = qt + integrator + scorecaster
        self._q_upper = self._qt_upper + integ_upper + self._last_scorecast_upper
        self._q_lower = self._qt_lower + integ_lower + self._last_scorecast_lower

        # Accumulate errors and update windows
        self._cum_err_upper += err_upper
        self._cum_err_lower += err_lower
        self._t += 1
        self._error_window.append(abs(e_upper))  # store magnitude for learning rate

    # ------------------------------------------------------------------
    # evaluate override for asymmetric coverage
    # ------------------------------------------------------------------

    def evaluate(  # type: ignore[override]
        self,
        scores: NDArray,
        point_forecasts: NDArray,
        burnin: int = 0,
    ) -> dict:
        """Rolling evaluation using asymmetric intervals.

        ``scores`` is interpreted as the **signed** error ``e = y - ŷ``.
        Coverage is checked asymmetrically:
            covered = (lo <= y <= hi) = (-q_lo <= e <= q_up)

        ``point_forecasts`` is used only to compute the Winkler score;
        the coverage check uses only ``scores`` (the signed errors).
        """
        if not self._fitted:
            raise RuntimeError('Call fit() before evaluate()')

        scores = np.asarray(scores, dtype=np.float64)
        T = len(scores)

        covereds: list[bool] = []
        widths: list[float] = []
        winkler: list[float] = []
        quantiles_out: list[float] = []
        q_lowers_out: list[float] = []
        q_uppers_out: list[float] = []

        for t_idx in range(T):
            e = float(scores[t_idx])  # signed error y - ŷ
            q_lo = max(self._q_lower, 0.0)
            q_up = max(self._q_upper, 0.0)
            width = q_lo + q_up

            covered = -q_lo <= e <= q_up
            quantiles_out.append(q_up)
            q_lowers_out.append(q_lo)
            q_uppers_out.append(q_up)

            if t_idx >= burnin:
                covereds.append(covered)
                widths.append(width if math.isfinite(width) else math.inf)
                if math.isfinite(width):
                    if covered:
                        w = width
                    else:
                        miss = max(-q_lo - e, e - q_up, 0.0)
                        w = width + (2.0 / self.alpha) * miss
                else:
                    w = math.inf
                if math.isfinite(w):
                    winkler.append(w)

            if math.isfinite(e):
                self.update_asymmetric(e, -e)

        finite_winkler = [w for w in winkler if math.isfinite(w)]
        finite_widths = [w for w in widths if math.isfinite(w)]
        covered_arr = np.array(covereds, dtype=bool)

        return {
            'coverage': float(np.mean(covered_arr))
            if len(covered_arr) > 0
            else float('nan'),
            'avg_width': float(np.mean(finite_widths))
            if finite_widths
            else float('nan'),
            'winkler_score': float(np.mean(finite_winkler))
            if finite_winkler
            else float('nan'),
            'quantiles': np.array(quantiles_out),
            'q_lowers': np.array(q_lowers_out),
            'q_uppers': np.array(q_uppers_out),
            'covered': covered_arr,
        }

    def batch_evaluate(
        self,
        scores: NDArray,
        point_forecasts: NDArray,
        burnin: int = 0,
    ) -> dict:
        """Vectorized batch evaluation for AcMCP.

        Performs a scan-style forward pass computing all quantiles, then
        vectorizes the metrics.  AcMCP uses asymmetric intervals so metrics
        are computed with the asymmetric coverage rule.
        """
        if not self._fitted:
            raise RuntimeError('Call fit() before batch_evaluate()')

        scores = np.asarray(scores, dtype=np.float64)
        T = len(scores)

        # Scan: compute upper/lower quantiles
        q_uppers = np.empty(T, dtype=float)
        q_lowers = np.empty(T, dtype=float)

        for t_idx in range(T):
            q_lowers[t_idx] = max(self._q_lower, 0.0)
            q_uppers[t_idx] = max(self._q_upper, 0.0)
            e = float(scores[t_idx])
            if math.isfinite(e):
                self.update_asymmetric(e, -e)

        # Vectorized metrics
        if burnin >= T:
            return {
                'coverage': float('nan'),
                'avg_width': float('nan'),
                'winkler_score': float('nan'),
                'quantiles': q_uppers,
                'q_lowers': q_lowers,
                'q_uppers': q_uppers,
                'covered': np.array([], dtype=bool),
            }

        s = scores[burnin:]
        q_lo = q_lowers[burnin:]
        q_up = q_uppers[burnin:]
        widths = q_lo + q_up
        covered = (-q_lo <= s) & (s <= q_up)

        # Exclude NaN/inf test scores from coverage
        finite_score = np.isfinite(s)
        valid = finite_score & np.isfinite(widths)
        coverage = float(np.mean(covered[valid])) if valid.any() else float('nan')

        finite_widths = widths[valid]
        avg_width = (
            float(np.mean(finite_widths)) if len(finite_widths) > 0 else float('nan')
        )

        miss = np.maximum(np.maximum(-q_lo - s, s - q_up), 0.0)
        winkler_all = np.where(valid, widths + (2.0 / self.alpha) * miss, np.inf)
        finite_winkler = winkler_all[np.isfinite(winkler_all)]
        winkler_score = (
            float(np.mean(finite_winkler)) if len(finite_winkler) > 0 else float('nan')
        )

        return {
            'coverage': coverage,
            'avg_width': avg_width,
            'winkler_score': winkler_score,
            'quantiles': q_uppers,
            'q_lowers': q_lowers,
            'q_uppers': q_uppers,
            'covered': covered,
        }
