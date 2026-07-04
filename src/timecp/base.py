"""Abstract base class for all score-based conformal predictors."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Vectorized metrics helpers (used by batch_evaluate)
# ---------------------------------------------------------------------------


def _vectorized_metrics(
    scores: NDArray, quantiles: NDArray, alpha: float, burnin: int = 0
) -> dict:
    """Compute coverage, width, and Winkler score from pre-computed quantiles.

    Parameters
    ----------
    scores : (T,) array of nonconformity scores.
    quantiles : (T,) array of predicted quantiles (may contain inf).
    alpha : float target miscoverage rate.
    burnin : int steps to skip at the beginning.

    Returns
    -------
    dict with coverage, avg_width, winkler_score, quantiles.
    """
    T = len(scores)
    if burnin >= T:
        return {
            'coverage': float('nan'),
            'avg_width': float('nan'),
            'winkler_score': float('nan'),
            'quantiles': quantiles,
        }

    s = scores[burnin:]
    q = quantiles[burnin:]

    covered = s <= q
    finite_mask = np.isfinite(q)
    widths = np.where(finite_mask, 2.0 * q, np.inf)

    # Coverage: exclude only NaN/inf test scores (missing ground truth).
    # Infinite quantiles DO count as covered (the interval is infinitely wide).
    finite_score = np.isfinite(s)
    coverage = (
        float(np.mean(covered[finite_score])) if finite_score.any() else float('nan')
    )

    # Width: average over finite-quantile steps only (avoid inf poisoning the mean)
    finite_widths = widths[finite_mask & finite_score]
    avg_width = (
        float(np.mean(finite_widths)) if len(finite_widths) > 0 else float('nan')
    )

    # Winkler score: width + (2/alpha) * penalty for misses
    penalty = np.where(covered, 0.0, (2.0 / alpha) * (s - q))
    winkler_all = np.where(finite_mask & finite_score, widths + penalty, np.inf)
    finite_winkler = winkler_all[np.isfinite(winkler_all)]
    winkler_score = (
        float(np.mean(finite_winkler)) if len(finite_winkler) > 0 else float('nan')
    )

    return {
        'coverage': coverage,
        'avg_width': avg_width,
        'winkler_score': winkler_score,
        'quantiles': quantiles,
        'q_lowers': quantiles,
        'q_uppers': quantiles,
        'covered': covered,
    }


def _vectorized_metrics_asymmetric(
    signed_errors: NDArray,
    q_lowers: NDArray,
    q_uppers: NDArray,
    alpha: float,
    burnin: int = 0,
) -> dict:
    """Compute coverage, width, and Winkler score for asymmetric intervals.

    Parameters
    ----------
    signed_errors : (T,) array of signed errors e = y - ŷ.
    q_lowers : (T,) array of lower half-widths q_lo (so the lower bound is ŷ − q_lo).
    q_uppers : (T,) array of upper half-widths q_up (so the upper bound is ŷ + q_up).
    alpha : float target miscoverage rate.
    burnin : int steps to skip at the beginning.

    Returns
    -------
    dict with coverage, avg_width, winkler_score, quantiles (= q_uppers).
    """
    T = len(signed_errors)
    if burnin >= T:
        return {
            'coverage': float('nan'),
            'avg_width': float('nan'),
            'winkler_score': float('nan'),
            'quantiles': q_uppers,
        }

    s = signed_errors[burnin:]
    q_lo = q_lowers[burnin:]
    q_up = q_uppers[burnin:]

    if s.ndim == 2 and s.shape[1] == 2:
        covered = (s[:, 0] <= q_lo) & (s[:, 1] <= q_up)
        widths = q_lo + q_up
        finite_score = np.isfinite(s[:, 0]) & np.isfinite(s[:, 1])
        finite_width = np.isfinite(widths)
        miss = np.maximum(s[:, 0] - q_lo, 0.0) + np.maximum(s[:, 1] - q_up, 0.0)
    else:
        covered = (-q_lo <= s) & (s <= q_up)
        widths = q_lo + q_up
        finite_score = np.isfinite(s)
        finite_width = np.isfinite(widths)
        miss = np.maximum(np.maximum(-q_lo - s, s - q_up), 0.0)

    # Coverage: exclude only NaN/inf test scores. Infinite quantiles count
    # as covered (the interval is infinitely wide).
    coverage = (
        float(np.mean(covered[finite_score])) if finite_score.any() else float('nan')
    )

    # Width: average over finite-quantile steps only
    valid_width = finite_score & finite_width
    finite_widths = widths[valid_width]
    avg_width = (
        float(np.mean(finite_widths)) if len(finite_widths) > 0 else float('nan')
    )

    winkler_all = np.where(valid_width, widths + (2.0 / alpha) * miss, np.inf)
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


class _WindowMixin:
    """Mixin providing simple rolling-window ``update`` / ``update_signed`` for
    :class:`ConformalPredictor` subclasses whose entire online state is a
    single ``_scores`` deque.

    Classes that need custom update logic (e.g. ACI, QuantileIntegrator)
    should **not** inherit this mixin; they implement their own ``update`` and
    ``update_signed``.

    Assumes the concrete class also inherits :class:`ConformalPredictor` (or
    sets ``_scores`` itself).  The MRO ensures these methods take priority over
    the no-op defaults in ``ConformalPredictor``.
    """

    _scores: deque  # declared for type checkers; set by _init_window

    def update(self, new_score: float) -> None:  # type: ignore[override]
        """Append *new_score* to the rolling window."""
        self._scores.append(new_score)

    def update_signed(self, e: float) -> None:  # type: ignore[override]
        """Append signed error *e = y − ŷ* to the window (asymmetric mode)."""
        self._scores.append(e)


class ConformalPredictor(ABC):
    """
    Unified score-based interface for conformal prediction on time series.

    **Symmetric mode** (default, ``asymmetric=False``)::

        predictor.fit(initial_scores)        # calibrate on |y - ŷ| scores
        for each new time step t:
            q   = predictor.predict_quantile()        # get threshold
            lo, hi = predictor.predict_interval(y_hat) # symmetric ŷ ± q
            predictor.update(score)                    # online update with |e|

    **Asymmetric mode** (``asymmetric=True``)::

        predictor.fit(signed_errors)         # calibrate on signed e = y - ŷ
        for each new time step t:
            q_lo, q_up = predictor.predict_quantile_pair()
            lo, hi = predictor.predict_interval(y_hat)  # [ŷ − q_lo, ŷ + q_up]
            predictor.update_signed(signed_error)        # online update with signed e

        Asymmetric mode uses ``alpha/2`` per tail, so the total miscoverage rate
        is still ``alpha``.  Scores passed to ``fit``, ``evaluate``, and
        ``batch_evaluate`` must be **signed** errors ``e = y − ŷ``.

    Parameters
    ----------
    alpha : float
        Target miscoverage rate in (0, 1).  Intervals aim for
        (1 - alpha) marginal coverage.
    asymmetric : bool
        If True, track separate lower and upper quantiles using signed errors.
        Default False (symmetric intervals).
    conformal_correction : bool
        If True, use the finite-sample conformal correction index level.
        Default True.
    bonferroni_horizon : int or None
        When set to H, divides ``alpha`` by H before computing quantiles so
        that a union bound (Bonferroni correction) gives joint coverage over
        all H horizon steps at the desired rate.  Equivalent to the CFRNN
        correction.  Default None (no correction).
    """

    natively_asymmetric: bool = False

    def __new__(cls, *args, **kwargs) -> ConformalPredictor:
        asymmetric = kwargs.get('asymmetric', False)
        if not asymmetric:
            import inspect

            try:
                sig = inspect.signature(cls.__init__)
                bound = sig.bind_partial(*args, **kwargs)
                asymmetric = bound.arguments.get('asymmetric', False)
            except Exception:
                pass

        if asymmetric and not getattr(cls, 'natively_asymmetric', False):
            instance = object.__new__(AsymmetricPredictor)
            instance.__init__(cls, *args, **kwargs)
            return instance

        return super().__new__(cls)

    def __init__(
        self,
        alpha: float,
        asymmetric: bool = False,
        conformal_correction: bool = True,
        bonferroni_horizon: int | None = None,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f'alpha must be in (0, 1), got {alpha}')
        if bonferroni_horizon is not None and bonferroni_horizon < 1:
            raise ValueError(
                f'bonferroni_horizon must be >= 1, got {bonferroni_horizon}'
            )
        self.alpha = float(alpha)
        self.asymmetric = bool(asymmetric)
        self.conformal_correction = bool(conformal_correction)
        self.bonferroni_horizon: int | None = bonferroni_horizon
        self._last_q: float | None = None
        self._last_q_pair: tuple[float, float] | None = None
        self._fitted: bool = False

    def clone(self) -> ConformalPredictor:
        """Return a fresh copy of this predictor with reset online state.

        Subclasses with additional mutable state MUST override and call super().
        """
        import copy

        new = copy.copy(self)
        if hasattr(self, '_scores'):
            if hasattr(self, 'window_size') and self.window_size is not None:
                new._scores = deque(maxlen=self.window_size)
            elif isinstance(self._scores, np.ndarray):
                new._scores = np.empty(0, dtype=float)
            else:
                new._scores = deque()
        new._fitted = False
        new._last_q = None
        new._last_q_pair = None
        return new

    @property
    def _effective_alpha(self) -> float:
        """Effective miscoverage rate after Bonferroni correction.

        Returns ``alpha / bonferroni_horizon`` when *bonferroni_horizon* is set,
        otherwise returns ``alpha`` unchanged.
        """
        if self.bonferroni_horizon is not None:
            return self.alpha / self.bonferroni_horizon
        return self.alpha

    # ------------------------------------------------------------------
    # Shared fit helper
    # ------------------------------------------------------------------

    def _init_window(self, cal_scores: NDArray, window_size: int | None) -> deque:
        """Build a calibration-score deque and reset prediction state.

        Converts *cal_scores* to a float64 deque bounded by *window_size*,
        resets ``_last_q``, ``_last_q_pair``, and sets ``_fitted = True``.

        Intended to be called from concrete ``fit()`` implementations to
        eliminate repeated boilerplate::

            def fit(self, cal_scores):
                self._scores = self._init_window(cal_scores, self.window_size)
                # ... any method-specific reset ...
                return self

        Parameters
        ----------
        cal_scores : array-like of shape (n,)
            Calibration nonconformity scores.
        window_size : int or None
            Maximum deque length.  ``None`` = unbounded.

        Returns
        -------
        deque[float]
        """
        arr = np.asarray(cal_scores, dtype=float)
        self._last_q = None
        self._last_q_pair = None
        self._fitted = True
        return deque(arr.tolist(), maxlen=window_size)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(self, cal_scores: NDArray) -> 'ConformalPredictor':
        """
        Calibrate on an array of historical nonconformity scores.

        Parameters
        ----------
        cal_scores : NDArray of shape (n,)
            Observed nonconformity scores.  For symmetric predictors these are
            unsigned scores ``|y_i - ŷ_i|``; for asymmetric predictors
            (``asymmetric=True``) these are **signed** errors ``y_i - ŷ_i``.

        Returns
        -------
        self
        """

    @abstractmethod
    def predict_quantile(self) -> float:
        """
        Return the current nonconformity threshold q̂.

        In asymmetric mode this returns the **upper** quantile q_up as a
        representative single value.  Use :meth:`predict_quantile_pair` to
        obtain both (q_lo, q_up).

        The result is cached internally so that ``update`` can reference
        it without requiring the caller to pass it again.

        Returns
        -------
        float
            q̂ >= 0.  May be ``math.inf`` when there are too few calibration
            samples to form a valid quantile.
        """

    # ------------------------------------------------------------------
    # Concrete helpers (override if needed)
    # ------------------------------------------------------------------

    def predict_quantile_pair(self) -> tuple[float, float]:
        """Return ``(q_lower, q_upper)`` for asymmetric interval construction.

        The interval at point forecast ŷ is ``[ŷ − q_lower, ŷ + q_upper]``.

        The default implementation returns ``(q, q)`` where
        ``q = predict_quantile()``, giving symmetric intervals.  Asymmetric
        subclasses override this to track separate tail quantiles.

        Returns
        -------
        tuple[float, float]
            ``(q_lower, q_upper)`` both >= 0.
        """
        q = self.predict_quantile()
        self._last_q_pair = (q, q)
        return (q, q)

    def predict_interval(self, point_forecast: Any) -> tuple[float, float]:
        """
        Return the prediction interval.

        In symmetric mode (``asymmetric=False``) returns ``(ŷ − q̂, ŷ + q̂)``.
        In asymmetric mode (``asymmetric=True``) returns
        ``(ŷ − q_lower, ŷ + q_upper)`` using :meth:`predict_quantile_pair`.

        Parameters
        ----------
        point_forecast : float
            Model point prediction ŷ.

        Returns
        -------
        tuple[float, float]
            (lower bound, upper bound)

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        if not self._fitted:
            raise RuntimeError('Call fit() before predict_interval()')
        if self.asymmetric:
            q_lo, q_up = self.predict_quantile_pair()
            return float(point_forecast) - q_lo, float(point_forecast) + q_up
        q = self.predict_quantile()
        return float(point_forecast - q), float(point_forecast + q)

    def update(self, new_score: float) -> None:
        """
        Online update with the observed nonconformity score (unsigned).

        Must be called **after** ``predict_quantile()`` / ``predict_interval()``
        for each time step, so the cached ``_last_q`` is available to methods
        that need it (e.g. ACI).

        The default implementation is a no-op; subclasses override it.

        Parameters
        ----------
        new_score : float
            Nonconformity score for the current time step, e.g. |y_t − ŷ_t|.
        """

    def update_signed(self, e: float) -> None:
        """Online update with a **signed** error ``e = y_t − ŷ_t``.

        Used in asymmetric mode.  The default implementation falls back to
        ``update(|e|)`` so that symmetric subclasses remain compatible.
        Asymmetric subclasses override this to update separate tail trackers.

        Parameters
        ----------
        e : float
            Signed residual ``y_t − ŷ_t`` for the current time step.
        """
        self.update(abs(e))

    def evaluate(
        self,
        scores: NDArray,
        point_forecasts: NDArray,
        burnin: int = 0,
    ) -> dict:
        """
        Rolling evaluation over a score / forecast sequence.

        In symmetric mode *scores* are unsigned nonconformity scores
        ``|y_t − ŷ_t|``.  In asymmetric mode (``asymmetric=True``) *scores*
        are **signed** errors ``y_t − ŷ_t`` and separate lower/upper coverage
        is checked.

        Iterates over time steps, calling the appropriate predict / update
        methods at each step.  Metrics are accumulated starting from *burnin*.

        .. important::
            This method mutates the predictor's internal state.  For clean
            benchmarking pass a freshly fitted predictor, or use
            :func:`~timecp.evaluation.compare_methods` which handles fresh
            instantiation for you.

        Parameters
        ----------
        scores : NDArray of shape (T,)
            Nonconformity scores (unsigned) or signed errors (asymmetric).
        point_forecasts : NDArray of shape (T,)
            Point predictions for each time step.
        burnin : int, optional
            Number of initial steps to skip when computing metrics.  Default 0.

        Returns
        -------
        dict with keys:
            coverage      — empirical marginal coverage
            avg_width     — mean interval width
            winkler_score — mean Winkler score (width + penalty for misses)
            quantiles     — NDArray of q̂ values (or q_upper in asymmetric mode)
        """
        if not self._fitted:
            raise RuntimeError('Call fit() before evaluate()')
        scores = np.asarray(scores, dtype=float)
        point_forecasts = np.asarray(point_forecasts, dtype=float)
        T = len(scores)

        covereds: list[bool] = []
        widths: list[float] = []
        winkler: list[float] = []
        quantiles_out: list[float] = []

        if self.asymmetric:
            for t in range(T):
                q_lo, q_up = self.predict_quantile_pair()

                if scores.ndim == 2 and scores.shape[1] == 2:
                    e_lo = float(scores[t, 0])
                    e_up = float(scores[t, 1])
                    covered = bool(e_lo <= q_lo and e_up <= q_up)
                    width = (
                        (q_lo + q_up)
                        if math.isfinite(q_lo) and math.isfinite(q_up)
                        else math.inf
                    )

                    quantiles_out.append(q_up)
                    if not hasattr(self, '_tmp_q_lowers'):
                        self._tmp_q_lowers = []
                    if not hasattr(self, '_tmp_q_uppers'):
                        self._tmp_q_uppers = []
                    self._tmp_q_lowers.append(q_lo)
                    self._tmp_q_uppers.append(q_up)

                    if t >= burnin:
                        covereds.append(covered)
                        widths.append(width)
                        if math.isfinite(width):
                            miss = max(e_lo - q_lo, 0.0) + max(e_up - q_up, 0.0)
                            w = width + (2.0 / self.alpha) * miss
                        else:
                            w = math.inf
                        winkler.append(w)

                    if math.isfinite(e_lo) and math.isfinite(e_up):
                        self.update_signed(scores[t])
                else:
                    e = float(scores[t])
                    covered = bool(-q_lo <= e <= q_up)
                    width = (
                        (q_lo + q_up)
                        if math.isfinite(q_lo) and math.isfinite(q_up)
                        else math.inf
                    )

                    quantiles_out.append(q_up)
                    if not hasattr(self, '_tmp_q_lowers'):
                        self._tmp_q_lowers = []
                    if not hasattr(self, '_tmp_q_uppers'):
                        self._tmp_q_uppers = []
                    self._tmp_q_lowers.append(q_lo)
                    self._tmp_q_uppers.append(q_up)

                    if t >= burnin:
                        covereds.append(covered)
                        widths.append(width)
                        if math.isfinite(width):
                            miss = max(-q_lo - e, e - q_up, 0.0)
                            w = width + (2.0 / self.alpha) * miss
                        else:
                            w = math.inf
                        winkler.append(w)

                    if math.isfinite(e):
                        self.update_signed(e)
        else:
            for t in range(T):
                q = self.predict_quantile()
                score = float(scores[t])
                covered = score <= q
                width = 2.0 * q if math.isfinite(q) else math.inf

                quantiles_out.append(q)
                if not hasattr(self, '_tmp_q_lowers'):
                    self._tmp_q_lowers = []
                if not hasattr(self, '_tmp_q_uppers'):
                    self._tmp_q_uppers = []
                self._tmp_q_lowers.append(q)
                self._tmp_q_uppers.append(q)

                if t >= burnin:
                    covereds.append(covered)
                    widths.append(width)
                    if math.isfinite(width):
                        w = (
                            width
                            if covered
                            else width + (2.0 / self.alpha) * (score - q)
                        )
                    else:
                        w = math.inf
                    winkler.append(w)

                if math.isfinite(score):
                    self.update(score)

        finite_widths = [w for w in widths if math.isfinite(w)]
        finite_winkler = [w for w in winkler if math.isfinite(w)]
        res = {
            'coverage': float(np.mean(covereds)) if covereds else float('nan'),
            'avg_width': float(np.mean(finite_widths))
            if finite_widths
            else float('nan'),
            'winkler_score': float(np.mean(finite_winkler))
            if finite_winkler
            else float('nan'),
            'quantiles': np.array(quantiles_out),
            'q_lowers': np.array(getattr(self, '_tmp_q_lowers', quantiles_out)),
            'q_uppers': np.array(getattr(self, '_tmp_q_uppers', quantiles_out)),
            'covered': np.array(covereds, dtype=bool),
        }
        if hasattr(self, '_tmp_q_lowers'):
            del self._tmp_q_lowers
        if hasattr(self, '_tmp_q_uppers'):
            del self._tmp_q_uppers
        return res

    def batch_evaluate(
        self,
        scores: NDArray,
        point_forecasts: NDArray,
        burnin: int = 0,
    ) -> dict:
        """
        Vectorized post-hoc evaluation over a score / forecast sequence.

        Computes all quantiles in a single forward scan and then vectorizes
        the metrics computation.  Produces identical results to :meth:`evaluate`
        but is significantly faster for large T.

        In symmetric mode *scores* are unsigned ``|y_t − ŷ_t|``.
        In asymmetric mode *scores* are signed ``y_t − ŷ_t``.

        .. important::
            This method mutates the predictor's internal state.

        Parameters
        ----------
        scores : NDArray of shape (T,)
            Nonconformity scores (unsigned) or signed errors (asymmetric).
        point_forecasts : NDArray of shape (T,)
            Point predictions for each time step.
        burnin : int, optional
            Number of initial steps to skip when computing metrics.  Default 0.

        Returns
        -------
        dict with keys:
            coverage      — empirical marginal coverage
            avg_width     — mean interval width
            winkler_score — mean Winkler score
            quantiles     — NDArray of q̂ values (or q_upper in asymmetric mode)
        """
        if not self._fitted:
            raise RuntimeError('Call fit() before batch_evaluate()')
        scores = np.asarray(scores, dtype=float)
        point_forecasts = np.asarray(point_forecasts, dtype=float)

        # Scan: compute all quantiles sequentially (subclasses override this)
        quantiles = self._scan_quantiles(scores)

        if self.asymmetric:
            # _scan_quantiles returns (T, 2): col 0 = q_lower, col 1 = q_upper
            return _vectorized_metrics_asymmetric(
                scores, quantiles[:, 0], quantiles[:, 1], self.alpha, burnin
            )
        return _vectorized_metrics(scores, quantiles, self.alpha, burnin)

    def batch_evaluate_series(
        self,
        cal_scores: NDArray,
        test_scores: NDArray,
    ) -> NDArray | tuple[NDArray, NDArray]:
        """Vectorized evaluation across N series.

        Parameters
        ----------
        cal_scores : (C, N) array
         test_scores : (T, N) array

        Returns
        -------
        quantiles : (T, N) array or tuple of ((T, N) array, (T, N) array)
        """
        C, N = cal_scores.shape
        T = test_scores.shape[0]
        if self.asymmetric:
            q_lowers = np.empty((T, N), dtype=float)
            q_uppers = np.empty((T, N), dtype=float)
            for n in range(N):
                p = self.clone()
                p.fit(cal_scores[:, n])
                pairs = p._scan_quantiles(test_scores[:, n])
                q_lowers[:, n] = pairs[:, 0]
                q_uppers[:, n] = pairs[:, 1]
            return q_lowers, q_uppers
        else:
            quantiles = np.empty((T, N), dtype=float)
            for n in range(N):
                p = self.clone()
                p.fit(cal_scores[:, n])
                quantiles[:, n] = p._scan_quantiles(test_scores[:, n])
            return quantiles

    def _scan_quantiles(self, scores: NDArray) -> NDArray:
        """Compute the quantile sequence via a forward scan.

        In symmetric mode returns a ``(T,)`` array of quantiles.
        In asymmetric mode returns a ``(T, 2)`` array where column 0 is
        ``q_lower`` and column 1 is ``q_upper``.

        Default implementation calls :meth:`predict_quantile` /
        :meth:`predict_quantile_pair` and :meth:`update` / :meth:`update_signed`
        in a loop.  Subclasses override this with optimized scan logic.

        Parameters
        ----------
        scores : NDArray of shape (T,)
            Full test score sequence.

        Returns
        -------
        NDArray of shape ``(T,)`` (symmetric) or ``(T, 2)`` (asymmetric).
        """
        T = len(scores)
        if self.asymmetric:
            pairs = np.empty((T, 2), dtype=float)
            for t in range(T):
                pairs[t] = self.predict_quantile_pair()
                self.update_signed(float(scores[t]))
            return pairs
        quantiles = np.empty(T, dtype=float)
        for t in range(T):
            quantiles[t] = self.predict_quantile()
            self.update(float(scores[t]))
        return quantiles


def _make_window(scores: NDArray, window_size: Optional[int] = None) -> deque:
    """Return a deque pre-filled with *scores*, bounded to *window_size*."""
    maxlen = window_size  # None → unbounded
    d: deque = deque(scores.tolist(), maxlen=maxlen)
    return d


class JointPredictor(ABC):
    """Abstract base for **joint** (simultaneous) conformal predictors.

    Unlike :class:`ConformalPredictor`, which is calibrated and evaluated
    independently per horizon step, a ``JointPredictor`` takes full
    prediction windows of shape ``(N, H)`` and produces an interval band
    that aims for *simultaneous* coverage across all ``H`` steps:

        Prob(y_t ∈ [ŷ_t − r_t, ŷ_t + r_t]  ∀ t = 1…H) ≥ 1 − alpha

    The evaluation metric is **simultaneous coverage** (fraction of test
    windows in which every horizon step is covered).

    Parameters
    ----------
    alpha : float
        Target miscoverage rate in (0, 1).
    """

    def __init__(self, alpha: float) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f'alpha must be in (0, 1), got {alpha}')
        self.alpha = float(alpha)
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(self, cal_point: NDArray, cal_gt: NDArray) -> 'JointPredictor':
        """Calibrate on a batch of calibration windows.

        Parameters
        ----------
        cal_point : NDArray of shape (C, N, H)
            Point predictions for C calibration windows, N series, H steps.
        cal_gt : NDArray of shape (C, N, H)
            Ground-truth values matching ``cal_point``.

        Returns
        -------
        self
        """

    @abstractmethod
    def predict_radii(self, point_preds: NDArray) -> NDArray:
        """Return per-horizon half-widths for a batch of test windows.

        Parameters
        ----------
        point_preds : NDArray of shape (N, H)
            Point predictions for one test window.

        Returns
        -------
        NDArray of shape (H,)
            Half-widths ``r_h`` so that the interval at step h is
            ``[ŷ_h − r_h, ŷ_h + r_h]``.
        """

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    def evaluate(self, test_point: NDArray, test_gt: NDArray) -> dict:
        """Evaluate joint coverage metrics over a sequence of test windows.

        Parameters
        ----------
        test_point : NDArray of shape (T, N, H)
            Point predictions for T test windows.
        test_gt : NDArray of shape (T, N, H)
            Ground-truth values.

        Returns
        -------
        dict with keys:
            joint_coverage      — fraction of windows fully covered (all H steps)
            marginal_coverage   — mean per-horizon coverage averaged over H, T, N
            per_horizon_coverage — (H,) array of per-step marginal coverage
            avg_width           — mean half-width averaged over horizons
            winkler_score       — mean Winkler score averaged over series & steps
        """
        if not self._fitted:
            raise RuntimeError('Call fit() before evaluate()')

        T, N, H = test_point.shape

        # Compute radii for every test window: (T, H)
        radii_list = [self.predict_radii(test_point[t]) for t in range(T)]
        radii = np.stack(radii_list, axis=0)  # (T, H)

        try:
            import jax
            import jax.numpy as jnp

            # Using JAX for JIT compilation to avoid massive memory allocations
            @jax.jit
            def _compute_metrics(tp, tgt, rad, alpha):
                lo = tp - rad[:, None, :]
                hi = tp + rad[:, None, :]
                covered_tnh = (tgt >= lo) & (tgt <= hi)

                joint_per_tn = covered_tnh.all(axis=2)
                joint_coverage = jnp.mean(joint_per_tn)

                per_horizon_coverage = jnp.mean(covered_tnh, axis=(0, 1))
                marginal_coverage = jnp.mean(covered_tnh)

                avg_width = jnp.mean(2.0 * rad)

                band = 2.0 * rad[:, None, :]
                miss = jnp.maximum(jnp.maximum(lo - tgt, tgt - hi), 0.0)
                winkler_tnh = band + (2.0 / alpha) * miss
                winkler_score = jnp.mean(winkler_tnh)

                return (
                    joint_coverage,
                    marginal_coverage,
                    per_horizon_coverage,
                    avg_width,
                    winkler_score,
                )

            jc, mc, phc, aw, ws = _compute_metrics(
                jnp.asarray(test_point),
                jnp.asarray(test_gt),
                jnp.asarray(radii),
                self.alpha,
            )
            return {
                'joint_coverage': float(jc),
                'marginal_coverage': float(mc),
                'per_horizon_coverage': np.array(phc),
                'avg_width': float(aw),
                'winkler_score': float(ws),
                'radii': np.asarray(radii),
            }

        except ImportError:
            # Broadcasting: (T, N, H)
            lo = test_point - radii[:, None, :]  # (T, N, H)
            hi = test_point + radii[:, None, :]  # (T, N, H)
            covered_tnh = (test_gt >= lo) & (test_gt <= hi)  # (T, N, H)

            # Joint coverage: all H steps covered per (t, n), averaged over t and n
            joint_per_tn = covered_tnh.all(axis=2)  # (T, N)
            joint_coverage = float(np.mean(joint_per_tn))

            # Marginal coverage: per-horizon and overall
            per_horizon_coverage = np.mean(covered_tnh, axis=(0, 1))  # (H,)
            marginal_coverage = float(np.mean(covered_tnh))

            avg_width = float(np.mean(2.0 * radii))

            # Winkler score — vectorised over (T, N, H)
            band = 2.0 * radii[:, None, :]  # (T, N, H) broadcast
            miss = np.maximum(np.maximum(lo - test_gt, test_gt - hi), 0.0)  # (T, N, H)
            winkler_tnh = band + (2.0 / self.alpha) * miss  # (T, N, H)
            winkler_score = float(np.mean(winkler_tnh))

            return {
                'joint_coverage': joint_coverage,
                'marginal_coverage': marginal_coverage,
                'per_horizon_coverage': per_horizon_coverage,
                'avg_width': avg_width,
                'winkler_score': winkler_score,
                'radii': radii,
            }


def conformal_quantile(
    scores: NDArray, alpha: float, conformal_correction: bool = True
) -> float:
    """Compute the empirical (1 - alpha) quantile of *scores*.

    Computes the ``ceil((n+1)(1-α)) / n`` level quantile of *scores*
    using the ``'higher'`` interpolation method if conformal_correction is True.
    Otherwise, computes the standard empirical quantile at level ``1-α``.

    Non-finite values (NaN, ±inf) are filtered before computing the quantile.

    Parameters
    ----------
    scores : NDArray of shape (n,)
        Nonconformity scores.
    alpha : float
        Target miscoverage rate in (0, 1).
    conformal_correction : bool
        Whether to use the finite sample correction. Default is True.

    Returns
    -------
    float
        The conformal quantile, or ``math.inf`` when *scores* has no finite
        values.
    """
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    n = len(scores)
    if n == 0:
        return math.inf
    if conformal_correction:
        level = math.ceil((n + 1) * (1.0 - alpha)) / n
        level = min(level, 1.0)
    else:
        level = 1.0 - alpha
    return float(np.quantile(scores, level, method='higher'))


class AsymmetricPredictor(ConformalPredictor):
    """Wraps two symmetric conformal predictors to handle asymmetric intervals.
    One predictor is trained on the lower violations, and one on the upper violations.
    """

    def __init__(self, base_class: type[ConformalPredictor], *args, **kwargs) -> None:
        import inspect

        sig = inspect.signature(base_class.__init__)
        bound = sig.bind_partial(*args, **kwargs)
        alpha = bound.arguments.get('alpha')
        if alpha is None:
            if len(args) > 0:
                alpha = args[0]
            else:
                raise ValueError("Could not find 'alpha' parameter in arguments.")

        cc = kwargs.get('conformal_correction', True)
        super().__init__(alpha, asymmetric=True, conformal_correction=cc)
        self.base_class = base_class
        self.args = args
        self.kwargs = kwargs

        # We pass asymmetric=False to the child instances
        kwargs_base = kwargs.copy()
        kwargs_base['asymmetric'] = False

        bound_lo = sig.bind_partial(*args, **kwargs_base)
        bound_lo.arguments['alpha'] = alpha / 2.0

        args_lo = bound_lo.args
        kwargs_lo = bound_lo.kwargs

        self.lower_ = base_class(*args_lo, **kwargs_lo)
        self.upper_ = base_class(*args_lo, **kwargs_lo)
        self.window_size = getattr(self.lower_, 'window_size', None)
        self._scores = deque(maxlen=self.window_size)

    def clone(self) -> AsymmetricPredictor:
        new = super().clone()
        new.lower_ = self.lower_.clone()
        new.upper_ = self.upper_.clone()
        new._scores = deque(maxlen=self.window_size)
        return new

    def fit(self, cal_scores: NDArray) -> AsymmetricPredictor:
        cal_scores = np.asarray(cal_scores, dtype=float)
        if cal_scores.ndim == 2 and cal_scores.shape[1] == 2:
            self.lower_.fit(cal_scores[:, 0])
            self.upper_.fit(cal_scores[:, 1])
        else:
            self.lower_.fit(-cal_scores)
            self.upper_.fit(cal_scores)
        self._scores = deque(cal_scores.tolist(), maxlen=self.window_size)
        self._fitted = True
        return self

    def predict_quantile(self) -> float:
        q = self.upper_.predict_quantile()
        self._last_q = q
        return q

    def predict_quantile_pair(self) -> tuple[float, float]:
        q_lo = self.lower_.predict_quantile()
        q_up = self.upper_.predict_quantile()
        self._last_q_pair = (q_lo, q_up)
        self._last_q = q_up
        return q_lo, q_up

    def update(self, new_score: float) -> None:
        self.update_signed(new_score)

    def update_signed(self, e: Any) -> None:
        if isinstance(e, (tuple, list, np.ndarray)) and len(e) == 2:
            e_lo, e_up = float(e[0]), float(e[1])
        else:
            e_lo, e_up = -float(e), float(e)
        self.lower_.update(e_lo)
        self.upper_.update(e_up)
        self._scores.append(e)

    def _scan_quantiles(self, scores: NDArray) -> NDArray:
        from timecp.methods._scan import _rebuild_deque

        scores = np.asarray(scores, dtype=float)
        if scores.ndim == 2 and scores.shape[1] == 2:
            q_lo_seq = self.lower_._scan_quantiles(scores[:, 0])
            q_up_seq = self.upper_._scan_quantiles(scores[:, 1])
        else:
            q_lo_seq = self.lower_._scan_quantiles(-scores)
            q_up_seq = self.upper_._scan_quantiles(scores)
        pairs = np.stack([q_lo_seq, q_up_seq], axis=1)
        if len(pairs) > 0:
            self._last_q_pair = (float(pairs[-1, 0]), float(pairs[-1, 1]))
            self._last_q = float(pairs[-1, 1])
        self._scores = _rebuild_deque(
            np.array(self._scores, dtype=float) if self._scores else None,
            scores,
            self.window_size,
        )
        return pairs

    def batch_evaluate_series(
        self,
        cal_scores: NDArray,
        test_scores: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """Vectorized asymmetric evaluation across N series.

        Returns
        -------
        q_lowers : (T, N) array
        q_uppers : (T, N) array
        """
        cal_scores = np.asarray(cal_scores, dtype=float)
        test_scores = np.asarray(test_scores, dtype=float)

        if cal_scores.ndim == 3 and cal_scores.shape[2] == 2:
            lower_cal = cal_scores[:, :, 0]
            upper_cal = cal_scores[:, :, 1]
            lower_test = test_scores[:, :, 0]
            upper_test = test_scores[:, :, 1]
        else:
            lower_cal = -cal_scores
            upper_cal = cal_scores
            lower_test = -test_scores
            upper_test = test_scores

        q_lowers = self.lower_.batch_evaluate_series(lower_cal, lower_test)
        q_uppers = self.upper_.batch_evaluate_series(upper_cal, upper_test)
        return q_lowers, q_uppers

    def predict_interval(self, point_forecast: Any) -> tuple[float, float]:
        if not self._fitted:
            raise RuntimeError('Call fit() before predict_interval()')
        from timecp.methods.cqr import CQRBase

        if issubclass(self.base_class, CQRBase):
            q_low, q_high = point_forecast
            q_lo, q_up = self.predict_quantile_pair()
            return float(q_low - q_lo), float(q_high + q_up)
        else:
            q_lo, q_up = self.predict_quantile_pair()
            return float(point_forecast) - q_lo, float(point_forecast) + q_up

    def __getattr__(self, name: str) -> Any:
        if name in (
            'lower_',
            'upper_',
            'base_class',
            'args',
            'kwargs',
            'window_size',
            '_scores',
            '_fitted',
            'alpha',
            'asymmetric',
            '_last_q',
            '_last_q_pair',
        ):
            if name in self.__dict__:
                return self.__dict__[name]
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

        if 'lower_' not in self.__dict__ or 'upper_' not in self.__dict__:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

        for suffix, target in [
            ('_lower', self.lower_),
            ('_upper', self.upper_),
            ('_lo', self.lower_),
            ('_up', self.upper_),
        ]:
            if name.endswith(suffix):
                cand = name[: -len(suffix)]
                if hasattr(target, cand):
                    return getattr(target, cand)
                if cand.startswith('_') and hasattr(target, cand):
                    return getattr(target, cand)

        if hasattr(self.upper_, name):
            return getattr(self.upper_, name)
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        if name in (
            'lower_',
            'upper_',
            'base_class',
            'args',
            'kwargs',
            'window_size',
            '_scores',
            '_fitted',
            'alpha',
            'asymmetric',
            '_last_q',
            '_last_q_pair',
        ):
            object.__setattr__(self, name, value)
            return

        if 'lower_' not in self.__dict__ or 'upper_' not in self.__dict__:
            object.__setattr__(self, name, value)
            return

        for suffix, target in [
            ('_lower', self.lower_),
            ('_upper', self.upper_),
            ('_lo', self.lower_),
            ('_up', self.upper_),
        ]:
            if name.endswith(suffix):
                cand = name[: -len(suffix)]
                if hasattr(target, cand):
                    setattr(target, cand, value)
                    return
                if cand.startswith('_') and hasattr(target, cand):
                    setattr(target, cand, value)
                    return

        if hasattr(self.upper_, name):
            setattr(self.upper_, name, value)
            if hasattr(self.lower_, name):
                setattr(self.lower_, name, value)
        else:
            object.__setattr__(self, name, value)
