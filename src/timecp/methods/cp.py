"""
Conformal Prediction methods.

Contains:
- SplitCP: Static Inductive Conformal Prediction.
- TrailingWindow: Rolling-window empirical quantile.
- WeightedCP: Rolling-window empirical quantile with dynamically computed geometric decay weights.
"""

from __future__ import annotations

from collections import deque
import math
from typing import Sequence

from numpy.typing import NDArray

from timecp.base import ConformalPredictor, _WindowMixin, conformal_quantile, np


class SplitCP(ConformalPredictor):
    """
    Split (Inductive) Conformal Prediction.

    Calibrates once on the calibration set and is strictly static (no online updates).

    Parameters
    ----------
    alpha : float
        Target miscoverage rate in (0, 1).
    asymmetric : bool
        Track separate lower/upper quantiles using signed errors.  Default False.
    conformal_correction : bool
        Use finite-sample conformal correction.  Default True.
    bonferroni_horizon : int or None
        If set, applies a Bonferroni correction over H horizons by using
        ``alpha / H`` as the effective level.  This is equivalent to CFRNN
        calibration.  Default None.
    """

    def __init__(
        self,
        alpha: float,
        asymmetric: bool = False,
        conformal_correction: bool = True,
        bonferroni_horizon: int | None = None,
    ) -> None:
        super().__init__(
            alpha,
            asymmetric=asymmetric,
            conformal_correction=conformal_correction,
            bonferroni_horizon=bonferroni_horizon,
        )
        self._scores: NDArray = np.empty(0, dtype=float)

    def fit(self, cal_scores: NDArray) -> 'SplitCP':
        cal_scores = np.asarray(cal_scores, dtype=float)
        self._scores = cal_scores[np.isfinite(cal_scores)]
        self._fitted = True
        return self

    def predict_quantile(self) -> float:
        q = conformal_quantile(
            self._scores, self._effective_alpha, self.conformal_correction
        )
        self._last_q = q
        return q

    def _scan_quantiles(self, scores: NDArray) -> NDArray:
        T = len(scores)
        q = self.predict_quantile()
        return np.full(T, q)


class TrailingWindow(_WindowMixin, ConformalPredictor):
    """
    Trailing-window empirical quantile (baseline "Trail" method).

    Keeps the last *window_size* observed nonconformity scores.
    Uses standard empirical quantile by default (no conformal correction).

    Parameters
    ----------
    alpha : float
        Target miscoverage rate in (0, 1).
    window_size : int
        Rolling window length.  Default 100.
    asymmetric : bool
        Track separate lower/upper quantiles.  Default False.
    conformal_correction : bool
        Use finite-sample conformal correction.  Default False.
    bonferroni_horizon : int or None
        Bonferroni correction over H horizons (uses ``alpha / H``).  Default None.
    """

    def __init__(
        self,
        alpha: float,
        window_size: int = 100,
        asymmetric: bool = False,
        conformal_correction: bool = False,
        bonferroni_horizon: int | None = None,
    ) -> None:
        super().__init__(
            alpha,
            asymmetric=asymmetric,
            conformal_correction=conformal_correction,
            bonferroni_horizon=bonferroni_horizon,
        )
        if window_size < 1:
            raise ValueError(f'window_size must be >= 1, got {window_size}')
        self.window_size = window_size
        self._scores: deque[float] = deque(maxlen=window_size)

    def fit(self, cal_scores: NDArray) -> 'TrailingWindow':
        self._scores = self._init_window(cal_scores, self.window_size)
        return self

    def predict_quantile(self) -> float:
        scores_arr = np.asarray(self._scores, dtype=float)
        q = conformal_quantile(
            scores_arr, self._effective_alpha, self.conformal_correction
        )
        self._last_q = q
        return q

    def _scan_quantiles(self, scores: NDArray) -> NDArray:
        from timecp.methods._scan import (
            _rebuild_deque,
            split_scan,
        )

        scores = np.asarray(scores, dtype=float)
        T = len(scores)
        initial_window = np.array(self._scores, dtype=float) if self._scores else None

        quantiles = split_scan(
            scores,
            alpha=self._effective_alpha,
            initial_window=initial_window,
            window_size=self.window_size,
            conformal_correction=self.conformal_correction,
        )
        self._last_q = quantiles[-1] if T > 0 else self._last_q
        self._scores = _rebuild_deque(initial_window, scores, self.window_size)
        return quantiles


class WeightedCP(_WindowMixin, ConformalPredictor):
    """
    Weighted Conformal Prediction with rolling window and geometric decay weights.

    Weighs scores in the window with a geometric decay function by default.
    ``weights[i] = decay ** (n - 1 - i)``

    Parameters
    ----------
    alpha : float
        Target miscoverage rate in (0, 1).
    window_size : int or None
        Rolling window length.  Defaults to 100 when ``weights`` is None.
    decay : float
        Geometric decay factor in (0, 1].  Default 0.99.
    test_weight : float
        Weight assigned to the test point in the weighted quantile.  Default 1.0.
    asymmetric : bool
        Track separate lower/upper quantiles.  Default False.
    conformal_correction : bool
        Use finite-sample conformal correction.  Default False.
    weights : array-like or None
        Custom static weight vector (overrides *decay* and *window_size*).
    bonferroni_horizon : int or None
        Bonferroni correction over H horizons (uses ``alpha / H``).  Default None.
    """

    def __init__(
        self,
        alpha: float,
        window_size: int | None = None,
        decay: float = 0.99,
        test_weight: float = 1.0,
        asymmetric: bool = False,
        conformal_correction: bool = False,
        weights: Sequence[float] | NDArray | None = None,
        bonferroni_horizon: int | None = None,
    ) -> None:
        super().__init__(
            alpha,
            asymmetric=asymmetric,
            conformal_correction=conformal_correction,
            bonferroni_horizon=bonferroni_horizon,
        )

        # Support both custom static weights and new geometric decay weights
        if weights is not None:
            self.weights = np.asarray(weights, dtype=float)
            if len(self.weights) == 0:
                raise ValueError('weights array must not be empty.')
            if np.any(self.weights < 0):
                raise ValueError('weights must be non-negative.')
            self.window_size = len(self.weights)
            self.decay = None
        else:
            if window_size is None:
                window_size = 100
            if window_size < 1:
                raise ValueError(f'window_size must be >= 1, got {window_size}')
            if not 0.0 < decay <= 1.0:
                raise ValueError(f'decay must be in (0, 1], got {decay}')
            self.window_size = window_size
            self.decay = float(decay)

        self.test_weight = float(test_weight)
        self._scores: deque[float] = deque(maxlen=self.window_size)

    def fit(self, cal_scores: NDArray) -> 'WeightedCP':
        self._scores = self._init_window(cal_scores, self.window_size)
        return self

    def _weighted_quantile(self, values: NDArray, target_p: float) -> float:
        values = np.asarray(values, dtype=float)
        finite_mask = np.isfinite(values)
        if not finite_mask.any():
            return math.inf
        if not finite_mask.all():
            values = values[finite_mask]

        n = len(values)
        if n == 0:
            return math.inf

        if self.decay is not None:
            w = self.decay ** np.arange(n - 1, -1, -1)
        else:
            w = self.weights[-n:]

        total_weight = float(np.sum(w)) + self.test_weight
        p = w / total_weight
        idx = np.argsort(values)
        sorted_vals = values[idx]
        sorted_p = p[idx]
        cumsum_p = np.cumsum(sorted_p)
        if target_p > cumsum_p[-1]:
            return math.inf
        idx_q = int(np.searchsorted(cumsum_p, target_p))
        return float(sorted_vals[idx_q])

    def predict_quantile(self) -> float:
        scores_arr = np.asarray(self._scores, dtype=float)
        n = len(scores_arr)
        if self.conformal_correction and n > 0:
            target_p = math.ceil((n + 1) * (1.0 - self._effective_alpha)) / n
            target_p = min(target_p, 1.0)
        else:
            target_p = 1.0 - self._effective_alpha
        q = self._weighted_quantile(scores_arr, target_p)
        self._last_q = q
        return q
