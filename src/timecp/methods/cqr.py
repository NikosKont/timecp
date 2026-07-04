"""
Conformalized Quantile Regression (CQR).

Reference
---------
Romano et al. (2019) "Conformalized Quantile Regression"
https://arxiv.org/abs/1905.03222
"""

from __future__ import annotations

import math
from collections import deque

from numpy.typing import NDArray

from timecp.base import ConformalPredictor, _WindowMixin, conformal_quantile, np


class CQRBase:
    """Mixin providing CQR-specific ``predict_interval`` and ``evaluate``.

    Overrides the symmetric-interval logic in
    :class:`~timecp.base.ConformalPredictor` to work with asymmetric
    quantile regression bounds ``(q_low, q_high)``.

    Must be combined (via multiple inheritance) with a concrete class
    that supplies:

    * ``self.alpha: float``
    * ``self._fitted: bool``
    * ``self.predict_quantile() -> float``
    * ``self.update(new_score: float) -> None``

    Usage::

        class CQR(CQRBase, ConformalPredictor):
            ...

        class AdaptiveCQR(CQRBase, ACI):
            ...
    """

    # Declared for type checkers — concrete values come from the co-inherited class.
    alpha: float
    _fitted: bool

    # ------------------------------------------------------------------

    def cqr_score(self, y_true: float, q_low: float, q_high: float) -> float:
        """Compute the CQR conformity score: ``max(q_low - y_true, y_true - q_high)``."""
        return max(q_low - y_true, y_true - q_high)

    def predict_interval(
        self, point_forecast: tuple[float, float] | NDArray | list[float]
    ) -> tuple[float, float]:
        """Return the CQR prediction interval ``(q_low - q̂, q_high + q̂)``.

        Parameters
        ----------
        point_forecast : tuple, list, or NDArray of shape (2,)
            The predicted lower and upper conditional quantiles
            ``(q_low, q_high)``.

        Returns
        -------
        tuple[float, float]
            ``(lower bound, upper bound)``
        """
        if not self._fitted:
            raise RuntimeError('Call fit() before predict_interval()')
        q_low, q_high = point_forecast
        q = self.predict_quantile()
        return float(q_low - q), float(q_high + q)

    def evaluate(
        self,
        scores: NDArray,
        point_forecasts: NDArray,
        burnin: int = 0,
    ) -> dict:
        """Rolling evaluation for CQR-family methods.

        Parameters
        ----------
        scores : NDArray of shape (T,)
            CQR conformity scores: ``max(q_low - y, y - q_high)``.
        point_forecasts : NDArray of shape (T, 2)
            Predicted ``(q_low, q_high)`` for each time step.
        burnin : int, optional
            Steps to skip when computing metrics.  Default 0.

        Returns
        -------
        dict
            ``coverage``, ``avg_width``, ``winkler_score``, ``quantiles``.
        """
        if not self._fitted:
            raise RuntimeError('Call fit() before evaluate()')

        scores = np.asarray(scores, dtype=float)
        point_forecasts = np.asarray(point_forecasts, dtype=float)

        if point_forecasts.ndim != 2 or point_forecasts.shape[1] != 2:
            raise ValueError(
                'point_forecasts must have shape (T, 2) for CQR (q_low, q_high)'
            )

        T = len(scores)

        covereds: list[bool] = []
        widths: list[float] = []
        winkler: list[float] = []
        quantiles_out: list[float] = []

        for t in range(T):
            q = self.predict_quantile()
            score = float(scores[t])
            covered = score <= q

            q_low, q_high = point_forecasts[t]

            if math.isfinite(q):
                width = float((q_high + q) - (q_low - q))
            else:
                width = math.inf

            quantiles_out.append(q)

            if t >= burnin:
                covereds.append(covered)
                widths.append(width)
                if math.isfinite(width):
                    if covered:
                        w = width
                    else:
                        w = width + (2.0 / self.alpha) * (score - q)
                else:
                    w = math.inf
                winkler.append(w)

            self.update(score)

        finite_winkler = [w for w in winkler if math.isfinite(w)]
        finite_widths = [w for w in widths if math.isfinite(w)]

        return {
            'coverage': float(np.mean(covereds)) if covereds else float('nan'),
            'avg_width': float(np.mean(finite_widths))
            if finite_widths
            else float('nan'),
            'winkler_score': float(np.mean(finite_winkler))
            if finite_winkler
            else float('nan'),
            'quantiles': np.array(quantiles_out),
        }

    def batch_evaluate(
        self,
        scores: NDArray,
        point_forecasts: NDArray,
        burnin: int = 0,
    ) -> dict:
        """Vectorized batch evaluation for CQR-family methods.

        Performs a scan-style forward pass to compute all quantiles, then
        vectorizes coverage/width/Winkler computation.

        Parameters
        ----------
        scores : NDArray of shape (T,)
            CQR conformity scores.
        point_forecasts : NDArray of shape (T, 2)
            Predicted (q_low, q_high) for each time step.
        burnin : int, optional
            Steps to skip.  Default 0.

        Returns
        -------
        dict with coverage, avg_width, winkler_score, quantiles.
        """
        if not self._fitted:
            raise RuntimeError('Call fit() before batch_evaluate()')

        scores = np.asarray(scores, dtype=float)
        point_forecasts = np.asarray(point_forecasts, dtype=float)

        if point_forecasts.ndim != 2 or point_forecasts.shape[1] != 2:
            raise ValueError(
                'point_forecasts must have shape (T, 2) for CQR (q_low, q_high)'
            )

        T = len(scores)

        # Scan: compute quantiles sequentially (state depends on previous steps)
        quantiles = np.empty(T, dtype=float)
        for t in range(T):
            quantiles[t] = self.predict_quantile()
            self.update(float(scores[t]))

        # Vectorized metrics
        if burnin >= T:
            return {
                'coverage': float('nan'),
                'avg_width': float('nan'),
                'winkler_score': float('nan'),
                'quantiles': quantiles,
            }

        s = scores[burnin:]
        q = quantiles[burnin:]
        q_low = point_forecasts[burnin:, 0]
        q_high = point_forecasts[burnin:, 1]

        covered = s <= q
        finite_mask = np.isfinite(q)
        widths = np.where(finite_mask, (q_high + q) - (q_low - q), np.inf)

        coverage = float(np.mean(covered))

        finite_widths = widths[finite_mask]
        avg_width = (
            float(np.mean(finite_widths)) if len(finite_widths) > 0 else float('nan')
        )

        penalty = np.where(covered, 0.0, (2.0 / self.alpha) * (s - q))
        winkler_all = np.where(finite_mask, widths + penalty, np.inf)
        finite_winkler = winkler_all[np.isfinite(winkler_all)]
        winkler_score = (
            float(np.mean(finite_winkler)) if len(finite_winkler) > 0 else float('nan')
        )

        return {
            'coverage': coverage,
            'avg_width': avg_width,
            'winkler_score': winkler_score,
            'quantiles': quantiles,
        }


class CQR(CQRBase, _WindowMixin, ConformalPredictor):
    """Online Conformalized Quantile Regression (CQR).

    Adapts the CQR method to an online setting using a sliding window
    of conformity scores, similar to SplitCP.

    The conformity score for CQR evaluates the error of the lower and upper bounds:
        E_t = max(q_low_t - y_t, y_t - q_high_t)
    where q_low_t and q_high_t are the conditional quantile predictions.

    Parameters
    ----------
    alpha : float
        Target miscoverage rate in (0, 1).
    window_size : int or None, optional
        Maximum number of calibration scores to keep. ``None`` keeps all scores.
    """

    def __init__(
        self,
        alpha: float,
        window_size: int | None = None,
        asymmetric: bool = False,
        conformal_correction: bool = True,
    ) -> None:
        super().__init__(
            alpha, asymmetric=asymmetric, conformal_correction=conformal_correction
        )
        self.window_size = window_size
        self._scores: deque[float] = deque(maxlen=window_size)

    def fit(self, cal_scores: NDArray) -> 'CQR':
        self._scores = self._init_window(cal_scores, self.window_size)
        return self

    def predict_quantile(self) -> float:
        n = len(self._scores)
        if n == 0:
            return math.inf
        scores_arr = np.asarray(self._scores, dtype=float)
        q = conformal_quantile(scores_arr, self.alpha)
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
            alpha=self.alpha,
            initial_window=initial_window,
            window_size=self.window_size,
        )
        self._last_q = quantiles[-1] if T > 0 else self._last_q
        self._scores = _rebuild_deque(initial_window, scores, self.window_size)
        return quantiles
