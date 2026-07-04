"""Sequential Predictive Conformal Inference (SPCI).

Reference
---------
Xu, C., and Xie, Y. (2023).
"Sequential Predictive Conformal Inference for Time Series."
ICML 2023.
"""

from __future__ import annotations

from collections import deque
import numpy as np
from numpy.typing import NDArray

try:
    from sklearn_quantile import RandomForestQuantileRegressor  # type: ignore

    HAS_SKLEARN_QUANTILE = True
except ImportError:
    HAS_SKLEARN_QUANTILE = False
    RandomForestQuantileRegressor = None  # type: ignore

from timecp.base import ConformalPredictor


class FallbackQuantileRegressor:
    """A simple wrapper around GradientBoostingRegressor to act like RandomForestQuantileRegressor."""

    def __init__(self, q, n_estimators=10, max_depth=2):
        from sklearn.ensemble import GradientBoostingRegressor

        self.q = q
        self.estimators = [
            GradientBoostingRegressor(
                loss='quantile',
                alpha=alpha,
                n_estimators=n_estimators,
                max_depth=max_depth,
            )
            for alpha in q
        ]

    def fit(self, X, y):
        for est in self.estimators:
            est.fit(X, y)
        return self

    def predict(self, X):
        import numpy as np

        preds = [est.predict(X) for est in self.estimators]
        return np.column_stack(preds)

    def set_params(self, **kwargs):
        pass


class SPCI(ConformalPredictor):
    natively_asymmetric = True

    """Sequential Predictive Conformal Inference.

    A per-horizon asymmetric conformal predictor that conditionally forecasts
    the error quantiles using a Quantile Random Forest on recent residual history.

    Parameters
    ----------
    alpha : float
        Target miscoverage rate.
    w : int
        Window size for the autoregressive residual features. Default 100.
    n_estimators : int
        Number of trees for the Quantile Random Forest. Default 10.
    max_depth : int
        Maximum depth for the QRF. Default 2.
    bins : int
        Number of bins for searching the shortest interval. Default 5.
    retrain_freq : int
        How often to retrain the QRF (in number of steps). Default 1 (retrain every step).
    """

    def __init__(
        self,
        alpha: float,
        w: int = 100,
        n_estimators: int = 10,
        max_depth: int = 2,
        bins: int = 5,
        retrain_freq: int = 1,
    ) -> None:
        super().__init__(alpha, asymmetric=True)

        self.w = w
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.bins = bins
        self.retrain_freq = retrain_freq

        self._history = deque()
        self._steps_since_retrain = 0

        self.beta_ls = np.linspace(start=1e-5, stop=max(1e-5, alpha - 1e-5), num=bins)
        self.full_alphas = np.concatenate([self.beta_ls, 1.0 - alpha + self.beta_ls])
        self.full_alphas = np.clip(self.full_alphas, 1e-5, 1.0 - 1e-5)

        if HAS_SKLEARN_QUANTILE and RandomForestQuantileRegressor is not None:
            self.rfqr = RandomForestQuantileRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                criterion='squared_error',
                q=self.full_alphas.tolist(),
                n_jobs=-1,
            )
        else:
            self.rfqr = FallbackQuantileRegressor(
                q=self.full_alphas.tolist(),
                n_estimators=n_estimators,
                max_depth=max_depth,
            )
        self._q_lower = 0.0
        self._q_upper = 0.0

    def clone(self) -> ConformalPredictor:
        import copy

        new = super().clone()
        new._history = deque()
        new._steps_since_retrain = 0
        new._q_lower = 0.0
        new._q_upper = 0.0
        new.rfqr = copy.deepcopy(self.rfqr)
        return new

    def fit(self, cal_scores: NDArray) -> 'SPCI':  # type: ignore[override]
        """Fit the initial SPCI QRF.

        Parameters
        ----------
        cal_scores : (C,) array of signed errors e = y - ŷ
        """
        self._history = deque(cal_scores.tolist())
        self._steps_since_retrain = 0
        self._retrain_qrf()
        self._fitted = True
        return self

    def _retrain_qrf(self) -> None:
        """Retrain the QRF and update current q_lower, q_upper thresholds."""
        n = len(self._history)
        if n <= self.w:
            if n > 0:
                self._q_lower = -np.percentile(self._history, self.alpha / 2 * 100)
                self._q_upper = np.percentile(self._history, (1 - self.alpha / 2) * 100)
            else:
                self._q_lower = 0.0
                self._q_upper = 0.0
            return

        history_arr = np.array(self._history, dtype=np.float64)

        from numpy.lib.stride_tricks import sliding_window_view

        windows = sliding_window_view(history_arr, window_shape=self.w)

        X_train = windows[:-1]
        Y_train = history_arr[self.w :]

        if len(X_train) == 0:
            return

        try:
            self.rfqr.fit(X_train, Y_train)
        except TypeError as e:
            if 'sample_weight' in str(e) and HAS_SKLEARN_QUANTILE:
                # sklearn-quantile is incompatible with scikit-learn 1.4+
                self.rfqr = FallbackQuantileRegressor(
                    q=self.full_alphas.tolist(),
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                )
                self.rfqr.fit(X_train, Y_train)
            else:
                raise e
        except ValueError as e:
            # Fallback for old sklearn versions where 'squared_error' is not valid
            if 'criterion' in str(e) and HAS_SKLEARN_QUANTILE:
                self.rfqr.set_params(criterion='mse')
                self.rfqr.fit(X_train, Y_train)
            else:
                raise e

        x_latest = windows[-1].reshape(1, -1)
        low_high_pred = self.rfqr.predict(x_latest)
        if low_high_pred.ndim > 1:
            low_high_pred = low_high_pred[0]

        num_mid = self.bins
        low_pred = low_high_pred[:num_mid]
        high_pred = low_high_pred[num_mid:]

        width = high_pred - low_pred
        i_star = np.argmin(width)

        self._q_lower = float(-low_pred[i_star].item())
        self._q_upper = float(high_pred[i_star].item())

    def predict_quantile(self) -> float:
        """Return the upper quantile (required by base class)."""
        self._last_q = float(self._q_upper)
        return self._last_q

    def predict_quantile_pair(self) -> tuple[float, float]:
        q_lo, q_up = float(self._q_lower), float(self._q_upper)
        self._last_q_pair = (q_lo, q_up)  # type: ignore
        return q_lo, q_up  # type: ignore

    def update_signed(self, e: float) -> None:
        self._history.append(e)
        self._steps_since_retrain += 1

        if self._steps_since_retrain >= self.retrain_freq:
            self._retrain_qrf()
            self._steps_since_retrain = 0
