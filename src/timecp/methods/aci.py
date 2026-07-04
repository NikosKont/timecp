"""
Adaptive Conformal Inference (ACI).

The quantile threshold adapts online: the effective miscoverage level α_t is
updated after each step to track the target α.  The rolling empirical quantile
of a window of past scores is recomputed each step at level ``1 − α_t``:

    err_t  = 1 if score_t > q̂_t else 0
    α_{t+1} = clip(α_t + γ · (α − err_t), 0, 1)

**Asymmetric mode** (``asymmetric=True``):
    Operates on signed errors ``e = y − ŷ``.  Two independent α_t trackers
    are maintained, one per tail, each targeting ``α/2``:

        err_upper_t = 1  if e_t > q_up   else 0
        err_lower_t = 1  if −e_t > q_lo  else 0
        α_upper_{t+1} = clip(α_upper_t + γ · (α/2 − err_upper_t), 0, 1)
        α_lower_{t+1} = clip(α_lower_t + γ · (α/2 − err_lower_t), 0, 1)

References
----------
Gibbs & Candès (2021) "Adaptive Conformal Inference Under Distribution Shift"
  https://arxiv.org/abs/2106.00170

Zaffran et al. (2022) "Adaptive Conformal Predictions for Time Series"
  https://arxiv.org/abs/2202.07282

Closely follows ``methods/ACI/models.py`` (method ``"ACP"``) and
``methods/conformal-PID/core/methods.py`` (function ``aci``).
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Optional

from numpy.typing import NDArray

from timecp.base import ConformalPredictor, np
from timecp.methods._scan import _quantile, _rebuild_deque


class ACI(ConformalPredictor):
    """
    Adaptive Conformal Inference.

    Maintains a rolling window of the most recent *window_size* nonconformity
    scores.  At each step the current quantile is computed at level ``1 − α_t``,
    where ``α_t`` is updated after observing whether the true score exceeded the
    quantile.

    Parameters
    ----------
    alpha : float
        Target miscoverage rate in (0, 1).
    gamma : float
        Step-size for the α_t update.  Larger values react faster to coverage
        errors but produce more volatile interval widths.  Typical range: 0.001–0.1.
    window_size : int or None
        Number of past scores used to compute the quantile.  ``None`` uses all
        scores seen so far (unbounded memory).  Default 100.
    asymmetric : bool
        If True, maintain two independent α_t trackers (one per tail) using
        signed errors.  Each tail targets ``alpha/2``.  Default False.
    """

    def __init__(
        self,
        alpha: float,
        gamma: float = 0.01,
        window_size: Optional[int] = 100,
        asymmetric: bool = False,
        conformal_correction: bool = True,
        min_guard: bool = False,
    ) -> None:
        super().__init__(
            alpha, asymmetric=asymmetric, conformal_correction=conformal_correction
        )
        if gamma < 0:
            raise ValueError(f'gamma must be >= 0, got {gamma}')
        self.gamma = float(gamma)
        self.window_size = window_size
        self.min_guard = min_guard
        # Symmetric state
        self.alpha_t: float = alpha
        self._scores: deque[Any] = deque(maxlen=window_size)

    def clone(self) -> ACI:
        new = super().clone()
        new.alpha_t = self.alpha
        return new

    def fit(self, cal_scores: NDArray) -> 'ACI':
        self._scores = self._init_window(cal_scores, self.window_size)
        self.alpha_t = self.alpha
        return self

    def predict_quantile(self) -> float:
        scores_arr = np.asarray(self._scores, dtype=float)
        q = _quantile(
            scores_arr,
            self.alpha_t,
            conformal_correction=self.conformal_correction,
            min_guard=self.min_guard,
        )
        self._last_q = q
        return q

    def update(self, new_score: float) -> None:
        """
        Update α_t and the score window.

        err_t = 1 if new_score > q̂_t else 0
        α_{t+1} = clip(α_t + γ · (α − err_t), 0, 1)
        """
        q = self._last_q if self._last_q is not None else math.inf
        err = 1 if new_score > q else 0
        self.alpha_t = float(
            np.clip(self.alpha_t + self.gamma * (self.alpha - err), 0.0, 1.0)
        )
        self._scores.append(new_score)

    def _scan_quantiles(self, scores: NDArray) -> NDArray:
        """Vectorized scan: compute all quantiles in one forward pass."""
        from timecp.methods._scan import (
            aci_scan,
        )

        scores = np.asarray(scores, dtype=float)
        T = len(scores)
        initial_window = np.array(self._scores, dtype=float) if self._scores else None

        # Symmetric path
        quantiles, alpha_t_final = aci_scan(
            scores,
            alpha=self.alpha,
            gamma=self.gamma,
            alpha_t_init=self.alpha_t,
            initial_window=initial_window,
            window_size=self.window_size,
            min_guard=self.min_guard,
        )
        self.alpha_t = alpha_t_final
        self._last_q = quantiles[-1] if T > 0 else self._last_q
        self._scores = _rebuild_deque(initial_window, scores, self.window_size)
        return quantiles
