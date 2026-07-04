"""Conformal-RNN (CFRNN).

Reference
---------
Stankevičiūtė et al. (2021) "Conformal Time-Series Forecasting"
https://proceedings.neurips.cc/paper/2021/hash/31e21b7ff320d3c054dc24846460c57c-Abstract.html

Two classes are provided:

``CFRNN``
    Convenience subclass of :class:`~timecp.methods.SplitCP` with a built-in
    Bonferroni correction.  Equivalent to
    ``SplitCP(alpha, bonferroni_horizon=horizon)``.  The Bonferroni correction
    is now a first-class feature of :class:`~timecp.base.ConformalPredictor`
    via the ``bonferroni_horizon`` parameter, so any method can target joint
    coverage — not just CFRNN.

``JointCFRNN``
    :class:`~timecp.base.JointPredictor` subclass that takes full prediction
    windows of shape ``(C, N, H)`` and produces per-horizon half-widths via a
    Bonferroni-corrected split-CP quantile.  This is the correct class to use
    for joint simultaneous-coverage evaluation via ``CPEvaluator.run_multi_step``.
"""

from __future__ import annotations

import math

from numpy.typing import NDArray

from timecp.base import (
    JointPredictor,
    np,
)
from timecp.methods.cp import SplitCP

# ---------------------------------------------------------------------------
# Convenience scalar-score variant (ConformalPredictor)
# ---------------------------------------------------------------------------


class CFRNN(SplitCP):
    """Conformal-RNN calibration — convenience wrapper around SplitCP.

    Equivalent to ``SplitCP(alpha, bonferroni_horizon=horizon)``.

    Applies a Bonferroni correction to the conformal quantile level so that
    a union bound over H independent per-horizon intervals gives joint
    simultaneous coverage at rate ``1 − alpha``.

    Use :class:`JointCFRNN` when working with the joint evaluation pipeline
    (``CPEvaluator.run_multi_step``); use this class when nonconformity scores
    are already pre-aggregated scalars (e.g. a single ``max |e_h|`` per window).

    Parameters
    ----------
    alpha : float
        Target joint miscoverage rate in (0, 1).
    horizon : int
        Forecasting horizon H. The effective alpha is ``alpha / H``
        (Bonferroni correction).
    """

    def __init__(self, alpha: float, horizon: int) -> None:
        if horizon < 1:
            raise ValueError(f'horizon must be >= 1, got {horizon}')
        self.horizon = int(horizon)
        # Pass bonferroni_horizon so the base class _effective_alpha property
        # returns alpha / horizon automatically.
        super().__init__(alpha, bonferroni_horizon=self.horizon)


# ---------------------------------------------------------------------------
# Joint variant (JointPredictor)
# ---------------------------------------------------------------------------


class JointCFRNN(JointPredictor):
    """Conformal-RNN for joint trajectory coverage (JointPredictor variant).

    Computes per-horizon nonconformity scores from full ``(C, N, H)`` windows,
    then applies a Bonferroni-corrected split-CP quantile to produce
    per-horizon half-widths that simultaneously cover all H steps.

    Specifically, for each calibration window ``i`` and series ``n``:
        ``s_{i,n,h} = |y_{i,n,h} − ŷ_{i,n,h}|``

    The Bonferroni quantile ``C_h`` at level ``1 − alpha/H`` is computed
    independently per horizon ``h`` from the pooled calibration scores.
    The joint coverage guarantee follows from the union bound:
        ``Prob(∀h: |e_h| ≤ C_h) ≥ 1 − sum_h alpha/H = 1 − alpha``.

    Parameters
    ----------
    alpha : float
        Target joint miscoverage rate.
    """

    def __init__(self, alpha: float) -> None:
        super().__init__(alpha)
        self._radii: NDArray | None = None  # (H,)

    def fit(self, cal_point: NDArray, cal_gt: NDArray) -> 'JointCFRNN':
        """Calibrate on full prediction windows.

        Parameters
        ----------
        cal_point : (C, N, H) array — point predictions.
        cal_gt    : (C, N, H) array — ground truth.
        """
        C, N, H = cal_point.shape
        residuals = np.abs(cal_gt - cal_point).astype(np.float64)  # (C, N, H)

        # Pool C*N scores per horizon → shape (C*N, H), then compute all H
        # Bonferroni-corrected quantiles in one vectorised call.
        corrected_alpha = self.alpha / H
        n = C * N
        level = min(math.ceil((n + 1) * (1.0 - corrected_alpha)) / n, 1.0)

        # Reshape to (C*N, H), transpose to (H, C*N) for quantile over axis 1
        scores_nh = residuals.reshape(C * N, H)  # (C*N, H)
        # np.quantile with axis=0 on (C*N, H) gives (H,)
        self._radii = np.quantile(scores_nh, level, axis=0, method='higher').astype(
            np.float64
        )  # (H,)

        self._fitted = True
        return self

    def predict_radii(self, point_preds: NDArray) -> NDArray:
        """Return Bonferroni-corrected per-horizon half-widths.

        Parameters
        ----------
        point_preds : (N, H) — unused (static method).

        Returns
        -------
        NDArray of shape (H,)
        """
        if not self._fitted or self._radii is None:
            raise RuntimeError('Call fit() before predict_radii()')
        return self._radii
