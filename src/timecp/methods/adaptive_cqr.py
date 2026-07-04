"""
Adaptive Conformalized Quantile Regression (AdaptiveCQR).

Combines CQR-style asymmetric nonconformity scores with a pluggable
calibration algorithm that adapts to distribution shift.

Calibration algorithms
----------------------
``"agaci"``  (default)
    Aggregated Adaptive Conformal Inference — ensemble of ACI experts with
    AdaHedge mixture weights.  Best overall robustness.
    Zaffran et al. (2022) https://arxiv.org/abs/2202.07282

``"aci"``
    Adaptive Conformal Inference — single step-size α_t update.
    Gibbs & Candès (2021) https://arxiv.org/abs/2106.00170

``"dtaci"``
    Distribution-free Time-series ACI — exponential-weighted expert ensemble,
    sampled (rather than averaged) at each step.
    Gibbs & Candes (2022) https://arxiv.org/abs/2208.08401

``"pid"``
    Conformal PID controller — gradient descent on the pinball loss with an
    optional integral correction term.
    Angelopoulos et al. (2024) https://arxiv.org/abs/2307.16895

References
----------
Romano et al. (2019) "Conformalized Quantile Regression"
  https://arxiv.org/abs/1905.03222
"""

from __future__ import annotations

from typing import Any, Literal

from numpy.typing import NDArray

from timecp.base import ConformalPredictor, np
from timecp.methods.cqr import CQRBase

# Calibration method type alias
CalibrationMethod = Literal['agaci', 'aci', 'dtaci', 'pid']

_VALID_METHODS = ('agaci', 'aci', 'dtaci', 'pid')


def _build_engine(
    method: CalibrationMethod,
    alpha: float,
    method_kwargs: dict[str, Any],
) -> ConformalPredictor:
    """Instantiate the calibration engine for the requested *method*."""
    if method == 'agaci':
        from timecp.methods.agaci import AgACI

        defaults: dict[str, Any] = dict(
            gammas=(0.001, 0.005, 0.01, 0.05, 0.1),
            window_size=100,
        )
        defaults.update(method_kwargs)
        return AgACI(alpha=alpha, **defaults)

    if method == 'aci':
        from timecp.methods.aci import ACI

        defaults = dict(gamma=0.01, window_size=100)
        defaults.update(method_kwargs)
        return ACI(alpha=alpha, **defaults)

    if method == 'dtaci':
        from timecp.methods.dtaci import DtACI

        defaults = dict(window_size=100)
        defaults.update(method_kwargs)
        return DtACI(alpha=alpha, **defaults)

    if method == 'pid':
        from timecp.methods.pid import QuantileIntegrator

        defaults = dict(lr=0.1, KI=0.0, window_size=100)
        defaults.update(method_kwargs)
        return QuantileIntegrator(alpha=alpha, **defaults)

    raise ValueError(
        f'Unknown calibration method {method!r}. Choose one of {_VALID_METHODS}.'
    )


class AdaptiveCQR(CQRBase, ConformalPredictor):
    """Adaptive Conformalized Quantile Regression.

    Wraps a pluggable calibration engine (ACI, AgACI, DtACI, or PID) with
    CQR-specific asymmetric nonconformity scoring and interval construction.

    The CQR nonconformity score at time *t* is::

        E_t = max(q_low_t − y_t,  y_t − q_high_t)

    The calibration engine uses this score to maintain an adaptive threshold
    q̂_t.  The prediction interval is ``(q_low − q̂_t, q_high + q̂_t)``.

    Parameters
    ----------
    alpha : float
        Target miscoverage rate in (0, 1).
    method : {"agaci", "aci", "dtaci", "pid"}
        Calibration algorithm.  Default ``"agaci"``.
    **method_kwargs
        Extra keyword arguments forwarded verbatim to the calibration engine.

        *agaci* — ``gammas``, ``window_size``, ``eps``
        *aci*   — ``gamma``, ``window_size``
        *dtaci* — ``gammas``, ``sigma``, ``eta``, ``eta_adapt``,
                  ``eta_lookback``, ``window_size``, ``seed``
        *pid*   — ``lr``, ``KI``, ``C_sat``, ``window_size``,
                  ``proportional_lr``, ``q_init``

    Examples
    --------
    >>> model = AdaptiveCQR(alpha=0.1)                    # AgACI (default)
    >>> model = AdaptiveCQR(alpha=0.1, method="aci", gamma=0.05)
    >>> model = AdaptiveCQR(alpha=0.1, method="pid", lr=0.2, KI=0.1)
    """

    def __init__(
        self,
        alpha: float,
        method: CalibrationMethod = 'agaci',
        asymmetric: bool = False,
        **method_kwargs: Any,
    ) -> None:
        # ConformalPredictor.__init__ validates alpha and sets self.alpha
        ConformalPredictor.__init__(self, alpha, asymmetric=asymmetric)

        if method not in _VALID_METHODS:
            raise ValueError(
                f'Unknown calibration method {method!r}. '
                f'Choose one of {_VALID_METHODS}.'
            )
        self.method = method
        method_kwargs['asymmetric'] = asymmetric
        self._method_kwargs = method_kwargs
        self._engine: ConformalPredictor = _build_engine(method, alpha, method_kwargs)

    def clone(self) -> ConformalPredictor:
        new = super().clone()
        new._engine = self._engine.clone()
        return new

    # ------------------------------------------------------------------
    # ConformalPredictor abstract interface — delegate to engine
    # ------------------------------------------------------------------

    def fit(self, cal_scores: NDArray) -> 'AdaptiveCQR':
        """Calibrate the engine on historical CQR nonconformity scores."""
        cal_scores = np.asarray(cal_scores, dtype=float)
        self._engine.fit(cal_scores)
        self._fitted = True
        return self

    def predict_quantile(self) -> float:
        """Return the current adaptive threshold q̂ from the engine."""
        q = self._engine.predict_quantile()
        self._last_q = q
        return q

    def update(self, new_score: float) -> None:
        """Forward the CQR score to the calibration engine for an online update."""
        self._engine.update(new_score)

    def _scan_quantiles(self, scores: NDArray) -> NDArray:
        """Delegate forward scan to the calibration engine."""
        quantiles = self._engine._scan_quantiles(scores)
        self._last_q = float(quantiles[-1]) if len(quantiles) > 0 else None
        return quantiles
