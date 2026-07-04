"""
Conformal PID methods from Angelopoulos et al. (2024).

QuantileIntegrator
    PID-style quantile controller.  The quantile q̂_t is updated by a
    proportional term (gradient on the pinball loss) plus an integral
    term that corrects for cumulative coverage drift::

        q_{t+1} = q_t  −  lr · grad_t  +  integrator_t

    where::

        grad_t      = α  if covered_t  else −(1−α)
        integrator_t = KI · tan_sat(Σ_{s≤t} err_s − t·α,  t,  C_sat)

    Setting ``KI = 0`` gives the pure proportional ("Quantile") variant.

References
----------
Angelopoulos et al. (2024) "Conformal PID Control for Time Series Prediction"
  https://arxiv.org/abs/2307.16895
  ``methods/conformal-PID/core/methods.py``.
"""

import math
from collections import deque
from typing import Any

from numpy.typing import NDArray

from timecp.base import ConformalPredictor, np
from timecp.methods._scan import _quantile, _rebuild_deque, _saturation_log

# ---------------------------------------------------------------------------
# QuantileIntegrator  (conformal PID)
# ---------------------------------------------------------------------------


class QuantileIntegrator(ConformalPredictor):
    """
    Conformal PID quantile controller.

    The threshold q̂_t is updated online by gradient descent on the
    pinball (quantile) loss, augmented with an integral correction term::

        q_{t+1} = q_t  −  lr_t · grad_t  +  I_t

        grad_t  = α          if covered_t,    else  −(1−α)
        lr_t    = lr · range(window)         if proportional_lr else lr
        I_t     = KI · tan_sat( Σerr − t·α,  t,  C_sat )

    A proportional learning rate (``proportional_lr=True``) scales ``lr``
    by the score range in the current window, making it invariant to
    the score magnitude.

    **Asymmetric mode** (``asymmetric=True``):
        Two independent PID controllers track upper and lower quantiles.
        Each controller targets ``alpha/2`` miscoverage:

            covered_upper = (e ≤ q_up),   grad_upper = α/2 if covered else −(1 − α/2)
            covered_lower = (−e ≤ q_lo),  grad_lower = α/2 if covered else −(1 − α/2)

    .. note::
        The integrator clamps infinite values (when ``tan`` saturates at
        ±π/2) to 0.0 as a safety measure.  The reference implementation
        allows infinite integrator values to propagate.

    Parameters
    ----------
    alpha : float
        Target miscoverage rate.
    lr : float
        Base learning rate.  When ``proportional_lr=True`` this is scaled
        by the score range.  Typical values: 0.01–1.0.
    KI : float
        Integral gain.  Set to 0 for the pure proportional (Quantile) method.
        Default 0 (no integrator).
    C_sat : float
        Saturation constant for the integrator.  Controls how aggressively
        the integral term is clipped.  Only relevant when KI > 0.
    window_size : int or None
        Window for computing the proportional learning rate.  Only used
        when ``proportional_lr=True``.  Default 100.
    proportional_lr : bool
        If True, scale ``lr`` by the range of recent scores, making the step
        size invariant to the score magnitude.  Default True.
    q_init : float or None
        Initial quantile value.  If None, the median of the calibration
        scores is used.
    asymmetric : bool
        If True, run two independent PID controllers (one per tail) on signed
        errors.  Default False.

    References
    ----------
    ``methods/conformal-PID/core/methods.py``,
    functions ``quantile`` and ``quantile_integrator_log``.
    """

    def __init__(
        self,
        alpha: float,
        lr: float = 0.1,
        KI: float = 0.0,
        C_sat: float = 1.0,
        window_size: int | None = 100,
        proportional_lr: bool = True,
        q_init: float | None = None,
        asymmetric: bool = False,
        conformal_correction: bool = True,
    ) -> None:
        super().__init__(
            alpha, asymmetric=asymmetric, conformal_correction=conformal_correction
        )
        self.lr = float(lr)
        self.KI = float(KI)
        self.C_sat = float(C_sat)
        self.window_size = window_size
        self.proportional_lr = proportional_lr
        self.q_init = q_init

        # Symmetric state
        self._qt: float = 0.0
        self._q: float = 0.0
        self._t: int = 0
        self._cum_err: float = 0.0
        self._recent_scores: deque[Any] = deque(maxlen=window_size)

    def clone(self) -> ConformalPredictor:
        new = super().clone()
        new._recent_scores = deque(maxlen=self.window_size)
        new._qt = 0.0
        new._q = 0.0
        new._t = 0
        new._cum_err = 0.0
        return new

    # ------------------------------------------------------------------

    def fit(self, cal_scores: NDArray) -> 'QuantileIntegrator':
        cal_scores = np.asarray(cal_scores, dtype=float)
        self._recent_scores = self._init_window(cal_scores, self.window_size)

        if self.q_init is not None:
            self._qt = float(self.q_init)
        elif len(cal_scores) > 0:
            q0 = float(
                _quantile(
                    cal_scores,
                    self.alpha,
                    conformal_correction=self.conformal_correction,
                )
            )
            if not math.isfinite(q0):
                finite_cal = cal_scores[np.isfinite(cal_scores)]
                q0 = float(np.max(finite_cal)) if len(finite_cal) > 0 else 0.0
            self._qt = q0
        else:
            self._qt = 0.0
        self._q = self._qt

        self._t = 0
        self._cum_err = 0.0
        return self

    def predict_quantile(self) -> float:
        q = float(self._q)
        self._last_q = q
        return q

    def update(self, new_score: float) -> None:
        """
        One gradient step on the pinball loss + integral correction.

        Matches the original ``quantile_integrator_log_scorecaster``
        from ``methods/conformal-PID/core/methods.py``:

        1. Observe coverage: ``covered_t = (q̂_t >= score_t)``
        2. Compute gradient: ``grad = α if covered else -(1-α)``
        3. Compute integrator on *past* cumulative error (before current step)
        4. Update: ``qt_{t+1} = qt_t - lr_t * grad``
                   ``q̂_{t+1} = qt_{t+1} + integrator``
        5. Add current error to cumulative count
        """
        q = self._last_q if self._last_q is not None else self._q

        if not math.isfinite(new_score):
            return

        covered = new_score <= q
        err = 0.0 if covered else 1.0

        # Proportional learning rate (range of recent scores)
        if self.proportional_lr and len(self._recent_scores) > 0:
            score_range = max(self._recent_scores) - min(self._recent_scores)
            lr_t = self.lr * score_range
        else:
            lr_t = self.lr

        # Gradient: covered → q shrinks; missed → q grows
        grad = self.alpha if covered else -(1.0 - self.alpha)

        # Integral correction — uses cumulative error BEFORE adding current step
        integrator_arg = self._cum_err - self._t * self.alpha
        integrator = _saturation_log(integrator_arg, self._t, self.C_sat, self.KI)
        if not math.isfinite(integrator):
            integrator = 0.0  # safety clamp

        # Update proportional part and output
        self._qt = self._qt - lr_t * grad
        self._q = self._qt + integrator

        self._cum_err += err
        self._t += 1

        self._recent_scores.append(new_score)

    def _scan_quantiles(self, scores: NDArray) -> NDArray:
        """Vectorized scan for PID quantile controller."""
        from timecp.methods._scan import pid_scan

        scores = np.asarray(scores, dtype=float)
        T = len(scores)
        initial_window = (
            np.array(self._recent_scores, dtype=float) if self._recent_scores else None
        )

        # Symmetric path
        quantiles, qt_final, q_final, cum_err_final, t_final = pid_scan(
            scores,
            alpha=self.alpha,
            lr=self.lr,
            qt_init=self._qt,
            proportional_lr=self.proportional_lr,
            KI=self.KI,
            C_sat=self.C_sat,
            initial_window=initial_window,
            window_size=self.window_size,
        )
        self._qt = qt_final
        self._q = q_final
        self._cum_err = cum_err_final
        self._t = t_final
        self._last_q = quantiles[-1] if T > 0 else self._last_q
        self._recent_scores = _rebuild_deque(initial_window, scores, self.window_size)
        return quantiles
