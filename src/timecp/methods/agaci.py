"""
Aggregated Adaptive Conformal Inference (AgACI).

Runs K independent ACI experts, one for each candidate step-size γ_k.
A mixture weight over the experts is updated via an online learning rule
(FTRL / AdaHedge-style), and the aggregate α_t is the weighted average of
expert α_t values.

At each step::

    α_t     = Σ_k w_k · α_t^k          (mixture prediction)
    q̂_t    = Quantile(window; 1 − α_t)
    err_t   = 1[score_t > q̂_t]
    expert losses → update weights
    α_{t+1}^k = α_t^k + γ_k · (α − err_t^k)

References
----------
Zaffran et al. (2022) "Adaptive Conformal Predictions for Time Series"
  https://arxiv.org/abs/2202.07282

Adapted from ``methods/CPTC/algos/agaci.py`` for the online, stateful API.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Sequence

from numba import njit, prange
from numpy.typing import NDArray

from timecp.base import ConformalPredictor, np
from timecp.methods._scan import _quantile, _rebuild_deque


class AgACI(ConformalPredictor):
    """
    Aggregated Adaptive Conformal Inference.

    Maintains an ensemble of K ACI experts with different step-sizes
    ``gammas``.  The aggregate α_t is a convex combination of expert α_t
    values, with weights updated by AdaHedge to down-weight poorly
    performing experts.

    Parameters
    ----------
    alpha : float
        Target miscoverage rate in (0, 1).
    gammas : sequence of float
        Candidate step-sizes for the K experts.  A log-spaced grid such as
        ``[0.001, 0.005, 0.01, 0.05, 0.1]`` works well in practice.
    window_size : int or None
        Rolling score window used to compute each expert's quantile.
        Default 100.
    eps : float
        Small constant for numerical stability in weight updates.
    asymmetric : bool
        If True, track separate lower and upper quantiles using signed errors.
        Default False (symmetric intervals).
    """

    def __init__(
        self,
        alpha: float,
        gammas: Sequence[float] = (0.001, 0.005, 0.01, 0.05, 0.1),
        window_size: int | None = 100,
        eps: float = 1e-3,
        asymmetric: bool = False,
        conformal_correction: bool = True,
    ) -> None:
        super().__init__(
            alpha, asymmetric=asymmetric, conformal_correction=conformal_correction
        )
        self.gammas = np.asarray(gammas, dtype=float)
        self.window_size = window_size
        self.eps = eps
        self.K = len(self.gammas)

        # Expert state — initialised in fit()
        self._expert_alpha: NDArray = np.full(self.K, alpha)
        self._expert_probs: NDArray = np.full(self.K, 1.0 / self.K)
        self._expert_sq_losses: NDArray = np.zeros(self.K)
        self._expert_max_losses: NDArray = np.zeros(self.K)
        self._expert_etas: NDArray = np.zeros(self.K)
        self._expert_l_values: NDArray = np.zeros(self.K)
        self._last_alpha_t: float = alpha
        self._scores: deque[float] = deque(maxlen=window_size)

    def clone(self) -> ConformalPredictor:
        new = super().clone()
        new._expert_alpha = self._expert_alpha.copy()
        new._expert_probs = self._expert_probs.copy()
        new._expert_sq_losses = self._expert_sq_losses.copy()
        new._expert_max_losses = self._expert_max_losses.copy()
        new._expert_etas = self._expert_etas.copy()
        new._expert_l_values = self._expert_l_values.copy()
        new._last_alpha_t = self.alpha
        return new

    # ------------------------------------------------------------------

    def fit(self, cal_scores: NDArray) -> 'AgACI':
        self._scores = self._init_window(cal_scores, self.window_size)

        # Reset expert state
        self._expert_alpha[:] = self.alpha
        self._expert_probs[:] = 1.0 / self.K
        self._expert_sq_losses[:] = 0.0
        self._expert_max_losses[:] = 0.0
        self._expert_etas[:] = 0.0
        self._expert_l_values[:] = 0.0
        self._last_alpha_t = self.alpha
        return self

    def predict_quantile(self) -> float:
        # Aggregate α_t = weighted sum of expert α values
        alpha_t = float(
            np.clip(np.dot(self._expert_probs, self._expert_alpha), 0.0, 1.0)
        )
        q = _quantile(
            np.asarray(self._scores, dtype=float),
            alpha_t,
            conformal_correction=self.conformal_correction,
        )
        self._last_q = q
        self._last_alpha_t = alpha_t
        return q

    def _update_experts(
        self,
        new_score: float,
        alpha: float,
        expert_alpha: NDArray,
        expert_probs: NDArray,
        expert_sq_losses: NDArray,
        expert_max_losses: NDArray,
        expert_etas: NDArray,
        expert_l_values: NDArray,
        last_alpha_t: float,
        q_t: float,
        sign: float = 1.0,
    ) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
        err_t = 1.0 if sign * new_score > q_t else 0.0
        expert_losses = (err_t - alpha) * (expert_alpha - last_alpha_t)

        expert_sq_losses += expert_losses**2
        expert_max_losses = np.maximum(expert_max_losses, np.abs(expert_losses))

        e_vals = 2.0 ** (np.ceil(np.log2(np.abs(expert_max_losses) + self.eps)) + 1)
        expert_l_values += 0.5 * (
            expert_losses * (1.0 + expert_etas * expert_losses)
            + e_vals * (expert_etas * expert_losses > 0.5)
        )
        expert_etas = np.minimum(
            1.0 / e_vals,
            np.sqrt(np.log(self.K) / np.maximum(expert_sq_losses, self.eps)),
        )
        max_val = np.max(expert_etas * expert_l_values)
        expert_weights = expert_etas * np.exp(-expert_etas * expert_l_values + max_val)
        expert_probs = expert_weights / (expert_weights.sum() + 1e-300)

        scores_arr = np.asarray(self._scores, dtype=float)
        n = len(scores_arr)
        if n == 0:
            per_expert_err = np.zeros(self.K)
        else:
            sorted_scores = np.sort(sign * scores_arr)
            expert_levels = np.clip(1.0 - expert_alpha, 0.0, 1.0)
            min_samples = np.ceil(1.0 / np.maximum(expert_alpha, 1e-9)).astype(int)

            float_indices = expert_levels * (n - 1)
            indices = np.minimum(np.ceil(float_indices).astype(int), n - 1)
            expert_qs = sorted_scores[indices].astype(float)
            expert_qs = np.where(n < min_samples, np.inf, expert_qs)
            per_expert_err = (sign * new_score > expert_qs).astype(float)

        expert_alpha = np.clip(
            expert_alpha + self.gammas * (alpha - per_expert_err), 0.0, 1.0
        )

        return (
            expert_alpha,
            expert_probs,
            expert_sq_losses,
            expert_max_losses,
            expert_etas,
            expert_l_values,
        )

    def update(self, new_score: float) -> None:
        """
        Update expert α values, mixture weights, and the score window.
        """
        alpha_t = self._last_alpha_t
        q_t = self._last_q if self._last_q is not None else math.inf

        (
            self._expert_alpha,
            self._expert_probs,
            self._expert_sq_losses,
            self._expert_max_losses,
            self._expert_etas,
            self._expert_l_values,
        ) = self._update_experts(
            new_score,
            self.alpha,
            self._expert_alpha,
            self._expert_probs,
            self._expert_sq_losses,
            self._expert_max_losses,
            self._expert_etas,
            self._expert_l_values,
            alpha_t,
            q_t,
            sign=1.0,
        )

        self._scores.append(new_score)


@njit(cache=True)
def _agaci_loop_jit(
    scores: NDArray,
    alpha: float,
    gammas: NDArray,
    conformal_correction: bool,
    window_size: int,
    expert_alpha: NDArray,
    expert_probs: NDArray,
    expert_sq_losses: NDArray,
    expert_max_losses: NDArray,
    expert_etas: NDArray,
    expert_l_values: NDArray,
    full_buf: NDArray,
    buf_len: int,
    eps: float,
) -> tuple[
    NDArray,  # quantiles
    NDArray,  # expert_alpha
    NDArray,  # expert_probs
    NDArray,  # expert_sq_losses
    NDArray,  # expert_max_losses
    NDArray,  # expert_etas
    NDArray,  # expert_l_values
]:
    T = len(scores)
    quantiles = np.empty(T, dtype=np.float64)
    K = len(gammas)
    max_buf = window_size if window_size >= 0 else (buf_len + T)

    for t in range(T):
        # Aggregate alpha_t
        alpha_t = 0.0
        for i in range(K):
            alpha_t += expert_probs[i] * expert_alpha[i]
        if alpha_t < 0.0:
            alpha_t = 0.0
        elif alpha_t > 1.0:
            alpha_t = 1.0

        total = buf_len + t
        start = max(total - max_buf, 0) if window_size >= 0 else 0
        window_slice = full_buf[start:total]
        q = _quantile(
            window_slice,
            alpha_t,
            conformal_correction=conformal_correction,
        )

        quantiles[t] = q
        score_t = scores[t]

        # Aggregate error
        err_t = 1.0 if score_t > q else 0.0

        # AdaHedge weight update
        e_vals = np.empty(K, dtype=np.float64)
        for i in range(K):
            expert_loss = (err_t - alpha) * (expert_alpha[i] - alpha_t)
            expert_sq_losses[i] += expert_loss**2
            abs_loss = abs(expert_loss)
            if abs_loss > expert_max_losses[i]:
                expert_max_losses[i] = abs_loss

            e_vals[i] = 2.0 ** (math.ceil(math.log2(expert_max_losses[i] + eps)) + 1)

            if expert_etas[i] * expert_loss > 0.5:
                l_term = expert_loss * (1.0 + expert_etas[i] * expert_loss) + e_vals[i]
            else:
                l_term = expert_loss * (1.0 + expert_etas[i] * expert_loss)
            expert_l_values[i] += 0.5 * l_term

            denom = expert_sq_losses[i]
            if denom < eps:
                denom = eps

            eta_val = np.sqrt(math.log(K) / denom)
            if 1.0 / e_vals[i] < eta_val:
                expert_etas[i] = 1.0 / e_vals[i]
            else:
                expert_etas[i] = eta_val

        # Compute weights
        expert_weights = np.empty(K, dtype=np.float64)
        max_val = -1e300
        for i in range(K):
            w = expert_etas[i] * expert_l_values[i]
            val = -w
            if val > max_val:
                max_val = val

        sum_weights = 0.0
        for i in range(K):
            val = expert_etas[i] * math.exp(
                -expert_etas[i] * expert_l_values[i] - max_val
            )
            expert_weights[i] = val
            sum_weights += val

        for i in range(K):
            expert_probs[i] = expert_weights[i] / (sum_weights + 1e-300)

        # Per-expert ACI alpha update
        n = total - start
        if n > 0:
            sorted_scores = np.sort(window_slice)
            for i in range(K):
                lvl = 1.0 - expert_alpha[i]
                if lvl < 0.0:
                    lvl = 0.0
                elif lvl > 1.0:
                    lvl = 1.0
                idx = int(math.ceil(lvl * (n - 1)))
                if idx > n - 1:
                    idx = n - 1
                elif idx < 0:
                    idx = 0

                min_samples = int(math.ceil(1.0 / max(expert_alpha[i], 1e-9)))
                if n < min_samples:
                    per_expert_err_i = 0.0
                else:
                    per_expert_err_i = 1.0 if score_t > sorted_scores[idx] else 0.0

                new_alpha = expert_alpha[i] + gammas[i] * (alpha - per_expert_err_i)
                if new_alpha < 0.0:
                    new_alpha = 0.0
                elif new_alpha > 1.0:
                    new_alpha = 1.0
                expert_alpha[i] = new_alpha
        else:
            for i in range(K):
                new_alpha = expert_alpha[i] + gammas[i] * alpha
                if new_alpha < 0.0:
                    new_alpha = 0.0
                elif new_alpha > 1.0:
                    new_alpha = 1.0
                expert_alpha[i] = new_alpha

        full_buf[buf_len + t] = score_t

    return (
        quantiles,
        expert_alpha,
        expert_probs,
        expert_sq_losses,
        expert_max_losses,
        expert_etas,
        expert_l_values,
    )

    def batch_evaluate_series(
        self,
        cal_scores: NDArray,
        test_scores: NDArray,
    ) -> NDArray:
        w_size = self.window_size if self.window_size is not None else -1
        return _agaci_batch_loop_jit(
            cal_scores,
            test_scores,
            self.alpha,
            self.gammas,
            self.conformal_correction,
            w_size,
            self.eps,
        )

    def _scan_quantiles(self, scores: NDArray) -> NDArray:
        scores = np.asarray(scores, dtype=float)
        T = len(scores)

        # Copy mutable state
        expert_alpha = self._expert_alpha.copy()
        expert_probs = self._expert_probs.copy()
        expert_sq_losses = self._expert_sq_losses.copy()
        expert_max_losses = self._expert_max_losses.copy()
        expert_etas = self._expert_etas.copy()
        expert_l_values = self._expert_l_values.copy()

        # Score buffer
        score_buf = np.array(self._scores, dtype=float)
        buf_len = len(score_buf)
        window_size_val = self.window_size if self.window_size is not None else -1
        full_buf = np.empty(buf_len + T, dtype=float)
        if buf_len > 0:
            full_buf[:buf_len] = score_buf

        (
            quantiles,
            expert_alpha,
            expert_probs,
            expert_sq_losses,
            expert_max_losses,
            expert_etas,
            expert_l_values,
        ) = _agaci_loop_jit(
            scores,
            self.alpha,
            self.gammas,
            self.conformal_correction,
            window_size_val,
            expert_alpha,
            expert_probs,
            expert_sq_losses,
            expert_max_losses,
            expert_etas,
            expert_l_values,
            full_buf,
            buf_len,
            self.eps,
        )

        # Update internal state
        self._expert_alpha = expert_alpha
        self._expert_probs = expert_probs
        self._expert_sq_losses = expert_sq_losses
        self._expert_max_losses = expert_max_losses
        self._expert_etas = expert_etas
        self._expert_l_values = expert_l_values
        self._last_alpha_t = float(np.dot(expert_probs, expert_alpha))
        self._last_q = quantiles[-1] if T > 0 else self._last_q
        self._scores = _rebuild_deque(
            score_buf if buf_len > 0 else None, scores, self.window_size
        )
        return quantiles


@njit(parallel=True, cache=True)
def _agaci_batch_loop_jit(
    cal_scores: NDArray,
    test_scores: NDArray,
    alpha: float,
    gammas: NDArray,
    conformal_correction: bool,
    window_size: int,
    eps: float,
) -> NDArray:
    C, N = cal_scores.shape
    T = test_scores.shape[0]
    quantiles = np.empty((T, N), dtype=np.float64)
    K = len(gammas)

    for n in prange(N):
        expert_alpha = np.full(K, alpha)
        expert_probs = np.full(K, 1.0 / K)
        expert_sq_losses = np.zeros(K)
        expert_max_losses = np.zeros(K)
        expert_etas = np.zeros(K)
        expert_l_values = np.zeros(K)

        buf_len = C
        full_buf = np.empty(buf_len + T, dtype=np.float64)
        for i in range(C):
            full_buf[i] = cal_scores[i, n]

        max_buf = window_size if window_size >= 0 else (buf_len + T)

        for t in range(T):
            alpha_t = 0.0
            for i in range(K):
                alpha_t += expert_probs[i] * expert_alpha[i]
            if alpha_t < 0.0:
                alpha_t = 0.0
            elif alpha_t > 1.0:
                alpha_t = 1.0

            total = buf_len + t
            start = max(total - max_buf, 0) if window_size >= 0 else 0
            window_slice = full_buf[start:total]
            q = _quantile(
                window_slice,
                alpha_t,
                conformal_correction=conformal_correction,
            )

            quantiles[t, n] = q
            score_t = test_scores[t, n]

            err_t = 1.0 if score_t > q else 0.0

            e_vals = np.empty(K, dtype=np.float64)
            for i in range(K):
                expert_loss = (err_t - alpha) * (expert_alpha[i] - alpha_t)
                expert_sq_losses[i] += expert_loss**2
                abs_loss = abs(expert_loss)
                if abs_loss > expert_max_losses[i]:
                    expert_max_losses[i] = abs_loss

                e_vals[i] = 2.0 ** (
                    math.ceil(math.log2(expert_max_losses[i] + eps)) + 1
                )

                if expert_etas[i] * expert_loss > 0.5:
                    l_term = (
                        expert_loss * (1.0 + expert_etas[i] * expert_loss) + e_vals[i]
                    )
                else:
                    l_term = expert_loss * (1.0 + expert_etas[i] * expert_loss)
                expert_l_values[i] += 0.5 * l_term

                denom = expert_sq_losses[i]
                if denom < eps:
                    denom = eps

                eta_val = np.sqrt(math.log(K) / denom)
                if 1.0 / e_vals[i] < eta_val:
                    expert_etas[i] = 1.0 / e_vals[i]
                else:
                    expert_etas[i] = eta_val

            expert_weights = np.empty(K, dtype=np.float64)
            max_val = -1e300
            for i in range(K):
                w = expert_etas[i] * expert_l_values[i]
                val = -w
                if val > max_val:
                    max_val = val

            sum_weights = 0.0
            for i in range(K):
                val = expert_etas[i] * math.exp(
                    -expert_etas[i] * expert_l_values[i] - max_val
                )
                expert_weights[i] = val
                sum_weights += val

            for i in range(K):
                expert_probs[i] = expert_weights[i] / (sum_weights + 1e-300)

            num_samples = total - start
            if num_samples > 0:
                sorted_scores = np.sort(window_slice)
                for i in range(K):
                    lvl = 1.0 - expert_alpha[i]
                    if lvl < 0.0:
                        lvl = 0.0
                    elif lvl > 1.0:
                        lvl = 1.0
                    idx = int(math.ceil(lvl * (num_samples - 1)))
                    if idx > num_samples - 1:
                        idx = num_samples - 1
                    elif idx < 0:
                        idx = 0

                    min_samples = int(math.ceil(1.0 / max(expert_alpha[i], 1e-9)))
                    if num_samples < min_samples:
                        per_expert_err_i = 0.0
                    else:
                        per_expert_err_i = 1.0 if score_t > sorted_scores[idx] else 0.0

                    new_alpha = expert_alpha[i] + gammas[i] * (alpha - per_expert_err_i)
                    if new_alpha < 0.0:
                        new_alpha = 0.0
                    elif new_alpha > 1.0:
                        new_alpha = 1.0
                    expert_alpha[i] = new_alpha
            else:
                for i in range(K):
                    new_alpha = expert_alpha[i] + gammas[i] * alpha
                    if new_alpha < 0.0:
                        new_alpha = 0.0
                    elif new_alpha > 1.0:
                        new_alpha = 1.0
                    expert_alpha[i] = new_alpha

            full_buf[buf_len + t] = score_t

    return quantiles
