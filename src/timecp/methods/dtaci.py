"""
Distribution-free Time-series Adaptive Conformal Inference (DtACI)

Reference
---------
Gibbs & Candes (2022) "Conformal Inference for Online Prediction with Arbitrary Distribution Shifts"
https://arxiv.org/abs/2208.08401
"""

from __future__ import annotations

import math
from collections import deque

from numba import njit, prange
from numpy.typing import NDArray

from timecp.base import ConformalPredictor, np
from timecp.methods._scan import _quantile, _rebuild_deque


class DtACI(ConformalPredictor):
    """
    Distribution-free Time-series Adaptive Conformal Inference (DtACI).

    Maintains an ensemble of ACI experts with different learning rates (gammas).
    Each expert maintains its own nominal error rate (expert_alpha). The final
    quantile is sampled from the experts according to exponential weights, which
    are updated based on the pinball loss of the nominal error rates compared
    to the empirical tail probability of the observed scores.

    Parameters
    ----------
    alpha : float
        Target miscoverage rate in (0, 1).
    gammas : list of float, optional
        Candidate learning rates for the experts. Default [0.001, 0.01, 0.1].
    sigma : float, optional
        Mixing parameter for expert weights. Default 1e-3.
    eta : float, optional
        Learning rate parameter for exponential weights. Default 2.72.
    eta_adapt : bool, optional
        Whether to adapt eta parameter online. Default False.
    eta_lookback : int, optional
        Lookback window for adaptive eta. Default 500.
    window_size : int or None, optional
        Rolling score window used to compute each expert's quantile. Default 100.
    seed : int or None, optional
        Random seed for reproducibility. Default None.
    asymmetric : bool, optional
        If True, track separate lower and upper quantiles using signed errors.
        Default False (symmetric intervals).
    """

    def __init__(
        self,
        alpha: float,
        gammas: list[float] | None = None,
        sigma: float = 1e-3,
        eta: float = 2.72,
        eta_adapt: bool = False,
        eta_lookback: int = 500,
        window_size: int | None = 100,
        seed: int | None = None,
        asymmetric: bool = False,
        conformal_correction: bool = True,
    ) -> None:
        super().__init__(
            alpha, conformal_correction=conformal_correction, asymmetric=asymmetric
        )
        if gammas is None:
            gammas = [0.001, 0.01, 0.1]
        self.gammas = np.array(gammas, dtype=float)
        self.sigma = float(sigma)
        self.eta_init = float(eta)
        self.eta_adapt = bool(eta_adapt)
        self.eta_lookback = int(eta_lookback)
        self.window_size = window_size
        self._rng = np.random.default_rng(seed)
        self.k = len(self.gammas)

        self.eta = self.eta_init
        self._expert_alphas = np.full(self.k, alpha)
        self._expert_ws = np.ones(self.k)
        self._expert_probs = np.full(self.k, 1.0 / self.k)
        self._cur_expert = 0
        self._expert_cumulative_losses = np.zeros(self.k)
        self._loss_window: deque[float] = deque(maxlen=self.eta_lookback)

        self._scores: deque[float] = deque(maxlen=window_size)

    def clone(self) -> ConformalPredictor:
        import copy

        new = super().clone()
        new._rng = copy.deepcopy(self._rng)
        new.eta = self.eta_init
        new._expert_alphas = self._expert_alphas.copy()
        new._expert_ws = self._expert_ws.copy()
        new._expert_probs = self._expert_probs.copy()
        new._cur_expert = 0
        new._expert_cumulative_losses = self._expert_cumulative_losses.copy()
        new._loss_window = deque(maxlen=self.eta_lookback)
        return new

    def fit(self, cal_scores: NDArray) -> 'DtACI':
        self._scores = self._init_window(cal_scores, self.window_size)

        self._expert_alphas = np.full(self.k, self.alpha)
        self._expert_ws = np.ones(self.k)
        self._expert_probs = np.full(self.k, 1.0 / self.k)
        self._cur_expert = self._rng.choice(self.k, p=self._expert_probs)
        self._expert_cumulative_losses = np.zeros(self.k)
        self._loss_window = deque(maxlen=self.eta_lookback)
        self.eta = self.eta_init

        return self

    def predict_quantile(self) -> float:
        expert_alpha = float(self._expert_alphas[self._cur_expert])
        q = _quantile(
            np.asarray(self._scores, dtype=float),
            expert_alpha,
            conformal_correction=self.conformal_correction,
        )
        self._last_q = q
        return q

    def _pinball_loss(self, u: NDArray, alpha: float) -> NDArray:
        return alpha * u - np.minimum(u, 0)

    def _update_experts(
        self,
        new_score: float,
        alpha: float,
        expert_alphas: NDArray,
        expert_ws: NDArray,
        expert_probs: NDArray,
        expert_cumulative_losses: NDArray,
        loss_window: deque,
        eta: float,
        sign: float = 1.0,
    ) -> tuple[NDArray, NDArray, NDArray, int, NDArray, float]:
        scores_arr = np.asarray(self._scores, dtype=float)
        n = len(scores_arr)
        if n == 0:
            beta_t = 1.0  # Conservative default if no history
        else:
            beta_t = float(np.mean(sign * scores_arr > sign * new_score))

        # DtACI evaluates experts by comparing the target error alpha to the empirical error beta_t
        expert_losses = self._pinball_loss(beta_t - expert_alphas, alpha)

        ensemble_loss = float(np.sum(expert_losses * expert_probs))
        loss_window.append(ensemble_loss)

        if self.eta_adapt and len(loss_window) == self.eta_lookback:
            lookback_losses = np.array(loss_window)
            eta = float(
                np.sqrt(
                    (np.log(2 * self.k * self.eta_lookback) + 1)
                    / max(np.sum(lookback_losses**2), 1e-12)
                )
            )

        indicator = (expert_alphas > beta_t).astype(float)
        expert_alphas = np.clip(
            expert_alphas + self.gammas * (alpha - indicator), 0.0, 1.0
        )

        if eta < np.inf:
            log_weights = np.log(expert_ws + 1e-300) - eta * expert_losses
            log_weights_stable = log_weights - np.max(log_weights)
            expert_bar_ws = np.exp(log_weights_stable)

            expert_bar_ws_sum = np.sum(expert_bar_ws) + 1e-300
            expert_next_ws = (
                1 - self.sigma
            ) * expert_bar_ws / expert_bar_ws_sum + self.sigma / self.k

            probs = expert_next_ws / (np.sum(expert_next_ws) + 1e-300)
            probs = np.clip(probs, 1e-300, 1.0)
            expert_probs = probs / np.sum(probs)

            cur_expert = self._rng.choice(self.k, p=expert_probs)
            expert_ws = expert_next_ws
        else:
            expert_cumulative_losses += expert_losses
            cur_expert = int(np.argmin(expert_cumulative_losses))

        return (
            expert_alphas,
            expert_ws,
            expert_probs,
            cur_expert,
            expert_cumulative_losses,
            eta,
        )

    def update(self, new_score: float) -> None:
        (
            self._expert_alphas,
            self._expert_ws,
            self._expert_probs,
            self._cur_expert,
            self._expert_cumulative_losses,
            self.eta,
        ) = self._update_experts(
            new_score,
            self.alpha,
            self._expert_alphas,
            self._expert_ws,
            self._expert_probs,
            self._expert_cumulative_losses,
            self._loss_window,
            self.eta,
            sign=1.0,
        )
        self._scores.append(new_score)


@njit(cache=True)
def _dtaci_loop_jit(
    scores: NDArray,
    alpha: float,
    gammas: NDArray,
    sigma: float,
    eta_adapt: bool,
    eta_lookback: int,
    window_size: int,
    conformal_correction: bool,
    # expert state:
    expert_alphas: NDArray,
    expert_ws: NDArray,
    expert_probs: NDArray,
    cur_expert: int,
    expert_cumulative_losses: NDArray,
    eta: float,
    # loss window state:
    loss_window_arr: NDArray,
    loss_window_ptr: int,
    loss_window_count: int,
    # score buffer:
    full_buf: NDArray,
    buf_len: int,
) -> tuple[
    NDArray,  # quantiles
    NDArray,  # expert_alphas
    NDArray,  # expert_ws
    NDArray,  # expert_probs
    int,  # cur_expert
    NDArray,  # expert_cumulative_losses
    float,  # eta
    NDArray,  # loss_window_arr
    int,  # loss_window_ptr
    int,  # loss_window_count
]:
    T = len(scores)
    quantiles = np.empty(T, dtype=np.float64)
    k = len(gammas)
    max_buf = window_size if window_size >= 0 else (buf_len + T)

    for t in range(T):
        total = buf_len + t
        start = max(total - max_buf, 0) if window_size >= 0 else 0
        window_slice = full_buf[start:total]

        expert_alpha = expert_alphas[cur_expert]
        q = _quantile(
            window_slice,
            expert_alpha,
            conformal_correction=conformal_correction,
        )

        quantiles[t] = q
        score_t = scores[t]

        # Compute beta_t
        n = total - start
        if n == 0:
            beta_t = 1.0
        else:
            count_greater = 0
            for i in range(n):
                if window_slice[i] > score_t:
                    count_greater += 1
            beta_t = count_greater / n

        # Pinball loss
        expert_losses = np.empty(k, dtype=np.float64)
        for i in range(k):
            u = beta_t - expert_alphas[i]
            val = alpha * u - min(u, 0.0)
            expert_losses[i] = val

        ensemble_loss = 0.0
        for i in range(k):
            ensemble_loss += expert_losses[i] * expert_probs[i]

        # Update loss window
        loss_window_arr[loss_window_ptr] = ensemble_loss
        loss_window_ptr = (loss_window_ptr + 1) % eta_lookback
        if loss_window_count < eta_lookback:
            loss_window_count += 1

        if eta_adapt and loss_window_count == eta_lookback:
            sq_sum = 0.0
            for i in range(eta_lookback):
                sq_sum += loss_window_arr[i] ** 2
            if sq_sum < 1e-12:
                sq_sum = 1e-12
            eta = np.sqrt((np.log(2 * k * eta_lookback) + 1.0) / sq_sum)

        for i in range(k):
            indicator = 1.0 if expert_alphas[i] > beta_t else 0.0
            new_a = expert_alphas[i] + gammas[i] * (alpha - indicator)
            if new_a < 0.0:
                new_a = 0.0
            elif new_a > 1.0:
                new_a = 1.0
            expert_alphas[i] = new_a

        if eta < np.inf:
            log_weights = np.empty(k, dtype=np.float64)
            max_log_w = -1e300
            for i in range(k):
                val = np.log(expert_ws[i] + 1e-300) - eta * expert_losses[i]
                log_weights[i] = val
                if val > max_log_w:
                    max_log_w = val

            expert_bar_ws = np.empty(k, dtype=np.float64)
            sum_bar_ws = 0.0
            for i in range(k):
                val = math.exp(log_weights[i] - max_log_w)
                expert_bar_ws[i] = val
                sum_bar_ws += val

            sum_bar_ws += 1e-300

            expert_next_ws = np.empty(k, dtype=np.float64)
            sum_next_ws = 0.0
            for i in range(k):
                val = (1.0 - sigma) * expert_bar_ws[i] / sum_bar_ws + sigma / k
                expert_next_ws[i] = val
                sum_next_ws += val

            sum_next_ws += 1e-300

            # Compute probs and sample next expert
            expert_probs = np.empty(k, dtype=np.float64)
            sum_probs = 0.0
            for i in range(k):
                val = expert_next_ws[i] / sum_next_ws
                if val < 1e-300:
                    val = 1e-300
                elif val > 1.0:
                    val = 1.0
                expert_probs[i] = val
                sum_probs += val
            for i in range(k):
                expert_probs[i] /= sum_probs

            # Choice sampler JIT
            r = np.random.random()
            cum = 0.0
            cur_expert = k - 1
            for i in range(k):
                cum += expert_probs[i]
                if r <= cum:
                    cur_expert = i
                    break
            expert_ws = expert_next_ws
        else:
            min_loss = 1e300
            cur_expert = 0
            for i in range(k):
                expert_cumulative_losses[i] += expert_losses[i]
                if expert_cumulative_losses[i] < min_loss:
                    min_loss = expert_cumulative_losses[i]
                    cur_expert = i

        full_buf[total] = score_t

    return (
        quantiles,
        expert_alphas,
        expert_ws,
        expert_probs,
        cur_expert,
        expert_cumulative_losses,
        eta,
        loss_window_arr,
        loss_window_ptr,
        loss_window_count,
    )

    def batch_evaluate_series(
        self,
        cal_scores: NDArray,
        test_scores: NDArray,
    ) -> NDArray:
        w_size = self.window_size if self.window_size is not None else -1
        return _dtaci_batch_loop_jit(
            cal_scores,
            test_scores,
            self.alpha,
            self.gammas,
            self.sigma,
            self.eta_adapt,
            self.eta_lookback,
            w_size,
            self.conformal_correction,
            self.eta_init,
        )

    def _scan_quantiles(self, scores: NDArray) -> NDArray:
        scores = np.asarray(scores, dtype=float)
        T = len(scores)

        # Copy mutable state
        expert_alphas = self._expert_alphas.copy()
        expert_ws = self._expert_ws.copy()
        expert_probs = self._expert_probs.copy()
        cur_expert = self._cur_expert
        expert_cumulative_losses = self._expert_cumulative_losses.copy()
        eta = self.eta

        # Score buffer
        score_buf = np.array(self._scores, dtype=float)
        buf_len = len(score_buf)
        window_size_val = self.window_size if self.window_size is not None else -1
        full_buf = np.empty(buf_len + T, dtype=float)
        if buf_len > 0:
            full_buf[:buf_len] = score_buf

        # Convert loss_window to numpy array
        loss_window_arr = np.zeros(self.eta_lookback, dtype=float)
        loss_window_list = list(self._loss_window)
        loss_window_count = len(loss_window_list)
        for i, val in enumerate(loss_window_list):
            loss_window_arr[i] = val
        loss_window_ptr = loss_window_count % self.eta_lookback

        (
            quantiles,
            expert_alphas,
            expert_ws,
            expert_probs,
            cur_expert,
            expert_cumulative_losses,
            eta,
            loss_window_arr,
            loss_window_ptr,
            loss_window_count,
        ) = _dtaci_loop_jit(
            scores,
            self.alpha,
            self.gammas,
            self.sigma,
            self.eta_adapt,
            self.eta_lookback,
            window_size_val,
            self.conformal_correction,
            expert_alphas,
            expert_ws,
            expert_probs,
            cur_expert,
            expert_cumulative_losses,
            eta,
            loss_window_arr,
            loss_window_ptr,
            loss_window_count,
            full_buf,
            buf_len,
        )

        # Update mutable state back
        self._expert_alphas = expert_alphas
        self._expert_ws = expert_ws
        self._expert_probs = expert_probs
        self._cur_expert = cur_expert
        self._expert_cumulative_losses = expert_cumulative_losses
        self.eta = eta

        # Rebuild loss_window deque
        self._loss_window.clear()
        if loss_window_count >= self.eta_lookback:
            for idx in range(self.eta_lookback):
                self._loss_window.append(
                    float(loss_window_arr[(loss_window_ptr + idx) % self.eta_lookback])
                )
        else:
            for idx in range(loss_window_count):
                self._loss_window.append(float(loss_window_arr[idx]))

        self._last_q = quantiles[-1] if T > 0 else self._last_q
        self._scores = _rebuild_deque(
            score_buf if buf_len > 0 else None, scores, self.window_size
        )
        return quantiles


@njit(parallel=True, cache=True)
def _dtaci_batch_loop_jit(
    cal_scores: NDArray,
    test_scores: NDArray,
    alpha: float,
    gammas: NDArray,
    sigma: float,
    eta_adapt: bool,
    eta_lookback: int,
    window_size: int,
    conformal_correction: bool,
    eta_init: float,
) -> NDArray:
    C, N = cal_scores.shape
    T = test_scores.shape[0]
    quantiles = np.empty((T, N), dtype=np.float64)
    k = len(gammas)

    for n in prange(N):
        expert_alphas = np.full(k, alpha)
        expert_ws = np.ones(k)
        expert_probs = np.full(k, 1.0 / k)

        r = np.random.random()
        cum = 0.0
        cur_expert = k - 1
        for i in range(k):
            cum += expert_probs[i]
            if r <= cum:
                cur_expert = i
                break

        expert_cumulative_losses = np.zeros(k)
        eta = eta_init

        loss_window_arr = np.zeros(eta_lookback, dtype=np.float64)
        loss_window_ptr = 0
        loss_window_count = 0

        buf_len = C
        full_buf = np.empty(buf_len + T, dtype=np.float64)
        for i in range(C):
            full_buf[i] = cal_scores[i, n]

        max_buf = window_size if window_size >= 0 else (buf_len + T)

        for t in range(T):
            total = buf_len + t
            start = max(total - max_buf, 0) if window_size >= 0 else 0
            window_slice = full_buf[start:total]

            expert_alpha = expert_alphas[cur_expert]
            q = _quantile(
                window_slice,
                expert_alpha,
                conformal_correction=conformal_correction,
            )

            quantiles[t, n] = q
            score_t = test_scores[t, n]

            num_samples = total - start
            if num_samples == 0:
                beta_t = 1.0
            else:
                count_greater = 0
                for i in range(num_samples):
                    if window_slice[i] > score_t:
                        count_greater += 1
                beta_t = count_greater / num_samples

            expert_losses = np.empty(k, dtype=np.float64)
            for i in range(k):
                u = beta_t - expert_alphas[i]
                val = alpha * u - min(u, 0.0)
                expert_losses[i] = val

            ensemble_loss = 0.0
            for i in range(k):
                ensemble_loss += expert_losses[i] * expert_probs[i]

            loss_window_arr[loss_window_ptr] = ensemble_loss
            loss_window_ptr = (loss_window_ptr + 1) % eta_lookback
            if loss_window_count < eta_lookback:
                loss_window_count += 1

            if eta_adapt and loss_window_count == eta_lookback:
                sq_sum = 0.0
                for i in range(eta_lookback):
                    sq_sum += loss_window_arr[i] ** 2
                if sq_sum < 1e-12:
                    sq_sum = 1e-12
                eta = np.sqrt((np.log(2 * k * eta_lookback) + 1.0) / sq_sum)

            for i in range(k):
                indicator = 1.0 if expert_alphas[i] > beta_t else 0.0
                new_a = expert_alphas[i] + gammas[i] * (alpha - indicator)
                if new_a < 0.0:
                    new_a = 0.0
                elif new_a > 1.0:
                    new_a = 1.0
                expert_alphas[i] = new_a

            if eta < np.inf:
                log_weights = np.empty(k, dtype=np.float64)
                max_log_w = -1e300
                for i in range(k):
                    val = np.log(expert_ws[i] + 1e-300) - eta * expert_losses[i]
                    log_weights[i] = val
                    if val > max_log_w:
                        max_log_w = val

                expert_bar_ws = np.empty(k, dtype=np.float64)
                sum_bar_ws = 0.0
                for i in range(k):
                    val = math.exp(log_weights[i] - max_log_w)
                    expert_bar_ws[i] = val
                    sum_bar_ws += val

                sum_bar_ws += 1e-300

                expert_next_ws = np.empty(k, dtype=np.float64)
                sum_next_ws = 0.0
                for i in range(k):
                    val = (1.0 - sigma) * expert_bar_ws[i] / sum_bar_ws + sigma / k
                    expert_next_ws[i] = val
                    sum_next_ws += val

                sum_next_ws += 1e-300

                expert_probs = np.empty(k, dtype=np.float64)
                sum_probs = 0.0
                for i in range(k):
                    val = expert_next_ws[i] / sum_next_ws
                    if val < 1e-300:
                        val = 1e-300
                    elif val > 1.0:
                        val = 1.0
                    expert_probs[i] = val
                    sum_probs += val
                for i in range(k):
                    expert_probs[i] /= sum_probs

                r_choice = np.random.random()
                cum_choice = 0.0
                cur_expert = k - 1
                for i in range(k):
                    cum_choice += expert_probs[i]
                    if r_choice <= cum_choice:
                        cur_expert = i
                        break
                expert_ws = expert_next_ws
            else:
                min_loss = 1e300
                cur_expert = 0
                for i in range(k):
                    expert_cumulative_losses[i] += expert_losses[i]
                    if expert_cumulative_losses[i] < min_loss:
                        min_loss = expert_cumulative_losses[i]
                        cur_expert = i

            full_buf[total] = score_t

    return quantiles
