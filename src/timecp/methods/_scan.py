"""Shared scan primitives for conformal prediction methods.

These functions implement the core forward-scan logic used by both the
marginal per-horizon methods (ACI, PID) and the joint CAFHT predictor.
Consolidating them here eliminates code duplication and ensures algorithmic
consistency.

Two variants are provided for each method:

- **Single-stream** (``aci_scan``, ``pid_scan``): operates on a 1-D score
  array with an optional initial score window (calibration history).  Used
  by the marginal ``ACI._scan_quantiles`` and ``QuantileIntegrator._scan_quantiles``.

- **Batch** (``aci_scan_batch``, ``pid_scan_batch``): operates on a 2-D
  ``(C, H)`` array — C independent trajectories of length H, no initial
  calibration window (starts from ``q0``).  Used by CAFHT's internal
  ``_base_run_batch``.

- **Asymmetric single-stream** (``aci_scan_asymmetric``,
  ``pid_scan_asymmetric``): like the single-stream variants but operate on
  **signed** errors ``e = y − ŷ`` and maintain two independent tail trackers
  that each target miscoverage ``alpha/2``.  Return a ``(T, 2)`` array where
  column 0 is ``q_lower`` and column 1 is ``q_upper``.

Internal helpers
----------------
``_saturation_log``    — logarithmically saturated integrator (PID formula).
``_init_scan_buffer``  — allocate and pre-fill the rolling score buffer.
``_rebuild_deque``     — reconstruct a rolling deque after a batch scan.
``_quantile``          — empirical quantile with optional conformal correction and minimum-sample guard.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
from numba import njit
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


@njit
def _saturation_log(x: float, t: int, C_sat: float, KI: float) -> float:
    """Logarithmically saturated integrator term (PID formula).

    Returns  KI · tan( x · log(t+1) / (C_sat · (t+1)) ).
    Saturates to ±∞ when the argument exceeds ±π/2.

    This is the formula used by the conformal-PID controller
    (Angelopoulos et al. 2024).  A separate variant with ``log(t)/t``
    is defined locally in ``acmcp.py``.
    """
    if KI == 0.0:
        return 0.0
    arg = x * math.log(t + 1) / (C_sat * (t + 1))
    if arg >= math.pi / 2.0:
        return math.inf
    if arg <= -math.pi / 2.0:
        return -math.inf
    return KI * math.tan(arg)


@njit
def _init_scan_buffer(
    initial_window: NDArray | None,
    T: int,
    window_size: int | None,
) -> tuple[NDArray, int, int]:
    """Allocate and pre-fill the rolling score buffer for a forward scan.

    Parameters
    ----------
    initial_window : (W,) array or None
        Calibration scores to pre-load into the buffer.
    T : int
        Number of test steps.
    window_size : int or None
        Maximum buffer length (None = unbounded).

    Returns
    -------
    full_buf : (buf_start + T,) float64 array — first ``buf_start`` entries
        contain ``initial_window``; positions ``buf_start:`` are uninitialised.
    buf_start : int — number of pre-loaded elements.
    max_buf : int — effective window capacity.
    """
    if initial_window is not None:
        iw = np.asarray(initial_window, dtype=np.float64)
        buf_start = len(iw)
        full_buf = np.empty(buf_start + T, dtype=np.float64)
        full_buf[:buf_start] = iw
    else:
        buf_start = 0
        full_buf = np.empty(T, dtype=np.float64)
    max_buf = window_size if window_size is not None else (buf_start + T)
    return full_buf, buf_start, max_buf


def _rebuild_deque(
    initial_window: NDArray | None,
    new_scores: NDArray,
    window_size: int | None,
) -> deque:
    """Reconstruct a rolling deque after a batch scan.

    Concatenates ``initial_window`` (if any) with ``new_scores``, then
    returns the last ``window_size`` elements as a bounded deque.

    Parameters
    ----------
    initial_window : (W,) array or None — pre-scan calibration history.
    new_scores : (T,) array — scores processed during the scan.
    window_size : int or None — deque maxlen (None = unbounded).

    Returns
    -------
    deque[float] with ``maxlen=window_size``.
    """
    ns = np.asarray(new_scores, dtype=np.float64)
    T = len(ns)
    if initial_window is not None:
        iw = np.asarray(initial_window, dtype=np.float64)
        buf_len = len(iw)
        if ns.ndim == 2:
            full_buf = np.empty((buf_len + T, ns.shape[1]), dtype=np.float64)
        else:
            full_buf = np.empty(buf_len + T, dtype=np.float64)
        full_buf[:buf_len] = iw
        full_buf[buf_len:] = ns
    else:
        buf_len = 0
        full_buf = ns
    total = buf_len + T
    max_buf = window_size if window_size is not None else total
    start = max(total - max_buf, 0)
    return deque(full_buf[start:total].tolist(), maxlen=window_size)


@njit
def _quantile(
    arr: NDArray,
    alpha_t: float,
    conformal_correction: bool = True,
    min_guard: bool = False,
) -> float:
    """Empirical quantile at level ``1 - alpha_t`` with optional conformal correction and minimum-sample guard.

    Returns ``math.inf`` when ``arr`` is empty or has fewer finite samples than
    required to place the ``(1 - alpha_t)`` quantile reliably
    (i.e. ``n < ceil(1 / alpha_t)``).

    Non-finite values (NaN, ±inf) are filtered before computing the quantile.

    Parameters
    ----------
    arr : array of scores (may be empty, may contain NaN/inf).
    alpha_t : effective miscoverage level in (0, 1].

    Returns
    -------
    float quantile, or ``math.inf`` if there are too few finite samples.
    """
    n = len(arr)
    if n == 0:
        return math.inf

    # Filter non-finite values (NaN, inf, -inf)
    count = 0
    for i in range(n):
        if math.isfinite(arr[i]):
            count += 1
    if count == 0:
        return math.inf

    if count < n:
        finite_arr = np.empty(count, dtype=np.float64)
        j = 0
        for i in range(n):
            if math.isfinite(arr[i]):
                finite_arr[j] = arr[i]
                j += 1
        arr = finite_arr

    n = count

    if conformal_correction:
        level = math.ceil((n + 1) * (1.0 - alpha_t)) / n
        if level > 1.0:
            level = 1.0
        elif level < 0.0:
            level = 0.0
        idx = int(math.ceil(level * (n - 1)))
    else:
        if min_guard:
            min_n = math.ceil(1.0 / max(alpha_t, 1e-9))
            if n < min_n:
                return math.inf

        p = 1.0 - alpha_t
        if p < 0.0:
            p = 0.0
        elif p > 1.0:
            p = 1.0

        idx = int(math.ceil(p * (n - 1)))

    idx = max(0, min(idx, n - 1))
    return np.sort(arr)[idx]


# ---------------------------------------------------------------------------
# ACI scan
# ---------------------------------------------------------------------------


@njit
def aci_scan(
    scores: NDArray,
    alpha: float,
    gamma: float,
    alpha_t_init: float | None = None,
    initial_window: NDArray | None = None,
    window_size: int | None = None,
    min_guard: bool = False,
) -> tuple[NDArray, float]:
    """Forward scan computing ACI quantiles over a 1-D score stream.

    Parameters
    ----------
    scores : (T,) array of test nonconformity scores.
    alpha : float target miscoverage rate.
    gamma : float ACI step size.
    alpha_t_init : initial adaptive alpha (defaults to alpha).
    initial_window : optional (W,) array of calibration scores pre-loaded
        into the rolling window.
    window_size : max window length (None = unbounded).

    Returns
    -------
    quantiles : (T,) array of predicted quantiles.
    alpha_t_final : float, the final alpha_t value after all updates.
    """
    scores = np.asarray(scores, dtype=np.float64)
    T = len(scores)
    quantiles = np.empty(T, dtype=np.float64)

    alpha_t = alpha_t_init if alpha_t_init is not None else alpha

    full_buf, buf_start, max_buf = _init_scan_buffer(initial_window, T, window_size)

    for t in range(T):
        total = buf_start + t
        start = max(total - max_buf, 0) if window_size is not None else 0
        q = _quantile(
            full_buf[start:total],
            alpha_t,
            conformal_correction=True,
            min_guard=min_guard,
        )
        quantiles[t] = q

        score_t = float(scores[t])
        if not math.isfinite(score_t):
            continue
        err = 1.0 if score_t > q else 0.0
        alpha_t = max(0.0, min(1.0, alpha_t + gamma * (alpha - err)))
        full_buf[buf_start + t] = score_t

    return quantiles, alpha_t


@njit
def aci_scan_batch(
    scores: NDArray,
    alpha: float,
    gamma: float,
    q0: float,
) -> NDArray:
    """Vectorised ACI scan over C independent trajectories of length H.

    Each trajectory starts fresh (no calibration window); the initial quantile
    is ``q0`` until enough scores accumulate.

    Parameters
    ----------
    scores : (C, H) array — C trajectories, each of H steps.
    alpha : float target miscoverage rate.
    gamma : float ACI step size.
    q0 : float initial quantile for steps where insufficient history.

    Returns
    -------
    qs : (C, H) array of per-trajectory per-horizon thresholds.
    """
    scores = np.asarray(scores, dtype=np.float64)
    C, H = scores.shape

    qs = np.full((C, H), q0, dtype=np.float64)
    alpha_t = np.full(C, alpha, dtype=np.float64)
    sorted_buf = np.zeros((C, H), dtype=np.float64)
    q = np.full(C, q0, dtype=np.float64)
    min_n = max(1, math.ceil(1.0 / max(alpha, 1e-9)))

    for h in range(H):
        s_h = scores[:, h]

        if h > 0:
            n = h
            for c in range(C):
                level = min(max(1.0 - alpha_t[c], 0.0), 1.0)
                idx_c = min(max(int(math.ceil(n * level)) - 1, 0), n - 1)
                window_q_c = sorted_buf[c, idx_c]
                if n < min_n:
                    q[c] = q0
                else:
                    q[c] = window_q_c
        else:
            for c in range(C):
                q[c] = q0

        qs[:, h] = q

        # Insert s_h into sorted_buf
        if h < H:
            for c in range(C):
                # Count how many elements in sorted_buf[:h] are less than s_h[c]
                p = 0
                for j in range(h):
                    if sorted_buf[c, j] < s_h[c]:
                        p += 1
                # Shift elements right to make room
                for j in range(h, p, -1):
                    sorted_buf[c, j] = sorted_buf[c, j - 1]
                sorted_buf[c, p] = s_h[c]

        # ACI alpha update
        for c in range(C):
            err_c = 1.0 if s_h[c] > q[c] else 0.0
            alpha_t[c] = min(max(alpha_t[c] + gamma * (alpha - err_c), 0.0), 1.0)

    return qs


# ---------------------------------------------------------------------------
# ACI asymmetric scan
# ---------------------------------------------------------------------------


@njit
def aci_scan_asymmetric(
    signed_errors: NDArray,
    alpha: float,
    gamma: float,
    alpha_t_lower_init: float | None = None,
    alpha_t_upper_init: float | None = None,
    initial_window: NDArray | None = None,
    window_size: int | None = None,
    min_guard: bool = False,
) -> tuple[NDArray, float, float]:
    """Forward scan computing asymmetric ACI quantile pairs over signed errors.

    Maintains two independent alpha_t trackers, one per tail, each targeting
    ``alpha / 2`` miscoverage.

    * Upper tail: ``err_upper = 1`` if ``e > q_upper``.
    * Lower tail: ``err_lower = 1`` if ``-e > q_lower`` (i.e. ``e < -q_lower``).

    The window stores **signed** errors; for the upper quantile the window is
    used directly, and for the lower quantile the negated window is used.

    Parameters
    ----------
    signed_errors : (T,) array of signed errors e = y − ŷ.
    alpha : float total target miscoverage rate; each tail targets alpha/2.
    gamma : float ACI step size.
    alpha_t_lower_init : initial lower alpha_t (defaults to alpha/2).
    alpha_t_upper_init : initial upper alpha_t (defaults to alpha/2).
    initial_window : optional (W,) array of initial signed errors.
    window_size : max window length (None = unbounded).

    Returns
    -------
    pairs : (T, 2) array — column 0 is q_lower, column 1 is q_upper.
    alpha_t_lower_final : float
    alpha_t_upper_final : float
    """
    signed_errors = np.asarray(signed_errors, dtype=np.float64)
    T = len(signed_errors)
    pairs = np.empty((T, 2), dtype=np.float64)

    half_alpha = alpha / 2.0
    alpha_t_lo = alpha_t_lower_init if alpha_t_lower_init is not None else half_alpha
    alpha_t_up = alpha_t_upper_init if alpha_t_upper_init is not None else half_alpha

    full_buf, buf_start, max_buf = _init_scan_buffer(initial_window, T, window_size)

    for t in range(T):
        total = buf_start + t
        start = max(total - max_buf, 0) if window_size is not None else 0
        window_slice = full_buf[start:total]

        q_up = _quantile(
            window_slice,
            alpha_t_up,
            conformal_correction=True,
            min_guard=min_guard,
        )
        q_lo = _quantile(
            -window_slice,
            alpha_t_lo,
            conformal_correction=True,
            min_guard=min_guard,
        )

        pairs[t, 0] = q_lo
        pairs[t, 1] = q_up

        e_t = float(signed_errors[t])
        if not math.isfinite(e_t):
            continue
        err_up = 1.0 if e_t > q_up else 0.0
        err_lo = 1.0 if -e_t > q_lo else 0.0
        alpha_t_up = max(0.0, min(1.0, alpha_t_up + gamma * (half_alpha - err_up)))
        alpha_t_lo = max(0.0, min(1.0, alpha_t_lo + gamma * (half_alpha - err_lo)))
        full_buf[buf_start + t] = e_t

    return pairs, alpha_t_lo, alpha_t_up


# ---------------------------------------------------------------------------
# SplitCP / TrailingWindow scan
# ---------------------------------------------------------------------------


@njit
def split_scan(
    scores: NDArray,
    alpha: float,
    initial_window: NDArray | None = None,
    window_size: int | None = None,
    conformal_correction: bool = True,
) -> NDArray:
    """Forward scan computing SplitCP (rolling) quantiles.

    Parameters
    ----------
    scores : (T,) test scores.
    alpha : target miscoverage rate.
    initial_window : optional (W,) calibration scores.
    window_size : max window length.
    conformal_correction: whether to use the finite sample correction.

    Returns
    -------
    quantiles : (T,) array of predicted quantiles.
    """
    scores = np.asarray(scores, dtype=np.float64)
    T = len(scores)
    quantiles = np.empty(T, dtype=np.float64)

    full_buf, buf_start, max_buf = _init_scan_buffer(initial_window, T, window_size)

    for t in range(T):
        total = buf_start + t
        start = max(total - max_buf, 0) if window_size is not None else 0
        window_slice = full_buf[start:total]
        q = _quantile(window_slice, alpha, conformal_correction=conformal_correction)
        quantiles[t] = q
        score_t = float(scores[t])
        if math.isfinite(score_t):
            full_buf[buf_start + t] = score_t

    return quantiles


@njit
def split_scan_asymmetric(
    signed_errors: NDArray,
    alpha: float,
    initial_window: NDArray | None = None,
    window_size: int | None = None,
    conformal_correction: bool = True,
) -> NDArray:
    """Forward scan computing asymmetric SplitCP (rolling) quantiles."""
    signed_errors = np.asarray(signed_errors, dtype=np.float64)
    T = len(signed_errors)
    pairs = np.empty((T, 2), dtype=np.float64)

    half_alpha = alpha / 2.0
    full_buf, buf_start, max_buf = _init_scan_buffer(initial_window, T, window_size)

    for t in range(T):
        total = buf_start + t
        start = max(total - max_buf, 0) if window_size is not None else 0
        window_slice = full_buf[start:total]

        q_up = _quantile(
            window_slice, half_alpha, conformal_correction=conformal_correction
        )
        q_lo = _quantile(
            -window_slice, half_alpha, conformal_correction=conformal_correction
        )

        pairs[t, 0] = q_lo
        pairs[t, 1] = q_up

        e_t = float(signed_errors[t])
        if math.isfinite(e_t):
            full_buf[buf_start + t] = e_t

    return pairs


# ---------------------------------------------------------------------------
# PID scan (proportional-only mode)
# ---------------------------------------------------------------------------


@njit
def pid_scan(
    scores: NDArray,
    alpha: float,
    lr: float,
    qt_init: float = 0.0,
    proportional_lr: bool = True,
    KI: float = 0.0,
    C_sat: float = 1.0,
    initial_window: NDArray | None = None,
    window_size: int | None = None,
) -> tuple[NDArray, float, float, float, int]:
    """Forward scan computing PID quantiles over a 1-D score stream.

    Parameters
    ----------
    scores : (T,) test scores.
    alpha : target miscoverage rate.
    lr : base learning rate.
    qt_init : initial value of the proportional accumulator.
    proportional_lr : whether to scale lr by score range.
    KI : integrator gain (0 = proportional only).
    C_sat : saturation constant for the integrator.
    initial_window : optional (W,) calibration scores for the recent-score buffer.
    window_size : max window length.

    Returns
    -------
    quantiles : (T,) array of predicted quantiles.
    qt_final : final proportional accumulator value.
    q_final : final output quantile.
    cum_err_final : cumulative error count.
    t_final : total steps processed.
    """
    scores = np.asarray(scores, dtype=np.float64)
    T = len(scores)
    quantiles = np.empty(T, dtype=np.float64)

    # Recover from a non-finite qt_init (e.g. insufficient calibration data
    # caused _safe_quantile to return inf).  Use the max of the finite
    # calibration scores as a finite fallback so the controller can adapt.
    qt = qt_init
    if not math.isfinite(qt):
        if initial_window is not None:
            iw = np.asarray(initial_window, dtype=np.float64)
            best = -math.inf
            for k in range(len(iw)):
                if math.isfinite(iw[k]) and iw[k] > best:
                    best = iw[k]
            qt = best if math.isfinite(best) else 0.0
        else:
            qt = 0.0
    q = qt  # output = qt + integrator
    cum_err = 0.0
    t_step = 0

    full_buf, buf_start, max_buf = _init_scan_buffer(initial_window, T, window_size)

    for i in range(T):
        quantiles[i] = q

        score_i = float(scores[i])

        # Skip non-finite test scores: don't update state or buffer
        if not math.isfinite(score_i):
            continue

        covered = score_i <= q
        err = 0.0 if covered else 1.0

        # Proportional learning rate
        total = buf_start + i
        start = max(total - max_buf, 0) if window_size is not None else 0
        n_recent = total - start
        if proportional_lr and n_recent > 0:
            window_slice = full_buf[start:total]
            score_range = float(window_slice.max() - window_slice.min())
            lr_t = lr * score_range
        else:
            lr_t = lr

        grad = alpha if covered else -(1.0 - alpha)

        # Integrator
        integrator_arg = cum_err - t_step * alpha
        integrator = _saturation_log(integrator_arg, t_step, C_sat, KI)
        if not math.isfinite(integrator):
            integrator = 0.0

        qt = qt - lr_t * grad
        q = qt + integrator

        cum_err += err
        t_step += 1
        full_buf[buf_start + i] = score_i

    return quantiles, qt, q, cum_err, t_step


@njit
def pid_scan_batch(
    scores: NDArray,
    alpha: float,
    gamma: float,
    q0: float,
) -> NDArray:
    """Vectorised PID (proportional-only) scan over C independent trajectories.

    Each trajectory starts from ``q0`` with no calibration history.

    Parameters
    ----------
    scores : (C, H) array of per-trajectory per-horizon scores.
    alpha : target miscoverage rate.
    gamma : base learning rate.
    q0 : initial quantile value.

    Returns
    -------
    qs : (C, H) array of per-trajectory per-horizon thresholds.
    """
    scores = np.asarray(scores, dtype=np.float64)
    C, H = scores.shape

    qs = np.zeros((C, H), dtype=np.float64)
    qt = np.full(C, q0, dtype=np.float64)
    q = qt.copy()
    score_buf = np.zeros((C, H), dtype=np.float64)

    for h in range(H):
        qs[:, h] = np.maximum(q, 0.0)

        s_h = scores[:, h]
        covered = s_h <= q
        grad = np.where(covered, alpha, -(1.0 - alpha))

        if h > 1:
            lr_t = np.empty(C, dtype=np.float64)
            for c in range(C):
                row = score_buf[c, :h]
                score_range = row.max() - row.min()
                lr_t[c] = gamma * score_range
        else:
            lr_t = np.full(C, gamma, dtype=np.float64)

        qt = qt - lr_t * grad
        q = np.maximum(qt, 0.0)
        score_buf[:, h] = s_h

    return qs


# ---------------------------------------------------------------------------
# PID asymmetric scan
# ---------------------------------------------------------------------------


@njit
def pid_scan_asymmetric(
    signed_errors: NDArray,
    alpha: float,
    lr: float,
    qt_lower_init: float = 0.0,
    qt_upper_init: float = 0.0,
    proportional_lr: bool = True,
    KI: float = 0.0,
    C_sat: float = 1.0,
    initial_window: NDArray | None = None,
    window_size: int | None = None,
) -> tuple[NDArray, float, float, float, float, float, float, int]:
    """Forward scan for asymmetric PID with two independent tail controllers.

    Each tail targets ``alpha / 2`` miscoverage:

    * Upper controller: ``covered_up = (e <= q_up)``.
    * Lower controller: ``covered_lo = (-e <= q_lo)`` (i.e. ``e >= -q_lo``).

    The proportional learning rate is computed from the range of
    ``|e|`` values in the buffer (shared between both controllers).

    Parameters
    ----------
    signed_errors : (T,) array of signed errors e = y − ŷ.
    alpha : total target miscoverage rate; each tail targets alpha/2.
    lr : base learning rate.
    qt_lower_init : initial proportional accumulator for lower tail.
    qt_upper_init : initial proportional accumulator for upper tail.
    proportional_lr : whether to scale lr by |e| range.
    KI : integrator gain (0 = proportional only).
    C_sat : saturation constant for the integrator.
    initial_window : optional (W,) array of initial signed errors.
    window_size : max window length.

    Returns
    -------
    pairs : (T, 2) array — column 0 is q_lower, column 1 is q_upper.
    qt_lower_final : float
    qt_upper_final : float
    q_lower_final : float
    q_upper_final : float
    cum_err_lower_final : float
    cum_err_upper_final : float
    t_final : int
    """
    signed_errors = np.asarray(signed_errors, dtype=np.float64)
    T = len(signed_errors)
    pairs = np.empty((T, 2), dtype=np.float64)

    half_alpha = alpha / 2.0

    # Recover from non-finite qt_init values (insufficient calibration data)
    qt_lo = qt_lower_init
    qt_up = qt_upper_init
    if not math.isfinite(qt_lo) or not math.isfinite(qt_up):
        if initial_window is not None:
            iw = np.asarray(initial_window, dtype=np.float64)
            best = -math.inf
            for k in range(len(iw)):
                if math.isfinite(iw[k]) and iw[k] > best:
                    best = iw[k]
            fallback = best if math.isfinite(best) else 0.0
        else:
            fallback = 0.0
        if not math.isfinite(qt_lo):
            qt_lo = fallback
        if not math.isfinite(qt_up):
            qt_up = fallback
    q_lo = qt_lo
    q_up = qt_up
    cum_err_lo = 0.0
    cum_err_up = 0.0
    t_step = 0

    full_buf, buf_start, max_buf = _init_scan_buffer(initial_window, T, window_size)

    for i in range(T):
        pairs[i, 0] = q_lo
        pairs[i, 1] = q_up

        e_i = float(signed_errors[i])

        # Skip non-finite test scores: don't update state or buffer
        if not math.isfinite(e_i):
            continue

        # Coverage checks
        covered_up = e_i <= q_up
        covered_lo = -e_i <= q_lo  # lower bound: e >= -q_lo

        err_up = 0.0 if covered_up else 1.0
        err_lo = 0.0 if covered_lo else 1.0

        # Proportional learning rate from |e| range in buffer
        total = buf_start + i
        start = max(total - max_buf, 0) if window_size is not None else 0
        n_recent = total - start
        if proportional_lr and n_recent > 0:
            window_slice = full_buf[start:total]
            abs_range = float(np.abs(window_slice).max() - np.abs(window_slice).min())
            lr_t = lr * abs_range
        else:
            lr_t = lr

        grad_up = half_alpha if covered_up else -(1.0 - half_alpha)
        grad_lo = half_alpha if covered_lo else -(1.0 - half_alpha)

        # Integrators (use past cumulative error, before current step)
        integ_up = 0.0
        integ_lo = 0.0
        if KI != 0.0 and t_step > 0:
            iu = _saturation_log(cum_err_up - t_step * half_alpha, t_step, C_sat, KI)
            il = _saturation_log(cum_err_lo - t_step * half_alpha, t_step, C_sat, KI)
            integ_up = iu if math.isfinite(iu) else 0.0
            integ_lo = il if math.isfinite(il) else 0.0

        qt_up = qt_up - lr_t * grad_up
        qt_lo = qt_lo - lr_t * grad_lo
        q_up = qt_up + integ_up
        q_lo = qt_lo + integ_lo

        cum_err_up += err_up
        cum_err_lo += err_lo
        t_step += 1
        full_buf[buf_start + i] = e_i

    return pairs, qt_lo, qt_up, q_lo, q_up, cum_err_lo, cum_err_up, t_step
