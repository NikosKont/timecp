"""CAFHT and NormMaxCP — joint conformal predictors for multi-step time series.

Two classes are provided:

``NormMaxCP``
    Simple static joint predictor based on ``Max_calibrate`` from
    methods/CAFHT.  Normalises per-horizon residuals by a scale vector σ_h,
    takes the max over horizons as a single nonconformity score per trajectory,
    and computes a single conformal quantile C.  Test-time radii are
    ``C × σ_h`` — fixed, independent of the test trajectory.

``CAFHT``
    Full two-stage Conformal Adaptive Forecaster for Heterogeneous
    Trajectories (Zhou et al. 2024).  Uses ACI or
    PID as an adaptive base method that runs *along the horizon dimension* of
    each trajectory, producing trajectory-specific band sizes.  Two-stage
    calibration selects the best adaptation step-size gamma and a calibrated
    inflation scalar eps_hat.  At test time each trajectory's bands are inflated
    multiplicatively by eps_hat to achieve joint coverage.

    Because the test-time interval for each trajectory depends on the base
    method's adaptive updates (which require observing horizon steps one at a
    time), ``predict_radii`` is not meaningful for CAFHT and raises
    ``NotImplementedError``.  Use ``evaluate`` directly, which has access to
    the full ``(test_point, test_gt)`` arrays.

References
----------
Zhou, Lindemann, and Sesia (2024) "Conformalized Adaptive Forecasting of Heterogeneous Trajectories"
  https://arxiv.org/abs/2402.09623
Reference implementation in methods/CAFHT/ConformalizedTS/methods.py.
"""

from __future__ import annotations

import math
from typing import Literal

from numpy.typing import NDArray

from timecp.base import JointPredictor, conformal_quantile, np
from timecp.methods._scan import aci_scan_batch, pid_scan_batch


# ---------------------------------------------------------------------------
# NormMaxCP (renamed from the original CAFHT placeholder)
# ---------------------------------------------------------------------------


class NormMaxCP(JointPredictor):
    """Joint conformal predictor via normalised maximum nonconformity score.

    Implements the ``Max_calibrate`` baseline from CAFHT.  Normalises
    per-horizon residuals by σ_h, takes max over horizons per trajectory
    as a single scalar score, then computes a single conformal quantile C.
    Test-time radii are the fixed vector ``C × σ_h``.

    Parameters
    ----------
    alpha : float
        Target miscoverage rate.
    normalize : {'mae', 'ones'} or NDArray of shape (H,)
        Per-horizon normalisation:

        * ``'mae'``  — calibration-set MAE per horizon (default).
        * ``'ones'`` — unit scales (no normalisation).
        * NDArray    — precomputed ``(H,)`` scale vector.
    """

    def __init__(
        self,
        alpha: float,
        normalize: str | NDArray = 'mae',
    ) -> None:
        super().__init__(alpha)
        self._normalize_spec = normalize
        self._scales: NDArray | None = None
        self._C: float | None = None

    def fit(self, cal_point: NDArray, cal_gt: NDArray) -> 'NormMaxCP':
        C, N, H = cal_point.shape
        residuals = np.abs(cal_gt - cal_point)  # (C, N, H)

        if isinstance(self._normalize_spec, str):
            if self._normalize_spec == 'mae':
                scales = np.mean(residuals, axis=(0, 1)).astype(np.float64)
                scales = np.where(scales < 1e-8, 1.0, scales)
            elif self._normalize_spec == 'ones':
                scales = np.ones(H, dtype=np.float64)
            else:
                raise ValueError(
                    f"Unknown normalize spec '{self._normalize_spec}'. "
                    "Use 'mae', 'ones', or a (H,) array."
                )
        else:
            scales = np.asarray(self._normalize_spec, dtype=np.float64)
            if scales.shape != (H,):
                raise ValueError(
                    f'normalize array must have shape ({H},), got {scales.shape}'
                )
            scales = np.where(scales < 1e-8, 1.0, scales)

        self._scales = scales

        normalised = residuals / scales[None, None, :]  # (C, N, H)
        scores = np.max(normalised, axis=2)  # (C, N)
        self._C = conformal_quantile(scores.reshape(-1), self.alpha)
        self._fitted = True
        return self

    def predict_radii(self, point_preds: NDArray) -> NDArray:
        if not self._fitted or self._scales is None or self._C is None:
            raise RuntimeError('Call fit() before predict_radii()')
        if math.isinf(self._C):
            return np.full(len(self._scales), math.inf)
        return self._C * self._scales


# ---------------------------------------------------------------------------
# Full CAFHT
# ---------------------------------------------------------------------------


class CAFHT(JointPredictor):
    """Conformal Adaptive Forecaster for Heterogeneous Trajectories.

    Two-stage calibration procedure:

    1. **Gamma selection** (first ``cal_split`` fraction of calibration
       windows): run the base method (ACI or PID) along H horizon steps for
       each trajectory, compute scaled violation scores, grid-search over
       ``gamma_grid`` to find ``gamma_hat`` that minimises mean band width.

    2. **Eps calibration** (remaining ``1 − cal_split`` fraction): re-run the
       base method with ``gamma_hat``, compute per-trajectory nonconformity
       scores ``s_i = max_h( max(0, |e_h| − q_h) / (2 · q_h) )``, take the
       finite-sample corrected ``(1−α)`` quantile → ``eps_hat``.

    3. **Test time**: run the base method on each test trajectory, inflate
       each band by ``(1 + eps_hat) × band_size_h`` multiplicatively.

    The coverage guarantee is::

        Prob( y_h ∈ [(1+eps_hat)·interval_h]  ∀ h=1…H ) ≥ 1 − α

    under exchangeability of calibration trajectories.

    Parameters
    ----------
    alpha : float
        Target miscoverage rate.
    base_model : {'aci', 'pid'}
        Adaptive base method.
    gamma_grid : sequence of float or None
        Step-size candidates for gamma selection.  Defaults to
        ``[0.001, 0.005, 0.01, 0.05, 0.1, 0.5]``.
    cal_split : float
        Fraction of calibration windows used for gamma selection (stage 1).
        The remaining ``1 − cal_split`` are used for eps calibration (stage 2).
        Default 0.5.
    q0 : float
        Initial threshold fed to the base method at ``h=0``.  Default 0.1.
    """

    def __init__(
        self,
        alpha: float,
        base_model: Literal['aci', 'pid'] = 'aci',
        gamma_grid: list[float] | None = None,
        cal_split: float = 0.5,
        q0: float = 0.1,
    ) -> None:
        super().__init__(alpha)
        if base_model not in ('aci', 'pid'):
            raise ValueError(f"base_model must be 'aci' or 'pid', got {base_model!r}")
        if not 0.0 < cal_split < 1.0:
            raise ValueError(f'cal_split must be in (0,1), got {cal_split}')
        self.base_model = base_model
        self.gamma_grid = gamma_grid or [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
        self.cal_split = cal_split
        self.q0 = q0

        # Set after fit():
        self._gamma_hat: float | None = None
        self._eps_hat: float | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _base_run(self, scores: NDArray, gamma: float) -> tuple[NDArray, NDArray]:
        """Run the base method on a single trajectory's per-horizon scores.

        Parameters
        ----------
        scores : (H,) float array — per-horizon nonconformity scores.
        gamma  : float — step-size / learning-rate.

        Returns
        -------
        qs : (H,) — threshold at each horizon step (before observing score_h).
        bands : (H,) — same as qs (symmetric intervals, half-width = threshold).
        """
        # Use batch runner with C=1, then squeeze
        scores_2d = scores[None, :]  # (1, H)
        qs = self._base_run_batch(scores_2d, gamma)[0]  # (H,)
        return qs, qs

    def _base_run_batch(self, mean_residuals: NDArray, gamma: float) -> NDArray:
        """Run the base method on all C trajectories in parallel.

        Parameters
        ----------
        mean_residuals : (C, H) — per-trajectory per-horizon mean residuals.
        gamma          : float  — step-size / learning-rate.

        Returns
        -------
        qs : (C, H) — per-trajectory per-horizon thresholds.
        """
        if self.base_model == 'aci':
            return aci_scan_batch(mean_residuals, self.alpha, gamma, self.q0)
        else:
            return pid_scan_batch(mean_residuals, self.alpha, gamma, self.q0)

    # ------------------------------------------------------------------
    # JointPredictor interface
    # ------------------------------------------------------------------

    def fit(self, cal_point: NDArray, cal_gt: NDArray) -> 'CAFHT':
        """Two-stage calibration.

        Parameters
        ----------
        cal_point : (C, N, H) — point predictions.
        cal_gt    : (C, N, H) — ground truth.
        """
        C, N, H = cal_point.shape
        c1 = max(1, int(math.floor(self.cal_split * C)))
        c1 = min(c1, C - 1)  # at least one window for stage 2

        stage1_pt = cal_point[:c1]
        stage1_gt = cal_gt[:c1]
        stage2_pt = cal_point[c1:]
        stage2_gt = cal_gt[c1:]

        # Reshape to (c1 * N, H) to treat each series-window as an independent trajectory
        stage1_mean_res = (
            np.abs(stage1_gt - stage1_pt).reshape(c1 * N, H).astype(np.float64)
        )
        stage2_mean_res = (
            np.abs(stage2_gt - stage2_pt).reshape((C - c1) * N, H).astype(np.float64)
        )

        # ------ Stage 1: gamma selection ------
        best_gamma = self.gamma_grid[0]
        best_width = math.inf

        try:
            from concurrent.futures import ThreadPoolExecutor

            def evaluate_gamma(gamma):
                qs = self._base_run_batch(stage1_mean_res, gamma)
                band_size = np.maximum(qs, 1e-8)
                violations = np.maximum(stage1_mean_res - qs, 0.0)
                scores1 = np.max(violations / band_size, axis=1)

                eps = conformal_quantile(scores1, self.alpha)
                if math.isinf(eps):
                    return gamma, math.inf

                width = float(np.mean((1.0 + eps) * 2.0 * qs))
                return gamma, width

            with ThreadPoolExecutor() as executor:
                results = executor.map(evaluate_gamma, self.gamma_grid)

            for gamma, width in results:
                if width < best_width:
                    best_width = width
                    best_gamma = gamma

        except Exception:
            for gamma in self.gamma_grid:
                qs = self._base_run_batch(stage1_mean_res, gamma)
                band_size = np.maximum(qs, 1e-8)
                violations = np.maximum(stage1_mean_res - qs, 0.0)
                scores1 = np.max(violations / band_size, axis=1)

                eps = conformal_quantile(scores1, self.alpha)
                if math.isinf(eps):
                    continue

                width = float(np.mean((1.0 + eps) * 2.0 * qs))
                if width < best_width:
                    best_width = width
                    best_gamma = gamma

        self._gamma_hat = best_gamma

        # ------ Stage 2: eps calibration ------
        qs2 = self._base_run_batch(stage2_mean_res, best_gamma)
        band_size2 = np.maximum(qs2, 1e-8)
        violations2 = np.maximum(stage2_mean_res - qs2, 0.0)
        scores2 = np.max(violations2 / band_size2, axis=1)

        self._eps_hat = conformal_quantile(scores2, self.alpha)

        self._fitted = True
        return self

    def predict_radii(self, point_preds: NDArray) -> NDArray:
        """Not meaningful for CAFHT — intervals are trajectory-specific.

        Use ``evaluate(test_point, test_gt)`` instead, which has access to
        the ground truth needed to run the adaptive base method.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError(
            'CAFHT intervals depend on ground truth at test time (the base '
            'method updates adaptively along the horizon). '
            'Use evaluate(test_point, test_gt) directly.'
        )

    def evaluate(self, test_point: NDArray, test_gt: NDArray) -> dict:
        """Evaluate joint coverage over test windows.

        Parameters
        ----------
        test_point : (T, N, H) — point predictions.
        test_gt    : (T, N, H) — ground truth.

        Returns
        -------
        dict with keys: joint_coverage, avg_width, winkler_score.
        """
        if not self._fitted:
            raise RuntimeError('Call fit() before evaluate()')

        gamma = self._gamma_hat
        eps = self._eps_hat
        if gamma is None or eps is None:
            raise RuntimeError('Internal state missing — call fit() first.')

        T, N, H = test_point.shape
        inf_eps = math.isinf(eps)

        if inf_eps:
            radii = np.full((T * N, H), math.inf, dtype=np.float64)
        else:
            # Batch all T * N test trajectories through the base runner at once.
            test_residuals = (
                np.abs(test_gt - test_point).reshape(T * N, H).astype(np.float64)
            )  # (T * N, H)
            qs = self._base_run_batch(test_residuals, gamma)  # (T * N, H)
            radii = (1.0 + eps) * qs  # (T * N, H)

        radii = radii.reshape(T, N, H)

        # radii: (T, N, H)
        lo = test_point - radii  # (T, N, H)
        hi = test_point + radii  # (T, N, H)
        covered_tnh = (test_gt >= lo) & (test_gt <= hi)  # (T, N, H)

        # Joint coverage: all H steps covered for each (t, n), then average over n, t
        joint_per_tn = covered_tnh.all(axis=2)  # (T, N)
        joint_coverage = float(np.mean(joint_per_tn))

        # Average width
        avg_width = float(np.mean(2.0 * radii)) if not inf_eps else math.inf

        # Winkler score — fully vectorised: (T, N, H)
        band = 2.0 * radii  # (T, N, H)
        miss = np.maximum(lo - test_gt, test_gt - hi)  # (T, N, H) ≥0 outside
        miss = np.maximum(miss, 0.0)
        winkler_tnh = band + (2.0 / self.alpha) * miss  # (T, N, H)
        winkler_score = float(np.mean(winkler_tnh)) if not inf_eps else math.inf

        return {
            'joint_coverage': joint_coverage,
            'avg_width': avg_width,
            'winkler_score': winkler_score,
        }
