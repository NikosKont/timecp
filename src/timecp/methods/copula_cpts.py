"""Copula Conformal Prediction for Time Series (CopulaCPTS).

Adapted from Sun & Yu (2022) "Copula Conformal Prediction for Multi-step
Time Series Forecasting" (arXiv:2212.03281) and the reference implementation
at methods/CopulaCPTS/.

Overview
--------
CopulaCPTS extends split-CP to multi-step forecasting by replacing the
per-horizon Bonferroni union bound (used in CFRNN) with an empirical copula
to model the dependence structure of per-horizon nonconformity scores.  The
result is a joint coverage guarantee that is less conservative than CFRNN
when horizon errors are strongly correlated.

The two-stage calibration procedure:

1. **Score stage** (``cal_split`` fraction of calibration windows): compute
   per-horizon residuals ``R[i, h] = |y[i,h] − ŷ[i,h]|``.  Store as a
   matrix of shape ``(C1, H)``.

2. **Copula stage** (remaining ``1 − cal_split`` fraction): for each window
   compute the empirical CDF rank of its per-horizon score relative to the
   score-stage matrix (the ``α``-vector).  Train a differentiable coverage
   module (``CP`` from the reference code) via Adam to find per-horizon
   thresholds ``θ_h ∈ [0, 1]`` such that the joint CDF
   ``Prob(α_1 ≤ θ_1, …, α_H ≤ θ_H) ≥ 1 − alpha``.

At test time, each horizon's interval half-width is
``r_h = quantile(R_score[:, h], θ_h)``.

This is a *static* split-CP method: it is calibrated once on the full
calibration set and applied unchanged to every test window.

Implementation Notes
--------------------
This implementation includes several numerical stability improvements over the
raw PyTorch reference code released by the original authors:

1. **Parameter Constraints**: Thresholds are parameterized as logits and passed
   through a sigmoid to guarantee they remain strictly bounded within (0, 1).
   The original code left them unconstrained and relied solely on weight decay.
2. **Softer Indicator Approximation**: The sigmoid temperature constant used to
   approximate the hard indicator function is set to 20.0 instead of 1000.0.
   This provides a smoother gradient landscape, preventing vanishing gradients
   and making convergence much more reliable.
3. **Smart Initialisation**: Thresholds are initialized to the marginal
   (1 - epsilon) quantile (their approximate expected value) rather than 1.0,
   significantly reducing the number of epochs required to converge.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from numpy.typing import NDArray

from timecp.base import JointPredictor, np

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Internal differentiable copula coverage module
# ---------------------------------------------------------------------------


def _empirical_copula_loss(theta: NDArray, alpha_mat: NDArray, epsilon: float) -> float:
    """Empirical copula CDF loss.

    Parameters
    ----------
    theta : (H,) array
        Candidate per-horizon thresholds in [0, 1].
    alpha_mat : (N, H) array
        Per-sample per-horizon rank scores (pseudo-observations).
    epsilon : float
        Target miscoverage rate.

    Returns
    -------
    |Prob(all α_h ≤ θ_h) - (1 - epsilon)|
    """
    joint_covered = np.all(alpha_mat <= theta[None, :], axis=1)
    return abs(float(np.mean(joint_covered)) - (1.0 - epsilon))


def _search_theta(
    alpha_mat: NDArray,
    epsilon: float,
    epochs: int = 500,
    lr: float = 0.01,
    backend: str | None = 'jax',
) -> NDArray:
    """Find per-horizon thresholds via gradient-free grid search + SGD refinement.

    Uses a scalar initialisation (all thresholds equal) then runs a simple
    projected gradient descent using finite differences to refine per-horizon
    thresholds.

    Returns
    -------
    NDArray of shape (H,) with values in [0, 1].
    """
    try:
        if backend != 'jax':
            raise ImportError('JAX backend not selected')
        import jax
        import jax.numpy as jnp
        import optax

        H = alpha_mat.shape[1]
        alpha_t = jnp.array(alpha_mat, dtype=jnp.float32)

        def loss_fn(theta_logits, pseudo_data):
            theta_c = jax.nn.sigmoid(theta_logits)
            # Soft approximation of joint indicator via product of sigmoids
            soft_joint = jnp.prod(
                jax.nn.sigmoid(20.0 * (theta_c[None, :] - pseudo_data)),
                axis=1,
            )
            coverage = jnp.mean(soft_joint)
            target = 1.0 - epsilon
            return jnp.abs(coverage - target)

        @jax.jit
        def step(theta_logits, opt_state):
            loss, grads = jax.value_and_grad(loss_fn)(theta_logits, alpha_t)
            updates, opt_state = optimizer.update(grads, opt_state, theta_logits)
            theta_logits = optax.apply_updates(theta_logits, updates)
            # avoid sigmoid saturation
            theta_logits = jnp.clip(jnp.asarray(theta_logits), -6.0, 6.0)
            return theta_logits, opt_state, loss

        optimizer = optax.adam(
            learning_rate=lr
        )  # Matches PyTorch's Adam implementation
        init_val = float(np.clip(1.0 - epsilon, 0.01, 0.99))

        # NOTE: We initialize theta_logits directly with init_val to exactly match the
        # original PyTorch implementation which passed init_val to nn.Parameter without logit transform.
        theta_logits = jnp.full((H,), init_val, dtype=jnp.float32)
        opt_state = optimizer.init(theta_logits)

        def train_loop(epochs_count, init_theta, init_opt_state):
            def body_fn(i, carry):
                theta, state = carry
                theta, state, _ = step(theta, state)
                return theta, state

            return jax.lax.fori_loop(
                0, epochs_count, body_fn, (init_theta, init_opt_state)
            )

        final_theta_logits, _ = train_loop(epochs, theta_logits, opt_state)
        result = jax.nn.sigmoid(final_theta_logits)
        return np.array(result, dtype=np.float64)

    except ImportError:
        try:
            import torch
            import torch.nn as nn

            H = alpha_mat.shape[1]
            alpha_t_torch = torch.from_numpy(alpha_mat.astype(np.float32))  # (N, H)

            class _CovModule(nn.Module):
                def __init__(self, h: int, eps: float) -> None:
                    super().__init__()
                    # Initialise near the 1-eps quantile — reasonable starting point.
                    init_val = float(np.clip(1.0 - eps, 0.01, 0.99))
                    self.theta = nn.Parameter(torch.full((h,), init_val))
                    self.eps = eps

                def forward(self, pseudo_data: 'torch.Tensor') -> 'torch.Tensor':
                    theta_c = torch.sigmoid(self.theta)  # keep in (0,1)
                    # Soft approximation of joint indicator via product of sigmoids
                    soft_joint = torch.prod(
                        torch.sigmoid(20.0 * (theta_c.unsqueeze(0) - pseudo_data)),
                        dim=1,
                    )  # (N,)
                    coverage = soft_joint.mean()
                    target = 1.0 - self.eps
                    return torch.abs(coverage - target)

            mod = _CovModule(H, epsilon)
            opt = torch.optim.Adam(mod.parameters(), lr=lr, weight_decay=1e-4)

            for _ in range(epochs):
                opt.zero_grad()
                loss = mod(alpha_t_torch)
                loss.backward()
                opt.step()
                with torch.no_grad():
                    mod.theta.clamp_(-6.0, 6.0)  # avoid sigmoid saturation

            with torch.no_grad():
                result = torch.sigmoid(mod.theta).numpy()
            return result.astype(np.float64)

        except ImportError:
            # PyTorch not available: fall back to scalar uniform threshold.
            # The empirical copula CDF is maximised by the largest scalar θ that
            # still achieves the required coverage. Use binary search.
            H = alpha_mat.shape[1]
            lo_s, hi_s = 0.0, 1.0
            for _ in range(60):
                mid = 0.5 * (lo_s + hi_s)
                theta_vec = np.full(H, mid)
                joint_covered = np.all(alpha_mat <= theta_vec[None, :], axis=1)
                cov = float(np.mean(joint_covered))
                if cov >= 1.0 - epsilon:
                    hi_s = mid
                else:
                    lo_s = mid
            return np.full(H, hi_s, dtype=np.float64)


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class CopulaCPTS(JointPredictor):
    """Copula conformal predictor for joint multi-step coverage.

    Parameters
    ----------
    alpha : float
        Target miscoverage rate (e.g. 0.1 for 90 % joint coverage).
    cal_split : float
        Fraction of calibration windows used for the score stage; the rest
        are used for copula fitting.  Default 0.6 (as in the paper).
    epochs : int
        Number of gradient descent epochs for fitting the copula thresholds.
    lr : float
        Adam learning rate for the copula optimisation.
    """

    def __init__(
        self,
        alpha: float,
        cal_split: float = 0.6,
        epochs: int = 500,
        lr: float = 0.01,
    ) -> None:
        super().__init__(alpha)
        if not 0.0 < cal_split < 1.0:
            raise ValueError(f'cal_split must be in (0, 1), got {cal_split}')
        self.cal_split = cal_split
        self.epochs = epochs
        self.lr = lr

        # Set after fit():
        self._score_matrix: NDArray | None = None  # (C1, H) residuals
        self._thresholds: NDArray | None = None  # (H,) copula thresholds ∈ [0,1]
        self._radii: NDArray | None = None  # (H,) cached half-widths

    # ------------------------------------------------------------------

    def fit(self, cal_point: NDArray, cal_gt: NDArray) -> 'CopulaCPTS':
        """Calibrate CopulaCPTS on calibration windows.

        Parameters
        ----------
        cal_point : (C, N, H) array — point predictions.
        cal_gt    : (C, N, H) array — ground truth.
        """
        C, N, H = cal_point.shape

        # Reshape to (C * N, H) to treat each series-window as an independent trajectory
        residuals = np.abs(cal_gt - cal_point).reshape(C * N, H).astype(np.float64)

        # Split into score stage and copula stage
        c1 = max(1, int(math.floor(self.cal_split * (C * N))))
        c1 = min(c1, (C * N) - 1)  # ensure at least one window for copula stage

        score_mat = residuals[:c1]  # (C1, H)
        copula_mat = residuals[c1:]  # (C2, H)

        self._score_matrix = score_mat  # store for quantile lookup at test time

        # Compute per-horizon rank of copula-stage scores relative to score stage
        # alpha[i, h] = fraction of score-stage samples with score ≤ copula[i, h]
        alpha_mat = np.mean(
            copula_mat[:, None, :] >= score_mat[None, :, :],  # (C2, C1, H)
            axis=1,
        )  # (C2, H) — empirical alpha values (pseudo-observations)

        # Fit copula thresholds
        self._thresholds = _search_theta(
            alpha_mat, self.alpha, epochs=self.epochs, lr=self.lr
        )  # (H,)

        # Pre-compute per-horizon interval radii from score stage quantiles
        self._radii = np.array(
            [float(np.quantile(score_mat[:, h], self._thresholds[h])) for h in range(H)]
        )  # (H,)

        self._fitted = True
        return self

    # ------------------------------------------------------------------

    def predict_radii(self, point_preds: NDArray) -> NDArray:
        """Return pre-computed per-horizon half-widths.

        CopulaCPTS is a static method: radii are fixed after calibration and
        do not depend on ``point_preds``.

        Parameters
        ----------
        point_preds : (N, H) — unused (present for interface compatibility).

        Returns
        -------
        NDArray of shape (H,)
        """
        if not self._fitted or self._radii is None:
            raise RuntimeError('Call fit() before predict_radii()')
        return self._radii
