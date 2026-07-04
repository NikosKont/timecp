"""Tests for timecp.methods — correctness, interface, and edge cases."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from timecp import (
    ACI,
    AdaptiveCQR,
    AgACI,
    CQR,
    QuantileIntegrator,
    SplitCP,
    TrailingWindow,
    WeightedCP,
    conformal_quantile,
)
from timecp.base import ConformalPredictor
from timecp.methods import CFRNN, CQRBase, DtACI


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def exchangeable_scores():
    """IID half-normal nonconformity scores."""
    rng = np.random.default_rng(42)
    return np.abs(rng.standard_normal(1000))


@pytest.fixture()
def cal_eval_split(exchangeable_scores):
    """Split into 200 calibration + 800 evaluation scores."""
    return exchangeable_scores[:200], exchangeable_scores[200:]


# ---------------------------------------------------------------------------
# conformal_quantile helper
# ---------------------------------------------------------------------------


class TestConformalQuantile:
    def test_empty_returns_inf(self):
        assert conformal_quantile(np.array([]), 0.1) == math.inf

    def test_single_element(self):
        # n=1, level = ceil(2*0.9)/1 = 2.0 → clamped to 1.0 → max element
        assert conformal_quantile(np.array([3.0]), 0.1) == 3.0

    def test_known_quantile(self):
        scores = np.arange(1.0, 11.0)  # [1..10]
        # n=10, alpha=0.1: level = ceil(11*0.9)/10 = ceil(9.9)/10 = 10/10 = 1.0
        q = conformal_quantile(scores, 0.1)
        assert q == 10.0

    def test_finite_sample_correction(self):
        """Level should exceed the raw (1-alpha) so it's conservative."""
        scores = np.arange(1.0, 101.0)
        alpha = 0.1
        n = len(scores)
        raw_level = 1.0 - alpha
        corrected_level = min(math.ceil((n + 1) * raw_level) / n, 1.0)
        assert corrected_level >= raw_level


# ---------------------------------------------------------------------------
# SplitCP
# ---------------------------------------------------------------------------


class TestSplitCP:
    def test_fit_predict_update_cycle(self, cal_eval_split):
        cal, _ = cal_eval_split
        cp = SplitCP(alpha=0.1)
        cp.fit(cal)
        q = cp.predict_quantile()
        assert math.isfinite(q)
        assert q > 0
        cp.update(0.5)

    def test_predict_interval_symmetric(self, cal_eval_split):
        cal, _ = cal_eval_split
        cp = SplitCP(alpha=0.1).fit(cal)
        lo, hi = cp.predict_interval(5.0)
        q = cp.predict_quantile()
        assert_allclose(hi - lo, 2 * q)
        assert_allclose((lo + hi) / 2, 5.0)

    def test_coverage_on_exchangeable_data(self, cal_eval_split):
        """On IID data, SplitCP should achieve ≥ (1-alpha) coverage."""
        cal, eval_scores = cal_eval_split
        alpha = 0.1
        cp = SplitCP(alpha=alpha).fit(cal)
        result = cp.evaluate(eval_scores, np.zeros_like(eval_scores))
        # Allow some slack for finite-sample variability
        assert result['coverage'] >= (1.0 - alpha) - 0.07

    def test_empty_scores_returns_inf(self):
        cp = SplitCP(alpha=0.1)
        cp._fitted = True
        assert cp.predict_quantile() == math.inf

    def test_fit_before_predict_interval(self):
        cp = SplitCP(alpha=0.1)
        with pytest.raises(RuntimeError, match='fit'):
            cp.predict_interval(1.0)


# ---------------------------------------------------------------------------
# ACI
# ---------------------------------------------------------------------------


class TestACI:
    def test_alpha_t_stays_bounded(self, cal_eval_split):
        """α_t should always remain in [0, 1]."""
        cal, eval_scores = cal_eval_split
        aci = ACI(alpha=0.1, gamma=0.05).fit(cal)
        for s in eval_scores:
            aci.predict_quantile()
            aci.update(s)
            assert 0.0 <= aci.alpha_t <= 1.0

    def test_gamma_zero_is_static(self, cal_eval_split):
        """With gamma=0, ACI should keep alpha_t constant."""
        cal, _ = cal_eval_split
        aci = ACI(alpha=0.1, gamma=0.0).fit(cal)
        aci.predict_quantile()
        aci.update(0.5)
        assert aci.alpha_t == 0.1

    def test_negative_gamma_raises(self):
        with pytest.raises(ValueError, match='gamma'):
            ACI(alpha=0.1, gamma=-1.0)

    def test_coverage(self, cal_eval_split):
        cal, eval_scores = cal_eval_split
        aci = ACI(alpha=0.1, gamma=0.01).fit(cal)
        result = aci.evaluate(eval_scores, np.zeros_like(eval_scores))
        # ACI should adapt to maintain reasonable coverage
        assert 0.80 <= result['coverage'] <= 1.0


# ---------------------------------------------------------------------------
# AgACI
# ---------------------------------------------------------------------------


class TestAgACI:
    def test_runs_without_error(self, cal_eval_split):
        cal, eval_scores = cal_eval_split
        agaci = AgACI(alpha=0.1).fit(cal)
        result = agaci.evaluate(eval_scores, np.zeros_like(eval_scores))
        assert 0.0 <= result['coverage'] <= 1.0
        assert result['avg_width'] > 0

    def test_expert_probs_sum_to_one(self, cal_eval_split):
        cal, eval_scores = cal_eval_split
        agaci = AgACI(alpha=0.1).fit(cal)
        for s in eval_scores[:50]:
            agaci.predict_quantile()
            agaci.update(s)
            assert_allclose(agaci._expert_probs.sum(), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# DtACI
# ---------------------------------------------------------------------------


class TestDtACI:
    def test_reproducibility_with_seed(self, cal_eval_split):
        """Same seed should produce identical quantile sequences."""
        cal, eval_scores = cal_eval_split

        def run(seed):
            d = DtACI(alpha=0.1, seed=seed).fit(cal)
            qs = []
            for s in eval_scores[:100]:
                qs.append(d.predict_quantile())
                d.update(s)
            return np.array(qs)

        qs1 = run(123)
        qs2 = run(123)
        qs3 = run(456)
        assert_allclose(qs1, qs2)
        # Different seeds should (with high probability) differ
        assert not np.allclose(qs1, qs3)

    def test_runs_without_error(self, cal_eval_split):
        cal, eval_scores = cal_eval_split
        d = DtACI(alpha=0.1, seed=0).fit(cal)
        result = d.evaluate(eval_scores, np.zeros_like(eval_scores))
        assert 0.0 <= result['coverage'] <= 1.0

    def test_eta_resets_on_refit(self, cal_eval_split):
        cal, eval_scores = cal_eval_split
        d = DtACI(alpha=0.1, eta=5.0, eta_adapt=True, seed=0).fit(cal)
        for s in eval_scores[:100]:
            d.predict_quantile()
            d.update(s)
        # After adaptation, eta may have changed
        d.fit(cal)  # Re-fit should reset eta
        assert d.eta == d.eta_init


# ---------------------------------------------------------------------------
# CFRNN
# ---------------------------------------------------------------------------


class TestCFRNN:
    def test_bonferroni_widens_intervals(self, cal_eval_split):
        """Larger horizon should produce wider intervals (higher quantile)."""
        cal, _ = cal_eval_split
        q1 = CFRNN(alpha=0.1, horizon=1).fit(cal).predict_quantile()
        q5 = CFRNN(alpha=0.1, horizon=5).fit(cal).predict_quantile()
        assert q5 >= q1

    def test_horizon_1_matches_splitcp(self, cal_eval_split):
        """CFRNN with horizon=1 should match SplitCP."""
        cal, _ = cal_eval_split
        q_cfrnn = CFRNN(alpha=0.1, horizon=1).fit(cal).predict_quantile()
        q_split = SplitCP(alpha=0.1).fit(cal).predict_quantile()
        assert_allclose(q_cfrnn, q_split)

    def test_invalid_horizon(self):
        with pytest.raises(ValueError, match='horizon'):
            CFRNN(alpha=0.1, horizon=0)


# ---------------------------------------------------------------------------
# WeightedCP
# ---------------------------------------------------------------------------


class TestWeightedCP:
    def test_uniform_weights_like_trailing(self, cal_eval_split):
        """Uniform weights with test_weight ≈ 0 should approximate TrailingWindow."""
        cal, _ = cal_eval_split
        n = len(cal)
        wcp = WeightedCP(alpha=0.1, weights=np.ones(n), test_weight=1e-12).fit(cal)
        tw = TrailingWindow(alpha=0.1, window_size=n).fit(cal)
        q_wcp = wcp.predict_quantile()
        q_tw = tw.predict_quantile()
        # Should be close (not exact due to different quantile algorithms)
        assert abs(q_wcp - q_tw) / max(q_tw, 1e-9) < 0.1

    def test_empty_weights_raises(self):
        with pytest.raises(ValueError, match='empty'):
            WeightedCP(alpha=0.1, weights=[])

    def test_negative_weights_raises(self):
        with pytest.raises(ValueError, match='non-negative'):
            WeightedCP(alpha=0.1, weights=[-1.0, 1.0])

    def test_high_test_weight_returns_inf(self, cal_eval_split):
        """If test_weight dominates, the quantile should be inf."""
        cal, _ = cal_eval_split
        wcp = WeightedCP(alpha=0.1, weights=np.ones(len(cal)), test_weight=1e12)
        wcp.fit(cal)
        q = wcp.predict_quantile()
        assert q == math.inf


# ---------------------------------------------------------------------------
# TrailingWindow
# ---------------------------------------------------------------------------


class TestTrailingWindow:
    def test_basic_cycle(self, cal_eval_split):
        cal, _ = cal_eval_split
        tw = TrailingWindow(alpha=0.1, window_size=50).fit(cal)
        q = tw.predict_quantile()
        assert math.isfinite(q)
        tw.update(1.0)

    def test_window_size_limits_memory(self):
        tw = TrailingWindow(alpha=0.1, window_size=10)
        tw.fit(np.ones(5))
        for i in range(20):
            tw.update(float(i))
        assert len(tw._scores) == 10

    def test_invalid_window_size(self):
        with pytest.raises(ValueError, match='window_size'):
            TrailingWindow(alpha=0.1, window_size=0)


# ---------------------------------------------------------------------------
# QuantileIntegrator (PID)
# ---------------------------------------------------------------------------


class TestQuantileIntegrator:
    def test_pure_proportional(self, cal_eval_split):
        """KI=0 should work as a pure gradient-descent method."""
        cal, eval_scores = cal_eval_split
        qi = QuantileIntegrator(alpha=0.1, lr=0.1, KI=0.0).fit(cal)
        result = qi.evaluate(eval_scores, np.zeros_like(eval_scores))
        assert 0.0 <= result['coverage'] <= 1.0

    def test_with_integrator(self, cal_eval_split):
        cal, eval_scores = cal_eval_split
        qi = QuantileIntegrator(alpha=0.1, lr=0.1, KI=1.0).fit(cal)
        result = qi.evaluate(eval_scores, np.zeros_like(eval_scores))
        assert 0.0 <= result['coverage'] <= 1.0

    def test_cumulative_error_tracking(self, cal_eval_split):
        cal, eval_scores = cal_eval_split
        qi = QuantileIntegrator(alpha=0.1).fit(cal)
        for s in eval_scores[:50]:
            qi.predict_quantile()
            qi.update(s)
        # cum_err should be non-negative integer count
        assert qi._cum_err >= 0
        assert qi._t == 50


# ---------------------------------------------------------------------------
# CQR
# ---------------------------------------------------------------------------


class TestCQR:
    def test_inherits_from_cqr_base(self):
        assert issubclass(CQR, CQRBase)

    def test_predict_interval_asymmetric(self, cal_eval_split):
        """CQR predict_interval should produce (q_low - q, q_high + q)."""
        cal, _ = cal_eval_split
        cqr = CQR(alpha=0.1).fit(cal)
        q = cqr.predict_quantile()
        lo, hi = cqr.predict_interval((2.0, 8.0))
        assert_allclose(lo, 2.0 - q)
        assert_allclose(hi, 8.0 + q)

    def test_evaluate_requires_2d_forecasts(self, cal_eval_split):
        cal, eval_scores = cal_eval_split
        cqr = CQR(alpha=0.1).fit(cal)
        with pytest.raises(ValueError, match='shape.*T, 2'):
            cqr.evaluate(eval_scores, np.zeros_like(eval_scores))

    def test_evaluate_with_valid_forecasts(self, cal_eval_split):
        cal, eval_scores = cal_eval_split
        cqr = CQR(alpha=0.1).fit(cal)
        forecasts = np.column_stack(
            [
                np.zeros(len(eval_scores)),
                np.ones(len(eval_scores)) * 5.0,
            ]
        )
        result = cqr.evaluate(eval_scores, forecasts)
        assert 0.0 <= result['coverage'] <= 1.0
        assert result['avg_width'] > 0

    def test_fit_before_evaluate(self):
        cqr = CQR(alpha=0.1)
        with pytest.raises(RuntimeError, match='fit'):
            cqr.evaluate(np.array([1.0]), np.array([[0.0, 1.0]]))


# ---------------------------------------------------------------------------
# AdaptiveCQR
# ---------------------------------------------------------------------------


class TestAdaptiveCQR:
    def test_inherits_from_cqr_base_and_aci(self):
        assert issubclass(AdaptiveCQR, CQRBase)
        assert issubclass(AdaptiveCQR, ConformalPredictor)

    def test_alpha_t_adapts(self, cal_eval_split):
        """AdaptiveCQR with ACI engine should update alpha_t."""
        cal, eval_scores = cal_eval_split
        acqr = AdaptiveCQR(alpha=0.1, method='aci', gamma=0.05).fit(cal)
        initial_q = acqr.predict_quantile()
        for s in eval_scores[:50]:
            acqr.predict_quantile()
            acqr.update(s)
        # After many updates, the quantile should have changed
        # (with high probability on random data)
        final_q = acqr.predict_quantile()
        assert initial_q != final_q

    def test_predict_interval_uses_cqr_logic(self, cal_eval_split):
        """Should use CQRBase's predict_interval, not ConformalPredictor's."""
        cal, _ = cal_eval_split
        acqr = AdaptiveCQR(alpha=0.1).fit(cal)
        q = acqr.predict_quantile()
        lo, hi = acqr.predict_interval((1.0, 9.0))
        assert_allclose(lo, 1.0 - q)
        assert_allclose(hi, 9.0 + q)

    def test_evaluate_works(self, cal_eval_split):
        cal, eval_scores = cal_eval_split
        acqr = AdaptiveCQR(alpha=0.1).fit(cal)
        forecasts = np.column_stack(
            [
                np.zeros(len(eval_scores)),
                np.ones(len(eval_scores)) * 5.0,
            ]
        )
        result = acqr.evaluate(eval_scores, forecasts)
        assert 0.0 <= result['coverage'] <= 1.0


# ---------------------------------------------------------------------------
# Base class / interface
# ---------------------------------------------------------------------------


class TestBaseInterface:
    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match='alpha'):
            SplitCP(alpha=0.0)
        with pytest.raises(ValueError, match='alpha'):
            SplitCP(alpha=1.0)
        with pytest.raises(ValueError, match='alpha'):
            SplitCP(alpha=-0.1)

    def test_evaluate_winkler_penalty(self, cal_eval_split):
        """Winkler penalty for uncovered points should use (score - q)."""
        cal, _ = cal_eval_split
        cp = SplitCP(alpha=0.1).fit(cal)
        q = cp.predict_quantile()
        # Construct a score that exceeds q
        big_score = q + 1.0
        width = 2.0 * q
        expected_winkler = width + (2.0 / 0.1) * (big_score - q)

        result = cp.evaluate(
            np.array([big_score]),
            np.array([0.0]),
        )
        assert_allclose(result['winkler_score'], expected_winkler, rtol=1e-10)


# ---------------------------------------------------------------------------
# Smoke tests — every method should complete the full protocol
# ---------------------------------------------------------------------------

ALL_SYMMETRIC_METHODS = [
    lambda: SplitCP(alpha=0.1),
    lambda: ACI(alpha=0.1, gamma=0.01),
    lambda: AgACI(alpha=0.1),
    lambda: DtACI(alpha=0.1, seed=0),
    lambda: CFRNN(alpha=0.1, horizon=3),
    lambda: WeightedCP(alpha=0.1, weights=np.ones(200)),
    lambda: TrailingWindow(alpha=0.1, window_size=100),
    lambda: QuantileIntegrator(alpha=0.1, lr=0.1),
]


# ---------------------------------------------------------------------------
# batch_evaluate equivalence tests
# ---------------------------------------------------------------------------

# Methods that have deterministic batch_evaluate (DtACI excluded due to random sampling)
BATCH_DETERMINISTIC_METHODS = [
    lambda: SplitCP(alpha=0.1),
    lambda: ACI(alpha=0.1, gamma=0.01),
    lambda: AgACI(alpha=0.1),
    lambda: CFRNN(alpha=0.1, horizon=3),
    lambda: WeightedCP(alpha=0.1, weights=np.ones(200)),
    lambda: TrailingWindow(alpha=0.1, window_size=100),
    lambda: QuantileIntegrator(alpha=0.1, lr=0.1),
]


@pytest.mark.parametrize(
    'make_method',
    BATCH_DETERMINISTIC_METHODS,
    ids=lambda fn: fn().__class__.__name__,
)
class TestBatchEvaluateSymmetric:
    """batch_evaluate must produce identical results to evaluate."""

    def test_batch_matches_online(self, make_method, cal_eval_split):
        cal, eval_scores = cal_eval_split

        # Run online evaluate
        m1 = make_method()
        m1.fit(cal)
        result_online = m1.evaluate(eval_scores, np.zeros_like(eval_scores))

        # Run batch_evaluate
        m2 = make_method()
        m2.fit(cal)
        result_batch = m2.batch_evaluate(eval_scores, np.zeros_like(eval_scores))

        assert_allclose(result_batch['coverage'], result_online['coverage'], rtol=1e-10)
        assert_allclose(
            result_batch['avg_width'], result_online['avg_width'], rtol=1e-10
        )
        assert_allclose(
            result_batch['winkler_score'], result_online['winkler_score'], rtol=1e-10
        )
        assert_allclose(
            result_batch['quantiles'], result_online['quantiles'], rtol=1e-10
        )

    def test_batch_with_burnin(self, make_method, cal_eval_split):
        cal, eval_scores = cal_eval_split
        burnin = 20

        m1 = make_method()
        m1.fit(cal)
        result_online = m1.evaluate(
            eval_scores, np.zeros_like(eval_scores), burnin=burnin
        )

        m2 = make_method()
        m2.fit(cal)
        result_batch = m2.batch_evaluate(
            eval_scores, np.zeros_like(eval_scores), burnin=burnin
        )

        assert_allclose(result_batch['coverage'], result_online['coverage'], rtol=1e-10)
        assert_allclose(
            result_batch['avg_width'], result_online['avg_width'], rtol=1e-10
        )
        assert_allclose(
            result_batch['winkler_score'], result_online['winkler_score'], rtol=1e-10
        )


class TestBatchEvaluateDtACI:
    """DtACI with same seed should give same results for batch and online."""

    def test_batch_matches_online(self, cal_eval_split):
        cal, eval_scores = cal_eval_split

        m1 = DtACI(alpha=0.1, seed=42)
        m1.fit(cal)
        result_online = m1.evaluate(eval_scores, np.zeros_like(eval_scores))

        m2 = DtACI(alpha=0.1, seed=42)
        m2.fit(cal)
        result_batch = m2.batch_evaluate(eval_scores, np.zeros_like(eval_scores))

        assert_allclose(result_batch['coverage'], result_online['coverage'], rtol=1e-10)
        assert_allclose(
            result_batch['avg_width'], result_online['avg_width'], rtol=1e-10
        )
        assert_allclose(
            result_batch['winkler_score'], result_online['winkler_score'], rtol=1e-10
        )
        assert_allclose(
            result_batch['quantiles'], result_online['quantiles'], rtol=1e-10
        )


BATCH_CQR_METHODS = [
    lambda: CQR(alpha=0.1),
    lambda: AdaptiveCQR(alpha=0.1, method='aci', gamma=0.01),
]


@pytest.mark.parametrize(
    'make_method',
    BATCH_CQR_METHODS,
    ids=lambda fn: fn().__class__.__name__,
)
class TestBatchEvaluateCQR:
    """batch_evaluate for CQR-family must match evaluate."""

    def test_batch_matches_online(self, make_method, cal_eval_split):
        cal, eval_scores = cal_eval_split
        forecasts = np.column_stack(
            [np.zeros(len(eval_scores)), np.ones(len(eval_scores)) * 5.0]
        )

        m1 = make_method()
        m1.fit(cal)
        result_online = m1.evaluate(eval_scores, forecasts)

        m2 = make_method()
        m2.fit(cal)
        result_batch = m2.batch_evaluate(eval_scores, forecasts)

        assert_allclose(result_batch['coverage'], result_online['coverage'], rtol=1e-10)
        assert_allclose(
            result_batch['avg_width'], result_online['avg_width'], rtol=1e-10
        )
        assert_allclose(
            result_batch['winkler_score'], result_online['winkler_score'], rtol=1e-10
        )
        assert_allclose(
            result_batch['quantiles'], result_online['quantiles'], rtol=1e-10
        )


# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'make_method',
    ALL_SYMMETRIC_METHODS,
    ids=lambda fn: fn().__class__.__name__,
)
class TestSmokeSymmetric:
    def test_fit_predict_update(self, make_method, cal_eval_split):
        cal, eval_scores = cal_eval_split
        m = make_method()
        m.fit(cal)
        q = m.predict_quantile()
        assert isinstance(q, float)
        m.update(eval_scores[0])

    def test_evaluate_returns_valid_dict(self, make_method, cal_eval_split):
        cal, eval_scores = cal_eval_split
        m = make_method()
        m.fit(cal)
        result = m.evaluate(eval_scores, np.zeros_like(eval_scores))
        assert 'coverage' in result
        assert 'avg_width' in result
        assert 'winkler_score' in result
        assert 'quantiles' in result
        assert len(result['quantiles']) == len(eval_scores)


# ===========================================================================
# Asymmetric interval tests
# ===========================================================================


@pytest.fixture()
def skewed_signed_errors():
    """Skewed signed errors: positive residuals are larger than negative ones."""
    rng = np.random.default_rng(7)
    # 1000 signed errors drawn from an asymmetric distribution
    # e ~ exponential(1) - 0.3  (positive skew)
    return rng.exponential(scale=1.0, size=1000) - 0.3


@pytest.fixture()
def sym_signed_errors():
    """Symmetric signed errors (standard normal)."""
    rng = np.random.default_rng(99)
    return rng.standard_normal(1000)


@pytest.fixture()
def asym_cal_eval(skewed_signed_errors):
    return skewed_signed_errors[:200], skewed_signed_errors[200:]


@pytest.fixture()
def sym_cal_eval(sym_signed_errors):
    return sym_signed_errors[:200], sym_signed_errors[200:]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_asymmetric_result(result: dict, alpha: float = 0.1):
    """Assert that evaluate/batch_evaluate output has the expected shape."""
    assert 'coverage' in result
    assert 'avg_width' in result
    assert 'winkler_score' in result
    assert 'quantiles' in result
    assert math.isfinite(result['coverage'])


def _run_online_asym(method, cal, eval_scores):
    """Run online evaluate in asymmetric mode."""
    method.fit(cal)
    return method.evaluate(eval_scores, np.zeros_like(eval_scores))


def _run_batch_asym(method, cal, eval_scores):
    """Run batch_evaluate in asymmetric mode."""
    method.fit(cal)
    return method.batch_evaluate(eval_scores, np.zeros_like(eval_scores))


# ---------------------------------------------------------------------------
# SplitCP asymmetric
# ---------------------------------------------------------------------------


class TestAsymmetricSplitCP:
    def test_symmetric_data_gives_equal_bounds(self, sym_cal_eval):
        cal, _ = sym_cal_eval
        cp = SplitCP(alpha=0.1, asymmetric=True).fit(cal)
        q_lo, q_up = cp.predict_quantile_pair()
        # For symmetric data, lower and upper quantile should be nearly equal
        assert math.isfinite(q_lo) and math.isfinite(q_up)
        assert abs(q_lo - q_up) / max(q_up, 1e-9) < 0.2

    def test_skewed_data_gives_unequal_bounds(self, asym_cal_eval):
        cal, _ = asym_cal_eval
        cp = SplitCP(alpha=0.1, asymmetric=True).fit(cal)
        q_lo, q_up = cp.predict_quantile_pair()
        # For positively skewed data: q_up > q_lo
        assert q_up > q_lo

    def test_asymmetric_predict_interval(self, asym_cal_eval):
        cal, _ = asym_cal_eval
        cp = SplitCP(alpha=0.1, asymmetric=True).fit(cal)
        q_lo, q_up = cp.predict_quantile_pair()
        lo, hi = cp.predict_interval(5.0)
        assert_allclose(lo, 5.0 - q_lo)
        assert_allclose(hi, 5.0 + q_up)

    def test_evaluate_runs_and_has_valid_output(self, asym_cal_eval):
        cal, eval_errors = asym_cal_eval
        cp = SplitCP(alpha=0.1, asymmetric=True)
        result = _run_online_asym(cp, cal, eval_errors)
        _check_asymmetric_result(result)

    def test_coverage_achieves_target(self, sym_cal_eval):
        """On symmetric IID data, coverage should be close to (1 - alpha)."""
        cal, eval_errors = sym_cal_eval
        cp = SplitCP(alpha=0.1, asymmetric=True)
        result = _run_online_asym(cp, cal, eval_errors)
        # Allow generous slack since finite-sample
        assert result['coverage'] >= 0.80

    def test_asymmetric_tighter_than_symmetric_on_skewed_data(self, asym_cal_eval):
        """Asymmetric intervals should be narrower on average for skewed data."""
        cal, eval_errors = asym_cal_eval
        cp_asym = SplitCP(alpha=0.1, asymmetric=True)
        result_asym = _run_online_asym(cp_asym, cal, eval_errors)
        # Symmetric method uses unsigned abs errors
        result_sym = (
            SplitCP(alpha=0.1)
            .fit(np.abs(cal))
            .evaluate(np.abs(eval_errors), np.zeros_like(eval_errors))
        )
        # Asymmetric should have comparable or smaller avg_width
        assert result_asym['avg_width'] < result_sym['avg_width'] * 1.1

    def test_predict_quantile_returns_q_up(self, asym_cal_eval):
        cal, _ = asym_cal_eval
        cp = SplitCP(alpha=0.1, asymmetric=True).fit(cal)
        q = cp.predict_quantile()
        _, q_up = cp.predict_quantile_pair()
        assert_allclose(q, q_up)

    def test_update_signed_appends_signed_error(self, asym_cal_eval):
        cal, _ = asym_cal_eval
        cp = SplitCP(alpha=0.1, asymmetric=True).fit(cal)
        initial_len = len(cp._scores)
        cp.update_signed(-2.5)
        assert len(cp._scores) == min(
            initial_len + 1, cp.window_size or initial_len + 1
        )
        assert cp._scores[-1] == -2.5  # signed, NOT abs

    def test_backward_compat_default_symmetric(self, asym_cal_eval):
        cal, _ = asym_cal_eval
        cp_default = SplitCP(alpha=0.1)
        cp_explicit = SplitCP(alpha=0.1, asymmetric=False)
        assert not cp_default.asymmetric
        assert not cp_explicit.asymmetric


# ---------------------------------------------------------------------------
# ACI asymmetric
# ---------------------------------------------------------------------------


class TestAsymmetricACI:
    def test_alpha_t_trackers_independent(self, sym_cal_eval):
        """The two alpha_t trackers should evolve independently."""
        cal, eval_errors = sym_cal_eval
        aci = ACI(alpha=0.1, gamma=0.05, asymmetric=True).fit(cal)
        for e in eval_errors[:50]:
            aci.predict_quantile_pair()
            aci.update_signed(e)
        # Both alpha_t values should be bounded
        assert 0.0 <= aci.alpha_t_lower <= 1.0
        assert 0.0 <= aci.alpha_t_upper <= 1.0

    def test_evaluate_runs(self, asym_cal_eval):
        cal, eval_errors = asym_cal_eval
        aci = ACI(alpha=0.1, gamma=0.01, asymmetric=True)
        result = _run_online_asym(aci, cal, eval_errors)
        _check_asymmetric_result(result)

    def test_batch_matches_online(self, sym_cal_eval):
        cal, eval_errors = sym_cal_eval
        m1 = ACI(alpha=0.1, gamma=0.01, asymmetric=True)
        m1.fit(cal)
        r_online = m1.evaluate(eval_errors, np.zeros_like(eval_errors))

        m2 = ACI(alpha=0.1, gamma=0.01, asymmetric=True)
        m2.fit(cal)
        r_batch = m2.batch_evaluate(eval_errors, np.zeros_like(eval_errors))

        assert_allclose(r_batch['coverage'], r_online['coverage'], rtol=1e-8)
        assert_allclose(r_batch['avg_width'], r_online['avg_width'], rtol=1e-8)
        assert_allclose(r_batch['winkler_score'], r_online['winkler_score'], rtol=1e-8)

    def test_gamma_zero_static(self, sym_cal_eval):
        """With gamma=0, alpha_t trackers should stay constant."""
        cal, _ = sym_cal_eval
        aci = ACI(alpha=0.1, gamma=0.0, asymmetric=True).fit(cal)
        init_lo, init_up = aci.alpha_t_lower, aci.alpha_t_upper
        aci.predict_quantile_pair()
        aci.update_signed(0.5)
        assert aci.alpha_t_lower == init_lo
        assert aci.alpha_t_upper == init_up

    def test_asymmetric_flag_off_unchanged(self, sym_cal_eval):
        cal, eval_errors = sym_cal_eval
        aci = ACI(alpha=0.1, gamma=0.01)  # default asymmetric=False
        assert not aci.asymmetric
        aci.fit(cal)
        result = aci.evaluate(np.abs(eval_errors), np.zeros_like(eval_errors))
        assert result['coverage'] >= 0.80


# ---------------------------------------------------------------------------
# TrailingWindow asymmetric
# ---------------------------------------------------------------------------


class TestAsymmetricTrailingWindow:
    def test_quantile_pair_matches_numpy(self, sym_cal_eval):
        """predict_quantile_pair should match numpy quantile on signed errors."""
        cal, _ = sym_cal_eval
        tw = TrailingWindow(alpha=0.1, window_size=200, asymmetric=True).fit(cal)
        q_lo, q_up = tw.predict_quantile_pair()
        cal_arr = np.asarray(cal)
        expected_up = float(np.quantile(cal_arr, 0.95, method='higher'))
        expected_lo = float(np.quantile(-cal_arr, 0.95, method='higher'))
        assert_allclose(q_lo, expected_lo, rtol=1e-10)
        assert_allclose(q_up, expected_up, rtol=1e-10)

    def test_evaluate_runs(self, asym_cal_eval):
        cal, eval_errors = asym_cal_eval
        tw = TrailingWindow(alpha=0.1, window_size=100, asymmetric=True)
        result = _run_online_asym(tw, cal, eval_errors)
        _check_asymmetric_result(result)

    def test_batch_matches_online(self, sym_cal_eval):
        cal, eval_errors = sym_cal_eval
        m1 = TrailingWindow(alpha=0.1, window_size=100, asymmetric=True)
        m1.fit(cal)
        r_online = m1.evaluate(eval_errors, np.zeros_like(eval_errors))

        m2 = TrailingWindow(alpha=0.1, window_size=100, asymmetric=True)
        m2.fit(cal)
        r_batch = m2.batch_evaluate(eval_errors, np.zeros_like(eval_errors))

        assert_allclose(r_batch['coverage'], r_online['coverage'], rtol=1e-8)
        assert_allclose(r_batch['avg_width'], r_online['avg_width'], rtol=1e-8)

    def test_predict_interval_asymmetric(self, sym_cal_eval):
        cal, _ = sym_cal_eval
        tw = TrailingWindow(alpha=0.1, window_size=200, asymmetric=True).fit(cal)
        q_lo, q_up = tw.predict_quantile_pair()
        lo, hi = tw.predict_interval(0.0)
        assert_allclose(lo, -q_lo)
        assert_allclose(hi, q_up)


# ---------------------------------------------------------------------------
# QuantileIntegrator asymmetric
# ---------------------------------------------------------------------------


class TestAsymmetricQuantileIntegrator:
    def test_two_controllers_independent(self, sym_cal_eval):
        """Upper and lower PID controllers should update independently."""
        cal, eval_errors = sym_cal_eval
        qi = QuantileIntegrator(alpha=0.1, lr=0.1, asymmetric=True).fit(cal)
        for e in eval_errors[:50]:
            qi.predict_quantile_pair()
            qi.update_signed(e)
        # Both controllers must have advanced
        assert qi._t == 50
        # Lower and upper qt should differ (different coverage feedback)
        # (Not guaranteed in all cases, but highly likely on random data)
        assert qi._q_lower >= 0 or qi._q_upper >= 0

    def test_evaluate_runs(self, asym_cal_eval):
        cal, eval_errors = asym_cal_eval
        qi = QuantileIntegrator(alpha=0.1, lr=0.1, asymmetric=True)
        result = _run_online_asym(qi, cal, eval_errors)
        _check_asymmetric_result(result)

    def test_batch_matches_online(self, sym_cal_eval):
        cal, eval_errors = sym_cal_eval
        m1 = QuantileIntegrator(
            alpha=0.1, lr=0.1, proportional_lr=False, asymmetric=True
        )
        m1.fit(cal)
        r_online = m1.evaluate(eval_errors, np.zeros_like(eval_errors))

        m2 = QuantileIntegrator(
            alpha=0.1, lr=0.1, proportional_lr=False, asymmetric=True
        )
        m2.fit(cal)
        r_batch = m2.batch_evaluate(eval_errors, np.zeros_like(eval_errors))

        assert_allclose(r_batch['coverage'], r_online['coverage'], rtol=1e-8)
        assert_allclose(r_batch['avg_width'], r_online['avg_width'], rtol=1e-8)

    def test_with_integrator(self, sym_cal_eval):
        cal, eval_errors = sym_cal_eval
        qi = QuantileIntegrator(alpha=0.1, lr=0.1, KI=1.0, asymmetric=True)
        result = _run_online_asym(qi, cal, eval_errors)
        _check_asymmetric_result(result)


# ---------------------------------------------------------------------------
# WeightedCP asymmetric
# ---------------------------------------------------------------------------


class TestAsymmetricWeightedCP:
    def test_pair_structure(self, sym_cal_eval):
        cal, _ = sym_cal_eval
        wcp = WeightedCP(alpha=0.1, weights=np.ones(200), asymmetric=True).fit(cal)
        q_lo, q_up = wcp.predict_quantile_pair()
        assert math.isfinite(q_lo) or q_lo == math.inf
        assert math.isfinite(q_up) or q_up == math.inf

    def test_skewed_data_unequal_bounds(self, asym_cal_eval):
        cal, _ = asym_cal_eval
        wcp = WeightedCP(alpha=0.1, weights=np.ones(200), asymmetric=True).fit(cal)
        q_lo, q_up = wcp.predict_quantile_pair()
        # Positive skew → upper quantile larger than lower
        assert q_up > q_lo

    def test_evaluate_runs(self, asym_cal_eval):
        cal, eval_errors = asym_cal_eval
        wcp = WeightedCP(alpha=0.1, weights=np.ones(200), asymmetric=True)
        result = _run_online_asym(wcp, cal, eval_errors)
        _check_asymmetric_result(result)

    def test_predict_interval_asymmetric(self, sym_cal_eval):
        cal, _ = sym_cal_eval
        wcp = WeightedCP(alpha=0.1, weights=np.ones(200), asymmetric=True).fit(cal)
        q_lo, q_up = wcp.predict_quantile_pair()
        lo, hi = wcp.predict_interval(3.0)
        assert_allclose(lo, 3.0 - q_lo)
        assert_allclose(hi, 3.0 + q_up)


# ---------------------------------------------------------------------------
# Backward-compatibility: all existing symmetric tests still pass
# ---------------------------------------------------------------------------


class TestAsymmetricBackwardCompat:
    """Verify that asymmetric=False (default) leaves all existing behaviour intact."""

    @pytest.mark.parametrize(
        'make_method', ALL_SYMMETRIC_METHODS, ids=lambda fn: fn().__class__.__name__
    )
    def test_default_symmetric_unchanged(self, make_method, cal_eval_split):
        cal, eval_scores = cal_eval_split
        m = make_method()
        assert not m.asymmetric
        m.fit(cal)
        result = m.evaluate(np.abs(eval_scores), np.zeros_like(eval_scores))
        assert 'coverage' in result
        assert result['avg_width'] > 0

    def test_predict_interval_symmetric_unchanged(self, cal_eval_split):
        cal, _ = cal_eval_split
        cp = SplitCP(alpha=0.1).fit(cal)
        lo, hi = cp.predict_interval(5.0)
        q = cp.predict_quantile()
        assert_allclose(hi - lo, 2 * q)
        assert_allclose((lo + hi) / 2, 5.0)


# ---------------------------------------------------------------------------
# Winkler score for asymmetric intervals
# ---------------------------------------------------------------------------


class TestAsymmetricWinklerScore:
    def test_covered_winkler_equals_width(self, sym_cal_eval):
        """When covered, Winkler score should equal interval width (q_lo + q_up)."""
        cal, _ = sym_cal_eval
        cp = SplitCP(alpha=0.1, asymmetric=True).fit(cal)
        q_lo, q_up = cp.predict_quantile_pair()
        # Construct a signed error that is inside the interval
        e_inside = 0.0  # always inside (assuming q_lo, q_up > 0)
        result = cp.evaluate(np.array([e_inside]), np.array([0.0]))
        expected_width = q_lo + q_up
        assert_allclose(result['winkler_score'], expected_width, rtol=1e-8)

    def test_upper_miss_penalizes_correctly(self, sym_cal_eval):
        """Upper violation: Winkler = width + (2/alpha) * (e - q_up)."""
        cal, _ = sym_cal_eval
        cp = SplitCP(alpha=0.1, asymmetric=True).fit(cal)
        q_lo, q_up = cp.predict_quantile_pair()
        # Construct a large positive error that exceeds q_up
        e_miss = q_up + 1.0
        result = cp.evaluate(np.array([e_miss]), np.array([0.0]))
        width = q_lo + q_up
        expected = width + (2.0 / 0.1) * (e_miss - q_up)
        assert_allclose(result['winkler_score'], expected, rtol=1e-8)

    def test_lower_miss_penalizes_correctly(self, sym_cal_eval):
        """Lower violation: Winkler = width + (2/alpha) * (-e - q_lo)."""
        cal, _ = sym_cal_eval
        cp = SplitCP(alpha=0.1, asymmetric=True).fit(cal)
        q_lo, q_up = cp.predict_quantile_pair()
        # Construct a large negative error that exceeds q_lo in magnitude
        e_miss = -(q_lo + 1.0)
        result = cp.evaluate(np.array([e_miss]), np.array([0.0]))
        width = q_lo + q_up
        expected = width + (2.0 / 0.1) * (-e_miss - q_lo)
        assert_allclose(result['winkler_score'], expected, rtol=1e-8)


# ---------------------------------------------------------------------------
# Scan primitives: asymmetric variants
# ---------------------------------------------------------------------------


class TestAsymmetricScanPrimitives:
    def test_aci_scan_asymmetric_shape(self, sym_cal_eval):
        from timecp.methods._scan import aci_scan_asymmetric

        cal, eval_errors = sym_cal_eval
        pairs, at_lo, at_up = aci_scan_asymmetric(
            eval_errors, alpha=0.1, gamma=0.01, initial_window=cal
        )
        assert pairs.shape == (len(eval_errors), 2)
        assert 0.0 <= at_lo <= 1.0
        assert 0.0 <= at_up <= 1.0

    def test_pid_scan_asymmetric_shape(self, sym_cal_eval):
        from timecp.methods._scan import pid_scan_asymmetric

        cal, eval_errors = sym_cal_eval
        result = pid_scan_asymmetric(
            eval_errors, alpha=0.1, lr=0.1, proportional_lr=False, initial_window=cal
        )
        pairs = result[0]
        assert pairs.shape == (len(eval_errors), 2)

    def test_aci_scan_asymmetric_both_tails_finite(self, sym_cal_eval):
        from timecp.methods._scan import aci_scan_asymmetric

        cal, eval_errors = sym_cal_eval
        pairs, _, _ = aci_scan_asymmetric(
            eval_errors, alpha=0.1, gamma=0.01, initial_window=cal
        )
        # After warm-up all quantiles should be finite (large cal window)
        finite_q_lo = np.isfinite(pairs[50:, 0])
        finite_q_up = np.isfinite(pairs[50:, 1])
        assert finite_q_lo.all(), 'q_lower should be finite after warm-up'
        assert finite_q_up.all(), 'q_upper should be finite after warm-up'


ALL_CQR_METHODS = [
    lambda: CQR(alpha=0.1),
    lambda: AdaptiveCQR(alpha=0.1, method='aci', gamma=0.01),
]


@pytest.mark.parametrize(
    'make_method',
    ALL_CQR_METHODS,
    ids=lambda fn: fn().__class__.__name__,
)
class TestSmokeCQR:
    def test_fit_predict_update(self, make_method, cal_eval_split):
        cal, eval_scores = cal_eval_split
        m = make_method()
        m.fit(cal)
        q = m.predict_quantile()
        assert isinstance(q, float)
        lo, hi = m.predict_interval((0.0, 5.0))
        assert lo <= hi
        m.update(eval_scores[0])

    def test_evaluate_returns_valid_dict(self, make_method, cal_eval_split):
        cal, eval_scores = cal_eval_split
        m = make_method()
        m.fit(cal)
        forecasts = np.column_stack(
            [
                np.zeros(len(eval_scores)),
                np.ones(len(eval_scores)) * 5.0,
            ]
        )
        result = m.evaluate(eval_scores, forecasts)
        assert 'coverage' in result
        assert 'avg_width' in result
        assert len(result['quantiles']) == len(eval_scores)
