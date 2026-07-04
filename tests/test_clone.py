from collections import deque

import numpy as np

from timecp.base import AsymmetricPredictor
from timecp.methods.aci import ACI
from timecp.methods.acmcp import AcMCP
from timecp.methods.adaptive_cqr import AdaptiveCQR
from timecp.methods.agaci import AgACI
from timecp.methods.cp import SplitCP, TrailingWindow, WeightedCP
from timecp.methods.cqr import CQR
from timecp.methods.dtaci import DtACI
from timecp.methods.pid import QuantileIntegrator
from timecp.methods.spci import SPCI


def test_clone_all_predictors():
    rng = np.random.default_rng(42)
    cal_data = rng.standard_normal(100)
    cal_signed = rng.standard_normal((100, 3))  # For AcMCP, etc.

    predictors = [
        SplitCP(alpha=0.1),
        TrailingWindow(alpha=0.1, window_size=20),
        WeightedCP(alpha=0.1, window_size=20),
        ACI(alpha=0.1, gamma=0.01, window_size=20),
        AgACI(alpha=0.1, gammas=[0.01, 0.05], window_size=20),
        DtACI(alpha=0.1, seed=42, window_size=20),
        QuantileIntegrator(alpha=0.1, lr=0.1, KI=0.01, window_size=20),
        CQR(alpha=0.1, window_size=20),
        AdaptiveCQR(alpha=0.1, method='aci', gamma=0.05, window_size=20),
        SPCI(alpha=0.1, w=10),
        AcMCP(alpha=0.1, h=3, ncal=20),
        AsymmetricPredictor(ACI, alpha=0.1, gamma=0.01, window_size=20),
    ]

    for p in predictors:
        # Fit
        if isinstance(p, AcMCP):
            p.fit(cal_signed[:, 2], cal_signed)
        else:
            p.fit(cal_data)

        # Clone
        cloned = p.clone()

        # Check instance type and properties
        assert type(cloned) is type(p)
        assert cloned.alpha == p.alpha
        assert cloned.asymmetric == p.asymmetric
        assert cloned.bonferroni_horizon == p.bonferroni_horizon
        assert cloned.conformal_correction == p.conformal_correction

        # Cloned should NOT be fitted
        assert not cloned._fitted
        assert cloned._last_q is None
        assert cloned._last_q_pair is None

        # Check independence (modifying cloned shouldn't modify original)
        for attr in ['_scores', '_recent_scores']:
            if hasattr(cloned, attr) and getattr(cloned, attr) is not None:
                orig_val = getattr(p, attr)
                cloned_val = getattr(cloned, attr)
                if isinstance(cloned_val, deque):
                    cloned_val.append(999.0)
                    assert 999.0 not in orig_val
                elif isinstance(cloned_val, np.ndarray):
                    # Replace array with new one to check they aren't same object
                    setattr(cloned, attr, np.array([999.0]))
                    assert 999.0 not in orig_val

        # Let's fit the clone too and verify it runs
        if isinstance(cloned, AcMCP):
            cloned.fit(cal_signed[:, 2], cal_signed)
        else:
            cloned.fit(cal_data)
        assert cloned._fitted


def _make_dummy_dataset(wdir, N, H, rng, alpha=0.1):
    import datasets as hf_datasets

    wdir.mkdir(parents=True, exist_ok=True)
    point = rng.standard_normal((N, H)).astype(np.float32)
    gt = rng.standard_normal((N, H)).astype(np.float32)
    lo_key = f'{alpha / 2:.4g}'.rstrip('0').rstrip('.')
    hi_key = f'{1 - alpha / 2:.4g}'.rstrip('0').rstrip('.')
    q_lo = (point - 0.5).astype(np.float32)
    q_hi = (point + 0.5).astype(np.float32)

    pred_rows = [
        {
            'predictions': point[n].tolist(),
            lo_key: q_lo[n].tolist(),
            hi_key: q_hi[n].tolist(),
        }
        for n in range(N)
    ]
    pred_ds = hf_datasets.Dataset.from_list(pred_rows)
    pred_dict = hf_datasets.DatasetDict({'target': pred_ds})
    pred_dict.save_to_disk(str(wdir / 'predictions'))

    gt_rows = [{'target': gt[n].tolist()} for n in range(N)]
    gt_ds = hf_datasets.Dataset.from_list(gt_rows)
    gt_dict = hf_datasets.DatasetDict({'target': gt_ds})
    gt_dict.save_to_disk(str(wdir / 'ground_truth'))


def test_thread_parallel_equivalence(tmp_path):
    import pandas as pd

    from timecp.cp_eval import CPEvaluator

    rng = np.random.default_rng(42)
    n_windows = 5
    H = 4
    N = 2

    # Build fake prediction windows
    for w in range(n_windows):
        wdir = tmp_path / f'window_{w}'
        _make_dummy_dataset(wdir, N=N, H=H, rng=rng)

    # Evaluator with 1 worker (sequential)
    eval_seq = CPEvaluator(
        predictions_dir=tmp_path,
        horizon=H,
        cal_windows=2,
        alpha=0.1,
        max_workers=1,
    )

    # Evaluator with 2 workers and min_horizon=1 (parallel)
    eval_par = CPEvaluator(
        predictions_dir=tmp_path,
        horizon=H,
        cal_windows=2,
        alpha=0.1,
        max_workers=2,
        parallel_min_horizon=1,
    )

    methods = {
        'SplitCP': [SplitCP(alpha=0.1) for _ in range(H)],
        'ACI': [ACI(alpha=0.1, gamma=0.05) for _ in range(H)],
    }

    # Run both and assert identical
    res_seq = eval_seq.run(methods)
    res_par = eval_par.run(methods)

    # Remove runtime columns for comparison
    res_seq_compare = res_seq.drop(columns=['runtime'])
    res_par_compare = res_par.drop(columns=['runtime'])

    pd.testing.assert_frame_equal(res_seq_compare, res_par_compare)
