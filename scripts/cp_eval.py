"""CP evaluation script: compare native quantile predictions vs conformal methods.

This script is the post-processing counterpart to ``scripts/forecast.py``.
It loads predictions saved to disk, splits them into calibration and test
windows (using the ``cal_windows`` field in ``metadata.json``), and runs
a battery of conformal prediction methods against the model's native
quantile predictions.

Three evaluation modes
----------------------
Methods passed to ``--methods`` are automatically classified as marginal or
joint based on the lists below, and dispatched to the appropriate evaluation
path.  Marginal and joint methods can be freely mixed in a single invocation.

**Marginal (default)**
    Per-horizon independent calibration.  One CP predictor per forecast
    horizon step h, calibrated independently.  Follows Wang & Hyndman (2024)
    and Szabadváry (2024).

    Supported methods: SplitCP, ACI, AgACI, DtACI, PID, AcMCP, SPCI.

**Single-step** (``--single-step``)
    One predictor per method across all H horizons.  Scores are flattened
    in window-major order (all horizons per window sequentially) and the
    predictor is calibrated and evaluated on this single stream.  Only
    affects marginal methods; joint/multi-step methods are unaffected.

**Joint / multi-step**
    Simultaneous coverage across all H horizon steps.  A single predictor is
    calibrated on full ``(C, N, H)`` windows.  The metric of interest is
    ``joint_coverage`` = fraction of test windows where every horizon step is
    simultaneously within the interval.

    Supported methods: JointCFRNN, CopulaCPTS, NormMaxCP, CAFHT, CAFHT_PID.

**External calibration** (``--calibration-dataset``)
    The original CFRNN approach (Stankeviciute et al. 2021, Algorithm 1).
    Methods are calibrated on ALL windows of a separate calibration dataset;
    the test dataset (``--predictions``) is evaluated with a fixed quantile
    without online recalibration.  Only batch methods (SplitCP, joint
    methods) are supported; online adaptive methods (ACI, AgACI, DtACI,
    PID, SPCI, AcMCP) are skipped with a warning.

Output layout
-------------
::

    <predictions_dir>/
      cp_results_a<alpha>_<score_type>_c<cal_windows>_<mode>.csv
                              -- per-horizon per-method metrics (marginal/single-step/cross mode)
      cp_summary_a<alpha>_<score_type>_c<cal_windows>_<mode>.csv
                              -- horizon-averaged summary (marginal/single-step/cross mode)
      cp_summary_a<alpha>_joint_c<cal_windows>_<mode>.csv
                              -- per-method joint metrics (joint mode; score_type='joint'
                                 because joint methods compute scores internally)

Joint methods are evaluated **once** per (alpha, cal_windows) combination,
independently of the ``--score-type`` argument, because they do not use the
external nonconformity score.  Their summaries use ``score_type=joint`` and
``mode=multi-step`` (or ``cross-sectional-cv``).

Marginal metrics
----------------
For each method and each horizon step:

* ``coverage``             -- empirical marginal coverage (CP method)
* ``joint_coverage``       -- fraction of test windows where all H horizons are
  simultaneously covered (method-level scalar, repeated on every horizon row)
* ``avg_width``            -- mean interval width (CP method)
* ``winkler_score``        -- mean Winkler score (CP method)

Joint metrics
-------------
For each method (same column schema as marginal summaries):

* ``coverage``             -- mean per-horizon coverage averaged over H steps
  (renamed from ``marginal_coverage`` for schema parity)
* ``joint_coverage``       -- fraction of windows where all H steps covered
* ``avg_width``            -- mean half-width averaged over horizons
* ``winkler_score``        -- mean Winkler score

The ``score_type`` column is set to ``'joint'`` and ``mode`` to ``'multi-step'``
(or ``'cross-sectional-cv'``).


Usage
-----
::

    # Marginal evaluation (default — all marginal methods)
    uv run python scripts/cp_eval.py \\
        --predictions results/tirex/ETTm2__T \\
        --alpha 0.2

    # Single-step evaluation (one predictor for all horizons)
    uv run python scripts/cp_eval.py \\
        --predictions results/tirex/ETTm2__T \\
        --alpha 0.2 \\
        --single-step

    # Joint evaluation only
    uv run python scripts/cp_eval.py \\
        --predictions results/tirex/ETTm2__T \\
        --alpha 0.2 \\
        --methods JointCFRNN CopulaCPTS CAFHT

    # Both marginal and joint methods in one pass (methods are auto-classified)
    uv run python scripts/cp_eval.py \\
        --predictions results/tirex/ETTm2__T \\
        --alpha 0.2 \\
        --methods SplitCP ACI AgACI PID DtACI JointCFRNN CopulaCPTS CAFHT

    # Single-step mode: ACI/AgACI run single-step, CopulaCPTS stays joint
    uv run python scripts/cp_eval.py \\
        --predictions results/tirex/ETTm2__T \\
        --alpha 0.2 \\
        --single-step \\
        --methods AgACI ACI CopulaCPTS

    # External calibration (CFRNN Algorithm 1 style) — marginal methods
    uv run python scripts/cp_eval.py \\
        --predictions results/tirex/ETTm2__T_test \\
        --calibration-dataset results/tirex/ETTm2__T_cal \\
        --methods SplitCP \\
        --alpha 0.2

    # External calibration — joint methods
    uv run python scripts/cp_eval.py \\
        --predictions results/tirex/ETTm2__T_test \\
        --calibration-dataset results/tirex/ETTm2__T_cal \\
        --methods JointCFRNN CAFHT \\
        --alpha 0.2

    # Evaluate all tasks under a model directory
    uv run python scripts/cp_eval.py \\
        --predictions results/tirex \\
        --alpha 0.2 \\
        --recursive
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from timecp import print_error, print_warning, set_global_seed
from timecp.cp_eval import _USER_SCORE_TYPES, CPEvaluator, _score_slug

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='cp_eval',
        description='Evaluate conformal prediction methods on saved forecast predictions.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        '--predictions',
        default=None,
        help=(
            'Path to a task-level predictions directory (containing window_N/ '
            'subdirectories and metadata.json), or a model-level directory '
            'when --recursive is used.'
        ),
    )
    p.add_argument(
        '--predictions_dir',
        default=None,
        help=(
            'Path to a task-level predictions directory (containing window_N/ '
            'subdirectories and metadata.json), or a model-level directory '
            'when --recursive is used. Preserves current behaviour and takes '
            'precedence over the new options.'
        ),
    )
    p.add_argument(
        '--base-dir',
        default='results',
        help='Base directory containing forester and dataset predictions.',
    )
    p.add_argument(
        '--forecasters',
        nargs='+',
        default=['chronos2'],
        help='List of forecasters/models to evaluate.',
    )
    p.add_argument(
        '--base-datasets',
        '--base_datasets',
        nargs='+',
        dest='base_datasets',
        default=['fev-bench_mini'],
        help='List of base datasets to evaluate.',
    )
    p.add_argument(
        '-T',
        '--config',
        default=None,
        help='Path to YAML config. Uses output_dir to auto-discover tasks.',
    )
    p.add_argument(
        '--alpha',
        nargs='+',
        type=float,
        default=[0.2],
        help='Target miscoverage rate (e.g. 0.2 for 80%% coverage).',
    )
    p.add_argument(
        '--score-type',
        '--score_type',
        nargs='+',
        choices=list(_USER_SCORE_TYPES),
        dest='score_type',
        default=['abs'],
        help=(
            'Nonconformity score type. Physical: abs, squared, signed. '
            'Scaled: iqr_scaled, mad_scaled. Quantile: cqr, scaled_cqr, '
            'distributional, cdf_tail. Transformed/stateful: log, diff. '
            '"signed" forces asymmetric=True automatically.'
        ),
    )
    p.add_argument(
        '--methods',
        nargs='+',
        default=None,
        metavar='METHOD',
        help=(
            'CP methods to evaluate.  '
            'Marginal choices: SplitCP, ACI, AgACI, DtACI, PID, AcMCP, SPCI. '
            'Joint/multi-step choices: JointCFRNN, CopulaCPTS, NormMaxCP, '
            'CAFHT, CAFHT_PID. '
            'Methods are automatically classified: marginal methods run as '
            'per-horizon independent calibration, joint methods run as '
            'simultaneous-coverage evaluation. '
            'Defaults to all marginal methods when not specified.'
        ),
    )
    # Step-mode: mutually exclusive --single-step / --marginal
    step_group = p.add_mutually_exclusive_group()
    step_group.add_argument(
        '--single-step',
        action='store_true',
        default=False,
        help=(
            'Evaluate marginal methods in single-step mode: one predictor '
            'instance per method across all H horizons.  Scores are flattened '
            'in window-major order (all horizons per window sequentially). '
            'Joint/multi-step methods are unaffected (they do not support a '
            'single-step variant).'
        ),
    )
    p.add_argument(
        '--cross-sectional-cv',
        action='store_true',
        help=(
            'Enable cross-sectional cross-validation (N-split) over time series entities '
            'instead of the default temporal split. Online adaptive methods will be skipped.'
        ),
    )
    p.add_argument(
        '--asymmetric',
        action='store_true',
        help='Run marginal CP methods in asymmetric mode (tracking separate upper and lower tail probabilities/quantiles).',
    )
    p.add_argument(
        '--online',
        action='store_true',
        default=False,
        help='Evaluate adaptive methods in online mode (updating state after each step within a window, instead of block/batch update per window).',
    )
    p.add_argument(
        '--cal-windows',
        '--cal_windows',
        nargs='+',
        type=int,
        dest='cal_windows',
        default=[100],
        metavar='M',
        help='Target number of calibration windows to use. Use -1 to use all available calibration windows.',
    )
    p.add_argument(
        '--min-cal-windows',
        type=int,
        default=10,
        metavar='MIN',
        help='Minimum total calibration windows needed to proceed with evaluation.',
    )
    p.add_argument(
        '--covariate-strategy',
        choices=['targets_only', 'covariates_only', 'all'],
        default='targets_only',
        help='Which series to use for calibration in --cross-sectional-cv.',
    )
    p.add_argument(
        '--use-cache',
        action='store_true',
        help=(
            'Cache interval bounds and metrics to .npz files in the task directory, '
            'and reload them if they already exist instead of re-evaluating.'
        ),
    )
    p.add_argument(
        '--save-intervals',
        '--save_intervals',
        action='store_true',
        dest='save_intervals',
        help=(
            'Save full prediction intervals (lower/upper bounds arrays) in the cache files. '
            'Warning: can use substantial disk space and RAM.'
        ),
    )
    p.add_argument(
        '--force',
        action='store_true',
        default=False,
        help='Bypass the cache read and overwrite the cache.',
    )
    p.add_argument(
        '--recursive',
        action='store_true',
        help=(
            'Treat --predictions as a model root and evaluate every '
            'subdirectory that contains a metadata.json.'
        ),
    )
    p.add_argument(
        '--output',
        default=None,
        help=(
            'Directory to write result CSVs.  Defaults to the predictions directory.'
        ),
    )
    # PID-specific
    p.add_argument(
        '--qi-lr',
        type=float,
        default=0.1,
        help='Learning rate for PID (scaled by score range when proportional_lr=True).',
    )
    p.add_argument(
        '--qi-ki',
        type=float,
        default=0.0,
        help='Integral gain for PID (0 = P-only variant).',
    )
    # ACI-specific
    p.add_argument(
        '--aci-gamma',
        type=float,
        default=0.005,
        help='Step size gamma for ACI.',
    )
    p.add_argument(
        '--window-size',
        type=int,
        default=100,
        help='Sliding window size for all methods that support it.',
    )
    # AcMCP-specific
    p.add_argument(
        '--acmcp-ncal',
        type=int,
        default=10,
        help='Burn-in length for AcMCP scorecaster training.',
    )
    p.add_argument(
        '--acmcp-lr',
        type=float,
        default=0.1,
        help='Base learning rate for AcMCP quantile tracking.',
    )
    p.add_argument(
        '--acmcp-no-integrate',
        action='store_true',
        help='Disable the integral (I) term in AcMCP.',
    )
    p.add_argument(
        '--acmcp-no-scorecast',
        action='store_true',
        help='Disable the scorecasting (D) term in AcMCP.',
    )
    p.add_argument(
        '--acmcp-ki',
        type=float,
        default=None,
        help='Integrator gain KI for AcMCP. Defaults to max calibration error.',
    )
    p.add_argument(
        '--acmcp-csat',
        type=float,
        default=None,
        help='Saturation constant Csat for AcMCP integrator.',
    )
    # CopulaCPTS-specific
    p.add_argument(
        '--copula-cal-split',
        type=float,
        default=0.6,
        help='Fraction of cal windows used for the score stage in CopulaCPTS.',
    )
    p.add_argument(
        '--copula-epochs',
        type=int,
        default=500,
        help='Gradient descent epochs for CopulaCPTS copula fitting.',
    )
    # CAFHT-specific
    p.add_argument(
        '--cafht-normalize',
        choices=['mae', 'ones'],
        default='mae',
        help='Per-horizon normalisation strategy for NormMaxCP.',
    )
    p.add_argument(
        '--cafht-base-model',
        choices=['aci', 'pid'],
        default='aci',
        help='Adaptive base method for CAFHT (aci or pid).',
    )
    p.add_argument(
        '--cafht-cal-split',
        type=float,
        default=0.5,
        help='Fraction of cal windows used for gamma selection in CAFHT.',
    )
    p.add_argument(
        '--cafht-q0',
        type=float,
        default=0.1,
        help='Initial threshold q0 for CAFHT base method.',
    )
    p.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Global random seed for reproducibility.',
    )
    return p


# ---------------------------------------------------------------------------
# Method factories
# ---------------------------------------------------------------------------

_VALID_METHODS = (
    'SplitCP',
    'ACI',
    'AgACI',
    'DtACI',
    'PID',
    'AcMCP',
    'SPCI',
    'TrailingWindow',
    'WeightedCP',
)

_VALID_MULTI_STEP_METHODS = (
    'JointCFRNN',
    'CopulaCPTS',
    'NormMaxCP',
    'CAFHT',
    'CAFHT_PID',
)

# Methods that perform online adaptation — they reduce to SplitCP under
# external calibration (fixed quantile, no updates) and are therefore skipped.
_ONLINE_ONLY_METHODS: frozenset[str] = frozenset(
    {'ACI', 'AgACI', 'DtACI', 'PID', 'SPCI', 'AcMCP'}
)


def make_postfix(config: CPEvalConfig) -> str:
    if config.single_step:
        eval_mode = 'single'
    elif config.cross_sectional_cv:
        eval_mode = 'cross'
    else:
        eval_mode = 'multi'

    slug = _score_slug(config.effective_score_type or config.score_type)
    postfix = f'_a{config.alpha}_{slug}_c{config.req_cal_windows}_{eval_mode}{"_asymm" if config.asymmetric else ""}{"_online" if config.online else ""}'
    return postfix


def _build_methods(
    method_names: list[str],
    horizon: int,
    alpha: float,
    aci_gamma: float,
    qi_lr: float,
    qi_ki: float,
    window_size: int,
    asymmetric: bool = False,
    acmcp_ncal: int = 10,
    acmcp_lr: float = 0.1,
    acmcp_integrate: bool = True,
    acmcp_scorecast: bool = True,
    acmcp_ki: float | None = None,
    acmcp_csat: float | None = None,
    cal_windows: int = 100,
    actual_horizon: int | None = None,
) -> tuple[dict[str, list], dict[str, list]]:
    """Instantiate one marginal predictor per horizon for each requested method.

    Returns
    -------
    (standard_methods, acmcp_methods)
        ``standard_methods`` maps name → list[ConformalPredictor] for methods
        that use the normal ``CPEvaluator.run()`` path.
        ``acmcp_methods`` maps name → list[AcMCP] for methods that use the
        ``CPEvaluator.run_acmcp()`` path.
    """
    from timecp.methods import (
        ACI,
        SPCI,
        AcMCP,
        AgACI,
        DtACI,
        QuantileIntegrator,
        SplitCP,
        TrailingWindow,
        WeightedCP,
    )

    standard: dict[str, list] = {}
    acmcp_dict: dict[str, list] = {}

    for name in method_names:
        if name == 'SplitCP':
            standard[name] = [
                SplitCP(alpha=alpha, asymmetric=asymmetric) for _ in range(horizon)
            ]
        elif name == 'ACI':
            standard[name] = [
                ACI(
                    alpha=alpha,
                    gamma=aci_gamma,
                    window_size=window_size,
                    asymmetric=asymmetric,
                )
                for _ in range(horizon)
            ]
        elif name == 'AgACI':
            standard[name] = [
                AgACI(alpha=alpha, window_size=window_size, asymmetric=asymmetric)
                for _ in range(horizon)
            ]
        elif name == 'PID':
            standard[name] = [
                QuantileIntegrator(
                    alpha=alpha,
                    lr=qi_lr,
                    KI=qi_ki,
                    window_size=window_size,
                    proportional_lr=True,
                    asymmetric=asymmetric,
                )
                for _ in range(horizon)
            ]
        elif name == 'DtACI':
            standard[name] = [
                DtACI(alpha=alpha, window_size=window_size, asymmetric=asymmetric)
                for _ in range(horizon)
            ]
        elif name == 'AcMCP':
            acmcp_dict[name] = [
                AcMCP(
                    alpha=alpha,
                    h=h + 1,
                    ncal=acmcp_ncal,
                    rolling=True,
                    integrate=acmcp_integrate,
                    scorecast=acmcp_scorecast,
                    lr=acmcp_lr,
                    Csat=acmcp_csat,
                    KI=acmcp_ki,
                )
                for h in range(horizon)
            ]
        elif name == 'SPCI':
            standard[name] = [
                SPCI(
                    alpha=alpha,
                    w=100,
                    n_estimators=10,
                    max_depth=2,
                    bins=1,
                    retrain_freq=1,
                )
                for _ in range(horizon)
            ]
        elif name == 'TrailingWindow':
            eff_horizon = actual_horizon if actual_horizon is not None else horizon
            baseline_window = max(cal_windows * eff_horizon, 1)
            standard[name] = [
                TrailingWindow(
                    alpha=alpha,
                    window_size=baseline_window,
                    asymmetric=asymmetric,
                )
                for _ in range(horizon)
            ]
        elif name == 'WeightedCP':
            eff_horizon = actual_horizon if actual_horizon is not None else horizon
            baseline_window = max(cal_windows * eff_horizon, 1)
            standard[name] = [
                WeightedCP(
                    alpha=alpha,
                    window_size=baseline_window,
                    decay=0.99,
                    asymmetric=asymmetric,
                )
                for _ in range(horizon)
            ]
        else:
            raise ValueError(
                f"Unknown method '{name}'. Valid choices: {_VALID_METHODS}"
            )

    return standard, acmcp_dict


def _build_multi_step_methods(
    method_names: list[str],
    alpha: float,
    copula_cal_split: float,
    copula_epochs: int,
    cafht_normalize: str,
    cafht_base_model: str,
    cafht_cal_split: float,
    cafht_q0: float,
) -> dict:
    """Instantiate one JointPredictor per requested joint method."""
    from timecp.methods import CAFHT, CopulaCPTS, JointCFRNN, NormMaxCP

    builders: dict = {}

    for name in method_names:
        if name == 'JointCFRNN':
            builders[name] = JointCFRNN(alpha=alpha)
        elif name == 'CopulaCPTS':
            builders[name] = CopulaCPTS(
                alpha=alpha,
                cal_split=copula_cal_split,
                epochs=copula_epochs,
            )
        elif name == 'NormMaxCP':
            builders[name] = NormMaxCP(alpha=alpha, normalize=cafht_normalize)
        elif name == 'CAFHT':
            builders[name] = CAFHT(
                alpha=alpha,
                base_model=cafht_base_model,  # type: ignore[arg-type]
                cal_split=cafht_cal_split,
                q0=cafht_q0,
            )
        elif name == 'CAFHT_PID':
            builders[name] = CAFHT(
                alpha=alpha,
                base_model='pid',
                cal_split=cafht_cal_split,
                q0=cafht_q0,
            )
        else:
            raise ValueError(
                f"Unknown joint method '{name}'. Valid choices: {_VALID_MULTI_STEP_METHODS}"
            )

    return builders


def _joint_asymmetric_suffix(method_names: list[str], asymmetric: bool) -> bool:
    """Determine whether joint methods warrant an ``_asymm`` filename suffix.

    The suffix is added only when asymmetry is actually relevant to the joint
    methods being evaluated, i.e. when the ``--asymmetric`` flag is set **and**
    at least one of the requested joint method constructors accepts an
    ``asymmetric`` parameter, or when a method is natively asymmetric.
    Currently no joint method satisfies either condition, so the suffix is
    never added — but the check is future-proof.
    """
    if not method_names:
        return False

    import inspect

    from timecp.methods import CAFHT, CopulaCPTS, JointCFRNN, NormMaxCP

    name_to_cls = {
        'JointCFRNN': JointCFRNN,
        'CopulaCPTS': CopulaCPTS,
        'NormMaxCP': NormMaxCP,
        'CAFHT': CAFHT,
        'CAFHT_PID': CAFHT,
    }

    any_ingests = False
    any_inherently = False
    for name in method_names:
        cls = name_to_cls.get(name)
        if cls is None:
            continue
        try:
            sig = inspect.signature(cls.__init__)
            if 'asymmetric' in sig.parameters:
                any_ingests = True
        except (ValueError, TypeError):
            pass
        if getattr(cls, 'natively_asymmetric', False):
            any_inherently = True

    return (asymmetric and any_ingests) or any_inherently


def _resolve_methods(
    methods: list[str] | None,
) -> tuple[list[str], list[str]]:
    """Classify raw ``--methods`` names into marginal and joint/multi-step lists.

    Names in :data:`_VALID_METHODS` are returned in the first list (marginal);
    names in :data:`_VALID_MULTI_STEP_METHODS` are returned in the second list
    (joint).  When *methods* is ``None`` the default is all marginal methods
    with no joint methods (backwards compatible).

    Raises
    ------
    ValueError
        If a name is not a recognised marginal or joint method.
    """
    method_names: list[str] = []
    multi_step_method_names: list[str] = []

    if methods is not None:
        for m in methods:
            if m in _VALID_METHODS:
                method_names.append(m)
            elif m in _VALID_MULTI_STEP_METHODS:
                multi_step_method_names.append(m)
            else:
                valid_choices = list(_VALID_METHODS) + list(_VALID_MULTI_STEP_METHODS)
                raise ValueError(
                    f'Unknown method "{m}". Valid choices: {valid_choices}'
                )
    else:
        method_names = list(_VALID_METHODS)
        multi_step_method_names = []

    return method_names, multi_step_method_names


# ---------------------------------------------------------------------------
# Single-task evaluation
# ---------------------------------------------------------------------------


@dataclass
class CPEvalConfig:
    alpha: float
    score_type: str
    method_names: list[str]
    run_multi_step: bool
    multi_step_method_names: list[str]
    aci_gamma: float
    qi_lr: float
    qi_ki: float
    window_size: int
    acmcp_ncal: int
    acmcp_lr: float
    acmcp_integrate: bool
    acmcp_scorecast: bool
    acmcp_ki: float | None
    acmcp_csat: float | None
    copula_cal_split: float
    copula_epochs: int
    cafht_normalize: str
    cafht_base_model: str
    cafht_cal_split: float
    cafht_q0: float
    single_step: bool
    cross_sectional_cv: bool
    req_cal_windows: int
    min_cal_windows: int
    covariate_strategy: str
    use_cache: bool
    asymmetric: bool
    online: bool
    force: bool
    save_intervals: bool = False
    effective_score_type: str = ''


def _get_num_series(task_dir: Path, meta: dict) -> int:
    """Retrieve or discover the number of series in the task predictions."""
    if 'num_series' in meta and meta['num_series'] is not None:
        return int(meta['num_series'])

    # Fallback to load shapes from files if num_series is not in metadata.
    try:
        import datasets as hf_datasets

        pred_dir = task_dir / 'predictions'
        if pred_dir.exists() and (pred_dir / 'dataset_dict.json').exists():
            ds_dict = hf_datasets.load_from_disk(str(pred_dir))
            for k in ds_dict.keys():
                return len(ds_dict[k])
    except Exception:
        pass

    try:
        import datasets as hf_datasets

        window_0 = task_dir / 'window_0'
        if window_0.exists():
            for name in ('predictions', 'ground_truth'):
                path = window_0 / name
                if path.exists() and (path / 'dataset_dict.json').exists():
                    ds_dict = hf_datasets.load_from_disk(str(path))
                    for k in ds_dict.keys():
                        return len(ds_dict[k])
    except Exception:
        pass

    try:
        import numpy as np

        if (task_dir / 'predictions.npz').exists():
            preds_npz = np.load(str(task_dir / 'predictions.npz'))
            for file_key in preds_npz.files:
                return preds_npz[file_key].shape[0]
    except Exception:
        pass

    return 1


def _evaluate_task(
    task_dir: Path,
    config: CPEvalConfig,
    output_dir: Path,
    cache: dict | None = None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Run CP evaluation for a single task directory."""

    meta_path = task_dir / 'metadata.json'
    if not meta_path.exists():
        print_warning(f'No metadata.json in {task_dir}. Skipping task.')
        return None, None

    with open(meta_path) as f:
        meta = json.load(f)

    n_series = _get_num_series(task_dir, meta)

    cal_windows: int = meta.get('cal_windows', 0)
    horizon: int = meta['horizon']
    total_windows: int = meta['total_windows']
    test_windows = total_windows - cal_windows

    active_cal_windows = cal_windows
    if not config.cross_sectional_cv:
        if config.req_cal_windows == -1:
            active_cal_windows = cal_windows
        elif cal_windows > config.req_cal_windows:
            active_cal_windows = config.req_cal_windows
        elif (
            cal_windows >= config.min_cal_windows
            and cal_windows < config.req_cal_windows
        ):
            print_warning(
                f'requested {config.req_cal_windows} calibration windows but only {cal_windows} available. Proceeding with {cal_windows}.'
            )
        elif cal_windows < config.min_cal_windows and cal_windows > 0:
            print_warning(
                f'Not enough calibration windows. Available {cal_windows} (< {config.min_cal_windows} minimum). Skipping task.'
            )
            return None, None

    df = None
    summary = None
    df_multi_step = None

    if cache is not None and not config.force:
        cache_key = (
            task_dir,
            config.alpha,
            config.score_type,
            active_cal_windows,
            config.single_step,
            config.cross_sectional_cv,
            config.asymmetric,
            config.online,
        )
        if cache_key in cache:
            print(
                f'  [CACHE HIT] Reusing previous results for active_cal_windows={active_cal_windows}'
            )
            df, summary, df_multi_step, eff_st = cache[cache_key]
            config.effective_score_type = eff_st

            output_dir.mkdir(parents=True, exist_ok=True)
            postfix = make_postfix(config)

            if df is not None:
                results_filename = f'cp_results{postfix}.csv'
                results_path = output_dir / results_filename
                summary_filename = f'cp_summary{postfix}.csv'
                summary_path = output_dir / summary_filename

                df.to_csv(results_path, index=False)
                print(f'  Saved per-horizon results -> {results_path}')
                if summary is not None:
                    summary.to_csv(summary_path, index=False)
                    print(f'  Saved marginal summary    -> {summary_path}')

                    print()
                    print(summary.to_string(index=False))
                    print()

            if df_multi_step is not None:
                multi_step_filename = f'cp_summary{postfix}.csv'
                multi_step_path = output_dir / multi_step_filename
                df_multi_step.to_csv(multi_step_path, index=False)
                print(f'  Saved joint summary        -> {multi_step_path}')

                print()
                print(df_multi_step.to_string(index=False))
                print()

            return summary, df_multi_step

    if config.cross_sectional_cv:
        print(
            f'  Task: {meta.get("task_name", task_dir.name)}\n'
            f'  H={horizon}  total_windows={total_windows}\n'
            f'  [cross-sectional CV: cal_windows={active_cal_windows}, cov_strategy={config.covariate_strategy}]',
            flush=True,
        )
    else:
        print(
            f'  Task: {meta.get("task_name", task_dir.name)}\n'
            f'  H={horizon}  cal={active_cal_windows}  test={test_windows}',
            flush=True,
        )

    if cal_windows == 0 and not config.cross_sectional_cv:
        print_warning(
            'cal_windows=0 — CP methods will have no calibration data.  '
            'Re-run forecast.py with --cp-cal-windows.'
        )

    if test_windows <= 0 and not config.cross_sectional_cv:
        print_warning(
            f'No test windows after cal split (total={total_windows}, cal={cal_windows}). Skipping task.'
        )
        return None, None

    output_dir.mkdir(parents=True, exist_ok=True)

    # print('DEBUG: Evaluator instantiation', flush=True)
    evaluator = CPEvaluator(
        predictions_dir=task_dir,
        horizon=horizon,
        cal_windows=cal_windows,
        active_cal_windows=active_cal_windows,
        alpha=config.alpha,
        score_type=config.score_type,
        save_intervals=config.save_intervals,
    )

    # Common kwargs for _build_methods
    _build_kw = dict(
        alpha=config.alpha,  # type: ignore[arg-type]
        aci_gamma=config.aci_gamma,  # type: ignore[arg-type]
        qi_lr=config.qi_lr,  # type: ignore[arg-type]
        qi_ki=config.qi_ki,  # type: ignore[arg-type]
        window_size=config.window_size,  # type: ignore[arg-type]
        asymmetric=config.asymmetric,
        acmcp_ncal=config.acmcp_ncal,  # type: ignore[arg-type]
        acmcp_lr=config.acmcp_lr,  # type: ignore[arg-type]
        acmcp_integrate=config.acmcp_integrate,  # type: ignore[arg-type]
        acmcp_scorecast=config.acmcp_scorecast,  # type: ignore[arg-type]
        acmcp_ki=config.acmcp_ki,
        acmcp_csat=config.acmcp_csat,
        cal_windows=active_cal_windows,
        actual_horizon=horizon,
    )

    summary = None
    df_multi_step = None

    # ------------------------------------------------------------------ #
    # Marginal evaluation                                                  #
    # ------------------------------------------------------------------ #
    if config.method_names:
        import pandas as pd

        all_dfs = []

        if config.single_step:
            # Single-step: build one predictor per method (horizon=1 trick),
            # then unwrap the single-element list.
            std_h1, acmcp_h1 = _build_methods(
                method_names=config.method_names,
                horizon=1,
                **_build_kw,  # type: ignore[arg-type]
            )
            if acmcp_h1:
                print_warning(
                    'AcMCP relies on per-horizon cross-correlations and '
                    'is not supported in single-step mode (skipped).'
                )
            single_methods = {name: preds[0] for name, preds in std_h1.items()}
            if single_methods:
                try:
                    df_ss = evaluator.run_single_step(
                        single_methods, use_cache=config.use_cache, online=config.online
                    )
                    all_dfs.append(df_ss)
                except Exception as exc:
                    print_error(f'Single-step evaluation failed: {exc}')

        elif config.cross_sectional_cv:
            # Cross-sectional CV mode
            standard_methods, acmcp_methods = _build_methods(
                method_names=config.method_names,
                horizon=horizon,
                **_build_kw,  # type: ignore[arg-type]
            )
            # Remove online adaptive methods
            batch_methods = {}
            for name, preds in standard_methods.items():
                if name in _ONLINE_ONLY_METHODS:
                    print_warning(
                        f'Skipping {name} in cross-sectional CV mode (online adaptive).'
                    )
                else:
                    batch_methods[name] = preds
            if acmcp_methods:
                print_warning(
                    'Skipping AcMCP in cross-sectional CV mode (online adaptive).'
                )

            if batch_methods:
                try:
                    df_cv = evaluator.run_cross_sectional_cv(
                        batch_methods,
                        cal_windows=config.req_cal_windows,
                        covariate_strategy=config.covariate_strategy,
                        use_cache=config.use_cache,
                    )
                    all_dfs.append(df_cv)
                except Exception as exc:
                    print_error(f'Cross-sectional CV evaluation failed: {exc}')

        else:
            # Default: multi-step (one predictor per horizon)
            # print('DEBUG: Building methods', flush=True)
            standard_methods, acmcp_methods = _build_methods(
                method_names=config.method_names,
                horizon=horizon,
                **_build_kw,  # type: ignore[arg-type]
            )

            if standard_methods:
                # print('DEBUG: Evaluator.run(standard_methods)', flush=True)
                try:
                    df_std = evaluator.run(standard_methods, use_cache=config.use_cache)
                    all_dfs.append(df_std)
                except Exception as exc:
                    print_error(f'Marginal evaluation failed: {exc}')

            if acmcp_methods:
                # print('DEBUG: Evaluator.run_acmcp(acmcp_methods)', flush=True)
                try:
                    df_acmcp = evaluator.run_acmcp(acmcp_methods)
                    all_dfs.append(df_acmcp)
                except Exception as exc:
                    print_error(f'AcMCP evaluation failed: {exc}')

        # Capture effective score type (may differ from requested if log fell back to abs)
        config.effective_score_type = evaluator._effective_score_type

        if all_dfs:
            df = pd.concat(all_dfs, ignore_index=True)

            postfix = make_postfix(config)
            results_filename = f'cp_results{postfix}.csv'
            results_path = output_dir / results_filename
            summary_filename = f'cp_summary{postfix}.csv'
            summary_path = output_dir / summary_filename

            df.to_csv(results_path, index=False)
            print(f'  Saved per-horizon results -> {results_path}')

            summary = (
                df.groupby('method')[
                    [
                        'coverage',
                        'joint_coverage',
                        'avg_width',
                        'winkler_score',
                        'runtime',
                    ]
                ]
                .mean()
                .reset_index()
            )
            if config.single_step:
                mode_val = 'single-step'
            elif config.cross_sectional_cv:
                mode_val = 'cross-sectional-cv'
            else:
                mode_val = 'multi-step'
            summary = _post_process_summary(
                df=summary,
                meta=meta,
                alpha=config.alpha,
                cal_windows=active_cal_windows,
                task_name=task_dir.name,
                n_series=n_series,
                score_type=config.effective_score_type or config.score_type,
                mode=mode_val,
            )

            summary.to_csv(summary_path, index=False)
            print(f'  Saved marginal summary    -> {summary_path}')

            print()
            print(summary.to_string(index=False))
            print()

    # ------------------------------------------------------------------ #
    # Joint evaluation                                                     #
    # ------------------------------------------------------------------ #
    if config.run_multi_step and config.multi_step_method_names:
        multi_step_methods = _build_multi_step_methods(
            method_names=config.multi_step_method_names,
            alpha=config.alpha,
            copula_cal_split=config.copula_cal_split,
            copula_epochs=config.copula_epochs,
            cafht_normalize=config.cafht_normalize,
            cafht_base_model=config.cafht_base_model,
            cafht_cal_split=config.cafht_cal_split,
            cafht_q0=config.cafht_q0,
        )

        try:
            if config.cross_sectional_cv:
                df_multi_step = evaluator.run_multi_step_cross_sectional_cv(
                    multi_step_methods,
                    cal_windows=cal_windows,
                    covariate_strategy=config.covariate_strategy,
                )
            else:
                df_multi_step = evaluator.run_multi_step(
                    multi_step_methods, use_cache=config.use_cache
                )
        except Exception as exc:
            print_error(f'Joint evaluation failed: {exc}')
        else:
            # Rename marginal_coverage → coverage to match the marginal summary schema.
            df_multi_step = df_multi_step.rename(
                columns={'marginal_coverage': 'coverage'}
            )
            # Reorder to match the marginal summary column order exactly.
            df_multi_step = df_multi_step[
                [
                    'method',
                    'coverage',
                    'joint_coverage',
                    'avg_width',
                    'winkler_score',
                    'runtime',
                ]
            ]
            joint_mode = (
                'cross-sectional-cv' if config.cross_sectional_cv else 'multi-step'
            )
            df_multi_step = _post_process_summary(
                df=df_multi_step,
                meta=meta,
                alpha=config.alpha,
                cal_windows=active_cal_windows,
                task_name=task_dir.name,
                n_series=n_series,
                score_type='joint',
                mode=joint_mode,
            )

            postfix = make_postfix(config)
            multi_step_filename = f'cp_summary{postfix}.csv'
            multi_step_path = output_dir / multi_step_filename
            df_multi_step.to_csv(multi_step_path, index=False)
            print(f'  Saved joint summary        -> {multi_step_path}')

            print()
            print(df_multi_step.to_string(index=False))
            print()

    if cache is not None:
        cache_key = (
            task_dir,
            config.alpha,
            config.score_type,
            active_cal_windows,
            config.single_step,
            config.cross_sectional_cv,
            config.asymmetric,
            config.online,
        )
        cache[cache_key] = (df, summary, df_multi_step, config.effective_score_type)

    return summary, df_multi_step  # type: ignore[return-value]


def categorize_horizon(horizon: int) -> str:
    """Categorize prediction horizon length into short, medium, or long.

    Thresholds:
    - Short: H <= 30
    - Medium: 30 < H <= 90
    - Long: H > 90
    """
    if horizon <= 30:
        return 'short'
    elif horizon <= 90:
        return 'medium'
    else:
        return 'long'


def _post_process_summary(
    df: pd.DataFrame,
    meta: dict,
    alpha: float,
    cal_windows: int,
    task_name: str,
    n_series: int,
    score_type: str | None = None,
    mode: str | None = None,
) -> pd.DataFrame:
    df = df.copy()
    df.insert(0, 'task', meta.get('task_name', task_name))
    df.insert(1, 'model', meta.get('model_name', ''))
    df.insert(2, 'alpha', alpha)

    idx = 3
    if score_type is not None:
        df.insert(idx, 'score_type', score_type)
        idx += 1
    if mode is not None:
        df.insert(idx, 'mode', mode)
        idx += 1

    horizon = meta.get('horizon', 0)
    df.insert(idx, 'horizon', horizon)
    idx += 1
    df.insert(idx, 'horizon_category', categorize_horizon(horizon))
    idx += 1

    df.insert(idx, 'cal_windows', cal_windows)
    df.insert(idx + 1, 'n_series', n_series)

    # Compute scale-normalized metrics
    native_row = df[df['method'] == 'Native']
    if not native_row.empty:
        native_width = float(list(native_row['avg_width'])[0])
        native_winkler = float(list(native_row['winkler_score'])[0])
        nw = native_width if native_width > 1e-8 else 1.0
        nws = native_winkler if native_winkler > 1e-8 else 1.0
        df['scaled_avg_width'] = df['avg_width'] / nw
        df['scaled_winkler_score'] = df['winkler_score'] / nws
    else:
        df['scaled_avg_width'] = float('nan')
        df['scaled_winkler_score'] = float('nan')

    df = df.drop(
        columns=[
            'avg_width',
            'winkler_score',
            'native_avg_width',
            'native_winkler_score',
        ],
        errors='ignore',
    )

    # Reorder columns: move 'runtime' to the end
    cols = [c for c in df.columns if c != 'runtime']
    if 'runtime' in df.columns:
        cols.append('runtime')
    df = df[cols]

    # Round metrics
    df = df.round(3)
    return df


def _aggregate_and_save(
    dfs: list[pd.DataFrame],
    group_cols: list[str],
    metrics: list[str],
    out_path: Path,
    label: str,
) -> None:
    global_df = pd.concat(dfs, ignore_index=True)
    group_cols = [c for c in group_cols if c in global_df.columns]
    metrics = [m for m in metrics if m in global_df.columns]

    agg_df = global_df.groupby(group_cols)[metrics].agg(['mean', 'median'])
    agg_df.columns = [f'{c[0]}_{c[1]}' if c[1] else c[0] for c in agg_df.columns]

    if 'task' in global_df.columns and 'n_series' in global_df.columns:
        task_n_series = global_df.groupby(group_cols + ['task'])['n_series'].first()
        agg_df['n_series'] = task_n_series.groupby(level=group_cols).sum().astype(int)

    agg_df = agg_df.reset_index()

    # Top-level across-task summary uses `horizon` as the column name for the
    # horizon category (short/medium/long), rather than `horizon_category`.
    if 'horizon_category' in agg_df.columns:
        agg_df = agg_df.rename(columns={'horizon_category': 'horizon'})

    # Sort the horizon column in reverse alphabetical order: short, medium, long.
    if 'horizon' in agg_df.columns:
        agg_df = agg_df.sort_values('horizon', ascending=False, kind='stable')

    if 'n_series' in agg_df.columns and 'cal_windows' in agg_df.columns:
        cols = list(agg_df.columns)
        cols.remove('n_series')
        loc = cols.index('cal_windows')
        cols.insert(loc + 1, 'n_series')
        agg_df = agg_df[cols]

    # Reorder columns: move runtime columns to the end
    cols = [c for c in agg_df.columns if not c.startswith('runtime')]
    runtime_cols = [c for c in agg_df.columns if c.startswith('runtime')]
    cols.extend(runtime_cols)
    agg_df = agg_df[cols]

    # Round metrics
    agg_df = agg_df.round(3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    agg_df.to_csv(out_path, index=False)
    print(f'\nSaved global {label} summary -> {out_path}')
    print_df = agg_df[[c for c in agg_df.columns if not c.endswith('_mean')]].copy()
    print_df.columns = [c.removesuffix('_median') for c in print_df.columns]
    print(print_df.to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    set_global_seed(args.seed)

    # Determine run targets: list of (predictions_root, is_recursive)
    run_targets: list[tuple[Path, bool]] = []

    if args.predictions_dir:
        run_targets.append((Path(args.predictions_dir), args.recursive))
    elif args.predictions:
        run_targets.append((Path(args.predictions), args.recursive))
    elif args.config:
        import yaml

        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        out_str = cfg.get('output_dir', 'results')
        run_targets.append((Path(out_str), True))
    else:
        # Base directory mode
        base_dir = Path(args.base_dir)
        if not base_dir.exists():
            print(f'Error: Base directory does not exist: {base_dir}', file=sys.stderr)
            return 1

        for forester in args.forecasters:
            for base_dataset in args.base_datasets:
                pair_dir = base_dir / forester / base_dataset
                if pair_dir.is_dir():
                    run_targets.append((pair_dir, True))
                else:
                    print_warning(
                        f'Pair directory {pair_dir} does not exist. Skipping.'
                    )

        if not run_targets:
            print_error(
                f"No valid model/base-dataset pair directories found in base directory '{base_dir}' "
                f'matching forecasters {args.forecasters} and base datasets {args.base_datasets}.'
            )
            return 1

    # Resolve methods into marginal and joint/multi-step based on --methods.
    # Methods are auto-classified: names in _VALID_METHODS run as marginal
    # (per-horizon) calibration; names in _VALID_MULTI_STEP_METHODS run as
    # joint (simultaneous-coverage) evaluation.  The two can be freely mixed
    # in a single --methods invocation.
    try:
        method_names, multi_step_method_names = _resolve_methods(args.methods)
    except ValueError as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1

    # --single-step only affects marginal methods; joint methods are unaffected.
    single_step: bool = args.single_step
    cross_sectional_cv: bool = args.cross_sectional_cv
    min_cal_windows: int = args.min_cal_windows
    covariate_strategy: str = args.covariate_strategy

    import itertools

    for predictions_root, is_recursive in run_targets:
        if not predictions_root.exists():
            print(f'Error: Predictions directory does not exist: {predictions_root}')
            return 1

        task_cache: dict = {}

        if is_recursive:
            meta_files = [
                m for m in predictions_root.rglob('metadata.json') if m.is_file()
            ]
            task_dirs = []
            for m in meta_files:
                try:
                    with open(m) as f:
                        meta = json.load(f)
                    if (
                        meta.get('test_windows', meta.get('total_windows', 0)) > 0
                        or meta.get('split') == 'test'
                        or args.cross_sectional_cv
                    ):
                        task_dirs.append(m.parent)
                except Exception:
                    pass
            if not task_dirs:
                print(f'No valid metadata.json found under {predictions_root}')
                if len(run_targets) > 1:
                    continue
                else:
                    return 1
            print(f'Found {len(task_dirs)} task(s) under {predictions_root}')
        else:
            task_dirs = [predictions_root]

        # Shared aggregation schema for both marginal and joint summaries.
        summary_group_cols = [
            'model',
            'alpha',
            'score_type',
            'mode',
            'horizon_category',
            'cal_windows',
            'method',
        ]
        summary_metrics = [
            'coverage',
            'joint_coverage',
            'avg_width',
            'winkler_score',
            'scaled_avg_width',
            'scaled_winkler_score',
            'runtime',
        ]

        # ------------------------------------------------------------------ #
        # Marginal evaluation: once per (alpha, score_type, cal_windows)      #
        # ------------------------------------------------------------------ #
        if method_names:
            for alpha, score_type, cal_win in itertools.product(
                args.alpha, args.score_type, args.cal_windows
            ):
                # Score-type-specific asymmetric handling
                forced_asymmetric = args.asymmetric
                if score_type == 'signed':
                    if not forced_asymmetric:
                        print(
                            "  score_type='signed' requires asymmetric intervals; "
                            'constructing marginal methods with asymmetric=True.'
                        )
                        forced_asymmetric = True
                elif score_type in ('distributional', 'cdf_tail'):
                    if forced_asymmetric:
                        print_warning(
                            f'score_type={score_type!r} does not support asymmetric '
                            'mode; using symmetric.'
                        )
                        forced_asymmetric = False

                active_method_names = method_names
                if score_type in ('distributional', 'cdf_tail'):
                    active_method_names = [m for m in method_names if m != 'SPCI']

                config = CPEvalConfig(
                    alpha=alpha,
                    score_type=score_type,
                    method_names=active_method_names,
                    run_multi_step=False,
                    multi_step_method_names=[],
                    aci_gamma=args.aci_gamma,  # type: ignore[arg-type]
                    qi_lr=args.qi_lr,  # type: ignore[arg-type]
                    qi_ki=args.qi_ki,  # type: ignore[arg-type]
                    window_size=args.window_size,  # type: ignore[arg-type]
                    acmcp_ncal=args.acmcp_ncal,  # type: ignore[arg-type]
                    acmcp_lr=args.acmcp_lr,  # type: ignore[arg-type]
                    acmcp_integrate=not args.acmcp_no_integrate,  # type: ignore[arg-type]
                    acmcp_scorecast=not args.acmcp_no_scorecast,  # type: ignore[arg-type]
                    acmcp_ki=args.acmcp_ki,
                    acmcp_csat=args.acmcp_csat,
                    copula_cal_split=args.copula_cal_split,  # type: ignore[arg-type]
                    copula_epochs=args.copula_epochs,  # type: ignore[arg-type]
                    cafht_normalize=args.cafht_normalize,
                    cafht_base_model=args.cafht_base_model,
                    cafht_cal_split=args.cafht_cal_split,  # type: ignore[arg-type]
                    cafht_q0=args.cafht_q0,  # type: ignore[arg-type]
                    single_step=single_step,
                    cross_sectional_cv=cross_sectional_cv,
                    req_cal_windows=cal_win,
                    min_cal_windows=min_cal_windows,
                    covariate_strategy=covariate_strategy,
                    use_cache=args.use_cache,
                    asymmetric=forced_asymmetric,
                    online=args.online,
                    force=args.force,
                    save_intervals=args.save_intervals,
                )

                all_summaries = []

                for task_dir in task_dirs:
                    output_dir = Path(args.output) if args.output else task_dir
                    print(
                        f'\n--- {task_dir} [alpha={alpha}, score_type={score_type}, cal_windows={cal_win}] ---'
                    )
                    summary, _ = _evaluate_task(
                        task_dir=task_dir,
                        config=config,
                        output_dir=output_dir,
                        cache=task_cache,
                    )
                    if summary is not None:
                        all_summaries.append(summary)

                if is_recursive and all_summaries:
                    _aggregate_and_save(
                        dfs=all_summaries,
                        group_cols=summary_group_cols,
                        metrics=summary_metrics,
                        out_path=predictions_root
                        / f'cp_summary{make_postfix(config)}.csv',
                        label='marginal',
                    )

        # ------------------------------------------------------------------ #
        # Joint evaluation: once per (alpha, cal_windows),                    #
        # score-type-independent (joint methods compute scores internally).  #
        # ------------------------------------------------------------------ #
        if multi_step_method_names:
            joint_asymm = _joint_asymmetric_suffix(
                multi_step_method_names, args.asymmetric
            )
            for alpha, cal_win in itertools.product(args.alpha, args.cal_windows):
                config = CPEvalConfig(
                    alpha=alpha,
                    score_type='joint',
                    method_names=[],
                    run_multi_step=True,
                    multi_step_method_names=multi_step_method_names,
                    aci_gamma=args.aci_gamma,  # type: ignore[arg-type]
                    qi_lr=args.qi_lr,  # type: ignore[arg-type]
                    qi_ki=args.qi_ki,  # type: ignore[arg-type]
                    window_size=args.window_size,  # type: ignore[arg-type]
                    acmcp_ncal=args.acmcp_ncal,  # type: ignore[arg-type]
                    acmcp_lr=args.acmcp_lr,  # type: ignore[arg-type]
                    acmcp_integrate=not args.acmcp_no_integrate,  # type: ignore[arg-type]
                    acmcp_scorecast=not args.acmcp_no_scorecast,  # type: ignore[arg-type]
                    acmcp_ki=args.acmcp_ki,
                    acmcp_csat=args.acmcp_csat,
                    copula_cal_split=args.copula_cal_split,  # type: ignore[arg-type]
                    copula_epochs=args.copula_epochs,  # type: ignore[arg-type]
                    cafht_normalize=args.cafht_normalize,
                    cafht_base_model=args.cafht_base_model,
                    cafht_cal_split=args.cafht_cal_split,  # type: ignore[arg-type]
                    cafht_q0=args.cafht_q0,  # type: ignore[arg-type]
                    single_step=False,
                    cross_sectional_cv=cross_sectional_cv,
                    req_cal_windows=cal_win,
                    min_cal_windows=min_cal_windows,
                    covariate_strategy=covariate_strategy,
                    use_cache=args.use_cache,
                    asymmetric=joint_asymm,
                    online=False,
                    force=args.force,
                    save_intervals=args.save_intervals,
                )

                all_multi_step = []

                for task_dir in task_dirs:
                    output_dir = Path(args.output) if args.output else task_dir
                    print(
                        f'\n--- {task_dir} [alpha={alpha}, score_type=joint, cal_windows={cal_win}] ---'
                    )
                    _, df_multi_step = _evaluate_task(
                        task_dir=task_dir,
                        config=config,
                        output_dir=output_dir,
                        cache=task_cache,
                    )
                    if df_multi_step is not None:
                        all_multi_step.append(df_multi_step)

                if is_recursive and all_multi_step:
                    _aggregate_and_save(
                        dfs=all_multi_step,
                        group_cols=summary_group_cols,
                        metrics=summary_metrics,
                        out_path=predictions_root
                        / f'cp_summary{make_postfix(config)}.csv',
                        label='joint',
                    )

    print('Done.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
