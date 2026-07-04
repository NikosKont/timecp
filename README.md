# time-cp

Benchmarking and inference toolkit for time-series foundation models, with built-in conformal prediction evaluation. Provides a unified `Forecaster` API, dataset loading utilities for [GiftEval](https://huggingface.co/datasets/Salesforce/GiftEval), [TIME](https://huggingface.co/datasets/Real-TSF/TIME), and [FEV](https://github.com/autogluon/fev) benchmarks, and a two-stage pipeline that (1) runs batch inference and (2) evaluates conformal prediction methods against the models' native quantile predictions.

- [time-cp](#time-cp)
  - [Quickstart](#quickstart)
  - [Project structure](#project-structure)
  - [Supported models](#supported-models)
  - [Forecaster API](#forecaster-api)
  - [Two-stage pipeline](#two-stage-pipeline)
    - [Stage 1 — Inference (`scripts/forecast.py`)](#stage-1--inference-scriptsforecastpy)
      - [forecast.py options](#forecastpy-options)
      - [Output layout](#output-layout)
    - [Stage 1b — Splitting (`scripts/split_predictions.py`)](#stage-1b--splitting-scriptssplit_predictionspy)
    - [Stage 2 — CP evaluation (`scripts/cp_eval.py`)](#stage-2--cp-evaluation-scriptscp_evalpy)
      - [Nonconformity scores](#nonconformity-scores)
      - [cp\_eval.py options](#cp_evalpy-options)
      - [Output layout](#output-layout-1)
  - [Conformal prediction methods](#conformal-prediction-methods)
    - [Base classes](#base-classes)
    - [Asymmetric intervals](#asymmetric-intervals)
    - [Marginal methods (`ConformalPredictor`)](#marginal-methods-conformalpredictor)
    - [Joint methods (`JointPredictor`)](#joint-methods-jointpredictor)
    - [CPEvaluator API](#cpevaluator-api)
  - [Calibration split strategy](#calibration-split-strategy)
  - [Dataset utilities](#dataset-utilities)
    - [Loading datasets](#loading-datasets)
    - [Building fev.Task objects](#building-fevtask-objects)
  - [Development](#development)

## Quickstart
We use `uv` for project management. If you haven't already, you should install it by following the instructions [here](https://docs.astral.sh/uv/getting-started/installation/).

To reproduce results for (multi-step adapted) `AgAci`, `DtACI`, `SplitCP` and `WeightedCP` using `distributional`, `cdf_tail`, `cqr`and `abs` scores on the `fev-bench mini` benchmark with the `chronos2` base model, run:
```bash
uv run python scripts/forecast.py \
    --model chronos2 \
    --tasks experiments/fev-bench_mini.yaml \
    --cal-windows 50

uv run python scripts/cp_eval.py \
    --forecasters chronos2 \
    --base-datasets fev-bench_mini \
    --alpha 0.2 \
    --methods AgACI DtACI SplitCP WeightedCP \
    --score-type distributional cdf_tail cqr abs \
    --min-cal-windows 1 \                       # Don't skip tasks even if 1 calibration window is available
    --cal-windows 50                            # Clip calibration windows to 50 if a task has more available

uv run python scripts/parse_summaries.py;uv run python scripts/generate_dashboard.py
```
The project environment will be automatically created by uv on the first run. Per-task calibration results will be saved to `results/chronos2/fev_bench_mini/<task>`. Global summaries will be saved to `results/chronos2/fev_bench_mini`, and an interactive dashboard will be saved to `results/dashboard.html`. Tested on a Windows 10 machine with an NVIDIA RTX 3080 GPU and 16GB RAM.

Full paper results can be reproduced with:
```bash
uv run python scripts/forecast.py \
    --model chronos2 tirex flowstate timesfm \
    --tasks experiments/fev-bench_mini.yaml \
    --cal-windows 50

uv run python scripts/cp_eval.py \
    --forecasters chronos2 tirex flowstate timesfm \
    --base-datasets fev-bench_mini \
    --alpha 0.2 \
    --methods ACI AgACI DtACI PID AcMCP TrailingWindow WeightedCP SplitCP \
    --score-type abs cqr squared signed iqr_scaled mad_scaled scaled_cqr distributional cdf_tail log diff \
    --min-cal-windows 1 \
    --cal-windows 50
```
## Project structure

```
src/
  timecp/
    data/          # Dataset loading, GiftEval/TIME→FEV conversion, task factory
    methods/       # CP methods (marginal and joint — see below)
    models/        # Forecasting model wrappers (each is a fev.ForecastingModel)
    base.py        # ConformalPredictor and JointPredictor base classes
    cp_eval.py     # CPEvaluator — marginal and joint CP evaluation
    evaluation.py  # compare_methods(), rolling_metrics() helpers
scripts/
  forecast.py           # Batch inference + ground truth saving CLI
  cp_eval.py            # Post-forecast conformal prediction evaluation CLI
  generate_dashboard.py # Generates interactive dashboard from summary CSVs
  parse_summaries.py    # Parses all summary CSVs into a single dashboard CSV
experiments/       # Benchmark YAML configs (FEV, TIME, GiftEval)
data/              # Local dataset cache
```

## Supported models

| Key                       | Model     | Default checkpoint                |
| ------------------------- | --------- | --------------------------------- |
| `chronos2` (or `chronos`) | Chronos-2 | `amazon/chronos-2`                |
| `tirex`                   | TiRex     | `NX-AI/TiRex`                     |
| `flowstate`               | FlowState | `ibm-research/flowstate`          |
| `timesfm`                 | TimesFM   | `google/timesfm-2.5-200m-pytorch` |

## Forecaster API

```python
from models import Forecaster
import numpy as np

# Factory pattern — returns the appropriate subclass
model = Forecaster('tirex')

# Single series (1-D)
past = np.random.randn(512)
point, quantiles = model.predict(past, horizon=24, output_format='numpy')
# point:     (24,)
# quantiles: (24, Q)  Q = number of quantile levels

# Batched (list of variable-length series)
batch = [np.random.randn(n) for n in [256, 512, 1024]]
point, quantiles = model.predict(batch, horizon=24, output_format='numpy')
# point:     (3, 24)
# quantiles: (3, 24, Q)

# Point forecast only
point = model.predict(past, horizon=24, forecast_type='point', output_format='numpy')

# Specific quantile levels
_, q = model.predict(past, horizon=24, forecast_type='quantile',
                     quantile_levels=[0.1, 0.5, 0.9], output_format='numpy')
```

## Two-stage pipeline

### Stage 1 — Inference (`scripts/forecast.py`)

Runs any model against any benchmark config, saves per-window predictions **and ground truth** to disk, and writes a `metadata.json` with the calibration/test split information.

When `--cp-cal-windows N` is passed, the first *N* windows of each task are reserved for conformal prediction calibration. If omitted, defaults to the task's configuration (using the `cal_windows` top-level parameter). Tasks where fewer than 10 calibration windows are possible are skipped with a warning. Set to `0` to disable calibration splitting entirely.

```bash
# Multi-model batch inference on FEV benchmark
uv run python scripts/forecast.py \
    --model tirex chronos2 timesfm \
    --tasks experiments/fev_bench.yaml \
    --output results/

# TIME benchmark (auto-evaluates short, medium, and long terms)
uv run python scripts/forecast.py \
    --model chronos2 \
    --tasks experiments/TIME_config.yaml \
    --split-series 0.5 \
    --output results/ \
    --evaluate

# With explicit CP calibration split override
uv run python scripts/forecast.py \
    --model tirex \
    --tasks experiments/TIME_config.yaml \
    --cp-cal-windows 30 \
    --output results/
```

#### forecast.py options

| Flag                    | Default       | Description                                                                                                                            |
| ----------------------- | ------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `--model`, `-M`         | *(required)*  | Model key(s): `tirex`, `chronos2`, `flowstate`, `timesfm`. Accepts multiple models.                                                    |
| `--model-id`            | model default | Override HuggingFace checkpoint id                                                                                                     |
| `--tasks`, `-T`         | *(required)*  | Path(s) to benchmark YAML config(s). Accepts multiple files.                                                                           |
| `--quantile-levels`     | `0.1…0.9`     | Quantile levels to predict (TIME configs only)                                                                                         |
| `--cp-cal-windows N`    | config        | Leading windows reserved for CP calibration. Overrides the config file value if set. Clamped per-dataset. Set to `0` to disable.       |
| `--split-series`        | None          | Ratio (e.g. `0.5`) to split the `N` series dimension into two mutually exclusive subsets, saving `<task>_cal` and `<task>_test` tasks. |
| `--force`               | off           | Run inference even if `metadata.json` exists in the target directory.                                                                  |
| `--device`              | auto          | Device override, e.g. `cuda`, `cpu`, `cuda:1`                                                                                          |
| `--batch-size`          | `256`         | Series per inference batch. *Automatically reduced if CUDA OutOfMemoryError occurs.*                                                   |
| `--gpu-memory-fraction` | `0.95`        | Restricts PyTorch max VRAM to force clean OOM errors rather than silently swapping to slow shared system memory on Windows.            |
| `--output`              | Config-based  | Output directory. Falls back to config `output_dir` or `results/`.                                                                     |
| `--evaluate`            | off           | Compute and save FEV evaluation metrics (on test windows only)                                                                         |

#### Output layout

```
<output_dir>/
  <task_name>/
    <model_name>/
      window_0/
        predictions/            # DatasetDict (Arrow): point + quantile forecasts
        ground_truth/           # DatasetDict (Arrow): actual target values
      window_1/
        ...
      metadata.json             # horizon, cal_windows, test_windows, quantile_levels
      evaluation.json           # FEV metrics (only with --evaluate)
  summary.csv                   # one row per task (only with --evaluate)
```

### Stage 1b — Splitting (`scripts/split_predictions.py`)

If you want to perform intra-dataset splitting (covariate/series splitting) after inference has already run, you can use the standalone split utility. It avoids re-running the heavy model inference.

```bash
uv run python scripts/split_predictions.py --input results/ETTm2__T/tirex --split 0.5 --seed 42
```
This produces `results/ETTm2__T_cal/tirex` and `results/ETTm2__T_test/tirex`.

### Stage 2 — CP evaluation (`scripts/cp_eval.py`)

Loads the saved predictions, splits them using `metadata.json`, calibrates CP methods on the calibration windows, and evaluates them on the test windows. Methods passed to `--methods` are automatically classified as marginal or joint and dispatched to the appropriate evaluation path:

- **Marginal** (default): per-horizon independent evaluation; tests how methods adapt step-by-step across windows.
- **Marginal single-step** (`--single-step`): flattens all `(W, H)` horizon steps into a single `(W*H)` sequence before evaluation; tests raw temporal adaptation without horizon-specific structures. Only affects marginal methods; joint methods are unaffected.
- **Marginal cross-sectional cross-validation** (`--cross-sectional-cv`): evaluates methods using N-split cross-validation across the series dimension instead of temporal splits. Online adaptive methods are skipped as they require temporal sequences.
- **Joint / multi-step**: simultaneous coverage over all H steps; one predictor over full `(C, N, H)` windows. Triggered automatically when joint methods (e.g. `CopulaCPTS`, `JointCFRNN`) are passed to `--methods`. Can be combined with `--cross-sectional-cv` for joint CV.

Marginal and joint methods can be freely mixed in a single `--methods` invocation.

```bash
# Marginal evaluation — all default methods
uv run python scripts/cp_eval.py \
    --predictions results/tirex \
    --alpha 0.1 \
    --recursive

# Joint evaluation only
uv run python scripts/cp_eval.py \
    --predictions results/tirex/electricity__H \
    --alpha 0.1 \
    --methods JointCFRNN CopulaCPTS CAFHT

# Mixed marginal + joint in one pass (methods are auto-classified)
uv run python scripts/cp_eval.py \
    --predictions results/tirex/electricity__H \
    --alpha 0.1 \
    --methods SplitCP ACI AgACI PID DtACI JointCFRNN CopulaCPTS CAFHT

# Single-step mode: ACI/AgACI run single-step, CopulaCPTS stays joint
uv run python scripts/cp_eval.py \
    --predictions results/tirex/electricity__H \
    --alpha 0.1 \
    --single-step \
    --methods AgACI ACI CopulaCPTS

# AcMCP (asymmetric scorecaster) alongside standard methods
uv run python scripts/cp_eval.py \
    --predictions results/tirex/electricity__H \
    --alpha 0.1 \
    --methods SplitCP ACI AgACI PID AcMCP DtACI

# CQR scores, specific marginal methods
uv run python scripts/cp_eval.py \
    --predictions results/tirex/electricity__H \
    --alpha 0.1 \
    --score-type cqr \
    --methods ACI AgACI

# Multiple score types in one pass
uv run python scripts/cp_eval.py \
    --predictions results/tirex/electricity__H \
    --alpha 0.1 \
    --score-type abs cqr iqr_scaled distributional cdf_tail \
    --methods SplitCP ACI
```

#### Nonconformity scores

| Score type       | Formula                                            | Interval inversion                           | Requirements                                       | When to use                                                                | Caveats                                                                            |
| ---------------- | -------------------------------------------------- | -------------------------------------------- | -------------------------------------------------- | -------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `abs`            | `                                                  | y - yhat                                     | `                                                  | `[yhat - q, yhat + q]`                                                     | point forecasts                                                                    | Default; simple residual score                      | —                                                                                                |
| `squared`        | `(y - yhat)^2`                                     | `[yhat - sqrt(q), yhat + sqrt(q)]`           | point forecasts                                    | When large errors should be penalised more                                 | Half-width is `sqrt(q)`                                                            |
| `signed`         | `y - yhat` (signed)                                | `[yhat - q_lo, yhat + q_up]`                 | point forecasts; **forces `--asymmetric`**         | When lower/upper tail errors differ                                        | Cannot run symmetric; CLI auto-forces asymmetric                                   |
| `iqr_scaled`     | `                                                  | y - yhat                                     | / (q_high - q_low)`                                | `[yhat - q * width, yhat + q * width]`                                     | native low/high quantiles                                                          | Locally adaptive intervals                          | Per-observation native width as denominator                                                      |
| `mad_scaled`     | `                                                  | y - yhat                                     | / rho_h`                                           | `[yhat - q * rho_h, yhat + q * rho_h]`                                     | point forecasts; calibration residuals                                             | Robust locally adaptive                             | `rho_h` = `1.4826 * MAD` per horizon                                                             |
| `cqr`            | `max(q_low - y, y - q_high)`                       | `[q_low - q, q_high + q]`                    | native low/high quantiles                          | Standard CQR for quantile models                                           | Negative `q` shrinks the interval                                                  |
| `scaled_cqr`     | `max((q_low - y)/sigma_lo, (y - q_high)/sigma_hi)` | `[q_low - q*sigma_lo, q_high + q*sigma_hi]`  | native low/high quantiles; calibration tail scales | Asymmetric scaled CQR                                                      | Separate per-tail robust scales                                                    |
| `distributional` | `mean_pinball(y) - min_pinball`                    | Convex root-finding inversion                | all available quantile columns                     | CRPS-like distributional calibration                                       | Inverted by binary search, not native-interval expansion; asymmetric not supported |
| `cdf_tail`       | `max(alpha/2 - Fhat(y), Fhat(y) - (1-alpha/2), 0)` | Probability-band inversion via quantile grid | all available quantile columns                     | **Recommended** distribution-native interval score for quantile-grid TSFMs | Calibrates tail probability, not pinball; asymmetric not supported                 |
| `log`            | `                                                  | log(y + eps) - log(yhat + eps)               | `                                                  | `[exp(log(yhat+eps) - q) - eps, exp(log(yhat+eps) + q) - eps]`             | positive target + point forecasts                                                  | Multiplicative-error data                           | Falls back to `abs` on nonpositive calibration/prediction; nonpositive test labels are uncovered |
| `diff`           | `                                                  | e_t - e_{t-1}                                | ` where `e_t = y_t - yhat_t`                       | `[yhat + e_{t-1} - q, yhat + e_{t-1} + q]`                                 | point forecasts; previous residual stream                                          | Online/stateful score for residual-change detection | Interval centered at `yhat + prev_error`; requires previous-window ground truth                  |

#### cp_eval.py options

**General**

| Flag                   | Default         | Description                                                                                                                                                  |
| ---------------------- | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `--predictions`        | *(required)*    | Task directory, or config file (with `--config`), or model root (with `--recursive`)                                                                         |
| `--config`, `-T`       | none            | Read `output_dir` and task list from YAML config to auto-discover tasks                                                                                      |
| `--alpha`              | `0.1`           | Target miscoverage rate (1 − coverage)                                                                                                                       |
| `--score-type`         | `abs`           | Nonconformity score (see table below). `signed` forces `--asymmetric`.                                                                                       |
| `--methods`            | all marginal    | CP methods to run; auto-classified as marginal (SplitCP, ACI, AgACI, DtACI, PID, AcMCP, SPCI) or joint (JointCFRNN, CopulaCPTS, NormMaxCP, CAFHT, CAFHT_PID) |
| `--recursive`          | off             | Evaluate all task subdirectories under `--predictions`                                                                                                       |
| `--output`             | predictions dir | Directory for output CSVs                                                                                                                                    |
| `--window-size`        | `100`           | Sliding window size for methods that support it                                                                                                              |
| `--single-step`        | off             | Evaluate marginal methods on a flattened `(W*H)` single-step sequence (joint methods unaffected)                                                             |
| `--online`             | off             | Evaluate adaptive methods in online mode (updating state after each step within a window, instead of block/batch update per window)                          |
| `--cross-sectional-cv` | off             | Enable N-split cross-validation over time series entities instead of temporal splits.                                                                        |
| `--cal-windows`        | `100`           | Target number of calibration windows to use. Use `-1` to use all available.                                                                                  |
| `--min-cal-windows`    | `10`            | Minimum total calibration windows needed to proceed with evaluation.                                                                                         |
| `--covariate-strategy` | `targets_only`  | Strategy for calibration series: `targets_only`, `covariates_only`, or `all`.                                                                                |

**Marginal method tuning**

| Flag                   | Default | Description                                                      |
| ---------------------- | ------- | ---------------------------------------------------------------- |
| `--aci-gamma`          | `0.005` | Step size γ for ACI                                              |
| `--qi-lr`              | `0.1`   | Learning rate for PID                                            |
| `--qi-ki`              | `0.0`   | Integral gain KI for PID (0 = P-only)                            |
| `--acmcp-ncal`         | `10`    | Burn-in length for AcMCP scorecaster training                    |
| `--acmcp-lr`           | `0.1`   | Base learning rate for AcMCP quantile tracking                   |
| `--acmcp-ki`           | auto    | Integrator gain KI for AcMCP (defaults to max calibration error) |
| `--acmcp-csat`         | `1.0`   | Saturation constant Csat for AcMCP integrator                    |
| `--acmcp-no-integrate` | off     | Disable the integral (I) term in AcMCP                           |
| `--acmcp-no-scorecast` | off     | Disable the scorecasting (D) term in AcMCP                       |

**Joint method tuning**

| Flag                 | Default | Description                                                |
| -------------------- | ------- | ---------------------------------------------------------- |
| `--copula-cal-split` | `0.6`   | Fraction of cal windows used for score stage in CopulaCPTS |
| `--copula-epochs`    | `500`   | Gradient descent epochs for CopulaCPTS copula fitting      |
| `--cafht-normalize`  | `mae`   | Per-horizon normalisation for NormMaxCP: `mae` or `ones`   |
| `--cafht-base-model` | `aci`   | Adaptive base method for CAFHT: `aci` or `pid`             |
| `--cafht-cal-split`  | `0.5`   | Fraction of cal windows used for gamma selection in CAFHT  |
| `--cafht-q0`         | `0.1`   | Initial threshold q0 for CAFHT base method                 |

#### Output layout

```
results/tirex/electricity__H/
  cp_results_a<alpha>_<score_type>_c<cal_windows>_<mode>[_online].csv  # per-horizon marginal metrics for every method + native quantiles
  cp_summary_a<alpha>_<score_type>_c<cal_windows>_<mode>[_online].csv  # horizon-averaged marginal summary
  cp_multi_step_results.csv  # per-method joint metrics (only when joint methods are requested)
```

**`cp_results_a<alpha>_<score_type>_c<cal_windows>_<mode>.csv` columns (per-horizon marginal metrics):**

| Column           | Description                                            |
| ---------------- | ------------------------------------------------------ |
| `horizon`        | Forecast step index (0-based)                          |
| `method`         | CP method name (including `Native`)                    |
| `coverage`       | Empirical marginal coverage                            |
| `avg_width`      | Mean interval width                                    |
| `winkler_score`  | Mean Winkler score (width + miscoverage penalty)       |
| `joint_coverage` | Fraction of test windows where all H steps are covered |
| `runtime`        | Method execution time in seconds                       |

**`cp_summary_a<alpha>_<score_type>_c<cal_windows>_<mode>.csv` columns (horizon-averaged marginal summary):**

| Column                 | Description                                                      |
| ---------------------- | ---------------------------------------------------------------- |
| `task` / `model`       | Task and model identifiers                                       |
| `alpha` / `score_type` | Evaluation configuration                                         |
| `mode`                 | Evaluation mode (e.g. multi-step, single-step)                   |
| `horizon`              | Exact forecast horizon length (omitted in global summary)        |
| `horizon`              | Horizon category: `short` (<=30), `medium` (31-90), `long` (>90) |
| `cal_windows`          | Number of calibration windows used                               |
| `n_tasks`              | Number of tasks aggregated in this row                           |
| `method`               | CP method name (including `Native`)                              |
| `coverage`             | Empirical marginal coverage                                      |
| `joint_coverage`       | Fraction of test windows where all H steps are covered           |
| `scaled_avg_width`     | Interval mean width scaled by the native interval                |
| `scaled_winkler_score` | Winkler score scaled by the native interval                      |
| `runtime`              | Method execution time in seconds                                 |

**`cp_multi_step_results.csv` columns (joint evaluation):**

| Column                 | Description                                                      |
| ---------------------- | ---------------------------------------------------------------- |
| `task` / `model`       | Task and model identifiers                                       |
| `alpha` / `mode`       | Evaluation configuration                                         |
| `horizon`              | Exact forecast horizon length (omitted in global summary)        |
| `horizon`              | Horizon category: `short` (<=30), `medium` (31-90), `long` (>90) |
| `cal_windows`          | Number of calibration windows used                               |
| `n_tasks`              | Number of tasks aggregated in this row                           |
| `method`               | CP method name (including `Native`)                              |
| `joint_coverage`       | Fraction of test windows where **all** H steps are covered       |
| `marginal_coverage`    | Mean per-horizon coverage averaged over H steps                  |
| `scaled_avg_width`     | Interval mean width scaled by the native interval                |
| `scaled_winkler_score` | Winkler score scaled by the native interval                      |
| `runtime`              | Method execution time in seconds                                 |

## Conformal prediction methods

### Base classes

All methods in `timecp.methods` implement one of two base classes:

**`ConformalPredictor`** — marginal per-horizon interface:
```python
# Symmetric mode (default) — operates on unsigned scores |y − ŷ|
predictor.fit(cal_scores)          # calibrate on (C,) history of |y − ŷ|
q = predictor.predict_quantile()   # threshold for current step
predictor.update(new_score)        # online update after observing |y − ŷ|

# Asymmetric mode — operates on signed errors e = y − ŷ
predictor.fit(signed_errors)             # calibrate on (C,) signed errors
q_lo, q_up = predictor.predict_quantile_pair()   # separate lower/upper thresholds
lo, hi = predictor.predict_interval(y_hat)       # [ŷ − q_lo, ŷ + q_up]
predictor.update_signed(e)               # online update with signed error
```

**`JointPredictor`** — simultaneous-coverage interface:
```python
predictor.fit(cal_point, cal_gt)   # calibrate on (C, N, H) windows
radii = predictor.predict_radii(point_preds)  # (H,) half-widths
result = predictor.evaluate(test_point, test_gt)  # joint_coverage, avg_width, winkler_score
```

### Asymmetric intervals

Five methods support **asymmetric intervals** via `asymmetric=True`.  Instead of the standard symmetric `ŷ ± q̂`, they produce `[ŷ − q_lo, ŷ + q_up]` by tracking separate lower and upper quantiles, each targeting `alpha/2` miscoverage (so the total miscoverage rate is still `alpha`).

This is most useful when forecast residuals are **skewed** — the asymmetric interval adapts independently to each tail, yielding narrower intervals without sacrificing coverage.

```python
from timecp.methods import SplitCP, ACI, TrailingWindow, QuantileIntegrator, WeightedCP
import numpy as np

rng = np.random.default_rng(0)
# Skewed signed errors (positive residuals are larger)
signed_errors = rng.exponential(scale=1.0, size=1000) - 0.3

cal, test = signed_errors[:200], signed_errors[200:]
forecasts = np.zeros(len(test))

# All five methods accept asymmetric=True
cp   = SplitCP(alpha=0.1, asymmetric=True).fit(cal)
aci  = ACI(alpha=0.1, gamma=0.01, asymmetric=True).fit(cal)
tw   = TrailingWindow(alpha=0.1, window_size=100, asymmetric=True).fit(cal)
pid  = QuantileIntegrator(alpha=0.1, lr=0.1, asymmetric=True).fit(cal)
wcp  = WeightedCP(alpha=0.1, window_size=200, decay=0.99, asymmetric=True).fit(cal)

# Inspect per-tail quantiles
q_lo, q_up = cp.predict_quantile_pair()
print(f"lower half-width: {q_lo:.3f},  upper half-width: {q_up:.3f}")

# predict_interval uses the asymmetric pair automatically
lo, hi = cp.predict_interval(y_hat=5.0)   # [5.0 − q_lo,  5.0 + q_up]

# evaluate / batch_evaluate accept signed errors directly
result = cp.evaluate(test, forecasts)
print(result['coverage'], result['avg_width'])
```

**Key differences from symmetric mode:**

|                           | Symmetric (`asymmetric=False`)   | Asymmetric (`asymmetric=True`)              |
| ------------------------- | -------------------------------- | ------------------------------------------- |
| `fit(scores)` input       | unsigned `\|y − ŷ\|`             | signed `y − ŷ`                              |
| `predict_quantile()`      | single `q̂`                       | returns upper quantile `q_up`               |
| `predict_quantile_pair()` | `(q, q)`                         | `(q_lo, q_up)` independently tracked        |
| `predict_interval(ŷ)`     | `[ŷ − q, ŷ + q]`                 | `[ŷ − q_lo, ŷ + q_up]`                      |
| `update(score)`           | appends unsigned score           | appends unsigned score (fallback)           |
| `update_signed(e)`        | calls `update(\|e\|)`            | appends signed error, updates both trackers |
| `evaluate(scores, ...)`   | scores are unsigned              | scores are signed errors                    |
| Coverage target per tail  | `alpha` (one-sided on abs score) | `alpha/2` per tail                          |

**`CPEvaluator.run()`** detects asymmetric predictors automatically and passes signed residuals `(gt − pred)` to them — no changes to calling code required:

```python
df = evaluator.run(methods={
    'SplitCP-sym':  [SplitCP(alpha=0.1)                    for _ in range(H)],
    'SplitCP-asym': [SplitCP(alpha=0.1, asymmetric=True)   for _ in range(H)],
    'ACI-asym':     [ACI(alpha=0.1, gamma=0.005, asymmetric=True) for _ in range(H)],
})
```

**Which method to use:**
- `SplitCP(asymmetric=True)` — static; best coverage guarantees under exchangeability.
- `ACI(asymmetric=True)` — adapts to distribution shift; two independent α_t trackers.
- `TrailingWindow(asymmetric=True)` — simple rolling quantile per tail; no α adaptation.
- `QuantileIntegrator(asymmetric=True)` — two independent PID controllers; set `KI > 0` for integral correction.
- `WeightedCP(asymmetric=True)` — weighted quantile per tail; useful with time-decay weights.
- `AcMCP` — always asymmetric (no flag needed); adds a scorecasting (D) term that forecasts future errors using MA and OLS models of past errors.

### Marginal methods (`ConformalPredictor`)

For multi-step forecasting, instantiate **one predictor per horizon step**:

```python
from timecp.methods import ACI, AgACI, DtACI, QuantileIntegrator, AcMCP

H = 24
methods = {
    'ACI':                [ACI(alpha=0.1, gamma=0.005)   for _ in range(H)],
    'AgACI':              [AgACI(alpha=0.1)               for _ in range(H)],
    'DtACI':              [DtACI(alpha=0.1)               for _ in range(H)],
    'PID':                [QuantileIntegrator(alpha=0.1)  for _ in range(H)],
    'AcMCP':              [AcMCP(alpha=0.1, h=h+1)        for h in range(H)],
}

# Fit on calibration scores for each horizon independently
for h in range(H):
    methods['ACI'][h].fit(cal_scores[:, h])
    methods['AcMCP'][h].fit(
        cal_scores[:, h],
        signed_errors=cal_signed[:, :h],  # shorter-horizon history for scorecaster
    )
```

| Class                | Asymmetric | Description                                                               | Reference                 |
| -------------------- | :--------: | ------------------------------------------------------------------------- | ------------------------- |
| `SplitCP`            |     ✓      | Split (inductive) CP; strictly static (no online updates)                 | Papadopoulos 2002         |
| `ACI`                |     ✓      | Adaptive CP — updates α_t online with step size γ                         | Gibbs & Candès 2021       |
| `AgACI`              |            | ACI ensemble with AdaHedge weighting (self-tuning γ)                      | Zaffran et al. 2022       |
| `DtACI`              |            | Exponential-weighted expert ensemble (self-tuning γ)                      | Gibbs & Candès 2022       |
| `TrailingWindow`     |     ✓      | Rolling empirical quantile of most recent window_size scores              | Angelopoulos et al. 2024  |
| `QuantileIntegrator` |     ✓      | PID controller on pinball loss (P + optional I)                           | Angelopoulos et al. 2024  |
| `AcMCP`              |   always   | Asymmetric PID with MA(h−1) + OLS scorecaster (D term)                    | Wang & Hyndman 2024       |
| `CFRNN`              |            | Scalar-score variant of CFRNN for use with pre-aggregated scores          | Stankeviciute et al. 2021 |
| `WeightedCP`         |     ✓      | Rolling weighted empirical quantile (defaults to geometric decay weights) | Tibshirani et al. 2019    |
| `CQR`                |            | Conformalized Quantile Regression                                         | Romano et al. 2019        |
| `AdaptiveCQR`        |            | CQR with ACI/AgACI/DtACI/PID adaptive α_t                                 | Romano + Gibbs & Candès   |

✓ = supports `asymmetric=True` flag. `AcMCP` is always asymmetric (separate lower/upper trackers + scorecaster; no flag needed).

**A note on `QuantileIntegrator` (PID) initialization:**
In this implementation, the PID controller immediately calculates and starts at the empirical `(1 - alpha)` quantile of the calibration set. This bypasses the need for an explicit burn-in phase, ensuring perfect initial calibration even on benchmarks with very short evaluation periods (e.g. `fev-bench_mini`). An alternative approach is to initialize the controller blindly (e.g. at the median) and simulate a burn-in phase by looping over the calibration scores with `.update()`, but this can severely under-cover if the calibration sequence is too short to reach the target quantile before the test phase begins.

### Joint methods (`JointPredictor`)

Target simultaneous coverage: $P(y_h \in interval_h  \forall h = 1…H) \ge 1 − \alpha$.

```python
from timecp.methods import JointCFRNN, CopulaCPTS, NormMaxCP, CAFHT

multi_step_methods = {
    'JointCFRNN':  JointCFRNN(alpha=0.1),
    'CopulaCPTS':  CopulaCPTS(alpha=0.1, cal_split=0.6),
    'NormMaxCP':   NormMaxCP(alpha=0.1, normalize='mae'),
    'CAFHT':       CAFHT(alpha=0.1, base_model='aci'),
}

# Fit on (C, N, H) calibration windows
for name, m in multi_step_methods.items():
    m.fit(cal_point, cal_gt)

# Evaluate on (T, N, H) test windows
for name, m in multi_step_methods.items():
    result = m.evaluate(test_point, test_gt)
    print(name, result['joint_coverage'], result['avg_width'])
```

| Class        | Description                                                                          | Reference                 |
| ------------ | ------------------------------------------------------------------------------------ | ------------------------- |
| `JointCFRNN` | Bonferroni-corrected per-horizon quantile; joint guarantee via union bound           | Stankeviciute et al. 2021 |
| `CopulaCPTS` | Two-stage: per-horizon score calibration + empirical copula threshold fitting        | Sun & Yu 2022             |
| `NormMaxCP`  | Max-over-horizons normalised score (Max_calibrate baseline); fixed radii `C × σ_h`   | CAFHT codebase            |
| `CAFHT`      | Two-stage: ACI/PID base method along horizon dimension + calibrated inflation scalar | Zhou et al. 2024          |

**Notes:**
- `JointCFRNN` and `NormMaxCP` are static methods; `predict_radii` returns fixed half-widths.
- `CAFHT` intervals are trajectory-specific (the base method adapts along the horizon using ground truth); `predict_radii` raises `NotImplementedError`. Use `evaluate(test_point, test_gt)` directly.
- `CopulaCPTS` requires PyTorch for the copula threshold optimisation; falls back to binary search if unavailable.

### CPEvaluator API

```python
from timecp.cp_eval import CPEvaluator
from timecp.methods import ACI, AgACI, AcMCP, DtACI, QuantileIntegrator, SplitCP
from timecp.methods import JointCFRNN, CopulaCPTS, NormMaxCP, CAFHT

H = 96
evaluator = CPEvaluator(
    predictions_dir='results/tirex/ETTm2__T',
    horizon=H,
    cal_windows=25,   # or read from metadata.json automatically via scripts/cp_eval.py
    alpha=0.1,
    score_type='abs',
)

# Marginal evaluation (per-horizon independent)
df = evaluator.run(methods={
    'ACI':                [ACI(alpha=0.1, gamma=0.005)   for _ in range(H)],
    'AgACI':              [AgACI(alpha=0.1)               for _ in range(H)],
    'DtACI':              [DtACI(alpha=0.1)               for _ in range(H)],
    'PID':                [QuantileIntegrator(alpha=0.1)  for _ in range(H)],
})
# df columns: horizon, method, coverage, avg_width, winkler_score, joint_coverage,
#             native_coverage, native_avg_width, native_winkler_score, native_joint_coverage

# AcMCP evaluation (asymmetric; needs signed-error wiring)
df_acmcp = evaluator.run_acmcp(predictors={
    'AcMCP': [AcMCP(alpha=0.1, h=h+1) for h in range(H)],
})

# Joint evaluation (simultaneous coverage)
df_multi_step = evaluator.run_multi_step(methods={
    'JointCFRNN': JointCFRNN(alpha=0.1),
    'CopulaCPTS': CopulaCPTS(alpha=0.1),
    'NormMaxCP':  NormMaxCP(alpha=0.1),
    'CAFHT':      CAFHT(alpha=0.1, base_model='aci'),
})
# df_multi_step columns: method, joint_coverage, avg_width, winkler_score,
#                   native_joint_coverage, native_avg_width, native_winkler_score

# Marginal single-step evaluation
df_single = evaluator.run_single_step(methods={
    'SplitCP': SplitCP(alpha=0.1),  # one predictor instance shared across all H horizons
})

# Cross-sectional cross-validation (N-split)
df_cv = evaluator.run_cross_sectional_cv(
    methods={'SplitCP': [SplitCP(alpha=0.1) for _ in range(H)]},
    cal_windows=100,
    covariate_strategy='targets_only'
)
```

## Calibration split strategy

The calibration split is determined by the `cal_windows` top-level parameter in the YAML config (defaults to `0`).

For **TIME-format configs**, if `cal_windows: val_length` is specified, the number of calibration windows is calculated dynamically per-dataset using `ceil(val_length / prediction_length)`.

Per-dataset constraints:
- The requested calibration windows are clamped to the maximum possible full windows before the test split.
- If fewer than 10 windows are available overall (and >0 were requested), the task is skipped entirely (warning emitted).
- If between 10 and the requested number of windows are available, all available windows are used (warning emitted).

Use `--cp-cal-windows N` in `scripts/forecast.py` to override the config file's strategy entirely.

The final `cal_windows` value used is stored in `metadata.json` and read automatically by `scripts/cp_eval.py`.

## Dataset utilities

### Loading datasets

```python
from timecp.data import load_dataset

# Checks data/<DatasetName>/ first, falls back to HuggingFace Hub
ds = load_dataset('Salesforce/GiftEval', subset='electricity/H')
ds = load_dataset('Real-TSF/TIME', subset='Crypto/D')
ds = load_dataset('autogluon/chronos_datasets', subset='m4_hourly')
```

### Building fev.Task objects

```python
from timecp.data import tasks_from_config

# Auto-detects format from top-level YAML key
tasks = tasks_from_config('experiments/fev_bench.yaml')
# For TIME configs, all defined terms (short, medium, long) are generated
tasks = tasks_from_config('experiments/TIME_config.yaml')
```

Each returned `fev.Task` has a `task.config_cal_windows` attribute with the number of leading windows requested for CP calibration, as well as `task.config_name` and `task.config_output_dir`.

## Development

```bash
# Install all dependencies (including model extras)
uv sync --all-groups

# Run tests
uv run pytest

# Lint and format
uv run ruff check .
uv run ruff format .
```
