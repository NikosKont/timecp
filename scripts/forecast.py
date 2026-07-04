"""Batch inference script for time series forecasting models.

Run predictions from any ``timecp.models.Forecaster`` on FEV, GiftEval, or TIME
benchmark tasks.  Predictions are always saved; evaluations are optional.

Supported config formats
------------------------
- **FEV YAML** (``experiments/fev_bench.yaml``) — top-level ``tasks:`` key.
- **TIME YAML** (``experiments/TIME_config.yaml``) — top-level ``datasets:``
  key with per-term ``prediction_length`` / ``test_length``.

CP calibration split
--------------------
When ``--cp-cal-windows N`` (or ``--cal-windows N``) is passed, a target of the first
*N* windows of each task are reserved for conformal prediction calibration.
If omitted, defaults to the task's configuration (using `cal_windows` top level param).
A ``metadata.json`` file is written alongside the window directories containing
the calibration split information required by ``scripts/cp_eval.py``.

Output layout
-------------
::

    <output_dir>/
      <task_name>/
        <model_name>/
          window_0/
            predictions/    ← DatasetDict saved to disk
            ground_truth/   ← DatasetDict with actual values
          window_1/
            ...
          evaluation.json   ← only when --evaluate is passed
          metadata.json     ← task metadata including cal_windows
      summary.csv           ← one row per task (only when --evaluate)

Usage
-----
::

    # Multi-model batch inference on FEV benchmark
    uv run python scripts/forecast.py \
        --model tirex chronos2 timesfm \
        --tasks experiments/fev_bench.yaml \
        --output results/

    # TIME benchmark (short term, with series split)
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
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import fev
from tqdm.auto import tqdm

from timecp import set_global_seed

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='forecast',
        description='Run inference with a Forecaster model on benchmark tasks.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    p.add_argument(
        '-M',
        '--model',
        nargs='+',
        action='append',
        required=True,
        help=(
            'Model type(s) to use: tirex, chronos2 (or chronos), flowstate, timesfm. '
            'May also be a full HuggingFace model id for non-default checkpoints '
            '(passed as the first positional arg to the model constructor). '
            'Multiple models can be specified separated by spaces.'
        ),
    )
    p.add_argument(
        '--model-id',
        default=None,
        help='Override the default HuggingFace model checkpoint id.',
    )

    # Tasks
    p.add_argument(
        '-T',
        '--tasks',
        nargs='+',
        required=True,
        help='Path(s) to benchmark YAML config(s) (FEV or TIME format).',
    )

    # Quantile levels
    p.add_argument(
        '--quantile-levels',
        nargs='+',
        type=float,
        default=None,
        metavar='Q',
        help=(
            'Quantile levels to predict and evaluate (e.g. 0.1 0.5 0.9). '
            'Defaults to [0.1, ..., 0.9] for TIME configs; taken from the '
            'YAML for FEV configs.'
        ),
    )

    # CP calibration split
    p.add_argument(
        '-C',
        '--cp-cal-windows',
        '--cal-windows',
        dest='cp_cal_windows',
        type=int,
        default=None,
        metavar='N',
        help=(
            'Target number of leading windows reserved for CP calibration. '
            'Overrides the config file value if set. '
            'The actual number used per task is clamped to '
            'what the dataset can provide before the test split.  Tasks '
            'where fewer than 10 calibration windows are possible are '
            'skipped with a warning.  Set to 0 to disable calibration '
            'windows entirely.'
        ),
    )

    # Hardware
    p.add_argument(
        '--device',
        default=None,
        help='Device override (e.g. "cuda", "cpu", "cuda:1"). '
        'Defaults to CUDA when available.',
    )
    p.add_argument(
        '--batch-size',
        type=int,
        default=None,
        metavar='N',
        help='Number of time series per inference batch.',
    )
    p.add_argument(
        '--max-context-length',
        type=int,
        default=1024,
        metavar='N',
        help='Maximum context length for the model.',
    )
    p.add_argument(
        '--gpu-memory-fraction',
        type=float,
        default=0.95,
        help=(
            'Maximum fraction of GPU memory to use. Setting this to < 1.0 (e.g. 0.95) '
            'prevents Windows from using slow shared system memory and instead forces '
            'a CUDA OOM error, triggering the automatic batch size reduction.'
        ),
    )

    # Output
    p.add_argument(
        '--output',
        default=None,
        help='Root directory for saving predictions (and evaluations). If not specified, uses output_dir from config or defaults to "results".',
    )
    p.add_argument(
        '--evaluate',
        action='store_true',
        help='Also compute and save evaluation metrics for each task.',
    )
    p.add_argument(
        '--force',
        action='store_true',
        help='Re-run inference even if results already exist for the task/model combination.',
    )
    p.add_argument(
        '--no-batch-windows',
        action='store_true',
        help='Disable cross-window batching and predict one window at a time (slower, for debugging).',
    )
    p.add_argument(
        '--split-series',
        type=float,
        default=None,
        metavar='RATIO',
        help=(
            'Split series into calibration and test subsets. '
            'E.g. 0.5 assigns ~50%% of series to a _cal task and the rest to _test. '
            'Creates three task directories per task: <task> (full), <task>_cal, and <task>_test. '
            'Use --split-seed to control reproducibility.'
        ),
    )
    p.add_argument(
        '--split-seed',
        type=int,
        default=42,
        help='Random seed for --split-series.',
    )
    p.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Global random seed for reproducibility.',
    )

    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_task_max_context(task: fev.Task) -> int:
    """Determine the maximum context length needed for a TimesFM task.

    Windows are ordered earliest→latest, so the last window always has the
    most past data (each window adds ``window_step_size`` extra observations).
    We only need to inspect that final window.
    """
    import os

    num_proc = 1 if os.name == 'nt' else fev.task.DEFAULT_NUM_PROC
    w = task.get_window(task.num_windows - 1, num_proc=num_proc)
    past_data, *_ = w.get_input_data()
    target_cols = task.target if isinstance(task.target, list) else [task.target]
    return max(
        (
            max(len(x) for x in past_data[col])
            for col in target_cols
            if col in past_data.column_names
        ),
        default=0,
    )


def get_tasks_max_context(tasks: list[fev.Task]) -> int:
    """Determine the maximum context length needed for a list of TimesFM tasks."""
    return max(get_task_max_context(task) for task in tasks)


def get_tasks_max_horizon(tasks: list[fev.Task]) -> int:
    """Determine the maximum forecast horizon needed for a list of tasks."""
    return max(task.horizon for task in tasks)


def get_task_num_series(task: fev.Task) -> int:
    """Determine the total number of series in a task without loading window data."""
    if hasattr(task, 'dataset_path') and task.dataset_path:
        import os

        import datasets as hf_datasets

        if os.path.isfile(task.dataset_path):
            n = len(hf_datasets.Dataset.from_file(task.dataset_path))
            gen = getattr(task, 'generate_univariate_targets_from', None)
            if gen is not None:
                return len(gen) * n
            return n

    if hasattr(task, 'dataset') and task.dataset is not None:
        return len(task.dataset)
    else:
        import os

        num_proc = 1 if os.name == 'nt' else fev.task.DEFAULT_NUM_PROC
        w0 = task.get_window(0, num_proc=num_proc)
        past_data_0, *_ = w0.get_input_data()
        return len(past_data_0)


def _load_model(model_type: str, batch_size: int, **kwargs):
    """Instantiate the Forecaster."""
    # Import fev/datasets/pyarrow before torch to avoid a Windows DLL conflict
    # where torch's bundled Arrow DLLs cause an access violation when pyarrow
    # tries to load its own Arrow DLLs afterwards.
    import pyarrow  # noqa: F401

    from timecp.models import Forecaster

    if model_type != 'timesfm':
        # kwargs.pop('max_context', None)  # only used by TimesFM
        kwargs.pop('max_horizon', None)  # only used by TimesFM
    if model_type == 'chronos':
        model_type = 'chronos2'  # alias for backward compatibility

    if kwargs.get('model_id') is not None:
        # Pass as first positional arg (model_id) to the subclass constructor
        model = Forecaster(
            model_type, kwargs.pop('model_id'), batch_size=batch_size, **kwargs
        )  # type: ignore[call-arg]
    else:
        model = Forecaster(model_type, batch_size=batch_size, **kwargs)  # type: ignore[call-arg]

    return model


def _load_tasks(args: argparse.Namespace):
    """Load fev.Task objects from the config file."""
    from timecp.data.tasks import tasks_from_config

    extra_kwargs: dict = {}
    if args.quantile_levels:
        extra_kwargs['quantile_levels'] = args.quantile_levels

    all_tasks = []
    for tasks_path in args.tasks:
        all_tasks.extend(tasks_from_config(tasks_path, **extra_kwargs))
    return all_tasks


def _safe_task_name(task_name: str) -> str:
    """Convert a task name to a filesystem-safe directory component."""
    return task_name.replace('/', '_').replace('\\', '_').replace(' ', '-')


def _build_combined_task(
    task: fev.Task, requested_cal_windows: int
) -> tuple[fev.Task, int]:
    """Reconstruct a FEV task extended to include calibration windows."""
    import math
    import os
    import warnings

    import pyarrow.compute as pc

    if requested_cal_windows == 0:
        return task, 0

    horizon = task.horizon
    num_test_windows = task.num_windows
    name = task.task_name or 'unknown'

    num_proc = min(2, os.cpu_count() or 1) if os.name != 'nt' else 1
    ds = task.load_full_dataset(num_proc=num_proc)

    target_cols = list(task.target_columns)
    series_col = next((c for c in target_cols if c in ds.column_names), None)
    if series_col is None:
        _meta = {'item_id', 'id', 'timestamp', 'start', 'freq'}
        series_col = next((c for c in ds.column_names if c not in _meta), None)
    if series_col is None:
        warnings.warn(
            f"'{name}': cannot determine series length — skipping calibration.",
            stacklevel=3,
        )
        return task, 0

    lengths = pc.list_value_length(ds.data.column(series_col))
    min_len = int(pc.min(lengths).as_py())

    # Maximum cal windows = all full windows that fit in the series before
    # the test split, keeping at least one window of context before them.
    max_cal = math.floor(min_len / horizon) - num_test_windows - 1
    max_cal = max(max_cal, 0)

    if requested_cal_windows == -1:
        n_cal = max_cal
    else:
        if max_cal < 10:
            warnings.warn(
                f"'{name}': only {max_cal} calibration window(s) possible "
                f'(minimum 10) — skipping task.',
                stacklevel=3,
            )
            return None, 0
        if max_cal < requested_cal_windows:
            warnings.warn(
                f"'{name}': only {max_cal} calibration window(s) available "
                f'(requested {requested_cal_windows}); using all {max_cal} available windows.',
                stacklevel=3,
            )
        n_cal = min(requested_cal_windows, max_cal)

    if n_cal == 0:
        return task, 0

    # Reconstruct with extended num_windows and earlier initial_cutoff.
    d = task.to_dict()
    d.pop('dataset', None)  # do not pass the loaded Dataset object
    d['num_windows'] = num_test_windows + n_cal

    cutoff_raw = task.initial_cutoff
    if isinstance(cutoff_raw, int):
        d['initial_cutoff'] = cutoff_raw - n_cal * horizon
    else:
        # Datetime-based cutoff: shift backwards by n_cal * horizon periods.
        import pandas as pd

        freq_str = task.freq or 'h'
        offset = n_cal * horizon * pd.tseries.frequencies.to_offset(freq_str)
        new_ts = pd.Timestamp(str(cutoff_raw)) - offset
        d['initial_cutoff'] = new_ts.isoformat()

    return fev.Task(**d), n_cal


def _save_predictions(
    predictions: list,
    ground_truths: list,
    task_dir: Path,
) -> Path:
    """Save all predictions and ground truths as unified Arrow DatasetDicts.

    Returns the task-level output directory.
    """
    import datasets as hf_datasets

    task_dir.mkdir(parents=True, exist_ok=True)

    # Convert predictions list of window dicts -> single DatasetDict
    if predictions and predictions[0] is not None:
        target_cols = list(predictions[0].keys())
        ds_dict = {}
        for col in target_cols:
            col_data = {}
            keys = list(predictions[0][col].keys())
            for key in keys:
                # Stack arrays across windows -> list of (N, H) arrays
                col_data[key] = [w[col][key] for w in predictions if w is not None]
            ds_dict[col] = hf_datasets.Dataset.from_dict(col_data)
        hf_datasets.DatasetDict(ds_dict).save_to_disk(str(task_dir / 'predictions'))

    # Convert ground_truths list of window dicts -> single DatasetDict
    if ground_truths and any(g is not None for g in ground_truths):
        valid_gt = next(g for g in ground_truths if g is not None)
        target_cols = list(valid_gt.keys())
        gt_ds_dict = {}
        for col in target_cols:
            col_data = {}
            keys = list(valid_gt[col].keys())
            for key in keys:
                col_data[key] = [
                    w[col][key] if w is not None else None for w in ground_truths
                ]
            gt_ds_dict[col] = hf_datasets.Dataset.from_dict(col_data)
        hf_datasets.DatasetDict(gt_ds_dict).save_to_disk(str(task_dir / 'ground_truth'))

    return task_dir


def _save_metadata(
    task,
    task_dir: Path,
    model_name: str,
    total_windows: int,
    cal_windows: int,
    test_windows: int,
    num_series: int | None = None,
    max_context: int | None = None,
    task_max_context: int | None = None,
    **extra_metrics,
) -> None:
    """Write task metadata."""
    metadata = {
        'task_name': task.task_name,
        'model_name': model_name,
        'horizon': task.horizon,
        'total_windows': total_windows,
        'cal_windows': cal_windows,
        'test_windows': test_windows,
        'quantile_levels': list(task.quantile_levels) if task.quantile_levels else [],
    }
    if num_series is not None:
        metadata['num_series'] = num_series
    if max_context is not None:
        metadata['max_context'] = max_context
    if task_max_context is not None:
        metadata['task_max_context'] = task_max_context
    metadata.update(extra_metrics)

    meta_path = task_dir / 'metadata.json'
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f'  Saved metadata  -> {meta_path}')


def _save_evaluation(summary: dict, task_dir: Path) -> None:
    """Write the evaluation summary dict as JSON."""
    eval_path = task_dir / 'evaluation.json'
    with open(eval_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'  Saved evaluation -> {eval_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.model = [model for group in args.model for model in group]

    set_global_seed(args.seed)

    # Import pyarrow before torch to avoid a Windows DLL conflict
    # where torch's bundled Arrow DLLs cause an access violation when pyarrow
    # tries to load its own Arrow DLLs afterwards.
    import pyarrow  # noqa: F401
    import torch

    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if torch.cuda.is_available() and args.gpu_memory_fraction < 1.0:
        for i in range(torch.cuda.device_count()):
            try:
                torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, i)
            except Exception as e:
                print(f'Warning: Could not set memory fraction on GPU {i}: {e}')

    if args.output is not None:
        output_dir_arg = Path(args.output)
    else:
        output_dir_arg = None

    print(f'Loading tasks from: {args.tasks}')
    try:
        tasks = _load_tasks(args)
    except Exception as exc:
        import traceback

        traceback.print_exc()
        print(f'Error: Failed to load tasks: {exc}')
        return 1

    print(f'  {len(tasks)} task(s) loaded')

    if not tasks:
        print('Error: No tasks loaded. Exiting with code 1.')
        return 1

    summaries: list[dict] = []

    global_max_horizon = get_tasks_max_horizon(tasks)
    for model_type in args.model:
        if args.batch_size is None:
            if model_type == 'tirex':
                batch_size = 8192
            else:
                batch_size = 256
        else:
            batch_size = args.batch_size
        print(
            f'\nLoading model {model_type} (batch_size={batch_size}, max_context={args.max_context_length})...',
            flush=True,
        )
        model = _load_model(
            model_type,
            batch_size,
            device=args.device,
            max_context=args.max_context_length,
            max_horizon=global_max_horizon,
        )

        # Track history mapping of context_length_bucket -> last_successful_batch_size.
        # This prevents repeated OOMs on large tasks and maximizes GPU utilization on small tasks.
        batch_size_history = {}

        for task_idx, base_task in enumerate(tasks):
            task_name = base_task.task_name or f'task_{task_idx}'

            config_output_dir = getattr(base_task, 'config_output_dir', None)
            config_name = getattr(base_task, 'config_name', 'results')

            if output_dir_arg is not None:
                output_dir = output_dir_arg
            elif config_output_dir:
                output_dir = Path(config_output_dir)
            else:
                output_dir = Path('results')

            task_dir = (
                output_dir / model_type / config_name / _safe_task_name(task_name)
            )

            if not args.force and (task_dir / 'metadata.json').exists():
                print(
                    f'[{task_idx + 1}/{len(tasks)}] {task_name}  [SKIP] Already complete for {model_type}'
                )
                continue

            if args.cp_cal_windows is not None:
                cp_cal_windows = int(args.cp_cal_windows)
            else:
                cp_cal_windows = getattr(base_task, 'config_cal_windows', 0)

            combined_task_res = _build_combined_task(base_task, cp_cal_windows)
            if combined_task_res[0] is None:
                continue
            task, cal_windows = combined_task_res
            test_windows = base_task.num_windows
            num_series = get_task_num_series(task)

            print(
                f'[{task_idx + 1}/{len(tasks)}] {task_name}  '
                f'(num_series={num_series}, '
                f'horizon={task.horizon}, '
                f'total_windows={task.num_windows}, '
                f'cal_windows={cal_windows}, '
                f'test_windows={test_windows})'
            )

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            task_max_context = get_task_max_context(task)
            current_context = min(task_max_context, args.max_context_length)

            # Compute power-of-2 context bucket to scale initial batch sizes across similar tasks.
            # E.g. context=100 -> bit_length=7 -> bucket=128
            bucket = (
                1 << (current_context - 1).bit_length() if current_context > 0 else 512
            )
            current_batch_size = batch_size_history.get(bucket, batch_size)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            while True:
                # Load or update model
                # Non-TimesFM models never need reloading—just a set_batch_size call
                need_reload = 'model' not in locals() or model is None
                if need_reload:
                    print(
                        f'Loading model {model_type} with batch_size={current_batch_size}'
                        + (
                            f', max_context={current_context}'
                            if model_type == 'timesfm'
                            else ''
                        )
                        + '...'
                    )
                    if 'model' in locals() and model is not None:
                        del model
                        import gc

                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    model = _load_model(
                        model_type,
                        current_batch_size,
                        device=args.device,
                        max_horizon=task.horizon,
                        max_context=current_context,
                    )
                else:
                    assert model is not None
                    model.set_batch_size(current_batch_size)
                    if hasattr(model, 'update_task_params'):
                        model.update_task_params(
                            max_context=current_context, max_horizon=task.horizon
                        )
                    model.max_context = current_context

                assert model is not None
                prev_inf_time = getattr(model, 'inference_time', 0.0)
                t_task = time.monotonic()

                try:
                    # Predict
                    predictions, ground_truths = model.fit_predict_with_gt(
                        task,
                        batch_windows=not args.no_batch_windows,
                        force_extract=args.force,
                    )
                    elapsed = time.monotonic() - t_task
                    current_batch_size = model.batch_size
                    # Update history mapping for this bucket based on the successful batch size
                    bucket = (
                        1 << (current_context - 1).bit_length()
                        if current_context > 0
                        else 512
                    )
                    batch_size_history[bucket] = current_batch_size
                    break
                except Exception as e:
                    if isinstance(e, torch.cuda.OutOfMemoryError) or (
                        isinstance(e, RuntimeError)
                        and 'out of memory' in str(e).lower()
                    ):
                        if model_type == 'timesfm':
                            new_context = current_context * 2 // 3
                            if new_context >= 128:
                                tqdm.write(
                                    f'  [OOM] Out of memory with batch_size=1. Reducing TimesFM max_context to {new_context} and retrying...'
                                )
                                current_context = new_context
                                model = None
                                continue
                            else:
                                tqdm.write(
                                    f'  [OOM] Out of memory with batch_size=1 and context={current_context}. Cannot reduce further.'
                                )
                                raise e
                        else:
                            tqdm.write(
                                '  [OOM] Out of memory with batch_size=1. Cannot reduce further.'
                            )
                            raise e
                    else:
                        raise

            task_model_time = getattr(model, 'inference_time', elapsed) - prev_inf_time

            peak_mem_mb = 0.0
            if torch.cuda.is_available():
                peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

            print(
                f'  Inference: {elapsed:.1f}s  ({task_model_time:.1f}s model time, peak GPU mem: {peak_mem_mb:.1f} MB)'
            )

            _save_predictions(
                predictions,
                ground_truths,
                task_dir,
            )
            print(f'  Saved predictions -> {task_dir}')
            _save_metadata(
                task,
                task_dir,
                model.model_name or model_type,
                total_windows=task.num_windows,
                cal_windows=cal_windows,
                test_windows=test_windows,
                num_series=num_series,
                inference_time_s=task_model_time,
                gpu_max_memory_mb=peak_mem_mb,
                batch_size=current_batch_size,
                max_context=current_context,
                task_max_context=task_max_context,
            )

            if args.evaluate:
                # Convert back to DatasetDict in memory for evaluation
                import datasets as hf_datasets

                test_preds = predictions[cal_windows:]
                test_preds_hf = []
                for p in test_preds:
                    ds_dict = {}
                    for col, data in p.items():
                        ds_dict[col] = hf_datasets.Dataset.from_dict(data)
                    test_preds_hf.append(hf_datasets.DatasetDict(ds_dict))

                summary = base_task.evaluation_summary(
                    test_preds_hf,
                    model_name=model.model_name or model_type,
                    inference_time_s=task_model_time,
                )
                summary['gpu_max_memory_mb'] = peak_mem_mb
                summary['batch_size'] = current_batch_size
                _save_evaluation(summary, task_dir)
                summaries.append(summary)

                primary = (
                    base_task.eval_metric
                    if isinstance(base_task.eval_metric, str)
                    else base_task.eval_metric.get('name', 'metric')
                )
                print(f'  {primary}: {summary.get("test_error", "n/a"):.4f}')

        # Free memory before next model
        del model
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Write aggregated summary CSV
    if args.evaluate and summaries:
        import pandas as pd

        summary_path = (
            output_dir_arg if output_dir_arg is not None else Path('results')
        ) / 'summary.csv'
        df = pd.DataFrame(summaries)
        df.to_csv(summary_path, index=False)
        print(f'\nSummary saved -> {summary_path}')

    print('\nDone.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
