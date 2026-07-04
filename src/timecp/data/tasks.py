"""Build ``fev.Task`` objects from TIME-format and FEV-format benchmark configs.

Supported config formats
------------------------
**FEV YAML** (``experiments/fev_bench.yaml``)
    Standard ``fev.Benchmark`` YAML with a ``tasks:`` list.

**TIME YAML** (``experiments/TIME_config.yaml``)
    A ``datasets:`` dict keyed by ``"DatasetName/freq"`` with per-term
    ``prediction_length`` and ``test_length`` / ``val_length`` values.
    GiftEval datasets can be specified using the same format — provide
    ``gifteval_datasets`` and ``time_datasets`` root keys (or just
    ``datasets:`` and pass ``source="gifteval"`` / ``source="time"``
    to disambiguate the local data path).

Usage::

    from timecp.data.tasks import tasks_from_config

    # Auto-detect format
    tasks = tasks_from_config("experiments/TIME_config.yaml", term="short")
    tasks = tasks_from_config("experiments/fev_bench.yaml")

    # Explicit source for TIME-format YAML pointing at GiftEval data
    tasks = tasks_from_config(
        "experiments/GiftEval_time_config.yaml",
        term="short",
        source="gifteval",
    )
"""

from __future__ import annotations

import math
import threading
from pathlib import Path

import datasets as hf_datasets
import fev  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import yaml

from timecp.data.convert import attach_fev_meta, to_fev
from timecp.data.loader import _DEFAULT_DATA_ROOT, load_dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Seasonality per (normalised) frequency alias
_SEASONALITY_MAP: dict[str, int] = {
    'A': 1,
    'Q': 4,
    'M': 12,
    'W': 52,
    'D': 7,
    'H': 24,
    'T': 1440,
    'S': 3600,
    'B': 5,
    'U': 1,
}

_SOURCE_ALIASES: dict[str, str] = {
    'gifteval': 'GiftEval',
    'time': 'TIME',
}


def _source_dir_name(source: str) -> str:
    return _SOURCE_ALIASES.get(source, source)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fev_cache_path(source: str, name: str, freq: str | None) -> Path:
    """Return the path where the converted FEV arrow file should be cached."""
    # Use a flat cache key derived from the source so ``org/repo`` doesn't
    # create a confusing ``org/repo-fev`` nesting.
    source_dir = _source_dir_name(source)
    if source_dir.startswith('data/'):
        source_dir = source_dir[5:]
    cache_key = source_dir.replace('/', '_')
    root = _DEFAULT_DATA_ROOT / f'{cache_key}-fev'
    if freq:
        return root / name / freq
    return root / name


_convert_lock = threading.Lock()


def _load_or_convert(source: str, name: str, freq: str | None) -> hf_datasets.Dataset:
    """Load a GiftEval/TIME dataset, converting to FEV format and caching.

    Returns
    -------
    ds : datasets.Dataset
        The dataset in FEV format.  For multivariate datasets the variate
        column names are stored in ``ds.info.description``; retrieve them
        with :func:`~timecp.data.convert.get_variate_names`.
    """
    fev_path = _fev_cache_path(source, name, freq)

    # Return cached conversion if it exists
    if fev_path.exists():
        loaded = hf_datasets.load_from_disk(str(fev_path))
        assert isinstance(loaded, hf_datasets.Dataset)
        attach_fev_meta(loaded)
        return loaded

    # Serialize conversion to avoid race conditions between threads
    with _convert_lock:
        # Double-check after acquiring lock
        if fev_path.exists():
            loaded = hf_datasets.load_from_disk(str(fev_path))
            assert isinstance(loaded, hf_datasets.Dataset)
            attach_fev_meta(loaded)
            return loaded

        # Fetch using standard load_dataset
        raw_loaded = load_dataset(source, subset=name, freq=freq, cache=True)
        assert isinstance(raw_loaded, hf_datasets.Dataset)

        # Convert to FEV format (variate_names stored in ds.info.description)
        ds = to_fev(raw_loaded)

        # Cache to disk
        fev_path.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(fev_path))

    return ds


def _min_series_length(ds: hf_datasets.Dataset, variate_names: list[str] | None) -> int:
    """Return the shortest series length in *ds* (FEV format)."""
    col = variate_names[0] if variate_names else 'target'
    lengths = pc.list_value_length(ds.data.column(col))  # type: ignore[attr-defined]
    return int(pc.min(lengths).as_py())  # type: ignore[attr-defined]


def _seasonality_for_freq(freq: str) -> int:
    """Return the seasonality for a given pandas frequency string."""
    # Normalise to upper-case legacy alias
    _alias_map = {
        'Y': 'A',
        'YE': 'A',
        'QE': 'Q',
        'ME': 'M',
        'h': 'H',
        'min': 'T',
        's': 'S',
        'us': 'U',
    }
    f = _alias_map.get(freq, freq).upper()
    return _SEASONALITY_MAP.get(f, 1)


def _gifteval_windows(min_len: int, prediction_length: int, name: str) -> int:
    """Compute number of rolling test windows following GiftEval logic."""
    if 'm4' in name.lower():
        return 1
    w = math.ceil(0.1 * min_len / prediction_length)
    return min(max(1, w), 20)


def _time_windows(test_length: int, prediction_length: int) -> int:
    """Compute number of rolling test windows following TIME logic."""
    return max(1, math.ceil(test_length / prediction_length))


def _find_arrow_file(path: Path) -> Path:
    if path.is_file() and path.suffix == '.arrow':
        return path
    arrow_files = list(path.glob('**/*.arrow'))
    if not arrow_files:
        raise FileNotFoundError(f'No arrow file found in {path}')
    return arrow_files[0]


def _resolve_fev_dataset_path(dataset_path: str, dataset_config: str | None) -> str:
    path = Path(dataset_path)
    if path.exists():
        return str(_find_arrow_file(path))

    # Try load_dataset
    _ = load_dataset(dataset_path, subset=dataset_config, cache=True)
    cache_dir = _DEFAULT_DATA_ROOT / dataset_path.split('/')[-1]
    local_path = cache_dir / (dataset_config if dataset_config is not None else '')
    return str(_find_arrow_file(local_path))


def _parse_name_freq(key: str) -> tuple[str, str | None]:
    """Split a ``"DatasetName/freq"`` key into ``(name, freq)``.

    A key without ``/`` returns ``(key, None)``.
    """
    parts = key.split('/', 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return key, None


# ---------------------------------------------------------------------------
# Config Parsers
# ---------------------------------------------------------------------------


def tasks_from_fev_yaml(
    config: dict, yaml_path: str | Path, cal_windows_raw, to_univariate: bool
) -> list[fev.Task]:
    """Load tasks from an FEV-format YAML benchmark file."""
    tasks = []
    for t_dict in config.get('tasks', []):
        d = dict(t_dict)
        dataset_path = d.pop('dataset_path')
        dataset_config = d.pop('dataset_config', None)

        arrow_path = _resolve_fev_dataset_path(dataset_path, dataset_config)
        d['dataset_path'] = arrow_path

        if to_univariate and isinstance(d.get('target'), list) and len(d['target']) > 1:
            d['generate_univariate_targets_from'] = d['target']
            d['target'] = 'target'

        if 'task_name' not in d and dataset_config is not None:
            d['task_name'] = dataset_config

        task = fev.Task(**d)
        task.config_cal_windows = (
            cal_windows_raw if isinstance(cal_windows_raw, int) else 0
        )
        tasks.append(task)
    return tasks


def tasks_from_time_yaml(
    config: dict,
    yaml_path: str | Path,
    cal_windows_raw,
    to_univariate: bool,
    quantile_levels: list[float] | None = None,
    extra_metrics: list[str] | None = None,
) -> list[fev.Task]:
    """Build ``fev.Task`` objects from a TIME-format benchmark YAML."""
    import warnings
    import concurrent.futures

    if quantile_levels is None:
        quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    if extra_metrics is None:
        extra_metrics = [
            'MASE',
            {'name': 'WAPE', 'epsilon': 1.0},
            {'name': 'WQL', 'epsilon': 1.0},
        ]  # type: ignore[list-item]

    source = config.get('source', 'data/TIME')
    datasets_cfg: dict = config.get('datasets', {})

    def _process_entry(item) -> list[fev.Task]:
        key, entry = item
        name, freq = _parse_name_freq(key)

        try:
            ds = _load_or_convert(source, name, freq)
        except Exception as exc:
            warnings.warn(
                f"Failed to load dataset '{key}': {exc} — skipping.", stacklevel=2
            )
            return []

        variate_names = getattr(ds, 'variate_names', None)
        min_len = _min_series_length(ds, variate_names)

        test_length = entry.get('test_length')
        val_length = entry.get('val_length')

        freq_str = freq or getattr(ds, 'freq', None) or 'H'
        seasonality = _seasonality_for_freq(freq_str)
        fev_path = _fev_cache_path(source, name, freq)
        dataset_path = str(_find_arrow_file(fev_path))

        base_task_name = f'{name}/{freq}' if freq else name

        entry_tasks = []
        for term in ['short', 'medium', 'long']:
            if term not in entry:
                continue

            prediction_length = entry[term]['prediction_length']

            if test_length is not None:
                num_windows = _time_windows(test_length, prediction_length)
                initial_cutoff = -test_length
            else:
                num_windows = _gifteval_windows(min_len, prediction_length, name)
                initial_cutoff = -(prediction_length * num_windows)

            if cal_windows_raw == 'val_length':
                if val_length is not None:
                    task_cal_windows = math.ceil(val_length / prediction_length)
                else:
                    task_cal_windows = 0
            else:
                task_cal_windows = (
                    int(cal_windows_raw) if cal_windows_raw is not None else 0
                )

            kwargs = dict(
                dataset_path=dataset_path,
                horizon=prediction_length,
                num_windows=num_windows,
                initial_cutoff=initial_cutoff,
                window_step_size=prediction_length,
                seasonality=seasonality,
                eval_metric='SQL' if quantile_levels else 'MASE',
                extra_metrics=extra_metrics,
                quantile_levels=quantile_levels,
                task_name=f'{base_task_name}_{term}',
            )

            if variate_names:
                if to_univariate:
                    kwargs['target'] = 'target'
                    kwargs['generate_univariate_targets_from'] = variate_names
                else:
                    kwargs['target'] = variate_names
            else:
                kwargs['target'] = 'target'

            task = fev.Task(**kwargs)
            task.config_cal_windows = task_cal_windows
            entry_tasks.append(task)

        return entry_tasks

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(_process_entry, datasets_cfg.items()))

    return [t for group in results for t in group]


def tasks_from_config(
    config_path: str | Path,
    *,
    quantile_levels: list[float] | None = None,
    extra_metrics: list[str] | None = None,
) -> list[fev.Task]:
    """Auto-detect config format and return ``fev.Task`` objects."""
    with open(config_path) as f:
        top = yaml.safe_load(f)

    output_dir = top.get('output_dir')
    name = top.get('name', Path(config_path).stem)
    to_univariate = top.get('to_univariate', False)
    cal_windows = top.get('cal_windows', 0)

    if 'tasks' in top:
        tasks = tasks_from_fev_yaml(top, config_path, cal_windows, to_univariate)
    elif 'datasets' in top:
        tasks = tasks_from_time_yaml(
            top, config_path, cal_windows, to_univariate, quantile_levels, extra_metrics
        )
    else:
        raise ValueError(
            f"Cannot detect config format in '{config_path}'. "
            "Expected a top-level 'tasks:' key (FEV) or 'datasets:' key (TIME)."
        )

    for task in tasks:
        task.config_output_dir = output_dir
        task.config_name = name

    return tasks
