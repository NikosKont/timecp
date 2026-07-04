"""Dataset loader with local-first caching for HuggingFace time series datasets.

Priority: local disk → HuggingFace Hub.

Default cache root: ``<project_root>/data/{dataset_name}`` where
``dataset_name`` is the part of the HF repo id after the slash
(e.g. ``Salesforce/GiftEval`` → ``GiftEval``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import datasets as hf_datasets

# Project root = four levels up from this file:
# src/timecp/data/loader.py → src/timecp/data → src/timecp → src → <root>
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DATA_ROOT = _PROJECT_ROOT / 'data'


def _try_load_from_disk(path: Path) -> Optional[hf_datasets.Dataset]:
    """Attempt to load a dataset from disk; return None if not found."""
    if not isinstance(path, Path):
        path = Path(path)
    if not path.exists():
        return None
    try:
        return hf_datasets.load_from_disk(str(path))
    except Exception:
        return None


def load_dataset(
    dataset_id: str,
    subset: Optional[str] = None,
    freq: Optional[str] = None,
    *,
    split: str = 'train',
    cache_dir: Optional[Path | str] = None,
    cache: bool = True,
    trust_remote_code: Optional[bool] = None,
) -> hf_datasets.Dataset:
    """Load a time series dataset, checking local disk before HuggingFace Hub.

    Supports both GiftEval-format datasets (``Salesforce/GiftEval``,
    ``Datadog/BOOM``) and FEV-format datasets (``autogluon/chronos_datasets``,
    ``autogluon/fev_datasets``).

    Parameters
    ----------
    dataset_id:
        HuggingFace repo id, e.g. ``"Salesforce/GiftEval"`` or
        ``"autogluon/chronos_datasets"``.
    subset:
        Dataset subset name passed to ``datasets.load_dataset(name, config)``.  For GiftEval this is a path
        like ``"electricity/H"``; for chronos_datasets it is a name like
        ``"m4_hourly"``.
    freq:
        Frequency of the time series data, e.g. ``"H"`` for hourly or ``"D"`` for daily.
    split:
        Dataset split to load (default: ``"train"``).
    cache_dir:
        Override the local cache directory.  When *None* the default path
        ``<project_root>/data/{DatasetName}[/{subset}[/{freq}]]`` is used.
    cache:
        If *True* (default), save the dataset to ``cache_dir`` after
        downloading from HuggingFace so future calls can load from disk.
    trust_remote_code:
        Passed through to ``datasets.load_dataset`` for script-based datasets.

    Returns
    -------
    datasets.Dataset
        The loaded split.
    """
    cache_dir = Path(cache_dir) if cache_dir is not None else _DEFAULT_DATA_ROOT
    cache_dir /= dataset_id.split('/')[-1]

    local_path = (
        cache_dir
        / (subset if subset is not None else '')
        / (freq if freq is not None else '')
    )

    # 1. Try local disk first
    local = _try_load_from_disk(local_path)
    if local is not None:
        return local

    # 2. Fall back to HuggingFace Hub
    try:
        hf_ds = hf_datasets.load_dataset(
            dataset_id,
            name=subset,
            split=split,
            trust_remote_code=trust_remote_code,
        )
    except ValueError:
        # probably a dataset without subset configs (e.g. GiftEval, BOOM)
        # manually construct the path to the subset if provided
        from huggingface_hub import snapshot_download

        cache_dir = snapshot_download(
            local_dir=cache_dir if cache else None,
            repo_id=dataset_id,
            repo_type='dataset',
            allow_patterns=f'{subset}/*' if subset else None,
        )
        local_path = (
            Path(cache_dir)
            / (subset if subset is not None else '')
            / (freq if freq is not None else '')
        )
        # print(f'Loaded dataset from HuggingFace Hub to {local_path}')
        local = _try_load_from_disk(local_path)
        if local is not None:
            # print(f'Loaded dataset from local cache at {local_path}')
            return local
        raise RuntimeError(
            f'snapshot_download succeeded but could not load dataset from {local_path}'
        )

    # 3. Optionally cache to disk
    if cache:
        # print(f'Caching dataset to {local_path}')
        local_path.mkdir(parents=True, exist_ok=True)
        hf_ds.save_to_disk(str(local_path))

    return hf_ds
