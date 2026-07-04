"""Time series dataset loading utilities for timecp.

Supports two dataset families:

GiftEval-format (Salesforce/GiftEval, Datadog/BOOM)
    Columns: ``item_id``, ``start``, ``freq``, ``target``.
    Use :class:`GiftEvalDataset` for GluonTS-compatible splits.
    Use :func:`to_fev` to convert to FEV format.

TIME-format (Real-TSF/TIME)
    Same as GiftEval, plus ``variate_names`` for multivariate datasets.
    Use :func:`to_fev` to convert; multivariate variates are stored as
    separate columns and ``dataset.variate_names`` / ``dataset.freq``
    are set on the returned object.

FEV-format (autogluon/chronos_datasets, autogluon/fev_datasets)
    Columns: ``id``, ``timestamp``, ``target``.
    Load directly with :func:`load_dataset`; no extra wrapper needed.

Quick start::

    from timecp.data import load_dataset, GiftEvalDataset, to_fev, attach_fev_meta

    # Load from local cache (falls back to HuggingFace Hub)
    hf_ds = load_dataset("Salesforce/GiftEval", subset="electricity/H")

    # GluonTS-compatible wrapper
    ds = GiftEvalDataset(hf_ds, name="electricity/H")
    train = ds.training_dataset
    test  = ds.test_data

    # Convert to FEV format
    fev_ds = to_fev(hf_ds)
    print(fev_ds.freq)           # "H"
    print(fev_ds.variate_names)  # None (univariate)

    # Multivariate TIME dataset
    hf_ds = load_dataset("Real-TSF/TIME", subset="Crypto/D")
    fev_ds = to_fev(hf_ds)
    print(fev_ds.variate_names)  # ["BitcoinCash", "Bitcoin", "Ethereum", "Litecoin"]

    # After save/load, restore attributes:
    import datasets
    reloaded = datasets.load_from_disk("path/to/cache")
    attach_fev_meta(reloaded)
    print(reloaded.freq, reloaded.variate_names)

    # FEV datasets load without any wrapper
    chronos_ds = load_dataset("autogluon/chronos_datasets", subset="m4_hourly")
"""

from timecp.data.convert import (
    attach_fev_meta,
    is_fev_format,
    is_gift_eval_format,
    to_fev,
)
from timecp.data.gift_eval import GiftEvalDataset, Term
from timecp.data.loader import load_dataset
from timecp.data.tasks import (
    tasks_from_config,
    tasks_from_fev_yaml,
    tasks_from_time_yaml,
)

__all__ = [
    'load_dataset',
    'GiftEvalDataset',
    'Term',
    'to_fev',
    'attach_fev_meta',
    'is_fev_format',
    'is_gift_eval_format',
    'tasks_from_config',
    'tasks_from_fev_yaml',
    'tasks_from_time_yaml',
]
