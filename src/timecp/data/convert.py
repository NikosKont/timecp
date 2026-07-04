"""Convert GiftEval-format datasets to FEV format.

GiftEval format (Salesforce/GiftEval, Datadog/BOOM):
    - ``item_id`` (str)       — series identifier
    - ``start``   (timestamp) — single start timestamp
    - ``freq``    (str)       — pandas frequency alias
    - ``target``  (list)      — time series values (1-D univariate or
                                2-D ``[n_dims, n_steps]`` multivariate)

TIME format (Real-TSF/TIME):
    Same as GiftEval, plus:
    - ``variate_names`` (list[str]) — names of the target dimensions

FEV format (autogluon/fev_datasets, autogluon/chronos_datasets):
    - ``id``        (str)            — series identifier
    - ``timestamp`` (list[datetime]) — explicit timestamp per observation
    - ``target``    (list)           — time series values

    For multivariate datasets converted for independent univariate
    evaluation, each variate is stored as a separate column named after
    the variate (or ``target_0``, ``target_1``, … when names are absent).
    Pass the list of variate column names to ``fev.Task`` via
    ``generate_univariate_targets_from`` to have FEV split them into
    individual univariate series at evaluation time.

Metadata attributes
-------------------
:func:`to_fev` attaches two attributes directly to the returned dataset:

- ``dataset.freq`` — pandas frequency string (e.g. ``"H"``, ``"D"``)
- ``dataset.variate_names`` — ``list[str]`` for multivariate datasets,
  ``None`` for univariate

These attributes are also serialised into ``dataset.info.description``
as JSON so they survive :func:`~datasets.Dataset.save_to_disk` /
:func:`~datasets.load_from_disk`.  After reloading from disk call
:func:`attach_fev_meta` to re-attach them as instance attributes.

Usage::

    from timecp.data.convert import to_fev, attach_fev_meta

    fev_ds = to_fev(raw_ds)
    print(fev_ds.freq)           # e.g. "H"
    print(fev_ds.variate_names)  # None or ["col1", "col2", ...]

    fev_ds.save_to_disk("path/to/cache")

    reloaded = datasets.load_from_disk("path/to/cache")
    attach_fev_meta(reloaded)    # re-attaches .freq and .variate_names
    print(reloaded.freq)
"""

from __future__ import annotations

import json

import datasets as hf_datasets
import pandas as pd


def _is_nested_list_feature(feature) -> bool:
    """Return True if *feature* is a List-of-Lists (multivariate covariate)."""
    return (
        isinstance(feature, hf_datasets.List)
        and isinstance(feature.feature, hf_datasets.List)
    ) or (
        # datasets also uses Sequence for fixed-length nested arrays
        isinstance(feature, hf_datasets.Sequence)
        and isinstance(feature.feature, (hf_datasets.Sequence, hf_datasets.List))
    )


def _is_multivariate_target(ds: hf_datasets.Dataset) -> bool:
    """Return True if *ds* has a nested (multivariate) target column."""
    return _is_nested_list_feature(ds.features['target'])


def _get_variate_names_from_raw(ds: hf_datasets.Dataset, n_dims: int) -> list[str]:
    """Return per-variate column names for a raw (pre-conversion) dataset.

    Uses the ``variate_names`` column when present (TIME format), otherwise
    falls back to ``target_0``, ``target_1``, … .
    """
    if 'variate_names' in ds.column_names:
        names = ds[0]['variate_names']
        if isinstance(names, (list, tuple)) and len(names) == n_dims:
            return [str(n) for n in names]
    return [f'target_{i}' for i in range(n_dims)]


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

_META_KEY = '_fev_meta'


def _write_meta(
    ds: hf_datasets.Dataset, freq: str, variate_names: list[str] | None
) -> None:
    """Serialise *freq* and *variate_names* into ``ds.info.description`` as JSON."""
    meta: dict = {'freq': freq}
    if variate_names is not None:
        meta['variate_names'] = variate_names
    ds.info.description = json.dumps(meta)


def _read_meta(ds: hf_datasets.Dataset) -> dict:
    """Parse the JSON metadata from ``ds.info.description``. Returns ``{}`` on failure."""
    try:
        return json.loads(ds.info.description or '{}')
    except (json.JSONDecodeError, AttributeError):
        return {}


def attach_fev_meta(ds: hf_datasets.Dataset) -> hf_datasets.Dataset:
    """Attach ``freq`` and ``variate_names`` as instance attributes on *ds*.

    Call this after reloading a converted FEV dataset from disk to restore
    the attributes that were set by :func:`to_fev`.

    Parameters
    ----------
    ds:
        A :class:`~datasets.Dataset` previously saved by :func:`to_fev`.

    Returns
    -------
    datasets.Dataset
        The same object with ``.freq`` and ``.variate_names`` set.
    """
    meta = _read_meta(ds)
    ds.freq = meta.get('freq')  # type: ignore[attr-defined]
    ds.variate_names = meta.get('variate_names')  # type: ignore[attr-defined]
    return ds


# ---------------------------------------------------------------------------
# Main conversion function
# ---------------------------------------------------------------------------


def to_fev(
    ds: hf_datasets.Dataset,
    *,
    id_column: str = 'id',
    timestamp_column: str = 'timestamp',
    keep_extra_columns: bool = True,
) -> hf_datasets.Dataset:
    """Convert a GiftEval-format dataset to the FEV dataset format.

    Each row is transformed:
    - ``item_id`` → ``id``  (renamed)
    - ``start`` + ``freq`` + ``len(target)`` → ``timestamp``  (expanded)
    - ``start`` and ``freq`` columns are dropped
    - flat extra columns (e.g. scalar metadata) are kept as-is when
      *keep_extra_columns* is True
    - nested-list columns like ``past_feat_dynamic_real`` (shape
      ``[n_dims, n_steps]``) are **transposed** to ``[n_steps, n_dims]``
      in-place so that the FEV validator's per-column length check passes
      (the column name is preserved).

    **Univariate datasets** (``target`` is a flat 1-D list):
    The ``target`` column is kept unchanged.  ``dataset.variate_names``
    is set to ``None``.

    **Multivariate datasets** (``target`` is a 2-D list ``[n_dims, n_steps]``):
    The nested ``target`` column is split into one flat column per variate.
    Column names come from the ``variate_names`` column when present (TIME
    format) or default to ``target_0``, ``target_1``, … .  The original
    ``target`` and ``variate_names`` columns are dropped.
    ``dataset.variate_names`` is set to the list of column names.

    In both cases ``dataset.freq`` is set to the pandas frequency string
    inferred from the source data, and both attributes are persisted in
    ``dataset.info.description`` as JSON so they survive
    :func:`~datasets.Dataset.save_to_disk` / :func:`~datasets.load_from_disk`.
    After reloading call :func:`attach_fev_meta` to restore the attributes.

    Parameters
    ----------
    ds:
        A ``datasets.Dataset`` in GiftEval/TIME format (columns ``item_id``,
        ``start``, ``freq``, ``target``).
    id_column:
        Name to use for the identifier column in the output (default: ``"id"``).
    timestamp_column:
        Name to use for the timestamp column in the output
        (default: ``"timestamp"``).
    keep_extra_columns:
        If *True* (default), columns beyond the core four
        (``item_id``, ``start``, ``freq``, ``target``) are included
        in the output.  Nested-list columns (shape ``[n_dims, n_steps]``)
        are transposed to ``[n_steps, n_dims]`` so their outer length
        aligns with ``timestamp``.

    Returns
    -------
    datasets.Dataset
        Dataset in FEV format with columns ``id``, ``timestamp``,
        and either ``target`` (univariate) or one column per variate
        (multivariate), plus any extras.
        The attributes ``dataset.freq`` (str) and ``dataset.variate_names``
        (``list[str] | None``) are set on the returned object.

    Raises
    ------
    ValueError
        If required columns (``item_id``, ``start``, ``freq``, ``target``)
        are missing from *ds*.
    """
    required = {'item_id', 'start', 'freq', 'target'}
    missing = required - set(ds.column_names)
    if missing:
        raise ValueError(
            f'Dataset is missing required GiftEval columns: {missing}. '
            f'Available columns: {ds.column_names}'
        )

    multivariate = _is_multivariate_target(ds)

    # Frequency string from the first row (uniform across all rows in GiftEval/TIME)
    freq: str = ds[0]['freq']

    # Determine variate column names for multivariate datasets
    variate_names: list[str] | None = None
    if multivariate:
        n_dims = len(ds[0]['target'])
        variate_names = _get_variate_names_from_raw(ds, n_dims)

    # Columns to always drop from extras
    drop_cols = {'item_id', 'start', 'freq', 'target'}
    if 'variate_names' in ds.column_names:
        drop_cols.add('variate_names')
    extra_cols = [c for c in ds.column_names if c not in drop_cols]

    # Classify extra columns: flat (pass through) vs nested-list (need transpose)
    flat_extra: list[str] = []
    nested_extra: list[str] = []
    if keep_extra_columns:
        for col in extra_cols:
            if _is_nested_list_feature(ds.features[col]):
                nested_extra.append(col)
            else:
                flat_extra.append(col)

    def _convert_row(row: dict) -> dict:
        start = row['start']
        row_freq = row['freq']
        target = row['target']

        if multivariate:
            # target is [n_dims, n_steps]; use first dim to get length
            n = len(target[0])
        else:
            n = len(target)

        # Build explicit timestamp list
        index = pd.date_range(start=start, periods=n, freq=row_freq)
        timestamps = index.tolist()  # list of pd.Timestamp

        out: dict = {
            id_column: row['item_id'],
            timestamp_column: timestamps,
        }

        if multivariate:
            assert variate_names is not None
            for col_name, variate_series in zip(variate_names, target):
                out[col_name] = list(variate_series)
        else:
            out['target'] = target

        # Flat extra columns pass through unchanged
        for col in flat_extra:
            out[col] = row[col]

        # Nested-list columns: transpose [n_dims, n_steps] → [n_steps, n_dims]
        for col in nested_extra:
            out[col] = [list(t) for t in zip(*row[col])]

        return out

    if multivariate:
        assert variate_names is not None
        new_columns = (
            [id_column, timestamp_column] + variate_names + flat_extra + nested_extra
        )
    else:
        new_columns = (
            [id_column, timestamp_column, 'target'] + flat_extra + nested_extra
        )

    converted = ds.map(
        _convert_row,
        remove_columns=ds.column_names,
        desc='Converting GiftEval/TIME → FEV',
    )
    converted = converted.select_columns(new_columns)

    # Persist freq and variate_names in info.description (survives save/load)
    _write_meta(converted, freq, variate_names)

    # Attach as instance attributes for immediate in-process use
    converted.freq = freq  # type: ignore[attr-defined]
    converted.variate_names = variate_names  # type: ignore[attr-defined]

    return converted


def is_fev_format(ds: hf_datasets.Dataset) -> bool:
    """Return True if *ds* already appears to be in FEV format.

    Checks for the presence of an ``id``/``timestamp`` column pair (the
    canonical FEV column names) and the absence of the GiftEval-specific
    ``start``/``freq`` columns.
    """
    cols = set(ds.column_names)
    has_fev = 'id' in cols or 'timestamp' in cols
    has_gift = 'start' in cols and 'freq' in cols and 'item_id' in cols
    return has_fev and not has_gift


def is_gift_eval_format(ds: hf_datasets.Dataset) -> bool:
    """Return True if *ds* appears to be in GiftEval format."""
    cols = set(ds.column_names)
    return {'item_id', 'start', 'freq', 'target'}.issubset(cols)
