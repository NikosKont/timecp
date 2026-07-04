from __future__ import annotations

from abc import ABC, ABCMeta, abstractmethod
from typing import Literal, Optional, Union

import datasets as hf_datasets
import fev
import numpy as np
import torch
from tqdm.auto import tqdm

# Canonical model type strings
ModelType = Literal['tirex', 'chronos2', 'flowstate', 'timesfm']

# Accepted aliases -> canonical name
_ALIASES: dict[str, str] = {
    'chronos': 'chronos2',
}


def _resolve_model_type(value: str) -> str:
    return _ALIASES.get(value, value)


ReturnType = Literal['both', 'point', 'quantile']
OutputFormat = Literal['torch', 'numpy']

# Shape conventions per model after normalization:
# point:     np.ndarray | Tensor  (batch, horizon)  or  (horizon,)
# quantile:  np.ndarray | Tensor  (batch, horizon, n_quantiles)  or  (horizon, n_quantiles)


def _default_device() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def _to_output(arr, fmt: OutputFormat):
    if fmt == 'numpy':
        if isinstance(arr, torch.Tensor):
            return arr.detach().cpu().numpy()
        return np.asarray(arr)
    else:
        if isinstance(arr, np.ndarray):
            return torch.from_numpy(arr)
        return arr


class _ForecasterMeta(ABCMeta):
    """Metaclass that makes ``Forecaster('tirex', ...)`` work as a factory
    without polluting __new__ / __init__ of subclasses."""

    def __call__(cls, *args, **kwargs):
        if cls is Forecaster:
            # Strip the leading model_type string before constructing the subclass.
            if args and isinstance(args[0], str):
                model_type = args[0]
                args = args[1:]
            else:
                model_type = kwargs.pop('model_type', None)
            if model_type is None:
                raise TypeError(
                    "Forecaster() requires 'model_type' when used as a factory"
                )
            resolved = _resolve_model_type(model_type)
            subcls = cls._registry.get(resolved)
            if subcls is None:
                raise ValueError(
                    f'No registered forecaster for model type {resolved!r}. '
                    f'Available: {list(cls._registry)}'
                )
            # Delegate to the subclass constructor with the stripped args.
            return subcls(*args, **kwargs)
        return super().__call__(*args, **kwargs)


class Forecaster(fev.ForecastingModel, ABC, metaclass=_ForecasterMeta):
    """Abstract base class for time-series forecasting models.

    Can also be used as a factory: ``Forecaster(model_type, **kwargs)``
    returns an instance of the appropriate subclass.
    """

    model_type: ModelType
    default_quantile_levels: list[float]
    max_context_length: int
    quantile_levels: list[float] | None
    batch_size: int
    # Subclasses that need quantile_levels forwarded into _predict kwargs should
    # override this to True.  Avoids model-type string checks in the base class.
    _pass_quantile_levels_to_predict: bool = False

    def __init_subclass__(cls, **kwargs):
        if hasattr(cls, 'model_type') and isinstance(cls.model_type, str):
            cls.model_name = cls.model_type
        super().__init_subclass__(**kwargs)

    def __class_getitem__(cls, model_type):
        """Allow ``Forecaster['tirex']`` syntax to retrieve the subclass type."""
        resolved = _resolve_model_type(model_type)
        subcls = cls._registry.get(resolved)
        if subcls is None:
            raise ValueError(f'No registered forecaster for model type {resolved!r}')
        return subcls

    def __init__(
        self,
        *,
        device: Optional[str] = None,
        max_context: Optional[int] = None,
        quantile_levels: Optional[list[float]] = None,
        batch_size: Optional[int] = None,
    ):
        super().__init__()
        if quantile_levels is not None:
            invalid = set(quantile_levels) - set(self.default_quantile_levels)
            if invalid:
                import warnings

                warnings.warn(
                    f'Invalid quantile levels {invalid} not in default_quantile_levels {self.default_quantile_levels}. Falling back to defaults.'
                )
                quantile_levels = self.default_quantile_levels
        self.device = device or _default_device()
        # Default to the subclass's full supported context length when not specified,
        # rather than hardcoding 1024, so models are not under-utilised by default.
        self.max_context = (
            min(max_context, self.max_context_length)
            if max_context is not None
            else self.max_context_length
        )
        self.quantile_levels = quantile_levels
        self.batch_size = batch_size or 256
        self.model = None

    def predict(
        self,
        past_values: Union[np.ndarray, torch.Tensor, list],
        horizon: int,
        *,
        forecast_type: ReturnType = 'both',
        output_format: OutputFormat = 'torch',
        quantile_levels: Optional[list[float] | str] = None,
        **kwargs,
    ) -> Union[
        tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray | tuple],
        torch.Tensor | np.ndarray | tuple,
    ]:
        """
        Args:
            past_values: 1-D array/tensor of shape (T,) or 2-D (B, T).
            horizon: forecast steps.
            forecast_type: 'both' -> (point, quantile); 'point' -> point only;
                           'quantile' -> quantile only.
            output_format: 'torch' or 'numpy'.
            quantile_levels: list of quantile_levels to return or 'all' for all quantiles.
            **kwargs: forwarded to the underlying model's prediction call.

        Returns:
            point    shape: (H,) for single input, (B, H) for batched
            quantile shape: (H, Q) for single input, (B, H, Q) for batched
        """
        quantile_levels = (
            quantile_levels or self.quantile_levels or self.default_quantile_levels
        )
        if self._pass_quantile_levels_to_predict:
            kwargs['quantile_levels'] = quantile_levels

        point, quantiles = self._predict(past_values, horizon, **kwargs)

        point = _to_output(point, output_format)
        if forecast_type == 'point':
            return point

        # internal shape: (Q, H) or (Q, B, H) -> transpose to (H, Q) or (B, H, Q)
        if isinstance(quantiles, torch.Tensor):
            if quantiles.ndim == 2:
                quantiles = quantiles.permute(1, 0)
            else:
                quantiles = quantiles.permute(1, 2, 0)
        else:
            quantiles = np.asarray(quantiles)
            if quantiles.ndim == 2:
                quantiles = quantiles.T
            else:
                quantiles = quantiles.transpose(1, 2, 0)

        if quantile_levels != 'all' and not self._pass_quantile_levels_to_predict:
            assert isinstance(quantile_levels, list)
            try:
                indices = [
                    self.default_quantile_levels.index(q) for q in quantile_levels
                ]
            except ValueError as e:
                raise ValueError(
                    f'One or more quantile levels from {quantile_levels} are not in default_quantile_levels: {self.default_quantile_levels}'
                ) from e
            quantiles = quantiles[..., indices]

        quantiles = _to_output(quantiles, output_format)

        if forecast_type == 'both':
            return point, quantiles
        else:
            return quantiles

    @abstractmethod
    def _predict(
        self,
        past_values: Union[np.ndarray, torch.Tensor, list],
        horizon: int,
        **kwargs,
    ) -> tuple:
        """Return (point, quantile) in internal shape convention.

        Internal shape convention:
            point:    (H,) or (B, H)
            quantile: (Q, H) or (Q, B, H)
        """
        ...
        raise NotImplementedError('Subclasses must implement _predict() method.')

    @staticmethod
    def _to_numpy_1d_or_batch(past_values) -> np.ndarray:
        """Normalise input to numpy, preserving batch dim if present."""
        if isinstance(past_values, torch.Tensor):
            arr = past_values.detach().cpu().numpy()
        else:
            arr = np.asarray(past_values, dtype=np.float32)
        return arr

    def _truncate_context(
        self,
        past_values: Union[list, np.ndarray],
        context_length: Optional[int] = None,
    ) -> Union[list, np.ndarray]:
        """Truncate *past_values* to the last *context_length* time steps.

        Centralises the per-model truncation logic so subclasses do not
        each need to replicate the list / 1-D / 2-D branch.

        Parameters
        ----------
        past_values:
            Either a list of 1-D arrays (ragged batch) or a numpy array of
            shape ``(T,)`` (single series) or ``(B, T)`` (uniform batch).
        context_length:
            Maximum number of most-recent time steps to keep.  Defaults to
            ``self.max_context``.  When ``None`` or ``0`` no truncation is
            applied.

        Returns
        -------
        Truncated input in the same container type as *past_values*.
        """
        ctx = context_length if context_length is not None else self.max_context
        if not ctx:
            return past_values
        if isinstance(past_values, list):
            return [c[-ctx:] for c in past_values]
        arr = self._to_numpy_1d_or_batch(past_values)
        if arr.ndim == 1:
            return arr[-ctx:]
        return arr[:, -ctx:]

    def set_batch_size(self, batch_size: int) -> None:
        """Set the batch size for future predictions."""
        self.batch_size = batch_size

    # ------------------------------------------------------------------
    # fev.ForecastingModel interface
    # ------------------------------------------------------------------

    def _run_inference(
        self,
        task: fev.Task,
        batch_windows: bool,
        extract_gt: bool,
        force_extract: bool = False,
    ) -> tuple[list[dict], list[dict | None]]:
        predictions_per_window: list[dict] = []
        ground_truths_per_window: list[dict | None] = []

        import os

        num_proc = 1 if os.name == 'nt' else 16

        if not batch_windows:
            for window in tqdm(
                task.iter_windows(num_proc=num_proc),
                total=task.num_windows,
                desc='Windows',
                leave=False,
                # dynamic_ncols=True,
                ncols=80,
            ):
                past_data, _future_data = window.get_input_data()
                while True:
                    try:
                        with self._record_inference_time():
                            preds = self._predict_window(past_data, task, window)
                        break
                    except Exception as e:
                        import gc
                        import sys

                        if isinstance(e, torch.cuda.OutOfMemoryError) or (
                            isinstance(e, RuntimeError)
                            and 'out of memory' in str(e).lower()
                        ):
                            new_batch_size = self.batch_size // 2
                            if new_batch_size == 0:
                                raise e
                            print(
                                f'\n  [OOM] Out of memory. Reducing batch_size from {self.batch_size} to {new_batch_size} and retrying current window...',
                                flush=True,
                            )
                            self.set_batch_size(new_batch_size)

                            e = None
                            sys.exc_info()
                            sys.last_traceback = None
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        else:
                            raise
                predictions_per_window.append(preds)

                if extract_gt:
                    gt = self._extract_ground_truth(window.get_ground_truth(), task)
                    ground_truths_per_window.append(gt)

            return predictions_per_window, ground_truths_per_window

        # Phase 1: Extract all data across windows
        import hashlib
        import json
        import pickle
        from pathlib import Path

        target_cols = list(task.target_columns)

        all_past_values = []
        window_sizes = []

        cache_key_data = {
            'task_name': getattr(task, 'task_name', 'unnamed'),
            'horizon': task.horizon,
            'num_windows': task.num_windows,
            'target_columns': target_cols,
            'max_context': self.max_context,
            'extract_gt': extract_gt,
        }

        key_str = json.dumps(cache_key_data, sort_keys=True)
        key_hash = hashlib.md5(key_str.encode()).hexdigest()
        safe_name = (
            getattr(task, 'task_name', 'unnamed')
            .replace('/', '_')
            .replace('\\', '_')
            .replace(' ', '-')
        )
        cache_dir = Path('.cache/timecp/windows')
        cache_file = cache_dir / f'{safe_name}_{key_hash}.pkl'

        cache_hit = False
        meta_per_window = []
        if not force_extract and cache_file.exists():
            try:
                tqdm.write(f'  [Cache] Loading extracted windows from {cache_file}...')
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                if 'all_past_values_concatenated' in cached_data:
                    concatenated = cached_data['all_past_values_concatenated']
                    lengths = cached_data['all_past_values_lengths']
                    if len(lengths) > 0:
                        offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
                        np.cumsum(lengths, out=offsets[1:])
                        all_past_values = [
                            concatenated[start:end]
                            for start, end in zip(offsets[:-1], offsets[1:])
                        ]
                    else:
                        all_past_values = []
                else:
                    all_past_values = cached_data['all_past_values']
                window_sizes = cached_data['window_sizes']
                meta_per_window = cached_data.get(
                    'meta_per_window', [{} for _ in window_sizes]
                )
                ground_truths_per_window = cached_data['ground_truths_per_window']
                cache_hit = True
            except Exception as e:
                tqdm.write(
                    f'  [Cache] Failed to load cache {cache_file}: {e}. Re-extracting...'
                )
                cache_hit = False

        if not cache_hit:
            all_past_values = []
            window_sizes = []
            meta_per_window = []
            # Note: ground_truths_per_window is already [] from before

            for window in tqdm(
                task.iter_windows(num_proc=num_proc),
                total=task.num_windows,
                desc='Extracting Data',
                leave=False,
                # dynamic_ncols=True,
                ncols=80,
            ):
                past_data, _future_data = window.get_input_data()
                past_dict = past_data.select_columns(target_cols).to_dict()
                ctx = self.max_context
                window_series = []
                for col in target_cols:
                    if ctx:
                        window_series.extend(
                            [
                                np.asarray(row[-ctx:], dtype=np.float32)
                                for row in past_dict[col]
                            ]
                        )
                    else:
                        window_series.extend(
                            [
                                np.asarray(row, dtype=np.float32)
                                for row in past_dict[col]
                            ]
                        )
                all_past_values.extend(window_series)
                window_sizes.append(len(window_series))

                meta_keys = {'item_id', 'id', 'start', 'timestamp', 'freq'}
                meta_cols = [c for c in past_data.column_names if c in meta_keys]
                meta_per_window.append(
                    past_data.select_columns(meta_cols).to_dict() if meta_cols else {}
                )

                if extract_gt:
                    gt = self._extract_ground_truth(window.get_ground_truth(), task)
                    ground_truths_per_window.append(gt)

            if all_past_values:
                lengths = np.array([len(x) for x in all_past_values], dtype=np.int32)
                concatenated = np.concatenate(all_past_values)
            else:
                lengths = np.array([], dtype=np.int32)
                concatenated = np.array([], dtype=np.float32)

            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                with open(cache_file, 'wb') as f:
                    pickle.dump(
                        {
                            'all_past_values_concatenated': concatenated,
                            'all_past_values_lengths': lengths,
                            'window_sizes': window_sizes,
                            'meta_per_window': meta_per_window,
                            'ground_truths_per_window': ground_truths_per_window,
                        },
                        f,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
            except Exception as e:
                tqdm.write(f'  [Cache] Failed to save cache {cache_file}: {e}')

        # Phase 2: Inference in batches
        all_point_list = []
        all_quantile_list = []

        chunk_start = 0
        n_total = len(all_past_values)

        pbar = tqdm(total=n_total, desc='Inference (Series)', leave=False, ncols=80)

        while chunk_start < n_total:
            chunk_end = min(chunk_start + self.batch_size, n_total)
            chunk = all_past_values[chunk_start:chunk_end]
            try:
                with self._record_inference_time():
                    point, quant = self._batch_predict(
                        chunk,
                        task.horizon,
                        task.quantile_levels,
                        freq=task.freq,
                        seasonality=getattr(task, 'seasonality', 1),
                    )
                all_point_list.append(point)
                all_quantile_list.append(quant)

                pbar.update(len(chunk))
                chunk_start = chunk_end
            except Exception as e:
                import gc
                import sys

                if isinstance(e, torch.cuda.OutOfMemoryError) or (
                    isinstance(e, RuntimeError) and 'out of memory' in str(e).lower()
                ):
                    new_batch_size = self.batch_size // 2
                    if new_batch_size == 0:
                        pbar.close()
                        raise e
                    print(
                        f'\n  [OOM] Out of memory. Reducing batch_size from {self.batch_size} to {new_batch_size} and retrying chunk...',
                        flush=True,
                    )
                    self.set_batch_size(new_batch_size)

                    e = None
                    sys.exc_info()
                    sys.last_traceback = None
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    pbar.close()
                    raise

        pbar.close()

        all_point = (
            np.concatenate(all_point_list, axis=0) if all_point_list else np.array([])
        )
        all_quantile = (
            np.concatenate(all_quantile_list, axis=0)
            if all_quantile_list
            else np.array([])
        )

        # Phase 3: Unflatten back into per-window dicts
        offset = 0
        for w_idx, n_series in enumerate(window_sizes):
            n_rows = n_series // len(target_cols)
            window_dict = {}
            window_meta = meta_per_window[w_idx] if meta_per_window else {}

            for col_idx, col in enumerate(target_cols):
                col_start = offset + col_idx * n_rows
                col_end = col_start + n_rows
                col_point = all_point[col_start:col_end]
                col_quantile = all_quantile[col_start:col_end]

                pred_dict: dict = {
                    'predictions': col_point,
                }
                for i, q in enumerate(task.quantile_levels):
                    pred_dict[str(q)] = col_quantile[:, :, i]

                for k, v in window_meta.items():
                    pred_dict[k] = v

                window_dict[col] = pred_dict

            predictions_per_window.append(window_dict)
            offset += n_series

        return predictions_per_window, ground_truths_per_window

    def _fit_predict(
        self, task: fev.Task, batch_windows: bool = True, force_extract: bool = False
    ) -> list[dict]:
        """Produce predictions for every evaluation window in *task*.

        Returns
        -------
        list[dict]
            One dictionary per window, containing a target key mapping
            to a dictionary with ``"predictions"`` (point forecast)
            and one key per quantile level.
        """
        preds, _ = self._run_inference(
            task, batch_windows, extract_gt=False, force_extract=force_extract
        )
        return preds

    def fit_predict_with_gt(
        self,
        task: fev.Task,
        batch_windows: bool = True,
        force_extract: bool = False,
    ) -> tuple[list[dict], list[dict | None]]:
        """Produce predictions and ground truth for every window in *task*.

        Extends :meth:`fit_predict` by also retrieving the ground-truth
        target values for each window, so that CP calibration can be
        performed offline without re-accessing the raw dataset.

        Returns
        -------
        predictions : list[dict]
            One dict per window (same structure as ``fit_predict``).
        ground_truths : list[dict | None]
            One dict per window with a single key equal to
            ``task.target_columns[0]`` containing a dict with a
            single key ``"target"`` of shape ``(N, H)``.  ``None`` when
            the future data is unavailable.
        """
        return self._run_inference(
            task, batch_windows, extract_gt=True, force_extract=force_extract
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _predict_window(
        self,
        past_data: hf_datasets.Dataset,
        task: fev.Task,
        window: fev.task.EvaluationWindow,  # type: ignore[name-defined]
    ) -> dict:
        """Predict all series for one evaluation window.

        Only the first (and for univariate tasks, only) target column is
        used.  For multivariate tasks with ``generate_univariate_targets_from``
        set, FEV has already expanded variates into individual rows with a
        single ``target`` column before this method is called.
        """
        horizon = window.horizon
        quantile_levels = task.quantile_levels
        target_cols = task.target_columns

        # Extract past values across all target columns
        past_dict = past_data.select_columns(target_cols).to_dict()
        ctx = self.max_context
        past_values_list = []
        for col in target_cols:
            if ctx:
                past_values_list.extend(
                    [np.asarray(row[-ctx:], dtype=np.float32) for row in past_dict[col]]
                )
            else:
                past_values_list.extend(
                    [np.asarray(row, dtype=np.float32) for row in past_dict[col]]
                )

        n_series = len(past_values_list)

        # Batch-predict across all series
        all_point, all_quantile = self._batch_predict(
            past_values_list,
            horizon,
            quantile_levels,
            freq=task.freq,
            seasonality=getattr(task, 'seasonality', 1),
        )

        meta_keys = {'item_id', 'id', 'start', 'timestamp', 'freq'}
        meta_cols = [c for c in past_data.column_names if c in meta_keys]
        meta_dict: dict = (
            past_data.select_columns(meta_cols).to_dict() if meta_cols else {}
        )  # type: ignore

        # Unflatten back into columns
        n_rows = n_series // len(target_cols)
        window_dict = {}
        for col_idx, col in enumerate(target_cols):
            col_start = col_idx * n_rows
            col_end = col_start + n_rows
            col_point = all_point[col_start:col_end]
            col_quantile = all_quantile[col_start:col_end]

            pred_dict: dict = {
                'predictions': col_point,
            }
            for i, q in enumerate(quantile_levels):
                pred_dict[str(q)] = col_quantile[:, :, i]

            for k, v in meta_dict.items():
                pred_dict[k] = v

            window_dict[col] = pred_dict

        return window_dict

    def _extract_ground_truth(
        self,
        future_data: hf_datasets.Dataset | None,
        task: fev.Task,
    ) -> dict | None:
        """Extract ground truth values from the window's future data.

        Parameters
        ----------
        future_data : Dataset or None
            The future portion of the window as returned by
            ``window.get_input_data()``.  Each row contains a 1-D sequence
            of length ``horizon`` for one series.

        Returns
        -------
        dict or None
            Dict with key ``target_col`` containing a dict with
            a single ``"target"`` key of numpy arrays, or ``None``
            if future_data is unavailable.
        """
        if future_data is None:
            return None

        target_cols = task.target_columns
        gt_dict = {}

        meta_keys = {'item_id', 'id', 'start', 'timestamp', 'freq'}
        meta_cols = [c for c in future_data.column_names if c in meta_keys]
        meta_dict: dict = (
            future_data.select_columns(meta_cols).to_dict() if meta_cols else {}
        )  # type: ignore

        for col in target_cols:
            if col in future_data.column_names:
                try:
                    gt_values: list[list[float]] = [
                        list(row) for row in future_data[col]
                    ]
                    gt_col_dict: dict = {
                        'target': np.asarray(gt_values, dtype=np.float32)
                    }
                    for k, v in meta_dict.items():
                        gt_col_dict[k] = v
                    gt_dict[col] = gt_col_dict
                except (KeyError, TypeError):
                    pass

        return gt_dict if gt_dict else None

    def _batch_predict(
        self,
        past_values_list: list[np.ndarray],
        horizon: int,
        quantile_levels: list[float],
        freq: str | None = None,
        seasonality: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run the forecaster over *past_values_list*.

        All models (TiRex, TimesFM, Chronos, and FlowState) handle sub-batching
        internally, so we forward the full list in a single call.

        Returns
        -------
        all_point : np.ndarray, shape (N, H)
        all_quantile : np.ndarray, shape (N, H, Q)
        """
        kwargs: dict = {}
        if self.model_type == 'flowstate':
            if freq is not None:
                kwargs['freq'] = freq
            kwargs['seasonality'] = seasonality

        # Pass the full list — the model sub-batches internally.
        point_raw, quantile_raw = self.predict(  # type: ignore[misc]
            past_values_list,
            horizon,
            forecast_type='both',
            output_format='numpy',
            quantile_levels=quantile_levels if quantile_levels else None,
            **kwargs,
        )
        all_point = np.atleast_2d(np.asarray(point_raw))  # (N, H)
        all_quantile = np.asarray(quantile_raw)
        if all_quantile.ndim == 2:  # (H, Q) -> (1, H, Q)
            all_quantile = all_quantile[np.newaxis]
        return all_point, all_quantile
