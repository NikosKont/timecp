from typing import Optional

import numpy as np
import torch

from timecp.models.base import Forecaster


class FlowStateForecaster(Forecaster):
    """Forecaster using the FlowState model (ibm-research/flowstate)."""

    model_type = 'flowstate'
    default_quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    max_context_length = 4096

    def __init__(
        self,
        model_id: str = 'ibm-research/flowstate',
        *,
        device: Optional[str] = None,
        max_context: Optional[int] = None,
        revision: str = 'r1.1',
        batch_size: int = 256,
        scale_factor_strategy: str = 'default',
        **kwargs,
    ):
        super().__init__(device=device, max_context=max_context, batch_size=batch_size)
        self.scale_factor_strategy = scale_factor_strategy
        self._init_model(model_id, revision, **kwargs)

    def _init_model(self, model_id: str, revision: str, **kwargs):
        from tsfm_public import FlowStateForPrediction

        if 'prediction_type' in kwargs:
            kwargs.pop('prediction_type')

        self.model = FlowStateForPrediction.from_pretrained(
            model_id, revision=revision, **kwargs
        ).to(self.device)
        self.model.eval()

        # Match the FEV reference model: use the median (0.5 quantile) as the
        # point prediction for quantile-loss tasks (SQL, WQL, ...). The default
        # in tsfm_public is 'mean', which is a weighted average of the
        # quantiles and disagrees with the FEV FlowStateModel on
        # asymmetric forecast distributions.
        if self.scale_factor_strategy == 'fev':
            self.model.config.prediction_type = 'median'

        context_length = getattr(self.model.config, 'context_length', 4096)
        self.max_context = min(self.max_context_length, context_length)

    def _predict(self, past_values, horizon: int, **kwargs):
        freq = kwargs.pop('freq', None)
        seasonality = kwargs.pop('seasonality', 1)
        if 'scale_factor' not in kwargs:
            if self.scale_factor_strategy == 'fev':
                scale_factor = 24.0 / _get_seasonal_period(freq or 'D', seasonality)
            else:
                scale_factor = 1.0
                if freq is not None:
                    import pandas as pd

                    try:
                        offset = pd.tseries.frequencies.to_offset(freq)
                        if offset is not None:
                            if offset.name.startswith('D'):
                                scale_factor = 3.43
                            elif offset.name.startswith('W'):
                                scale_factor = 0.46
                            elif offset.name.startswith('M'):
                                scale_factor = 2.0
                            else:
                                delta = pd.Timedelta(offset)
                                hours = delta.total_seconds() / 3600.0
                                if hours > 0:
                                    scale_factor = hours
                    except Exception:
                        pass
            kwargs['scale_factor'] = scale_factor

        # batch_first=True -> expects (B, T, C)
        kwargs.setdefault('batch_first', True)

        # Adjust max_context dynamically if FEV strategy is used
        if self.scale_factor_strategy == 'fev':
            current_max_context = int(
                self.model.config.context_length / kwargs['scale_factor']
            )
        else:
            current_max_context = self.max_context

        # Truncate to max_context using the shared base helper.
        past_values = self._truncate_context(
            past_values, context_length=current_max_context
        )

        if isinstance(past_values, list):
            # FlowState does not have internal batching, so we sub-batch here
            # using self.batch_size so that OOM recovery via set_batch_size()
            # takes effect even when _predict is called directly with a large list.
            n_series = len(past_values)
            bs = self.batch_size or n_series  # fall back to single pass if unset
            point_chunks: list[torch.Tensor] = []
            quant_chunks: list[torch.Tensor] = []

            # SORT BY LENGTH to minimize padding overhead (unless FEV parity is required)
            if self.scale_factor_strategy == 'fev':
                sorted_indices = np.arange(n_series)
            else:
                lengths = [len(c) for c in past_values]
                sorted_indices = np.argsort(lengths)

            # Create a reverse mapping to restore original order
            reverse_indices = np.empty_like(sorted_indices)
            reverse_indices[sorted_indices] = np.arange(n_series)

            sorted_contexts = [past_values[i] for i in sorted_indices]

            for batch_contexts in _dynamic_batchify(
                sorted_contexts, bs, self.model.config.context_length
            ):
                batch_tensor, mask = _prepare_batch(
                    batch_contexts, current_max_context, self.device
                )
                chunk_kwargs = dict(kwargs, past_observed_mask=mask)

                with torch.no_grad():
                    forecast = self.model(
                        batch_tensor, prediction_length=horizon, **chunk_kwargs
                    )
                pred = forecast['prediction_outputs'].detach().cpu()
                quants = forecast['quantile_outputs'].detach().cpu()

                # FEV heuristic: enforce non-negative when all past values are non-negative
                if self.scale_factor_strategy == 'fev':
                    pred_np = pred.numpy()
                    quants_np = quants.numpy()
                    for i, c in enumerate(batch_contexts):
                        if np.all(np.nan_to_num(c, nan=1.0) >= 0):
                            pred_np[i] = np.maximum(pred_np[i], 0.0)
                            quants_np[i] = np.maximum(quants_np[i], 0.0)
                    pred = torch.from_numpy(pred_np)
                    quants = torch.from_numpy(quants_np)

                point_chunks.append(pred)
                quant_chunks.append(quants)

            # Check dimensions of the first chunk to know how to concatenate
            dim_point = point_chunks[0].dim()
            if dim_point == 3:
                # Shape: (B, H, 1)
                point_raw = torch.cat(point_chunks, dim=0)  # (N, H, 1)
                quant_raw = torch.cat(
                    quant_chunks, dim=0
                )  # (N, Q, H, 1) (Assuming (B, Q, H, 1))

                point_raw = point_raw[reverse_indices, :, :]
                quant_raw = quant_raw[reverse_indices, :, :, :]

                point = point_raw.squeeze(-1)  # (N, H)
                quant = quant_raw.squeeze(-1)  # (N, Q, H)
                quant = quant.permute(1, 0, 2)  # (Q, N, H)
            else:
                # Shape: (1, B, H, 1)
                point_raw = torch.cat(point_chunks, dim=1)  # (1, N, H, 1)
                quant_raw = torch.cat(quant_chunks, dim=1)  # (1, N, Q, H, 1)

                point_raw = point_raw[:, reverse_indices, :, :]
                quant_raw = quant_raw[:, reverse_indices, :, :, :]

                point = point_raw.squeeze(-1).squeeze(0)  # (N, H)
                quant = quant_raw.squeeze(-1).squeeze(0)  # (N, Q, H)
                quant = quant.permute(1, 0, 2)  # (Q, N, H)
        else:
            arr = self._to_numpy_1d_or_batch(past_values)
            arr_ndim = arr.ndim
            if arr_ndim == 1:
                ctx_arr = arr[np.newaxis, :, np.newaxis]  # (1, T, 1)
            else:
                ctx_arr = arr[:, :, np.newaxis]  # (B, T, 1)
            ctx = torch.from_numpy(ctx_arr).to(self.device)

            forecast = self.model(ctx, prediction_length=horizon, **kwargs)
            # prediction_outputs: (C, B, H, 1)  -- C=num_channels, always 1 here
            # quantile_outputs:   (C, B, Q, H, 1)
            point_raw = forecast['prediction_outputs'].detach().cpu()
            quant_raw = forecast['quantile_outputs'].detach().cpu()

            if arr_ndim == 1:
                point = point_raw.squeeze()  # (1,1,H,1) -> (H,)
                quant = quant_raw.squeeze()  # (1,1,Q,H,1) -> (Q,H)
            else:
                point = point_raw.squeeze(-1).squeeze(0)  # (B,H)
                quant = quant_raw.squeeze(-1).squeeze(0)  # (B,Q,H)
                quant = quant.permute(1, 0, 2)  # (Q,B,H)

        return point, quant


def _dynamic_batchify(
    contexts: list[np.ndarray], base_batch_size: int, pretrain_context: int
):
    """Yield batches with dynamic size — reduce batch size for long contexts, never exceed base."""
    i = 0
    while i < len(contexts):
        ctx_len = len(contexts[i])
        batch_size = min(
            base_batch_size, max(1, int(base_batch_size * pretrain_context / ctx_len))
        )
        yield contexts[i : i + batch_size]
        i += batch_size


def _prepare_batch(contexts: list[np.ndarray], max_context: int, device: str):
    """Left-pad, truncate, and ffill NaN values."""
    import torch

    truncated = [ctx[-max_context:] for ctx in contexts]
    max_len = max(len(c) for c in truncated)

    batch = np.full((len(truncated), max_len, 1), np.nan, dtype=np.float32)
    for i, ctx in enumerate(truncated):
        batch[i, max_len - len(ctx) :, 0] = ctx

    tensor = torch.from_numpy(batch).to(device)
    mask = ~torch.isnan(tensor[:, :, 0])

    # Optional FEV ffill logic
    nan_mask = torch.isnan(tensor[:, :, 0])
    if nan_mask.any():
        for i in range(tensor.shape[0]):
            seq = tensor[i, :, 0]
            m = torch.isnan(seq)
            if m.any():
                indices = torch.where(
                    m,
                    torch.zeros_like(m, dtype=torch.long),
                    torch.arange(len(seq), device=device),
                )
                indices = torch.cummax(indices, dim=0).values
                tensor[i, :, 0] = seq[indices]
                first_valid = (~m).nonzero(as_tuple=True)[0]
                if len(first_valid) > 0:
                    tensor[i, : first_valid[0], 0] = seq[first_valid[0]]
                else:
                    tensor[i, :, 0] = 0.0

    return tensor, mask


def _get_seasonal_period(freq: str, seasonality: int) -> int:
    """Determine the seasonal period for FlowState's scale_factor."""
    if seasonality > 1:
        return seasonality

    import pandas as pd

    offset = pd.tseries.frequencies.to_offset(freq)

    if isinstance(offset, (pd.offsets.YearBegin, pd.offsets.YearEnd)):
        return 4
    if isinstance(offset, (pd.offsets.QuarterBegin, pd.offsets.QuarterEnd)):
        return 4
    if isinstance(offset, (pd.offsets.MonthBegin, pd.offsets.MonthEnd)):
        return 12
    if isinstance(offset, pd.offsets.Week):
        return 52
    if isinstance(offset, (pd.offsets.Day, pd.offsets.BusinessDay)):
        return 365
    try:
        nanos_per_day = 24 * 3600 * 10**9
        return max(1, round(nanos_per_day / offset.nanos))
    except (ValueError, AttributeError):
        return 1
