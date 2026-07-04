from functools import partial
from typing import Optional

import numpy as np
import torch

from timecp.models.base import Forecaster


class Chronos2Forecaster(Forecaster):
    """Forecaster using the Chronos-2 model (amazon/chronos-2)."""

    model_type = 'chronos2'
    default_quantile_levels = [
        0.01,
        0.05,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
        0.99,
    ]
    max_context_length = 8196
    # Tell the base class to inject resolved quantile_levels into _predict kwargs.
    _pass_quantile_levels_to_predict = True

    def __init__(
        self,
        model_id: str = 'amazon/chronos-2',
        *,
        device: Optional[str] = None,
        batch_size: int = 256,
        max_context: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(device=device, max_context=max_context, batch_size=batch_size)
        self._init_model(model_id, **kwargs)

    def _init_model(self, model_id: str, **kwargs):
        from chronos import BaseChronosPipeline

        self.pipeline = BaseChronosPipeline.from_pretrained(
            model_id, device_map=self.device, **kwargs
        )
        self.model = self.pipeline.model
        model_context_length = getattr(self.pipeline, 'model_context_length', 4096)
        self.max_context = min(self.max_context, model_context_length, 4096)

    def _predict(self, past_values, horizon: int, **kwargs):
        pipeline_type = type(self.pipeline).__name__

        if pipeline_type in ('ChronosPipeline', 'Chronos2Pipeline'):
            kwargs.setdefault('batch_size', self.batch_size)
        elif 'batch_size' in kwargs:
            kwargs.pop('batch_size')

        # quantile_levels was injected by the base class via _pass_quantile_levels_to_predict.
        quantile_levels = (
            kwargs.pop('quantile_levels', None) or self.default_quantile_levels
        )

        # Pre-truncate to max_context so the pipeline receives exactly the
        # tokens it needs — avoids internal truncation and potential OOM
        # from oversized context tensors.
        past_values = self._truncate_context(past_values)

        predict_func = partial(
            self.pipeline.predict_quantiles,
            prediction_length=horizon,
            quantile_levels=quantile_levels,
            # batch_size controls Chronos2Dataset.convert_inputs(), which builds
            # a DataLoader that sub-batches the full input list internally.
            # context_length is already ≤ max_context (inputs were pre-truncated
            # above), so Chronos2 won't silently reset it.
            **kwargs,
        )
        # Convert all formats to a list of 1-D torch tensors
        if isinstance(past_values, list):
            ctx = [
                torch.from_numpy(np.asarray(c, dtype=np.float32)) for c in past_values
            ]
            is_single = False
        else:
            arr = self._to_numpy_1d_or_batch(past_values)
            if arr.ndim == 1:
                ctx = [torch.from_numpy(arr)]
                is_single = True
            else:
                ctx = [torch.from_numpy(row) for row in arr]
                is_single = False

        if pipeline_type in ('ChronosPipeline', 'Chronos2Pipeline'):
            quantiles_out, mean_out = predict_func(
                ctx, context_length=max(c.shape[-1] for c in ctx)
            )
            point = torch.stack([m.squeeze(0) for m in mean_out])  # (B, H)
            quant = torch.stack([q.squeeze(0) for q in quantiles_out])  # (B, H, Q)
        else:
            quantiles_out, mean_out = predict_func(ctx)
            point = mean_out
            quant = quantiles_out

        quant = quant.permute(2, 0, 1)  # (B, H, Q) -> (Q, B, H)

        if is_single:
            point = point.squeeze(0)  # (1, H) -> (H,)
            quant = quant.squeeze(1)  # (Q, 1, H) -> (Q, H)
        return point, quant
