from functools import partial
from typing import Literal, Optional

from timecp.models.base import Forecaster


class TiRexForecaster(Forecaster):
    """Forecaster using the TiRex model (NX-AI/TiRex)."""

    model_type = 'tirex'
    default_quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    max_context_length = 2048
    BackendType = Literal['torch', 'cuda']

    def __init__(
        self,
        *,
        device: Optional[str] = None,
        batch_size: int = 2048,
        max_context: Optional[int] = None,
        backend: BackendType = 'torch',
        compile: bool = False,
        **kwargs,
    ):
        super().__init__(device=device, max_context=max_context, batch_size=batch_size)
        self._init_model(backend, compile, **kwargs)

    def _init_model(self, backend: BackendType, compile: bool, **kwargs):
        from tirex import load_model

        # self.model: ForecastModel = load_model(
        self.model = load_model(
            'NX-AI/TiRex',
            device=self.device,
            backend=backend,
            compile=compile,
            **kwargs,
        )

    def _predict(self, past_values, horizon: int, **kwargs):
        kwargs.setdefault('batch_size', self.batch_size)
        # Pre-truncate all input types so the model never receives tokens
        # beyond max_context (numpy paths relied on TiRex's internal
        # _adjust_context_length, which pads/truncates after the fact).
        past_values = self._truncate_context(past_values)
        predict_func = partial(self.model.forecast, prediction_length=horizon, **kwargs)

        if isinstance(past_values, list):
            inputs = past_values
            is_single = False
        else:
            arr = self._to_numpy_1d_or_batch(past_values)
            inputs = arr
            is_single = arr.ndim == 1

        if is_single:
            quantile, mean = predict_func(inputs)
            point = mean.squeeze(0)  # (1, H) -> (H,)
            quant = quantile.squeeze(0).permute(1, 0)  # (1, H, Q) -> (H, Q) -> (Q, H)
        else:
            quantile, mean = predict_func(inputs)
            point = mean  # (B, H)
            quant = quantile.permute(2, 0, 1)  # (B, H, Q) -> (B, H, Q) -> (Q, B, H)
        return point, quant
