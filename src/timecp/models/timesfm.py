import numpy as np

from timecp.models.base import Forecaster


class TimesFMForecaster(Forecaster):
    """Forecaster using the TimesFM-2.5 model (google/timesfm-2.5-200m-pytorch)."""

    model_type = 'timesfm'
    default_quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    max_context_length = 4096

    def __init__(
        self,
        model_id: str = 'google/timesfm-2.5-200m-pytorch',
        *,
        device: str | None = None,
        batch_size: int = 256,
        max_context: int | None = None,
        max_horizon: int = 1024,
        torch_compile: bool = False,
        normalize_inputs: bool = True,
        **config_kwargs,
    ):
        super().__init__(device=device, max_context=max_context, batch_size=batch_size)

        config_kwargs.setdefault('per_core_batch_size', batch_size)
        config_kwargs.setdefault('normalize_inputs', normalize_inputs)
        config_kwargs.setdefault('max_context', min(self.max_context, 4096))
        config_kwargs.setdefault('max_horizon', max_horizon)
        config_kwargs.setdefault('use_continuous_quantile_head', True)
        config_kwargs.setdefault('force_flip_invariance', True)
        config_kwargs.setdefault('infer_is_positive', True)
        config_kwargs.setdefault('fix_quantile_crossing', True)
        self.config = config_kwargs

        self._init_model(
            model_id,
            torch_compile,
        )

    def _init_model(
        self,
        model_id: str,
        torch_compile: bool,
    ):
        import timesfm

        m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(  # type: ignore[attr-defined]
            model_id, torch_compile=torch_compile, device=self.device
        )
        m.compile(
            timesfm.ForecastConfig(  # type: ignore[attr-defined]
                **self.config,
            )
        )
        self.model = m

    def _predict(self, past_values, horizon: int, **kwargs):
        # Truncate to max_context before building the input list so the model
        # never wastes computation on tokens it will discard internally.
        past_values = self._truncate_context(past_values)  # type: ignore[arg-type]

        if isinstance(past_values, list):
            inputs = [np.asarray(c, dtype=np.float32) for c in past_values]
            arr_ndim = 2
        else:
            arr = self._to_numpy_1d_or_batch(past_values)
            if arr.ndim == 1:
                inputs = [arr]
            else:
                inputs = list(arr)
            arr_ndim = arr.ndim

        point_raw_list = []
        quant_raw_list = []

        chunk_size = self.batch_size
        for i in range(0, len(inputs), chunk_size):
            chunk = inputs[i : i + chunk_size]
            # Dynamically set batch size to avoid the upstream library
            # padding the batch with dummy zeros up to self.batch_size.
            self.model.global_batch_size = len(chunk)
            p, q = self.model.forecast(horizon=horizon, inputs=chunk, **kwargs)
            point_raw_list.append(p)
            quant_raw_list.append(q)

        # Restore original batch size
        self.model.global_batch_size = self.batch_size

        point_raw = np.concatenate(point_raw_list, axis=0)
        quant_raw = np.concatenate(quant_raw_list, axis=0)

        # point_raw: (B, H) np; quant_raw: (B, H, Q) np
        # quant_raw is already a numpy array; transpose axes to (Q, B, H)
        quant_raw = np.asarray(quant_raw, dtype=np.float32)
        # remove mean as first quantile
        quant = quant_raw.transpose(2, 0, 1)[1:]  # -> (Q, B, H)
        point = np.asarray(point_raw, dtype=np.float32)

        if arr_ndim == 1:
            point = point.squeeze(0)  # (H,)
            quant = quant.squeeze(1)  # (Q, H)
        return point, quant

    def set_batch_size(self, batch_size: int) -> None:
        """Set the batch size for future predictions."""
        self.batch_size = batch_size
        self.model.global_batch_size = batch_size

    def update_task_params(self, max_context: int, max_horizon: int) -> None:
        """Update context and horizon dynamically without reloading."""
        self.max_context = min(max_context, 4096)
        self.config['max_context'] = self.max_context
        self.config['max_horizon'] = max_horizon
        import timesfm

        self.model.compile(
            timesfm.ForecastConfig(  # type: ignore[attr-defined]
                **self.config,
            )
        )
