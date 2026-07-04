"""
timecp — score-based conformal prediction for time series.

All methods share a unified interface:
  fit(cal_scores)          — calibrate on historical nonconformity scores
  predict_quantile()       — return the current q̂
  predict_interval(y_hat)  — return (y_hat - q̂, y_hat + q̂)
  update(new_score)        — online update with observed nonconformity score

Methods:
  SplitCP            — static calibration set (standard ICP)
  ACI                — adaptive conformal inference (Gibbs & Candès, 2021)
  AgACI              — aggregated ACI over a gamma grid (Zaffran et al., 2022)
  DtACI              — distribution-free time-series ACI (Gibbs & Candès, 2022)
  TrailingWindow     — simple rolling empirical quantile (baseline)
  QuantileIntegrator — PID-style quantile controller (Angelopoulos et al., 2024)
  AcMCP              — asymmetric PID + MA/OLS scorecaster (Wang & Hyndman, 2024)
  SPCI               — sequential predictive conformal inference (Xu & Xie, 2023)
  CFRNN              — conformal-RNN with Bonferroni correction (Stankeviciute et al., 2021)
  WeightedCP         — weighted conformal prediction (Tibshirani et al., 2019)
  CQR                — conformalized quantile regression (Romano et al., 2019)
  AdaptiveCQR        — adaptive CQR combining ACI/AgACI/DtACI/PID with CQR scores
"""

from timecp.base import ConformalPredictor, conformal_quantile
from timecp.evaluation import compare_methods, rolling_metrics
from timecp.methods import (
    ACI,
    CFRNN,
    CQR,
    AdaptiveCQR,
    AgACI,
    CQRBase,
    DtACI,
    QuantileIntegrator,
    SplitCP,
    TrailingWindow,
    WeightedCP,
)
from timecp.models import (
    Chronos2Forecaster,
    FlowStateForecaster,
    Forecaster,
    TimesFMForecaster,
    TiRexForecaster,
)
from timecp.utils import print_error, print_warning, set_global_seed

__all__ = [
    # Base
    'ConformalPredictor',
    'conformal_quantile',
    'CQRBase',
    # Evaluation
    'compare_methods',
    'rolling_metrics',
    # Methods
    'SplitCP',
    'ACI',
    'AgACI',
    'DtACI',
    'TrailingWindow',
    'QuantileIntegrator',
    'CFRNN',
    'WeightedCP',
    'CQR',
    'AdaptiveCQR',
    # Models
    'Forecaster',
    'Chronos2Forecaster',
    'FlowStateForecaster',
    'TimesFMForecaster',
    'TiRexForecaster',
    'set_global_seed',
    'print_error',
    'print_warning',
]
