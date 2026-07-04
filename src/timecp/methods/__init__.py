"""Methods package init."""

from timecp.methods.aci import ACI
from timecp.methods.acmcp import AcMCP
from timecp.methods.adaptive_cqr import AdaptiveCQR
from timecp.methods.agaci import AgACI
from timecp.methods.cafht import CAFHT, NormMaxCP
from timecp.methods.cfrnn import CFRNN, JointCFRNN
from timecp.methods.copula_cpts import CopulaCPTS
from timecp.methods.cp import SplitCP, TrailingWindow, WeightedCP
from timecp.methods.cqr import CQR, CQRBase
from timecp.methods.dtaci import DtACI
from timecp.methods.pid import QuantileIntegrator
from timecp.methods.spci import SPCI

__all__ = [
    # Marginal per-horizon methods (ConformalPredictor)
    'SplitCP',
    'ACI',
    'AgACI',
    'DtACI',
    'TrailingWindow',
    'QuantileIntegrator',
    'AcMCP',
    'SPCI',
    'CFRNN',
    'WeightedCP',
    'CQR',
    'CQRBase',
    'AdaptiveCQR',
    # Joint simultaneous-coverage methods (JointPredictor)
    'JointCFRNN',
    'CopulaCPTS',
    'NormMaxCP',
    'CAFHT',
]
