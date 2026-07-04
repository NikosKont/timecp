import random
import sys

import numpy as np


def set_global_seed(seed: int | None) -> None:
    """Set global seed for reproducibility across random, numpy, and torch."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def print_warning(msg: str) -> None:
    """Print a warning message to stderr."""
    import warnings

    warnings.warn(msg, UserWarning, stacklevel=2)
    print(f'  [\033[93mWARN\033[0m] {msg}', file=sys.stderr, flush=True)


def print_error(msg: str) -> None:
    """Print an error message to stderr."""
    print(f'  [\033[91mERROR\033[0m] {msg}', file=sys.stderr, flush=True)
