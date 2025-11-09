"""Global seed setting for deterministic evaluation and training."""

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int = 42) -> None:
    """
    Set all random seeds for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CUDA operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variables for additional determinism
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # Enable PyTorch deterministic mode (may impact performance)
    # This forces operations to use deterministic algorithms
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        # Some environments may not support this; gracefully degrade
        pass
