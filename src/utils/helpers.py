"""
Helper utility functions.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_results(
    results: Dict[str, Any],
    path: Union[str, Path],
    indent: int = 2,
) -> None:
    """Save results to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=indent, default=str)


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_device(gpu_id: Optional[int] = None) -> torch.device:
    """Get the best available device."""
    if gpu_id is not None and torch.cuda.is_available():
        if gpu_id < torch.cuda.device_count():
            return torch.device(f"cuda:{gpu_id}")
        else:
            print(f"Warning: GPU {gpu_id} not available. Using CPU.")
    
    if torch.cuda.is_available():
        return torch.device("cuda")
    
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_number(num: int) -> str:
    """Format large numbers with commas."""
    return f"{num:,}"
