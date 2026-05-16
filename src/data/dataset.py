"""
Dataset classes for hotel analytics.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset


class HotelDataset(Dataset):
    """PyTorch Dataset for hotel data."""
    
    def __init__(
        self,
        features: np.ndarray,
        targets: Optional[np.ndarray] = None,
        images: Optional[List[np.ndarray]] = None,
    ):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets) if targets is not None else None
        self.images = images
        
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        if self.images is not None:
            image = torch.FloatTensor(self.images[idx])
            if self.targets is not None:
                return (self.features[idx], image), self.targets[idx]
            return (self.features[idx], image)
        
        if self.targets is not None:
            return self.features[idx], self.targets[idx]
        return self.features[idx]
