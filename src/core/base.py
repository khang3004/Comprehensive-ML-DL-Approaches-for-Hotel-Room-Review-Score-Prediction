"""
Base classes and abstractions for the ML system.

This module provides foundational classes that all models, trainers, and data loaders inherit from.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
import logging
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# Configure logging
logger = logging.getLogger(__name__)


class BaseConfig:
    """Base configuration class for all components."""
    
    def __init__(self, **kwargs):
        self.seed = kwargs.get("seed", 42)
        self.device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.paths = kwargs.get("paths", {})
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self.__dict__.copy()
    
    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info(f"Configuration saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "BaseConfig":
        """Load configuration from JSON file."""
        path = Path(path)
        with open(path, "r") as f:
            config_dict = json.load(f)
        return cls(**config_dict)


class BaseModel(ABC, nn.Module):
    """Abstract base class for all neural network models."""
    
    def __init__(self, config: BaseConfig):
        super().__init__()
        self.config = config
        self._is_compiled = False
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        pass
    
    def compile(self) -> None:
        """Compile the model for optimization (optional)."""
        if hasattr(torch, "compile") and not self._is_compiled:
            try:
                self._model = torch.compile(self._model)
                self._is_compiled = True
                logger.info("Model compiled successfully")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save(self, path: Union[str, Path], metadata: Optional[Dict] = None) -> None:
        """Save model weights and metadata."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": self.config.to_dict(),
            "timestamp": datetime.now().isoformat(),
            "num_parameters": self.count_parameters(),
        }
        
        if metadata:
            checkpoint["metadata"] = metadata
            
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path} ({self.count_parameters():,} parameters)")
    
    def load(self, path: Union[str, Path], strict: bool = True) -> "BaseModel":
        """Load model weights from checkpoint."""
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.config.device, weights_only=False)
        self.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        logger.info(f"Model loaded from {path}")
        return self


class BaseTrainer(ABC):
    """Abstract base class for all trainers."""
    
    def __init__(
        self,
        model: BaseModel,
        config: BaseConfig,
        optimizer: Optional[torch.optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None,
    ):
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.criterion = criterion
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_metric": [],
            "val_metric": [],
        }
        
    @abstractmethod
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        pass
    
    @abstractmethod
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        pass
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        early_stopping_patience: int = 10,
        save_best: bool = True,
        save_path: Optional[Path] = None,
    ) -> Dict[str, List[float]]:
        """Full training loop with validation and early stopping."""
        best_val_loss = float("inf")
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.validate(val_loader)
            
            # Update history
            self.history["train_loss"].append(train_metrics.get("loss", 0))
            self.history["val_loss"].append(val_metrics.get("loss", 0))
            
            # Logging
            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_metrics.get('loss', 0):.4f} | "
                f"Val Loss: {val_metrics.get('loss', 0):.4f}"
            )
            
            # Early stopping check
            if val_metrics.get("loss", float("inf")) < best_val_loss:
                best_val_loss = val_metrics.get("loss", float("inf"))
                patience_counter = 0
                
                if save_best and save_path:
                    self.model.save(save_path, metadata={"epoch": epoch, "val_loss": best_val_loss})
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break
        
        return self.history
    
    def save_history(self, path: Union[str, Path]) -> None:
        """Save training history."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        import pandas as pd
        df = pd.DataFrame(self.history)
        df.to_csv(path, index=False)
        logger.info(f"Training history saved to {path}")


class BaseDataLoader(ABC):
    """Abstract base class for all data loaders."""
    
    def __init__(self, config: BaseConfig):
        self.config = config
        self.dataset: Optional[Dataset] = None
        
    @abstractmethod
    def load_data(self, path: Union[str, Path]) -> Any:
        """Load raw data from disk."""
        pass
    
    @abstractmethod
    def preprocess(self, data: Any) -> Any:
        """Preprocess raw data."""
        pass
    
    def get_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
    ) -> DataLoader:
        """Create PyTorch DataLoader."""
        if self.dataset is None:
            raise ValueError("Dataset not initialized. Call prepare() first.")
            
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=shuffle,
        )
    
    def prepare(self, data_path: Union[str, Path]) -> None:
        """Full data preparation pipeline."""
        raw_data = self.load_data(data_path)
        processed_data = self.preprocess(raw_data)
        self.dataset = self.create_dataset(processed_data)
        logger.info(f"Data prepared from {data_path}")
    
    @abstractmethod
    def create_dataset(self, data: Any) -> Dataset:
        """Create PyTorch Dataset from processed data."""
        pass


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")
