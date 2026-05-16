"""
Regression models for hotel review score prediction.
"""

from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from ..core.base import BaseModel, BaseTrainer, BaseConfig

logger = logging.getLogger(__name__)


class RegressionModel(BaseModel):
    """
    Neural network model for regression tasks.
    
    Supports:
    - Simple MLP for tabular data
    - Multi-modal with image features
    """
    
    def __init__(
        self,
        config: BaseConfig,
        input_dim: int,
        hidden_dims: List[int] = [128, 64, 32],
        dropout: float = 0.3,
        use_images: bool = False,
        image_feature_dim: int = 512,
    ):
        super().__init__(config)
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.use_images = use_images
        
        # Build layers
        layers = []
        prev_dim = input_dim + (image_feature_dim if use_images else 0)
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        # Output layer (single value for regression)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        logger.info(f"RegressionModel created with {self.count_parameters():,} parameters")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Handle tuple input (features, images)
        if isinstance(x, tuple):
            features, images = x
            x = torch.cat([features, images], dim=1)
        
        return self.network(x).squeeze(-1)


class RegressionTrainer(BaseTrainer):
    """Trainer for regression models."""
    
    def __init__(
        self,
        model: RegressionModel,
        config: BaseConfig,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
    ):
        criterion = nn.MSELoss()
        
        if optimizer is None:
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        
        super().__init__(model, config, optimizer, criterion)
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        for batch in dataloader:
            if len(batch) == 2:
                inputs, targets = batch
            else:
                continue
            
            inputs = inputs.to(self.config.device)
            targets = targets.to(self.config.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * len(targets)
            all_preds.extend(outputs.detach().cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(dataloader.dataset)
        rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
        mae = mean_absolute_error(all_targets, all_preds)
        r2 = r2_score(all_targets, all_preds)
        
        return {"loss": avg_loss, "rmse": rmse, "mae": mae, "r2": r2}
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 2:
                    inputs, targets = batch
                else:
                    continue
                
                inputs = inputs.to(self.config.device)
                targets = targets.to(self.config.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item() * len(targets)
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(dataloader.dataset)
        rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
        mae = mean_absolute_error(all_targets, all_preds)
        r2 = r2_score(all_targets, all_preds)
        
        return {"loss": avg_loss, "rmse": rmse, "mae": mae, "r2": r2}
