"""
Classification models for hotel quality assessment.
"""

from typing import Any, Dict, List, Optional
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from ..core.base import BaseModel, BaseTrainer, BaseConfig

logger = logging.getLogger(__name__)


class ClassificationModel(BaseModel):
    """Neural network model for classification tasks."""
    
    def __init__(
        self,
        config: BaseConfig,
        input_dim: int,
        num_classes: int,
        hidden_dims: List[int] = [128, 64, 32],
        dropout: float = 0.3,
    ):
        super().__init__(config)
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        logger.info(f"ClassificationModel created with {self.count_parameters():,} parameters")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if isinstance(x, tuple):
            features, images = x
            x = torch.cat([features, images], dim=1)
        
        return self.network(x)


class ClassificationTrainer(BaseTrainer):
    """Trainer for classification models."""
    
    def __init__(
        self,
        model: ClassificationModel,
        config: BaseConfig,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
    ):
        criterion = nn.CrossEntropyLoss()
        
        if optimizer is None:
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        
        super().__init__(model, config, optimizer, criterion)
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        for batch in dataloader:
            if len(batch) != 2:
                continue
            
            inputs, targets = batch
            inputs = inputs.to(self.config.device)
            targets = targets.long().to(self.config.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * len(targets)
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader.dataset)
        accuracy = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
        
        return {"loss": avg_loss, "accuracy": accuracy, "f1": f1}
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for batch in dataloader:
                if len(batch) != 2:
                    continue
                
                inputs, targets = batch
                inputs = inputs.to(self.config.device)
                targets = targets.long().to(self.config.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item() * len(targets)
                all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())
        
        avg_loss = total_loss / len(dataloader.dataset)
        accuracy = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
        
        # Calculate ROC-AUC for binary/multiclass
        try:
            if self.model.num_classes == 2:
                auc = roc_auc_score(all_targets, [p[1] for p in all_probs])
            else:
                auc = roc_auc_score(all_targets, all_probs, multi_class='ovr', average='weighted')
        except ValueError:
            auc = 0.0
        
        return {"loss": avg_loss, "accuracy": accuracy, "f1": f1, "auc": auc}
