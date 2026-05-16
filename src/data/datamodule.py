"""
DataModule for hotel analytics data.

Provides a clean interface for loading, preprocessing, and creating DataLoaders.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from ..core.base import BaseConfig, BaseDataLoader, set_seed

logger = logging.getLogger(__name__)


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


class HotelDataModule(BaseDataLoader):
    """
    DataModule for hotel analytics.
    
    Handles:
    - Loading CSV data
    - Feature preprocessing (scaling, encoding)
    - Image loading (if available)
    - Train/val/test splitting
    - Creating DataLoaders
    """
    
    def __init__(self, config: BaseConfig):
        super().__init__(config)
        self.data_path: Optional[Path] = None
        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[Dict[str, np.ndarray]] = None
        self.feature_columns: List[str] = []
        self.target_column: str = "review_score"
        
    def load_data(self, path: Union[str, Path]) -> pd.DataFrame:
        """Load raw data from CSV file."""
        self.data_path = Path(path)
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        logger.info(f"Loading data from {self.data_path}")
        self.raw_data = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(self.raw_data)} samples with {len(self.raw_data.columns)} columns")
        
        return self.raw_data
    
    def preprocess(
        self,
        data: pd.DataFrame,
        num_features: Optional[List[str]] = None,
        cat_features: Optional[List[str]] = None,
        target: str = "review_score",
    ) -> Dict[str, np.ndarray]:
        """
        Preprocess raw data.
        
        Args:
            data: Raw DataFrame
            num_features: List of numerical feature columns
            cat_features: List of categorical feature columns
            target: Target column name
            
        Returns:
            Dictionary with 'features', 'targets', and optionally 'images'
        """
        logger.info("Preprocessing data...")
        
        # Handle missing values
        data = data.copy()
        
        # Separate features and target
        if num_features is None:
            num_features = data.select_dtypes(include=[np.number]).columns.tolist()
            if target in num_features:
                num_features.remove(target)
        
        if cat_features is None:
            cat_features = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        self.feature_columns = num_features + cat_features
        self.target_column = target
        
        # Process numerical features
        X_num = data[num_features].fillna(data[num_features].median())
        
        # Normalize numerical features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_num_scaled = scaler.fit_transform(X_num)
        
        # Process categorical features
        if cat_features:
            X_cat = pd.get_dummies(data[cat_features], drop_first=True)
            X = np.hstack([X_num_scaled, X_cat.values])
        else:
            X = X_num_scaled
        
        # Extract target
        y = data[target].values if target in data.columns else None
        
        self.processed_data = {
            "features": X,
            "targets": y,
            "feature_names": self.feature_columns + X_cat.columns.tolist() if cat_features else self.feature_columns,
        }
        
        logger.info(f"Preprocessed data shape: {X.shape}")
        if y is not None:
            logger.info(f"Target shape: {y.shape}")
        
        return self.processed_data
    
    def create_dataset(
        self,
        data: Dict[str, np.ndarray],
        images: Optional[List[np.ndarray]] = None,
    ) -> HotelDataset:
        """Create PyTorch Dataset from processed data."""
        return HotelDataset(
            features=data["features"],
            targets=data.get("targets"),
            images=images,
        )
    
    def get_dataloaders(
        self,
        batch_size: int = 32,
        val_split: float = 0.1,
        test_split: float = 0.1,
        num_workers: int = 4,
        shuffle: bool = True,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create train, validation, and test DataLoaders.
        
        Args:
            batch_size: Batch size for DataLoaders
            val_split: Fraction of data for validation
            test_split: Fraction of data for test
            num_workers: Number of worker processes for data loading
            shuffle: Whether to shuffle training data
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        if self.dataset is None:
            raise ValueError("Dataset not initialized. Call prepare() first.")
        
        # Calculate split sizes
        n_total = len(self.dataset)
        n_test = int(n_total * test_split)
        n_val = int(n_total * val_split)
        n_train = n_total - n_test - n_val
        
        # Split dataset
        train_dataset, val_dataset, test_dataset = random_split(
            self.dataset,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(self.config.seed),
        )
        
        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        
        logger.info(
            f"DataLoader splits - Train: {n_train}, Val: {n_val}, Test: {n_test}"
        )
        
        return train_loader, val_loader, test_loader
