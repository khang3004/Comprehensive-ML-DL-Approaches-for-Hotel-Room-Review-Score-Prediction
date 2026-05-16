"""
Booking.com Hotel Analytics - Professional ML/DL System

A production-grade machine learning system for comprehensive hotel analytics.
"""

__version__ = "2.0.0"
__author__ = "Khang et al."
__email__ = "gausseuler159357@gmail.com"

from .core.base import BaseTrainer, BaseModel, BaseDataLoader
from .data.datamodule import HotelDataModule
from .models.regression import RegressionModel
from .models.classification import ClassificationModel
from .models.clustering import ClusteringModel

__all__ = [
    "BaseTrainer",
    "BaseModel",
    "BaseDataLoader",
    "HotelDataModule",
    "RegressionModel",
    "ClassificationModel",
    "ClusteringModel",
]
