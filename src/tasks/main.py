"""
Main entry point for running ML experiments.

Usage:
    python -m src.tasks.main --task regression --model mlp --epochs 100
    python -m src.tasks.main --task classification --model mlp --num_classes 3
    python -m src.tasks.main --task clustering --algorithm kmeans --n_clusters 5
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from ..core.base import BaseConfig, set_seed
from ..data.datamodule import HotelDataModule
from ..models.regression import RegressionModel, RegressionTrainer
from ..models.classification import ClassificationModel, ClassificationTrainer
from ..models.clustering import ClusteringModel
from ..utils.logging_config import setup_logging
from ..utils.metrics import MetricsCalculator
from ..utils.helpers import save_results


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Hotel Analytics ML System")
    
    # Task configuration
    parser.add_argument("--task", type=str, required=True,
                        choices=["regression", "classification", "clustering"],
                        help="Task type")
    
    # Data configuration
    parser.add_argument("--data_path", type=str, default="data/booking_images.csv",
                        help="Path to dataset")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    
    # Model configuration
    parser.add_argument("--model", type=str, default="mlp",
                        choices=["mlp", "kmeans", "dbscan", "hierarchical"],
                        help="Model architecture")
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[128, 64, 32],
                        help="Hidden layer dimensions for MLP")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout rate")
    
    # Training configuration
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--early_stopping", type=int, default=10,
                        help="Early stopping patience")
    
    # Clustering specific
    parser.add_argument("--algorithm", type=str, default="kmeans",
                        choices=["kmeans", "dbscan", "hierarchical"],
                        help="Clustering algorithm")
    parser.add_argument("--n_clusters", type=int, default=3,
                        help="Number of clusters")
    
    # Classification specific
    parser.add_argument("--num_classes", type=int, default=3,
                        help="Number of classes for classification")
    
    # General
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU ID to use")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Output directory")
    parser.add_argument("--log_file", type=str, default=None,
                        help="Log file path")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    
    return parser.parse_args()


def run_regression(args: argparse.Namespace, config: BaseConfig) -> Dict[str, Any]:
    """Run regression task."""
    logger = logging.getLogger(__name__)
    logger.info("Starting regression task...")
    
    # Prepare data
    datamodule = HotelDataModule(config)
    datamodule.prepare(args.data_path)
    datamodule.preprocess(datamodule.raw_data, target="review_score")
    datamodule.create_dataset(datamodule.processed_data)
    train_loader, val_loader, test_loader = datamodule.get_dataloaders(
        batch_size=args.batch_size
    )
    
    # Create model
    input_dim = datamodule.processed_data["features"].shape[1]
    model = RegressionModel(
        config=config,
        input_dim=input_dim,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
    )
    
    # Create trainer
    trainer = RegressionTrainer(
        model=model,
        config=config,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Train
    save_path = Path(args.output_dir) / "regression" / "best_model.pt"
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping,
        save_best=True,
        save_path=save_path,
    )
    
    # Evaluate on test set
    test_metrics = trainer.validate(test_loader)
    logger.info(f"Test metrics:\n{MetricsCalculator.format_metrics(test_metrics)}")
    
    return {
        "task": "regression",
        "test_metrics": test_metrics,
        "history": history,
    }


def run_classification(args: argparse.Namespace, config: BaseConfig) -> Dict[str, Any]:
    """Run classification task."""
    logger = logging.getLogger(__name__)
    logger.info("Starting classification task...")
    
    # Prepare data
    datamodule = HotelDataModule(config)
    datamodule.prepare(args.data_path)
    datamodule.preprocess(datamodule.raw_data, target="star")  # Using star as target
    datamodule.create_dataset(datamodule.processed_data)
    train_loader, val_loader, test_loader = datamodule.get_dataloaders(
        batch_size=args.batch_size
    )
    
    # Create model
    input_dim = datamodule.processed_data["features"].shape[1]
    model = ClassificationModel(
        config=config,
        input_dim=input_dim,
        num_classes=args.num_classes,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
    )
    
    # Create trainer
    trainer = ClassificationTrainer(
        model=model,
        config=config,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Train
    save_path = Path(args.output_dir) / "classification" / "best_model.pt"
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping,
        save_best=True,
        save_path=save_path,
    )
    
    # Evaluate on test set
    test_metrics = trainer.validate(test_loader)
    logger.info(f"Test metrics:\n{MetricsCalculator.format_metrics(test_metrics)}")
    
    return {
        "task": "classification",
        "test_metrics": test_metrics,
        "history": history,
    }


def run_clustering(args: argparse.Namespace, config: BaseConfig) -> Dict[str, Any]:
    """Run clustering task."""
    logger = logging.getLogger(__name__)
    logger.info("Starting clustering task...")
    
    # Prepare data
    datamodule = HotelDataModule(config)
    datamodule.prepare(args.data_path)
    datamodule.preprocess(datamodule.raw_data)
    datamodule.create_dataset(datamodule.processed_data)
    
    # Get features
    X = datamodule.processed_data["features"]
    
    # Create and fit model
    model = ClusteringModel(
        config=config,
        algorithm=args.algorithm,
        n_clusters=args.n_clusters,
    )
    model.fit(X)
    
    # Evaluate
    metrics = model.evaluate(X)
    logger.info(f"Clustering metrics:\n{MetricsCalculator.format_metrics(metrics)}")
    
    return {
        "task": "clustering",
        "metrics": metrics,
    }


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    """Run experiment based on task type."""
    # Setup
    setup_logging(log_file=args.log_file)
    set_seed(args.seed)
    
    # Create config
    config = BaseConfig(
        seed=args.seed,
        device=f"cuda:{args.gpu}" if args.gpu is not None else "cuda" if torch.cuda.is_available() else "cpu",
        paths={"output": args.output_dir},
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Running on device: {config.device}")
    
    # Run task
    if args.task == "regression":
        results = run_regression(args, config)
    elif args.task == "classification":
        results = run_classification(args, config)
    elif args.task == "clustering":
        results = run_clustering(args, config)
    else:
        raise ValueError(f"Unknown task: {args.task}")
    
    # Save results
    output_path = Path(args.output_dir) / f"{args.task}_results.json"
    save_results(results, output_path)
    logger.info(f"Results saved to {output_path}")
    
    return results


def main():
    """Main entry point."""
    args = parse_args()
    results = run_experiment(args)
    print("\n" + "="*60)
    print("Experiment completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
