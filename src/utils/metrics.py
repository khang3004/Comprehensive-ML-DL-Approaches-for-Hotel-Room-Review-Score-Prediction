"""
Metrics calculation utilities.
"""

from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    silhouette_score,
    davies_bouldin_score,
)


class MetricsCalculator:
    """Unified metrics calculator for all task types."""
    
    @staticmethod
    def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics."""
        return {
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mse": float(mean_squared_error(y_true, y_pred)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
        }
    
    @staticmethod
    def classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        average: str = "weighted",
    ) -> Dict[str, float]:
        """Calculate classification metrics."""
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        }
        
        if y_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    metrics["auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
                else:
                    metrics["auc"] = float(
                        roc_auc_score(y_true, y_proba, multi_class="ovr", average=average)
                    )
            except ValueError:
                metrics["auc"] = 0.0
        
        # Confusion matrix
        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
        
        return metrics
    
    @staticmethod
    def clustering_metrics(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate clustering metrics."""
        n_clusters = len(np.unique(labels))
        
        if n_clusters <= 1:
            return {
                "silhouette_score": 0.0,
                "davies_bouldin_score": float("inf"),
                "n_clusters": n_clusters,
            }
        
        return {
            "silhouette_score": float(silhouette_score(X, labels)),
            "davies_bouldin_score": float(davies_bouldin_score(X, labels)),
            "n_clusters": n_clusters,
        }
    
    @staticmethod
    def format_metrics(metrics: Dict[str, Any], decimals: int = 4) -> str:
        """Format metrics dictionary as a readable string."""
        lines = []
        for key, value in metrics.items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.{decimals}f}")
            elif isinstance(value, dict):
                lines.append(f"  {key}:")
                for k, v in value.items():
                    if isinstance(v, float):
                        lines.append(f"    {k}: {v:.{decimals}f}")
                    else:
                        lines.append(f"    {k}: {v}")
            else:
                lines.append(f"  {key}: {value}")
        return "\n".join(lines)
