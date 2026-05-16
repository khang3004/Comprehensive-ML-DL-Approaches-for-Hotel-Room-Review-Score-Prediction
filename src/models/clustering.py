"""
Clustering models for hotel market segmentation.
"""

from typing import Any, Dict, List, Optional
import logging

import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from ..core.base import BaseConfig

logger = logging.getLogger(__name__)


class ClusteringModel:
    """
    Clustering model for unsupervised learning tasks.
    
    Supports:
    - KMeans
    - DBSCAN
    - Hierarchical Clustering
    """
    
    def __init__(
        self,
        config: BaseConfig,
        algorithm: str = "kmeans",
        n_clusters: int = 3,
        **kwargs,
    ):
        self.config = config
        self.algorithm = algorithm
        self.n_clusters = n_clusters
        self.model = None
        self.labels_ = None
        
        # Initialize model based on algorithm
        if algorithm == "kmeans":
            self.model = KMeans(
                n_clusters=n_clusters,
                random_state=config.seed,
                n_init=10,
                max_iter=300,
                **kwargs,
            )
        elif algorithm == "dbscan":
            self.model = DBSCAN(
                eps=kwargs.get("eps", 0.5),
                min_samples=kwargs.get("min_samples", 5),
                **kwargs,
            )
        elif algorithm == "hierarchical":
            self.model = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=kwargs.get("linkage", "ward"),
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown clustering algorithm: {algorithm}")
        
        logger.info(f"ClusteringModel initialized with {algorithm} algorithm")
    
    def fit(self, X: np.ndarray) -> "ClusteringModel":
        """Fit the clustering model."""
        logger.info(f"Fitting {self.algorithm} clustering on {X.shape[0]} samples")
        self.labels_ = self.model.fit_predict(X)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if self.algorithm == "dbscan":
            # DBSCAN doesn't have a predict method
            raise NotImplementedError("DBSCAN doesn't support predict()")
        
        return self.model.fit_predict(X) if self.labels_ is None else self.model.predict(X)
    
    def evaluate(self, X: np.ndarray, labels: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Evaluate clustering quality."""
        if labels is None:
            labels = self.labels_
        
        if labels is None:
            raise ValueError("No labels available for evaluation")
        
        metrics = {}
        
        # Silhouette Score (works for all algorithms)
        if len(np.unique(labels)) > 1:
            metrics["silhouette_score"] = silhouette_score(X, labels)
        else:
            metrics["silhouette_score"] = 0.0
        
        # Davies-Bouldin Index (only for non-DBSCAN)
        if self.algorithm != "dbscan" and len(np.unique(labels)) > 1:
            metrics["davies_bouldin_score"] = davies_bouldin_score(X, labels)
        
        # Calinski-Harabasz Score (only for non-DBSCAN)
        if self.algorithm != "dbscan" and len(np.unique(labels)) > 1:
            metrics["calinski_harabasz_score"] = calinski_harabasz_score(X, labels)
        
        # Number of clusters
        metrics["n_clusters"] = len(np.unique(labels))
        
        # Cluster distribution
        unique, counts = np.unique(labels, return_counts=True)
        metrics["cluster_distribution"] = dict(zip(unique.astype(int), counts.astype(int)))
        
        logger.info(f"Clustering evaluation: {metrics}")
        return metrics
    
    def save(self, path: str) -> None:
        """Save the clustering model."""
        import joblib
        joblib.dump(self.model, path)
        logger.info(f"Clustering model saved to {path}")
    
    def load(self, path: str) -> "ClusteringModel":
        """Load a pre-trained clustering model."""
        import joblib
        self.model = joblib.load(path)
        logger.info(f"Clustering model loaded from {path}")
        return self
