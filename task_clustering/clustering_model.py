import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from base_model import BaseModel
import logging
import os
from sklearn.metrics import silhouette_score

class ClusteringModel(BaseModel):
    """Implementation cụ thể cho clustering models"""
    def __init__(self, args):
        super().__init__(args)
        self.algorithm = None
        self.param_search_results = {}
        
    def fit(self, X, algorithm='kmeans'):
        """Fit clustering model với hyperparameter optimization"""
        self._validate_input(X)
        self.algorithm = algorithm
        
        if algorithm == 'kmeans':
            self.best_params = self._find_optimal_kmeans(X)
            self.model = KMeans(**self.best_params, random_state=42)
        elif algorithm == 'dbscan':
            self.best_params = self._find_optimal_dbscan(X)
            self.model = DBSCAN(**self.best_params)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
            
        logging.info(f"Training final {algorithm} model with params: {self.best_params}")
        self.model.fit(X)
        
        # Log training results
        labels = self.model.labels_
        metrics = self._calculate_metrics(X, labels)
        self._log_training_step(metrics, self.best_params)
        
        return labels
        
    def predict(self, X):
        """Predict clusters cho dữ liệu mới"""
        self._validate_input(X)
        if self.model is None:
            raise RuntimeError("Model hasn't been trained yet")
            
        if isinstance(self.model, KMeans):
            return self.model.predict(X)
        else:  # DBSCAN
            return self.model.fit_predict(X)
            
    def evaluate(self, X, labels=None):
        """Evaluate clustering results"""
        self._validate_input(X)
        if labels is None:
            labels = self.predict(X)
            
        return self._calculate_metrics(X, labels)
        
    def _find_optimal_kmeans(self, X, k_range=range(2, 11)):
        """Tìm số clusters tối ưu cho KMeans"""
        logging.info("Finding optimal number of clusters for KMeans...")
        
        results = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X)
            metrics = self._calculate_metrics(X, labels)
            
            results.append({
                'k': k,
                'inertia': kmeans.inertia_,
                **metrics
            })
            
        # Convert to DataFrame for easier analysis
        results_df = pd.DataFrame(results)
        self.param_search_results['kmeans'] = results_df
        
        # Plot parameter search results
        self._plot_kmeans_optimization(results_df)
        
        # Find optimal k using silhouette score
        best_k = results_df.loc[results_df['silhouette'].idxmax(), 'k']
        logging.info(f"Optimal number of clusters: {best_k}")
        
        return {'n_clusters': int(best_k)}
        
    def _find_optimal_dbscan(self, X):
        """Tìm parameters tối ưu cho DBSCAN"""
        logging.info("Finding optimal parameters for DBSCAN...")
        
        # Define parameter grid
        param_grid = ParameterGrid({
            'eps': np.arange(0.1, 2.1, 0.1),
            'min_samples': range(3, 11)
        })
        
        results = []
        for params in param_grid:
            dbscan = DBSCAN(**params)
            labels = dbscan.fit_predict(X)
            
            # Skip if all points are noise
            if len(np.unique(labels)) < 2:
                continue
                
            metrics = self._calculate_metrics(X, labels)
            results.append({
                **params,
                **metrics,
                'n_clusters': len(np.unique(labels[labels != -1])),
                'noise_ratio': np.sum(labels == -1) / len(labels)
            })
            
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        self.param_search_results['dbscan'] = results_df
        
        # Plot parameter search results
        self._plot_dbscan_optimization(results_df)
        
        # Find optimal parameters
        best_params = results_df.loc[results_df['silhouette'].idxmax()].to_dict()
        best_params = {
            'eps': best_params['eps'],
            'min_samples': int(best_params['min_samples'])
        }
        
        logging.info(f"Optimal DBSCAN parameters: {best_params}")
        return best_params
        
    def _calculate_metrics(self, X, labels):
        """Tính toán các clustering metrics"""
        if len(np.unique(labels)) < 2:
            logging.warning("Only one cluster found!")
            return None
            
        # Chỉ tính inertia cho KMeans
        metrics = {
            'silhouette': silhouette_score(X, labels)
        }
        
        if isinstance(self.model, KMeans):
            metrics['inertia'] = self.model.inertia_
        
        # Log metrics
        for metric_name, score in metrics.items():
            logging.info(f"{metric_name}: {score:.4f}")
            
        return metrics
        
    def _plot_kmeans_optimization(self, results_df):
        """Visualize KMeans parameter optimization"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Elbow curve
        axes[0].plot(results_df['k'], results_df['inertia'], 'bo-')
        axes[0].set_xlabel('Number of Clusters (k)')
        axes[0].set_ylabel('Inertia')
        axes[0].set_title('Elbow Method')
        
        # Silhouette score
        axes[1].plot(results_df['k'], results_df['silhouette'], 'ro-')
        axes[1].set_xlabel('Number of Clusters (k)')
        axes[1].set_ylabel('Silhouette Score')
        axes[1].set_title('Silhouette Analysis')
        
        plt.tight_layout()
        plt.savefig(f'results/kmeans_optimization_{self.timestamp}.png')
        plt.close()
        
    def _plot_dbscan_optimization(self, results_df):
        """Visualize DBSCAN parameter optimization"""
        # Chỉ plot silhouette score heatmap
        plt.figure(figsize=(10, 8))
        pivot_silhouette = results_df.pivot(
            index='eps', 
            columns='min_samples', 
            values='silhouette'
        )
        sns.heatmap(pivot_silhouette, cmap='viridis')
        plt.title('Silhouette Score by Parameters')
        plt.xlabel('Min Samples')
        plt.ylabel('Epsilon')
        
        plt.tight_layout()
        plt.savefig(f'results/dbscan_optimization_{self.timestamp}.png')
        plt.close()
        
    def _log_training_step(self, metrics, params=None):
        """Log training metrics và parameters"""
        # Log best results
        if self.algorithm == 'kmeans':
            logging.info("\nBest KMeans Results:")
            logging.info(f"Number of clusters: {params['n_clusters']}")
            logging.info(f"Inertia: {metrics.get('inertia', 'N/A'):.4f}")
            logging.info(f"Silhouette score: {metrics['silhouette']:.4f}")
        else:  # DBSCAN
            logging.info("\nBest DBSCAN Results:")
            logging.info(f"Epsilon: {params['eps']:.2f}")
            logging.info(f"Min samples: {params['min_samples']}")
            logging.info(f"Silhouette score: {metrics['silhouette']:.4f}")
        
    def _validate_hyperparameters(self):
        """Validate model hyperparameters"""
        if self.algorithm == 'kmeans':
            if not isinstance(self.best_params.get('n_clusters'), int):
                raise ValueError("n_clusters must be an integer")
            if self.best_params['n_clusters'] < 2:
                raise ValueError("n_clusters must be >= 2")
                
        elif self.algorithm == 'dbscan':
            if not isinstance(self.best_params.get('eps'), (int, float)):
                raise ValueError("eps must be numeric")
            if not isinstance(self.best_params.get('min_samples'), int):
                raise ValueError("min_samples must be an integer")
            if self.best_params['eps'] <= 0:
                raise ValueError("eps must be > 0")
            if self.best_params['min_samples'] < 1:
                raise ValueError("min_samples must be >= 1")