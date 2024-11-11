import os
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, 
    davies_bouldin_score
)
import joblib
import logging
from abc import ABC, abstractmethod

class BaseModel(ABC):
    """Abstract base class cho các model clustering"""
    def __init__(self, args):
        self.args = args
        self.model = None
        self.best_params = None
        self.history = {'metrics': [], 'params': []}
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Tạo thư mục cho model artifacts
        for dir_name in ['models', 'results', 'logs']:
            os.makedirs(dir_name, exist_ok=True)
            
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Thiết lập logging cho model"""
        log_file = os.path.join('logs', f'model_{self.timestamp}.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
    @abstractmethod
    def fit(self, X, y=None):
        """Train model"""
        pass
        
    @abstractmethod
    def predict(self, X):
        """Predict clusters"""
        pass
        
    @abstractmethod
    def evaluate(self, X, labels=None):
        """Evaluate model performance"""
        pass
        
    def save_model(self, path=None):
        """Lưu model và parameters"""
        if path is None:
            path = os.path.join('models', f'model_{self.timestamp}.joblib')
            
        model_data = {
            'model': self.model,
            'best_params': self.best_params,
            'history': self.history,
            'args': self.args
        }
        
        joblib.dump(model_data, path)
        logging.info(f"Saved model to {path}")
        
    def load_model(self, path):
        """Load model từ file"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
            
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.best_params = model_data['best_params']
        self.history = model_data['history']
        logging.info(f"Loaded model from {path}")
        
    def _calculate_metrics(self, X, labels):
        """Tính toán các clustering metrics"""
        if len(np.unique(labels)) < 2:
            logging.warning("Only one cluster found!")
            return None
            
        metrics = {
            'silhouette': silhouette_score(X, labels),
            'calinski_harabasz': calinski_harabasz_score(X, labels),
            'davies_bouldin': davies_bouldin_score(X, labels)
        }
        
        # Log metrics
        for metric_name, score in metrics.items():
            logging.info(f"{metric_name}: {score:.4f}")
            
        return metrics
        
    def _log_training_step(self, metrics, params=None):
        """Log training metrics và parameters"""
        self.history['metrics'].append(metrics)
        if params:
            self.history['params'].append(params)
            
    def plot_training_history(self):
        """Plot lịch sử training"""
        if not self.history['metrics']:
            logging.warning("No training history available")
            return
            
        metrics_df = pd.DataFrame(self.history['metrics'])
        
        plt.figure(figsize=(12, 6))
        for metric in metrics_df.columns:
            plt.plot(metrics_df[metric], label=metric)
            
        plt.title('Training History')
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        
        # Lưu plot
        plot_path = os.path.join('results', f'training_history_{self.timestamp}.png')
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Saved training history plot to {plot_path}")
        
    def cross_validate(self, X, n_splits=5):
        """Thực hiện cross-validation"""
        from sklearn.model_selection import KFold
        
        cv_metrics = []
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            logging.info(f"\nFold {fold}/{n_splits}")
            
            # Train trên training fold
            X_train_fold = X[train_idx]
            self.fit(X_train_fold)
            
            # Evaluate trên validation fold
            X_val_fold = X[val_idx]
            val_labels = self.predict(X_val_fold)
            fold_metrics = self.evaluate(X_val_fold, val_labels)
            
            if fold_metrics:
                cv_metrics.append(fold_metrics)
                
        # Tính mean và std của metrics
        cv_results = {}
        for metric in cv_metrics[0].keys():
            values = [m[metric] for m in cv_metrics]
            cv_results[metric] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }
            
        # Log kết quả
        logging.info("\nCross-validation Results:")
        for metric, stats in cv_results.items():
            logging.info(f"{metric}: {stats['mean']:.4f} (±{stats['std']:.4f})")
            
        return cv_results
        
    def _validate_input(self, X):
        """Validate input data"""
        if not isinstance(X, (np.ndarray, pd.DataFrame)):
            raise TypeError("Input must be numpy array or pandas DataFrame")
            
        if np.isnan(X).any():
            raise ValueError("Input contains NaN values")
            
    def _validate_hyperparameters(self):
        """Validate hyperparameters"""
        raise NotImplementedError
