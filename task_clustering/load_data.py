import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

class BaseDataLoader:
    """Base class cho data loading và preprocessing"""
    def __init__(self, args):
        self.args = args
        self.data = None
        self.features = None
        self.scaler = None
        self.num_imputer = None
        self.cat_imputer = None
        
    def load_data(self):
        """Load dữ liệu từ file"""
        raise NotImplementedError
        
    def preprocess_features(self):
        """Xử lý features"""
        raise NotImplementedError
        
    def get_train_val_test_split(self):
        """Chia tập train/val/test"""
        raise NotImplementedError

class ClusteringDataLoader(BaseDataLoader):
    """Data loader specific cho clustering tasks"""
    def __init__(self, args):
        super().__init__(args)
        self.load_data()
        self.preprocess_features()
        
    def load_data(self):
        """Load và validate dữ liệu"""
        print("Loading data...")
        self.data = pd.read_csv(os.path.join(self.args.task_dir, f'{self.args.dataset}.csv'))
        self.review_scores = self.data['review_score'].values
        
        # Validate features
        missing_num = set(self.args.num_features) - set(self.data.columns)
        missing_cat = set(self.args.cat_features) - set(self.data.columns)
        if missing_num or missing_cat:
            raise ValueError(f"Missing features in dataset: {missing_num.union(missing_cat)}")
            
    def preprocess_features(self):
        """Xử lý và chuẩn hóa features"""
        print("Preprocessing features...")
        
        # Tách và xử lý numeric features
        num_features = self.data[self.args.num_features].values
        self.num_imputer = SimpleImputer(strategy='mean')
        num_features = self.num_imputer.fit_transform(num_features)
        
        # Tách và xử lý categorical features
        cat_features = self.data[self.args.cat_features].values
        self.cat_imputer = SimpleImputer(strategy='most_frequent')
        cat_features = self.cat_imputer.fit_transform(cat_features)
        
        # Kết hợp và chuẩn hóa features
        self.features = np.concatenate([num_features, cat_features], axis=1)
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)
        
        print(f"Final feature matrix shape: {self.features.shape}")
        
    def get_train_val_test_split(self):
        """Chia dữ liệu thành train/val/test sets"""
        np.random.seed(self.args.random_state)
        indices = np.random.permutation(len(self.features))
        
        test_size = int(0.2 * len(indices))
        val_size = int(0.1 * len(indices))
        
        test_idx = indices[:test_size]
        val_idx = indices[test_size:test_size + val_size]
        train_idx = indices[test_size + val_size:]
        
        return {
            'train': (self.features[train_idx], self.review_scores[train_idx]),
            'val': (self.features[val_idx], self.review_scores[val_idx]),
            'test': (self.features[test_idx], self.review_scores[test_idx])
        }
        
    def get_kfold_splits(self, n_splits=5):
        """Generator cho K-fold cross validation"""
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.args.random_state)
        for train_idx, val_idx in kf.split(self.features):
            yield {
                'train': (self.features[train_idx], self.review_scores[train_idx]),
                'val': (self.features[val_idx], self.review_scores[val_idx])
            }
            
    def visualize_data(self):
        """Tạo các visualization cho dữ liệu"""
        self._plot_feature_distributions()
        self._plot_correlation_matrix()
        
    def _plot_feature_distributions(self):
        """Plot phân phối của các features"""
        # Tạo thư mục output nếu chưa tồn tại
        os.makedirs(self.args.output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for idx, feature in enumerate(self.args.num_features[:6]):
            sns.histplot(data=self.data, x=feature, ax=axes[idx])
            axes[idx].set_title(f'Distribution of {feature}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.args.output_dir, 'feature_distributions.png'))
        plt.close()
        
    def _plot_correlation_matrix(self):
        """Plot correlation matrix"""
        # Tạo thư mục output nếu chưa tồn tại
        os.makedirs(self.args.output_dir, exist_ok=True)
        
        corr_matrix = self.data[self.args.num_features].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlations')
        plt.tight_layout()
        plt.savefig(os.path.join(self.args.output_dir, 'feature_correlations.png'))
        plt.close()
        
    def visualize_clusters(self, labels, method='tsne'):
        """Visualize clusters using dimension reduction"""
        # Đảm bảo labels có cùng kích thước với features
        if len(labels) != len(self.features):
            raise ValueError(f"Number of labels ({len(labels)}) must match number of samples ({len(self.features)})")
        
        # Chỉ reduce dimensions cho các điểm có labels
        reduced_features = self._reduce_dimensions(method)
        
        plt.figure(figsize=(10, 8))
        
        # Tạo scatter plot với colormap
        scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1],
                             c=labels, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='Cluster')
        
        # Thêm review scores dưới dạng annotations
        for i, txt in enumerate(self.review_scores):
            plt.annotate(f'{txt:.1f}', 
                        (reduced_features[i, 0], reduced_features[i, 1]),
                        fontsize=8, alpha=0.5)
        
        plt.title(f'Cluster Visualization using {method.upper()}')
        plt.xlabel('First Component')
        plt.ylabel('Second Component')
        
        # Tạo thư mục output nếu chưa tồn tại
        os.makedirs(self.args.output_dir, exist_ok=True)
        plt.savefig(os.path.join(self.args.output_dir, f'clusters_{method}.png'))
        plt.close()
        
    def _reduce_dimensions(self, method='tsne'):
        """Reduce dimensions for visualization"""
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=self.args.random_state)
        elif method == 'pca':
            reducer = PCA(n_components=2, random_state=self.args.random_state)
        else:
            raise ValueError(f"Unknown dimension reduction method: {method}")
            
        return reducer.fit_transform(self.features)

    def get_features(self):
        """Return processed features"""
        return self.features