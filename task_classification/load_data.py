import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def convert_review_score_to_class(score):
    """Chuyển đổi review_score thành biến phân loại"""
    score = float(score)  # Đảm bảo score là float
    if score < 4:
        return 0  # phòng kém
    elif score < 7:
        return 1  # phòng bth
    elif score < 9:
        return 2  # phòng tốt
    else:
        return 3  # phòng vip

class Supervised_ML_CustomDataLoader:
    def __init__(self, args):
        self.args = args
        
        # Đọc dữ liệu
        print("Loading data...")
        self.data = pd.read_csv(os.path.join(args.task_dir, f'{args.dataset}.csv'))
        
        # Chuẩn bị features
        print("Preparing features...")
        
        # Tách numeric và categorical features
        num_features = self.data[args.num_features].values
        cat_features = self.data[args.cat_features].values
        
        # Xử lý missing values cho numeric features
        print("Handling missing values in numeric features...")
        num_imputer = SimpleImputer(strategy='mean')
        num_features = num_imputer.fit_transform(num_features)
        
        # Xử lý missing values cho categorical features
        print("Handling missing values in categorical features...")
        cat_imputer = SimpleImputer(strategy='most_frequent')
        cat_features = cat_imputer.fit_transform(cat_features)
        
        # Kết hợp features
        self.features = np.concatenate([num_features, cat_features], axis=1)
        
        # Chuẩn hóa features
        print("Normalizing features...")
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)
        
        # Xử lý target variable
        print("Processing target variable...")
        if args.task_type == 'classification':
            review_scores = self.data['review_score'].fillna(self.data['review_score'].mean())
            self.targets = np.array([convert_review_score_to_class(score) 
                                   for score in review_scores], dtype=int)
            print(f"Unique classes in target: {np.unique(self.targets)}")
            print(f"Class distribution: {np.bincount(self.targets)}")
        else:
            self.targets = self.data['review_score'].fillna(self.data['review_score'].mean()).values
        
        # Kiểm tra NaN
        print("\nChecking for NaN values after preprocessing:")
        print(f"Features contains NaN: {np.isnan(self.features).any()}")
        print(f"Targets contains NaN: {np.isnan(self.targets).any()}")

    def get_train_val_test_split(self):
        """Chia dữ liệu thành train/val/test sets"""
        total_samples = len(self.features)
        indices = np.random.permutation(total_samples)
        
        train_size = int(0.7 * total_samples)
        val_size = int(0.15 * total_samples)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        X_train = self.features[train_indices]
        y_train = self.targets[train_indices]
        
        X_val = self.features[val_indices]
        y_val = self.targets[val_indices]
        
        X_test = self.features[test_indices]
        y_test = self.targets[test_indices]
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)