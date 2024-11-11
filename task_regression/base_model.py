import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score, 
                             precision_score, recall_score, roc_auc_score, 
                             average_precision_score, silhouette_score, r2_score, mean_squared_error)
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from load_data import *
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.cluster import KMeans
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import KFold 
import pandas as pd
from models import *

class MLModelWrapper:
    def __init__(self, args):
        self.args = args
        self.task_type = args.task_type
        self.device = args.device
        
        # Khởi tạo model dựa trên args.model
        if args.model == 'Vanilla_LinearRegression':
            self.model = LinearRegression()
            self.need_feature_selection = True
        elif args.model == 'Ridge_Regression':
            self.model = Ridge(alpha=args.alpha)
            self.need_feature_selection = False  # Ridge không cần feature selection
        elif args.model == 'Elastic_Regression':
            self.model = ElasticNet(alpha=args.alpha, l1_ratio=args.l1_ratio)
            self.need_feature_selection = False  # ElasticNet tự xử lý feature selection
        else:
            raise ValueError(f"Unsupported model: {args.model}")

        self.is_fitted = False
        self.selected_features = None
        self.vif_threshold = args.vif_threshold
        self.max_features = args.max_features

    def train_with_kfold(self, train_loader, n_splits=5):
        # Lấy toàn bộ dữ liệu
        X_train = []
        y_train = []
        for batch in train_loader:
            inputs, targets = batch
            if isinstance(inputs, tuple):
                inputs = inputs[1]
            X_train.append(inputs.numpy())
            y_train.append(targets.numpy())
        
        X_train = np.concatenate(X_train)
        y_train = np.concatenate(y_train).ravel()
        
        # Chọn features
        if self.need_feature_selection:
            X_train_selected = self.feature_selection(X_train)
            print("\nFeatures được chọn sau khi xử lý VIF:")
            print(self.selected_features)
        else:
            X_train_selected = X_train
            self.selected_features = self.args.num_features + self.args.cat_features
        
        # K-fold cross validation
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.args.random_state)
        fold_metrics = []
        
        print(f"\nBắt đầu {n_splits}-fold Cross Validation...")
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_selected)):
            X_fold_train = X_train_selected[train_idx]
            X_fold_val = X_train_selected[val_idx]
            y_fold_train = y_train[train_idx]
            y_fold_val = y_train[val_idx]
            
            self.model.fit(X_fold_train, y_fold_train)
            y_pred = self.model.predict(X_fold_val)
            
            rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))
            r2 = r2_score(y_fold_val, y_pred)
            adj_r2 = 1 - (1-r2)*(len(y_fold_val)-1)/(len(y_fold_val)-X_fold_val.shape[1]-1)
            
            fold_metrics.append({
                'rmse': rmse,
                'r2': r2,
                'adj_r2': adj_r2
            })
            print(f"\nFold {fold+1}:")
            print(f"RMSE: {rmse:.4f}")
            print(f"R²: {r2:.4f}")
            print(f"Adjusted R²: {adj_r2:.4f}")
        
        # Tính trung bình metrics
        avg_metrics = {
            'rmse': np.mean([m['rmse'] for m in fold_metrics]),
            'r2': np.mean([m['r2'] for m in fold_metrics]),
            'adj_r2': np.mean([m['adj_r2'] for m in fold_metrics])
        }
        
        print("\nKết quả trung bình qua {}-fold Cross Validation:".format(n_splits))
        print("-" * 50)
        print(f"RMSE: {avg_metrics['rmse']:.4f}")
        print(f"R²: {avg_metrics['r2']:.4f}")
        print(f"Adjusted R²: {avg_metrics['adj_r2']:.4f}")
        
        # Train final model trên toàn bộ dữ liệu
        self.model.fit(X_train_selected, y_train)
        self.is_fitted = True  # Set is_fitted = True sau khi train xong
        
        # Print model info
        self._print_model_info()
        
        return fold_metrics

    def _print_model_info(self):
        print("\nThông tin Model:")
        print("-"*50)
        if isinstance(self.model, LinearRegression):
            print("Model: Linear Regression (với xử lý VIF)")
        elif isinstance(self.model, Ridge):
            print("Model: Ridge Regression")
            print(f"Alpha (regularization strength): {self.model.alpha}")
        elif isinstance(self.model, ElasticNet):
            print("Model: Elastic Net Regression")
            print(f"Alpha: {self.model.alpha}")
            print(f"L1 ratio: {self.model.l1_ratio}")
        
        print("\nHệ số hồi quy:")
        for feature, coef in zip(self.selected_features, self.model.coef_):
            print(f"{feature}: {coef:.4f}")
        print(f"Intercept: {self.model.intercept_:.4f}")

    def evaluate(self, test_loader):
        """Đánh giá model trên tập test"""
        if not self.is_fitted:
            raise ValueError("Model needs to be trained before evaluation")
            
        # Lấy toàn bộ dữ liệu test
        X_test = []
        y_test = []
        for batch in test_loader:
            inputs, targets = batch
            X_test.append(inputs.numpy())
            y_test.append(targets.numpy())
        
        X_test = np.concatenate(X_test)
        y_test = np.concatenate(y_test)
        
        # Chọn features tương ứng nếu cần
        if self.need_feature_selection:
            feature_names = self.args.num_features + self.args.cat_features
            X_test_df = pd.DataFrame(X_test, columns=feature_names)
            X_test = X_test_df[self.selected_features].values
        
        # Dự đoán và tính metrics
        y_pred = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        adj_r2 = 1 - (1-r2)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
        
        return {
            'rmse': rmse,
            'r2': r2,
            'adj_r2': adj_r2
        }

    def _get_features_labels(self, data_loader):
        """
        Trích xuất features và labels từ DataLoader
        """
        features, labels = [], []
        for (_, feat), lab in data_loader:
            features.append(feat.numpy())
            labels.append(lab.numpy())
        return np.vstack(features), np.concatenate(labels)

    def _calculate_metrics(self, y_true, y_pred):
        """
        Tính toán các metrics dựa trên task_type
        """
        metrics = {}
        if self.task_type == 'classification':
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='macro')
            metrics['recall'] = recall_score(y_true, y_pred, average='macro')
            metrics['f1'] = f1_score(y_true, y_pred, average='macro')
        elif self.task_type == 'clustering':
            metrics['silhouette'] = silhouette_score(y_true, y_pred)
        
        return metrics

    def calculate_vif(self, X):
        """Tính toán VIF cho các features"""
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        return vif_data.sort_values('VIF', ascending=False)

    def feature_selection(self, X):
        """Chọn features dựa trên VIF"""
        feature_names = self.args.num_features + self.args.cat_features
        X_df = pd.DataFrame(X, columns=feature_names)
        
        print("\nKết quả phân tích VIF ban đầu:")
        print("-" * 50)
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X_df.columns
        vif_data["VIF"] = [variance_inflation_factor(X_df.values, i) 
                        for i in range(X_df.shape[1])]
        print(vif_data.sort_values('VIF', ascending=False))
        
        selected_features = []
        max_vif = float('inf')
        
        while max_vif > self.vif_threshold and len(X_df.columns) > 0:
            vif_data = pd.DataFrame()
            vif_data["Feature"] = X_df.columns
            vif_data["VIF"] = [variance_inflation_factor(X_df.values, i) 
                            for i in range(X_df.shape[1])]
            
            if max(vif_data["VIF"]) <= self.vif_threshold:
                break
                
            worst_feature = vif_data.loc[vif_data["VIF"].idxmax(), "Feature"]
            print(f"\nLoại bỏ biến {worst_feature} với VIF = {max(vif_data['VIF']):.2f}")
            X_df = X_df.drop(worst_feature, axis=1)
            max_vif = max(vif_data["VIF"])
        
        print("\nKết quả phân tích VIF sau khi loại bỏ đa cộng tuyến:")
        print("-" * 50)
        final_vif = pd.DataFrame({
            "Feature": X_df.columns,
            "VIF": [variance_inflation_factor(X_df.values, i) 
                    for i in range(X_df.shape[1])]
        })
        print(final_vif.sort_values('VIF', ascending=False))
        
        self.selected_features = X_df.columns.tolist()
        return X_df.values

    def load_model(self, path):
        """
        Load model từ file
        """
        if os.path.exists(path):
            self.model = joblib.load(path)
            print(f"Model loaded from {path}")
        else:
            raise FileNotFoundError(f"Model path {path} not found")

    def save_model(self, path=None):
        """
        Lưu model
        """
        path = path or self.args.model_path
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")
#_____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, average_precision_score, silhouette_score)
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.cluster import KMeans

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, 
                             roc_auc_score, average_precision_score, mean_squared_error)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt



class DLModelWrapper:
    def __init__(self, args):
        """
        Khởi tạo DL model wrapper
        """
        self.args = args
        self.task_type = args.task_type
        self.device = args.device
        
        # Khởi tạo model
        self.model = self._initialize_model()
        self.model.to(self.device)
        
        # Khởi tạo optimizer và scheduler
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()
        
        # Chỉ load model nếu args.load_model là True
        if args.load_model and args.model_path:
            self.load_model(args.model_path)

    def _initialize_model(self):
        """
        Khởi tạo model architecture
        """
        return CombinedModel(
            num_features=len(self.args.num_features) + len(self.args.cat_features),
            num_classes=self.args.num_classes,
            task_type=self.args.task_type
        )

    def _get_optimizer(self):
        """
        Khởi tạo optimizer
        """
        if self.args.optimizer == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay
            )
        return optim.SGD(
            self.model.parameters(),
            lr=self.args.lr,
            momentum=0.9,
            weight_decay=self.args.weight_decay
        )

    def _get_scheduler(self):
        """
        Khởi tạo learning rate scheduler
        """
        if not self.args.use_scheduler:
            return None
        return optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max' if self.task_type == 'classification' else 'min',
            patience=5
        )

    def train(self, train_loader, criterion, args):
        """
        Huấn luyện mô hình với thanh tiến trình và trả về metrics
        """
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        total_loss = 0
        predictions = []
        true_labels = []
        
        # Tqdm cho epochs
        epoch_pbar = tqdm(range(args.n_epoch), desc='Training Epochs')
        for epoch in epoch_pbar:
            running_loss = 0.0
            # Tqdm cho batches
            batch_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.n_epoch}', 
                             leave=False)
            
            for batch_idx, (data, target) in enumerate(batch_pbar):
                images, features = data
                images = images.to(self.device)
                features = features.to(self.device)
                target = target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model((images, features))
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                # Thu thập predictions và labels cho metrics
                predictions.extend(output.detach().cpu().numpy())
                true_labels.extend(target.cpu().numpy())
                
                # Cập nhật loss cho thanh tiến trình
                running_loss += loss.item()
                batch_pbar.set_postfix({'loss': f'{running_loss/(batch_idx+1):.4f}'})
            
            # Cập nhật loss trung bình cho epoch
            avg_loss = running_loss / len(train_loader)
            epoch_pbar.set_postfix({'avg_loss': f'{avg_loss:.4f}'})
            total_loss = avg_loss

        # Tính toán metrics và trả v
        metrics = {
            'loss': total_loss,
            'rmse': np.sqrt(mean_squared_error(true_labels, predictions))
        }
        
        return metrics

    def evaluate(self, test_loader):
        """
        Đánh giá model
        """
        self.model.eval()
        total_loss = 0
        predictions = []
        true_labels = []
        criterion = self._get_criterion()
        
        with torch.no_grad():
            for (images, features), labels in test_loader:
                images = images.to(self.device)
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model((images, features))
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                predictions.extend(outputs.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        metrics = self._calculate_metrics(true_labels, predictions)
        metrics['loss'] = total_loss / len(test_loader)
        return metrics

    def _get_criterion(self):
        """
        Lấy loss function dựa trên task_type
        """
        if self.task_type == 'classification':
            return nn.CrossEntropyLoss()
        return nn.MSELoss()

    def _calculate_metrics(self, y_true, y_pred):
        """
        Tính toán các metrics
        """
        metrics = {}
        if self.task_type == 'classification':
            y_pred = np.argmax(y_pred, axis=1)
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='macro')
            metrics['recall'] = recall_score(y_true, y_pred, average='macro')
            metrics['f1'] = f1_score(y_true, y_pred, average='macro')
        else:
            metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
            
        return metrics

    def save_model(self, path):
        """
        Lưu model vào file
        Args:
            path: Đường dẫn để lưu model
        """
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Lưu model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'task_type': self.task_type,
        }, path)
        print(f"Đã lưu model tại: {path}")

    def load_model(self, path):
        """
        Load model từ file
        Args:
            path: Đường dẫn tới file model
        """
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Đã load model từ: {path}")
        else:
            raise FileNotFoundError(f"Không tìm thấy file model tại: {path}")



