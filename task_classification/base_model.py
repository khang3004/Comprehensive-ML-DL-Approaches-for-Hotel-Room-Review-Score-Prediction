import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc,
    mean_squared_error, silhouette_score, confusion_matrix,
    average_precision_score
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib
from itertools import cycle

class MLModelWrapper:
    def __init__(self, args):
        self.args = args
        self.task_type = args.task_type
        self.device = args.device
        
        
        # Khởi tạo model dựa trên args.model
        if args.model == 'LogisticRegression':
            self.model = LogisticRegression(
                solver='lbfgs',
                max_iter=1000,
                random_state=args.random_state,
                class_weight='balanced'
            )
            self.is_ensemble = False
        elif args.model == 'Ensemble':
            self.model = None
            self.base_models = {
                'svm': SVC(
                    kernel='rbf',
                    decision_function_shape='ovo',
                    probability=True,
                    random_state=args.random_state,
                    class_weight='balanced'
                ),
                'knn': KNeighborsClassifier(
                    n_neighbors=5,
                    weights='distance',
                    algorithm='auto'
                ),
                'dt': DecisionTreeClassifier(
                    criterion='entropy',
                    random_state=args.random_state,
                    class_weight='balanced'
                ),
                'rf': RandomForestClassifier(
                    n_estimators=100,
                    criterion='entropy',
                    random_state=args.random_state,
                    class_weight='balanced'
                )
            }
            self.meta_model = LogisticRegression(
                solver='lbfgs',
                max_iter=1000,
                random_state=args.random_state,
                class_weight='balanced'
            )
            self.is_ensemble = True
        else:
            raise ValueError(f"Unsupported model: {args.model}")

        self.is_fitted = False

    def _tune_base_model(self, model, X_train, y_train, model_name):
        """Tune hyperparameters cho base models"""
        # Kiểm tra số lượng mẫu trong mỗi class
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        min_samples = np.min(class_counts)
        
        # Điều chỉnh số folds dựa trên số lượng mẫu nhỏ nhất
        n_splits = min(min_samples, 3)  # Không vượt quá 3 folds
        if n_splits < 2:
            print(f"Warning: Không đủ mẫu để cross-validate cho class {unique_classes[np.argmin(class_counts)]}.")
            print(f"Số mẫu nhỏ nhất: {min_samples}. Sẽ sử dụng train-test split thay vì CV.")
            n_splits = 2  # Sử dụng train-test split

        if model_name == 'svm':
            param_grid = {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto', 0.1, 1],
            }
        elif model_name == 'dt':
            param_grid = {
                'max_depth': [3, 5, 7, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif model_name == 'rf':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        else:
            return model  # Trả về nguyên model nếu không cần tune

        # Sử dụng RandomizedSearchCV với số folds đã điều chỉnh
        random_search = RandomizedSearchCV(
            model,
            param_distributions=param_grid,
            n_iter=5,
            cv=n_splits,  # Sử dụng số folds đã điều chỉnh
            n_jobs=-1,
            random_state=self.args.random_state,
            scoring='accuracy'
        )
        
        print(f"\nTuning hyperparameters for {model_name}...")
        print(f"Số lượng mẫu trong mỗi class: {dict(zip(unique_classes, class_counts))}")
        print(f"Sử dụng {n_splits}-fold cross-validation")
        
        random_search.fit(X_train, y_train)
        print(f"Best parameters for {model_name}: {random_search.best_params_}")
        
        return random_search.best_estimator_

    def _get_meta_features(self, X, models):
        """Tạo meta-features từ predictions của base models"""
        meta_features = []
        for model in models.values():
            meta_features.append(model.predict_proba(X))
        return np.hstack(meta_features)

    def _merge_minority_class(self, y):
        """
        Gộp class 0 vào class 1 và cập nhật lại labels
        """
        y_merged = y.copy()
        
        # Gộp class 0 vào class 1
        y_merged[y_merged == 0] = 1
        
        # Kiểm tra các classes có trong dữ liệu
        present_classes = sorted(np.unique(y_merged))
        
        # Lưu thông tin về mapping
        self.original_classes = present_classes
        self.class_mapping = {1: 1, 2: 2, 3: 3}  # Giữ nguyên labels
        
        print("\nThông tin gộp class:")
        print("- Class 0 đã được gộp vào Class 1")
        print("- Giữ nguyên các labels:")
        print("  Class 1: Phòng bình thường (review_score < 7)")
        print("  Class 2: Phòng tốt (7 ≤ review_score < 9)")
        print("  Class 3: Phòng xuất sắc (review_score ≥ 9)")
        
        # In phân phối sau khi gộp
        unique, counts = np.unique(y_merged, return_counts=True)
        print("\nPhân phối sau khi gộp:")
        for u, c in zip(unique, counts):
            print(f"Class {u}: {c} mẫu")
        
        # Kiểm tra tính hợp lệ của dữ liệu sau khi gộp
        assert set(np.unique(y_merged)) == set([1, 2, 3]), \
            "Lỗi: Dữ liệu sau khi gộp phải chỉ chứa các class 1, 2, 3"
        
        return y_merged

    def train_with_kfold(self, X_train, y_train):
        """
        Train model với K-fold Cross Validation
        """
        # Xác định số lượng classes thực tế từ dữ liệu
        unique_classes = np.unique(y_train)
        n_classes = len(unique_classes)
        print(f"Số lượng classes thực tế: {n_classes}")
        print(f"Các classes: {unique_classes}")
        
        # Kiểm tra số lượng mẫu trong mỗi class
        class_counts = np.bincount(y_train)
        print("\nPhân bố classes trong tập training:")
        for i, count in enumerate(class_counts):
            print(f"Class {i}: {count} mẫu")
        
        # Gộp class nếu cần
        if np.min(class_counts) < 5:
            print("\nCảnh báo: Một số classes có quá ít mẫu!")
            print("Tiến hành gộp class 0 vào class gần nhất...")
            y_train = self._merge_minority_class(y_train)
            
            # Cập nhật thông tin classes
            unique_classes = np.unique(y_train)
            n_classes = len(unique_classes)
            class_counts = np.bincount(y_train)
            
            print("\nPhân b classes sau khi gộp:")
            for i, count in enumerate(class_counts):
                if count > 0:  # Chỉ in các classes còn tồn tại
                    print(f"Class {i}: {count} mẫu")
        
            # Lưu thông tin về việc gộp class
            self.merged_classes = True
            self.n_classes = n_classes
        else:
            self.merged_classes = False
        
        # Sử dụng 5-fold cross validation
        n_splits = 5
        print(f"\nSử dụng {n_splits}-fold cross validation")
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Lưu metrics cho mỗi fold
        fold_metrics = []
        
        # Train và evaluate trên từng fold
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
            print(f"\nFold {fold}/{n_splits}:")
            
            # Chia dữ liệu theo fold
            X_train_fold = X_train[train_idx]
            y_train_fold = y_train[train_idx]  # Đã được gộp class nếu cần
            X_val_fold = X_train[val_idx]
            y_val_fold = y_train[val_idx]  # Đã được gộp class nếu cần
            
            if self.is_ensemble:
                # Train base models
                base_predictions = []
                
                # Khởi tạo và train các base models với class_weight='balanced'
                base_model_configs = [
                    ('svm', SVC(
                        kernel='rbf',
                        decision_function_shape='ovo',
                        probability=True,
                        random_state=42,
                        class_weight='balanced'
                    )),
                    ('knn', KNeighborsClassifier(
                        n_neighbors=5,
                        weights='distance'
                    )),
                    ('dt', DecisionTreeClassifier(
                        criterion='entropy',
                        random_state=42,
                        class_weight='balanced'
                    )),
                    ('rf', RandomForestClassifier(
                        n_estimators=100,
                        criterion='entropy',
                        random_state=42,
                        class_weight='balanced'
                    ))
                ]
                
                for name, model in base_model_configs:
                    print(f"Training {name}...")
                    self.base_models[name] = model
                    self.base_models[name].fit(X_train_fold, y_train_fold)
                    pred = self.base_models[name].predict_proba(X_val_fold)
                    base_predictions.append(pred)
                
                # Train meta model
                meta_features_train = np.hstack(base_predictions)
                print(f"Meta features shape: {meta_features_train.shape}")
                
                print("Training meta model...")
                self.meta_model = LogisticRegression(
                    solver='lbfgs',
                    max_iter=1000,
                    random_state=42,
                    class_weight='balanced'
                )
                self.meta_model.fit(meta_features_train, y_val_fold)
                
                # Predict trên validation set
                y_pred = self.meta_model.predict(meta_features_train)
                y_pred_proba = self.meta_model.predict_proba(meta_features_train)
            else:
                # Train Logistic Regression
                self.model = LogisticRegression(
                    solver='lbfgs',
                    max_iter=1000,
                    random_state=42,
                    class_weight='balanced'
                )
                self.model.fit(X_train_fold, y_train_fold)
                y_pred = self.model.predict(X_val_fold)
                y_pred_proba = self.model.predict_proba(X_val_fold)
            
            # In thông tin về phân phối classes trong fold hiện tại
            print("\nPhân bố classes trong fold hiện tại:")
            unique, counts = np.unique(y_train_fold, return_counts=True)
            for u, c in zip(unique, counts):
                print(f"Class {u}: {c} mẫu")
            
            # Tính metrics cho fold hiện tại
            fold_metric = self._calculate_metrics(y_val_fold, y_pred, y_pred_proba)
            fold_metrics.append(fold_metric)
            
            print(f"\nMetrics cho fold {fold}:")
            for metric_name, value in fold_metric.items():
                print(f"{metric_name}: {value:.4f}")
        
        # Train final model trên toàn bộ dữ liệu (đã được gộp class)
        print("\nTraining final model trên toàn bộ dữ liệu...")
        if self.is_ensemble:
            base_predictions = []
            
            # Train final base models
            for name, model_class in [
                ('svm', SVC(probability=True, random_state=42, class_weight='balanced')),
                ('knn', KNeighborsClassifier(n_neighbors=5, weights='distance')),
                ('dt', DecisionTreeClassifier(random_state=42, class_weight='balanced')),
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
            ]:
                print(f"Training final {name}...")
                self.base_models[name] = model_class
                self.base_models[name].fit(X_train, y_train)
                pred = self.base_models[name].predict_proba(X_train)
                base_predictions.append(pred)
            
            # Train final meta model
            meta_features = np.hstack(base_predictions)
            print("Training final meta model...")
            self.meta_model = LogisticRegression(random_state=42, class_weight='balanced')
            self.meta_model.fit(meta_features, y_train)
            self.is_fitted = True
        else:
            self.model = LogisticRegression(random_state=42, class_weight='balanced')
            self.model.fit(X_train, y_train)
            self.is_fitted = True
            
            # In và lưu thông tin về model parameters
        self._log_model_parameters(X_train)
        
        # Tính trung bình và độ lệch chuẩn của metrics
        avg_metrics = {}
        std_metrics = {}
        for metric in fold_metrics[0].keys():
            values = [fold[metric] for fold in fold_metrics]
            avg_metrics[metric] = np.mean(values)
            std_metrics[metric] = np.std(values)
        
        return {
            'avg_metrics': avg_metrics,
            'std_metrics': std_metrics,
            'fold_metrics': fold_metrics
        }

    def evaluate(self, X_test, y_test):
        """
        Đánh giá model trên tập test
        """
        try:
            # Gộp class trong tập test nếu đã gộp trong tập train
            if hasattr(self, 'merged_classes') and self.merged_classes:
                y_test = self._merge_minority_class(y_test)
                print("\nPhân bố classes trong tập test sau khi gộp:")
                unique, counts = np.unique(y_test, return_counts=True)
                for u, c in zip(unique, counts):
                    print(f"Class {u}: {c} mẫu")
            
            if self.is_ensemble:
                # Lấy predictions từ base models
                base_predictions = []
                
                print("Đang predict với các base models...")
                for name, model in self.base_models.items():
                    pred = model.predict_proba(X_test)
                    base_predictions.append(pred)
                
                # Tạo meta features
                meta_features = np.hstack(base_predictions)
                print(f"Meta features shape cho test set: {meta_features.shape}")
                
                # Predict với meta model
                print("Đang predict với meta model...")
                y_pred = self.meta_model.predict(meta_features)
                y_pred_proba = self.meta_model.predict_proba(meta_features)
            else:
                print("Đang predict với Logistic Regression...")
                y_pred = self.model.predict(X_test)
                y_pred_proba = self.model.predict_proba(X_test)
            
            # In thông tin về phân phối classes
            print("\nPhân phối classes trong tập test:")
            unique, counts = np.unique(y_test, return_counts=True)
            for u, c in zip(unique, counts):
                print(f"Class {u}: {c} mẫu")
            
            print("\nPhân phối classes được dự đoán:")
            unique, counts = np.unique(y_pred, return_counts=True)
            for u, c in zip(unique, counts):
                print(f"Class {u}: {c} mẫu")
            
            # Tính và trả về metrics
            metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
            print("\nKết quả đánh giá trên tập test:")
            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value:.4f}")
            
            return metrics
            
        except Exception as e:
            print(f"Lỗi khi evaluate model: {str(e)}")
            raise

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
        
        print("\nH số hồi quy:")
        for feature, coef in zip(self.selected_features, self.model.coef_):
            print(f"{feature}: {coef:.4f}")
        print(f"Intercept: {self.model.intercept_:.4f}")

    def _get_features_labels(self, data_loader):
        """
        Trích xuất features và labels từ DataLoader
        """
        features, labels = [], []
        for (_, feat), lab in data_loader:
            features.append(feat.numpy())
            labels.append(lab.numpy())
        return np.vstack(features), np.concatenate(labels)

    

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
        
        print("\nKết quả phân tch VIF sau khi loại bỏ đa cộng tuyến:")
        print("-" * 50)
        final_vif = pd.DataFrame({
            "Feature": X_df.columns,
            "VIF": [variance_inflation_factor(X_df.values, i) 
                    for i in range(X_df.shape[1])]
        })
        print(final_vif.sort_values('VIF', ascending=False))
        
        self.selected_features = X_df.columns.tolist()
        return X_df.values

    def save_model(self):
        """Lưu model vào file"""
        if not self.is_fitted:
            print("Model chưa được train!")
            return
            
        os.makedirs('models', exist_ok=True)
        model_path = os.path.join('models', f'{self.args.model}.joblib')
        joblib.dump(self.model, model_path)
        print(f"Đã lưu model tại: {model_path}")

    def load_model(self):
        """Load model từ file"""
        model_path = os.path.join('models', f'{self.args.model}.joblib')
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            self.is_fitted = True
            print(f"Đã load model từ: {model_path}")
        else:
            print(f"Không tìm thấy model tại: {model_path}")

    def _plot_roc_curves(self, y_true, y_pred_proba):
        """Vẽ đường cong ROC cho từng class"""
        plt.figure(figsize=(10, 8))
        n_classes = y_pred_proba.shape[1]
        
        # Chuyển y_true thành one-hot encoding
        y_true_bin = np.eye(n_classes)[y_true.astype(int)]
        
        # Tính và vẽ ROC curve cho từng class
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green'])
        for i, color in zip(range(n_classes), colors):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc = roc_auc_score(y_true_bin[:, i], y_pred_proba[:, i])
            plt.plot(fpr, tpr, color=color, lw=2,
                    label=f'ROC curve of class {i} (AUC = {roc_auc:0.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig('roc_curves.png')
        plt.close()

    def _plot_pr_curves(self, y_true, y_pred_proba):
        """Vẽ đường cong Precision-Recall cho tng class"""
        plt.figure(figsize=(10, 8))
        n_classes = y_pred_proba.shape[1]
        
        # Chuyển y_true thành one-hot encoding
        y_true_bin = np.eye(n_classes)[y_true.astype(int)]
        
        # Tính và vẽ PR curve cho từng class
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green'])
        for i, color in zip(range(n_classes), colors):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_proba[:, i])
            avg_precision = average_precision_score(y_true_bin[:, i], y_pred_proba[:, i])
            plt.plot(recall, precision, color=color, lw=2,
                    label=f'PR curve of class {i} (AP = {avg_precision:0.2f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.savefig('pr_curves.png')
        plt.close()

    def _print_model_stats(self, X, y):
        """In thống kê chi tiết của model"""
        print("\nLogistic Regression Model Statistics:")
        print("=" * 50)
        
        # Tính null deviance (với null model)
        n_classes = len(np.unique(y))
        null_probs = np.bincount(y) / len(y)
        null_predictions = np.tile(null_probs, (len(y), 1))
        null_deviance = -2 * np.sum([np.log(null_predictions[i, y[i]]) 
                                    for i in range(len(y))])
        
        # Tính model deviance
        y_pred_proba = self.model.predict_proba(X)
        model_deviance = -2 * np.sum([np.log(y_pred_proba[i, y[i]]) 
                                     for i in range(len(y))])
        
        # Tính LRT statistic
        lrt_statistic = null_deviance - model_deviance
        df = (n_classes - 1) * X.shape[1]  # degrees of freedom
        p_value = chi2.sf(lrt_statistic, df)
        
        print("\n1. Model Parameters:")
        print("-" * 30)
        print(f"Number of features: {X.shape[1]}")
        print(f"Number of classes: {n_classes}")
        print(f"Solver: {self.model.solver}")
        print(f"Max iterations: {self.model.max_iter}")
        print(f"Convergence achieved: {self.model.n_iter_ < self.model.max_iter}")
        print(f"Number of iterations: {self.model.n_iter_}")
        
        print("\n2. Model Coefficients:")
        print("-" * 30)
        feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
        for i, coef in enumerate(self.model.coef_):
            print(f"\nClass {i}:")
            for fname, c in zip(feature_names, coef):
                odds_ratio = np.exp(c)
                print(f"{fname:15} | Coef: {c:8.4f} | Odds Ratio: {odds_ratio:8.4f}")
        
        print("\n3. Likelihood Statistics:")
        print("-" * 30)
        print(f"Null Deviance (-2LL₀): {null_deviance:.4f}")
        print(f"Model Deviance (-2LL₁): {model_deviance:.4f}")
        print(f"Improvement (Δ): {null_deviance - model_deviance:.4f}")
        
        print("\n4. Likelihood Ratio Test:")
        print("-" * 30)
        print(f"LRT Statistic: {lrt_statistic:.4f}")
        print(f"Degrees of Freedom: {df}")
        print(f"P-value: {p_value:.4e}")
        
        # Thêm McFadden's R²
        mcfadden_r2 = 1 - (model_deviance / null_deviance)
        print(f"\nMcFadden's R²: {mcfadden_r2:.4f}")

    def _calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """
        Tính toán các metrics cho multi-class classification
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }

        try:
            # Kiểm tra các classes có trong dữ liệu
            present_classes = sorted(np.unique(y_true))
            
            # Chỉ tính AUC cho các classes thực sự có trong dữ liệu
            y_true_bin = label_binarize(y_true, classes=present_classes)
            y_pred_proba_filtered = y_pred_proba[:, [i-1 for i in present_classes]]  # Điều chỉnh index vì classes là 1,2,3
            
            # Tính AUC-ROC
            metrics['auc_roc'] = roc_auc_score(
                y_true_bin, 
                y_pred_proba_filtered,
                multi_class='ovr',
                average='weighted'
            )
            
            # Tính AUC-PR
            metrics['auc_pr'] = average_precision_score(
                y_true_bin,
                y_pred_proba_filtered,
                average='weighted'
            )

        except Exception as e:
            print(f"Warning: Không thể tính metrics: {str(e)}")
            
        return metrics

    def _log_model_parameters(self, X_train):
        """
        In và lưu parameters của model (cả LogisticRegression và Ensemble)
        """
        # Tạo thư mục results nếu chưa tồn tại
        os.makedirs('results', exist_ok=True)
        current_time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Lấy tên thực tế của các features
        feature_names = self.args.num_features + self.args.cat_features
        
        if self.is_ensemble:
            log_file = os.path.join('results', 'ensemble_model_params.log')
            log_content = [
                f"\nEnsemble Model Parameters - {current_time}",
                "="*50,
                "\nBase Models Configuration:"
            ]
            
            # Log thông tin về SVM
            log_content.extend([
                "\nSVM Model:",
                f"Kernel: {self.base_models['svm'].kernel}",
                f"C: {self.base_models['svm'].C}",
                f"Number of support vectors: {self.base_models['svm'].n_support_}"
            ])
            
            # Log thông tin về KNN
            log_content.extend([
                "\nKNN Model:",
                f"n_neighbors: {self.base_models['knn'].n_neighbors}",
                f"weights: {self.base_models['knn'].weights}",
                f"algorithm: {self.base_models['knn'].algorithm}"
            ])
            
            # Log thông tin về Decision Tree
            log_content.extend([
                "\nDecision Tree Model:",
                f"max_depth: {self.base_models['dt'].get_depth()}",
                f"n_leaves: {self.base_models['dt'].get_n_leaves()}",
                f"Feature importances:"
            ])
            for fname, imp in zip(feature_names, self.base_models['dt'].feature_importances_):
                log_content.append(f"{fname:<20}: {imp:.4f}")
            
            # Log thông tin về Random Forest
            log_content.extend([
                "\nRandom Forest Model:",
                f"n_estimators: {self.base_models['rf'].n_estimators}",
                f"max_depth: {self.base_models['rf'].max_depth}",
                f"Feature importances:"
            ])
            for fname, imp in zip(feature_names, self.base_models['rf'].feature_importances_):
                log_content.append(f"{fname:<20}: {imp:.4f}")
            
            # Log thông tin về Meta Model
            log_content.extend([
                "\nMeta Model (LogisticRegression):",
                f"Solver: {self.meta_model.solver}",
                f"Max iterations: {self.meta_model.max_iter}",
                f"Convergence achieved: {self.meta_model.n_iter_ < self.meta_model.max_iter}",
                f"Number of iterations: {self.meta_model.n_iter_}",
                "\nMeta Model Coefficients:"
            ])
            
            # Thêm thông tin về coefficients của meta model
            meta_feature_names = []
            for model in ['svm', 'knn', 'dt', 'rf']:
                for i in range(len(self.meta_model.classes_)):
                    meta_feature_names.append(f"{model}_class_{i}")
            
            log_content.append(f"{'Meta Feature':<20} | " + 
                             " | ".join(f"Class {i:>8}" for i in range(len(self.meta_model.classes_))))
            log_content.append("-"*80)
            
            for i, feature in enumerate(meta_feature_names):
                coef_values = [f"{coef:>8.4f}" for coef in self.meta_model.coef_[:, i]]
                log_content.append(f"{feature:<20} | " + " | ".join(coef_values))
            
            # Thêm thông tin về classes
            log_content.extend([
                "\nClasses:",
                f"Number of classes: {len(self.meta_model.classes_)}",
                f"Class labels: {self.meta_model.classes_}"
            ])
        
        else:
            # LogisticRegression
            log_file = os.path.join('results', 'logistic_regression_params.log')
            log_content = [
                f"\nLogistic Regression Parameters - {current_time}",
                "="*50,
                "\nModel Configuration:",
                f"Solver: {self.model.solver}",
                f"Max iterations: {self.model.max_iter}",
                f"Convergence achieved: {self.model.n_iter_ < self.model.max_iter}",
                f"Number of iterations: {self.model.n_iter_}",
                
                "\nModel Parameters:",
                "-"*30,
                "Intercept terms:"
            ]
            
            # Thêm thông tin về intercept
            for i, intercept in enumerate(self.model.intercept_):
                log_content.append(f"Class {i+1}: {intercept:.4f}")
            
            # Thêm thông tin về coefficients với tên features thực tế
            log_content.append("\nFeature coefficients:")
            log_content.append(f"{'Feature':<20} | " + 
                             " | ".join(f"Class {i+1:>8}" for i in range(len(self.model.classes_))))
            log_content.append("-"*80)
            
            for i, feature in enumerate(feature_names):
                coef_values = [f"{coef:>8.4f}" for coef in self.model.coef_[:, i]]
                log_content.append(f"{feature:<20} | " + " | ".join(coef_values))
            
            # Thêm thông tin về classes
            log_content.extend([
                "\nClasses:",
                f"Number of classes: {len(self.model.classes_)}",
                f"Class labels: {self.model.classes_}",
                "\nClass meanings:",
                "Class 1: Phòng bình thường (review_score < 7)",
                "Class 2: Phòng tốt (7 ≤ review_score < 9)",
                "Class 3: Phòng xuất sắc (review_score ≥ 9)"
            ])
        
        # In ra console
        print("\n".join(log_content))
        
        # Lưu vào file
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write("\n".join(log_content))
            f.write("\n\n" + "="*80 + "\n")
        
        print(f"\nĐã lưu thông tin parameters vào: {log_file}")
