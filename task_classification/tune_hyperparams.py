import argparse
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from base_model import MLModelWrapper
from load_data import Supervised_ML_CustomDataLoader
import pandas as pd
from utils import set_seed

def create_param_grid():
    """Tạo grid parameters cho từng loại model"""
    param_grids = {
        'Ridge_Regression': {
            'alpha': np.logspace(-3, 3, 20)  # 20 giá trị alpha từ 0.001 đến 1000
        },
        'Elastic_Regression': {
            'alpha': np.logspace(-3, 3, 10),
            'l1_ratio': np.linspace(0.1, 0.9, 9)  # 9 giá trị từ 0.1 đến 0.9
        }
    }
    return param_grids

def custom_scorer(y_true, y_pred):
    """Custom scorer kết hợp RMSE và R2"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    # Trọng số cho RMSE và R2 (có thể điều chỉnh)
    return 0.7 * (1 - rmse/10) + 0.3 * r2

def tune_hyperparameters(args, X_train, y_train):
    """Tối ưu hyperparameters sử dụng GridSearchCV"""
    param_grids = create_param_grid()
    
    if args.model not in param_grids:
        raise ValueError(f"Model {args.model} không hỗ trợ tune hyperparameters")
    
    # Khởi tạo model
    if args.model == 'Ridge_Regression':
        model = Ridge()
    elif args.model == 'Elastic_Regression':
        model = ElasticNet()
    
    # Tạo custom scorer
    scorer = make_scorer(custom_scorer)
    
    # GridSearchCV với cross-validation
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grids[args.model],
        scoring=scorer,
        cv=5,
        n_jobs=-1,
        verbose=2
    )
    
    print(f"\nBắt đầu tìm hyperparameters tối ưu cho {args.model}...")
    grid_search.fit(X_train, y_train)
    
    # In kết quả
    print("\nKết quả tối ưu hyperparameters:")
    print("-" * 50)
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_:.4f}")
    
    # In bảng so sánh kết quả
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df = results_df.sort_values('rank_test_score')
    print("\nTop 5 bộ hyperparameters tốt nhất:")
    print(results_df[['params', 'mean_test_score', 'std_test_score']].head())
    
    return grid_search.best_params_

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for regression models")
    # Thêm các argument từ evaluate.py
    parser.add_argument('--task_dir', type=str, default='data')
    parser.add_argument('--dataset', type=str, default='booking_images')
    parser.add_argument('--task_type', type=str, default='regression')
    parser.add_argument('--model_type', type=str, default='ml')
    parser.add_argument('--model', type=str, choices=['Ridge_Regression', 'Elastic_Regression'])
    parser.add_argument('--random_state', type=int, default=42)
    
    args = parser.parse_args()
    set_seed(args.random_state)
    
    # Load data
    data_loader = Supervised_ML_CustomDataLoader(args)
    train_loader, _, _ = data_loader.get_loaders()
    
    # Chuẩn bị dữ liệu
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
    
    # Tune hyperparameters
    best_params = tune_hyperparameters(args, X_train, y_train)
    
    # Lưu kết quả
    with open(f'results/{args.model}_best_params.txt', 'w') as f:
        f.write(str(best_params))

if __name__ == '__main__':
    main() 