import os
import argparse
import torch
import numpy as np
from base_model import *
from utils import *
from load_data import *
from constants import *
from models import *
from sklearn.metrics import mean_squared_error, silhouette_score
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd

# Khởi tạo parser cho các tham số
parser = argparse.ArgumentParser(description="Evaluation for ML and DL models")
parser.add_argument('--task_dir', type=str, default='D:\StatisticalMachineLearning\pythonProject1\Final_Project_StatisticalML\data', help='Directory to dataset')
parser.add_argument('--dataset', type=str, default='booking_images', help='Dataset name')


parser.add_argument('--task_type', type=str, choices=['classification', 'regression', 'clustering'], required=True, help='Task type')
parser.add_argument('--model_type', type=str, choices=['ml', 'dl'], required=True, help='Model type: ml for Machine Learning, dl for Deep Learning')


parser.add_argument('--gpu', type=int, default=0, help='GPU id to load (if available)')
parser.add_argument('--n_epoch', type=int, default=50, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--save_model', action='store_true', help='Save the model after training')
parser.add_argument('--load_model', action='store_true', help='Load a pre-trained model')
parser.add_argument('--vif_threshold', type=float, default=5.0, help='Threshold for Variance Inflation Factor (VIF)')
parser.add_argument('--max_features', type=int, default=10, help='Maximum number of features to select based on ranking')
parser.add_argument('--seed', type=int, default=1234, help='Random seed for reproducibility')
# parser.add_argument('--csv_path', default=f'D:\StatisticalMachineLearning\pythonProject1\Final_Project_StatisticalML\data\{booking_images.csv}', type=str, help='Path to the CSV data file')
parser.add_argument('--img_dir', type=str, default='D:\StatisticalMachineLearning\pythonProject1\Final_Project_StatisticalML\data\hotel_images\hotel_images', help='Directory for images')
# Thêm các tham số mặc định cho các đặc trưng số và phân loại
parser.add_argument('--num_features', nargs='+', default=['price', 'review_count', 'Comfort', 'Cleanliness', 'Facilities'], help='List of numerical feature columns')
parser.add_argument('--cat_features', nargs='+', default=['star', 'Pool', 'No_smoking_room', 'Families_room', 'Room_service', 'Free_parking', 'Breakfast', '24h_front_desk', 'Airport_shuttle'], help='List of categorical feature columns')
parser.add_argument('--test_size', type=float, default=0.2, help='Test split ratio for train-test split')
parser.add_argument('--random_state', type=int, default=42, help='Random state for reproducibility')
parser.add_argument('--model_name', type=str, default='resnet18', help='Name of the model architecture (e.g., resnet18, efficientnet)')
parser.add_argument('--num_classes', type=int, default=4, 
                    help='Number of output classes (4 for hotel review classification)')
parser.add_argument('--alpha', type=float, default=1.0, 
                    help='Regularization strength for Ridge/Elastic regression')
parser.add_argument('--l1_ratio', type=float, default=0.5,
                    help='L1 ratio for Elastic regression (0 <= l1_ratio <= 1)')
parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer type (e.g., adam, sgd)')
parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay for optimizer')
parser.add_argument('--use_scheduler', action='store_true', help='Use learning rate scheduler during training')

parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for computation')

parser.add_argument('--model_path', type=str, 
                   help='Path to saved model (default: results/model_type_task_type_dataset_best_model.pt)')

parser.add_argument('--model', type=str, default='LogisticRegression',
                   choices=['LogisticRegression', 'Ensemble'],
                   help='ML model to use (LogisticRegression or Ensemble)')
args = parser.parse_args()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.cuda.set_device(device)



def run_supervised_ml_model(args):
    set_seed(args.seed)
    
    print("Initializing data loader...")
    data_loader = Supervised_ML_CustomDataLoader(args)
    
    print("Getting train/val/test splits...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_loader.get_train_val_test_split()
    
    print("Initializing model...")
    model = MLModelWrapper(args)
    
    # In thông tin về dữ liệu
    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    print("\nBắt đầu huấn luyện với K-fold Cross Validation...")
    # Train model
    fold_metrics = model.train_with_kfold(X_train, y_train)
    
    # In kết quả validation
    print("\nKết quả Cross-validation:")
    print("-" * 50)
    metrics_list = [
        ('accuracy', 'Accuracy'),
        ('precision', 'Precision'),
        ('recall', 'Recall'),
        ('f1', 'F1-Score'),
        ('auc_roc', 'AUC-ROC (macro)'),
        ('auc_pr', 'AUC-PR (macro)')
    ]
    
    for metric_key, metric_name in metrics_list:
        if metric_key in fold_metrics['avg_metrics']:
            print(f"{metric_name}: {fold_metrics['avg_metrics'][metric_key]:.4f} "
                  f"(±{fold_metrics['std_metrics'][metric_key]:.4f})")
    
    # Lưu model nếu được yêu cầu
    if args.save_model:
        model.save_model()
    
    # Đánh giá trên tập test
    print("\nĐánh giá trên tập test:")
    print("-" * 50)
    test_metrics = model.evaluate(X_test, y_test)
    
    return model, fold_metrics, test_metrics

def run_dl_model(args):
    set_seed(args.seed)
    
    # Khởi tạo DataLoader và History
    dataloader = DL_CustomDataLoader(args)
    train_loader, val_loader, test_loader = dataloader.get_data_loaders(val_split=0.1)
    history = History(metric_name='rmse')
    
    model = DLModelWrapper(args)

    best_rmse = float('inf')
    for epoch in range(args.n_epoch):
        # Training
        if args.model_type == 'dl':
            train_metrics = model.train(train_loader, 
                                      criterion=torch.nn.MSELoss() if args.task_type == 'regression' else torch.nn.CrossEntropyLoss(), 
                                      args=args)
        
        # Đánh giá mỗi 2 epoch
        if epoch % 2 == 0:
            val_metrics = model.evaluate(val_loader)
            test_metrics = model.evaluate(test_loader)
            
            # Cập nhật history
            history.update(train_metrics, val_metrics, test_metrics)
            
            # In kết quả
            out_str = (f"Epoch {epoch+1}/{args.n_epoch}\n"
                      f"Train - Loss: {train_metrics['loss']:.4f}, RMSE: {train_metrics['rmse']:.4f}\n"
                      f"Val   - Loss: {val_metrics['loss']:.4f}, RMSE: {val_metrics['rmse']:.4f}\n"
                      f"Test  - Loss: {test_metrics['loss']:.4f}, RMSE: {test_metrics['rmse']:.4f}\n")
            
            print(out_str)
            with open(os.path.join('results', f"{args.dataset}_training_log.txt"), 'a') as f:
                f.write(out_str + "\n")
            
            # Lưu model tốt nhất
            if val_metrics['rmse'] < best_rmse:
                best_rmse = val_metrics['rmse']
                if args.save_model:
                    model.save_model(os.path.join('results', f"{args.dataset}_best_model.pt"))
    
    # Lưu history
    history.save_history(os.path.join('results', f"{args.dataset}_history.csv"))
    history.plot_metrics('results')
    
    print(f"Best validation RMSE: {best_rmse:.4f}")

# Thực thi mô hình
if __name__ == '__main__':
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Cập nhật model_path trước khi chạy run_model
    if args.model_path is None:
        args.model_path = get_default_model_path(args)
    
    # Run model với các tham số đã được chỉ định
    if args.model_type == 'dl':
        run_dl_model(args)
    elif args.model_type == 'ml':
        run_supervised_ml_model(args)
