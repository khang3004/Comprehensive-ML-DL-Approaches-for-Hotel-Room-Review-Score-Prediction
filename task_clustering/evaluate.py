import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import KFold
from clustering_model import ClusteringModel
from load_data import ClusteringDataLoader
import joblib
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns

def setup_logging(args):
    """Thiết lập logging"""
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'clustering_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Log các parameters
    logging.info("Running with parameters:")
    for arg, value in vars(args).items():
        logging.info(f"{arg}: {value}")

def save_results(results, model_name, timestamp):
    """Lưu kết quả thí nghiệm"""
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Tạo dict mới chỉ chứa các dữ liệu có thể serialize
    serializable_results = {}
    for algorithm in results:
        serializable_results[algorithm] = {
            'fold_metrics': results[algorithm]['fold_metrics'],
            'best_score': results[algorithm]['best_score'],
            'test_metrics': results[algorithm].get('test_metrics', {}),
            'best_params': results[algorithm]['best_model'].best_params if results[algorithm]['best_model'] else None,
            'timestamp': timestamp
        }
    
    # Lưu kết quả dưới dạng JSON
    results_file = os.path.join(results_dir, f'{model_name}_results_{timestamp}.json')
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=4)
    
    logging.info(f"Saved results to {results_file}")
    
    # Lưu các model riêng biệt
    for algorithm in results:
        if results[algorithm]['best_model']:
            model_path = os.path.join('models', f'{algorithm}_best_model_{timestamp}.joblib')
            joblib.dump(results[algorithm]['best_model'].model, model_path)
            logging.info(f"Saved {algorithm} model to {model_path}")

def plot_fold_results(fold_metrics, model_name, timestamp):
    """Vẽ biểu đồ kết quả cross-validation"""
    plt.figure(figsize=(12, 6))
    
    # Tạo box plot cho mỗi metric
    metrics_df = pd.DataFrame(fold_metrics)
    metrics_df.boxplot()
    
    plt.title(f'{model_name} Cross-validation Results')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    
    # Lưu plot
    plt.tight_layout()
    plt.savefig(f'results/{model_name}_cv_results_{timestamp}.png')
    plt.close()

def run_clustering_with_cv(args):
    # Thiết lập logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    setup_logging(args)
    
    # Khởi tạo data loader
    logging.info("Initializing data loader...")
    data_loader = ClusteringDataLoader(args)
    
    # Visualize dữ liệu ban đầu
    logging.info("Visualizing initial data distributions...")
    data_loader.visualize_data()
    
    # Lấy features đã được xử lý
    features = data_loader.get_features()
    
    # Chia tập test
    np.random.seed(42)
    test_size = 0.2
    indices = np.random.permutation(len(features))
    test_size_idx = int(len(features) * test_size)
    
    test_idx = indices[:test_size_idx]
    train_idx = indices[test_size_idx:]
    
    X_train = features[train_idx]
    X_test = features[test_idx]
    
    # K-fold Cross Validation
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Dictionary để lưu kết quả
    results = {
        'kmeans': {'fold_metrics': [], 'best_model': None, 'best_score': -1},
        'dbscan': {'fold_metrics': [], 'best_model': None, 'best_score': -1}
    }
    
    # Thử nghiệm cả KMeans và DBSCAN
    for algorithm in ['kmeans', 'dbscan']:
        logging.info(f"\nRunning {algorithm.upper()} clustering with {n_splits}-fold CV...")
        
        for fold, (train_fold_idx, val_fold_idx) in enumerate(kf.split(X_train), 1):
            logging.info(f"\nFold {fold}/{n_splits}")
            
            # Chia dữ liệu theo fold
            X_train_fold = X_train[train_fold_idx]
            X_val_fold = X_train[val_fold_idx]
            
            # Khởi tạo và train model
            model = ClusteringModel(args)
            
            # Fit model và lấy labels
            train_labels = model.fit(X_train_fold, algorithm=algorithm)
            val_labels = model.model.predict(X_val_fold) if algorithm == 'kmeans' else model.model.fit_predict(X_val_fold)
            
            # Evaluate kết quả
            metrics = model.evaluate(X_val_fold, val_labels)
            results[algorithm]['fold_metrics'].append(metrics)
            
            # Lưu model tốt nhất dựa trên silhouette score
            if metrics['silhouette'] > results[algorithm]['best_score']:
                results[algorithm]['best_score'] = metrics['silhouette']
                results[algorithm]['best_model'] = model
                
                # Lưu model
                model_path = f'models/{algorithm}_best_model_{timestamp}.joblib'
                if not os.path.exists('models'):
                    os.makedirs('models')
                joblib.dump(model.model, model_path)
                logging.info(f"Saved best model to {model_path}")
        
        # Plot kết quả cross-validation
        plot_fold_results(results[algorithm]['fold_metrics'], algorithm, timestamp)
        
        # Evaluate trên tập test với model tốt nhất
        logging.info(f"\nEvaluating best {algorithm} model on test set...")
        best_model = results[algorithm]['best_model']
        test_labels = best_model.model.predict(X_test) if algorithm == 'kmeans' else best_model.model.fit_predict(X_test)
        test_metrics = best_model.evaluate(X_test, test_labels)
        results[algorithm]['test_metrics'] = test_metrics
        
        # Lưu trữ dữ liệu gốc
        original_features = data_loader.features.copy()
        original_scores = data_loader.review_scores.copy()
        
        # Visualize kết quả cuối cùng
        for dim_red_method in ['tsne', 'pca']:
            try:
                # Chỉ visualize trên tập test
                data_loader.features = X_test
                data_loader.review_scores = data_loader.review_scores[test_idx]
                
                # Tạo visualization
                data_loader.visualize_clusters(
                    test_labels, 
                    method=dim_red_method
                )
                
                logging.info(f"Created {dim_red_method.upper()} visualization for {algorithm}")
                
            except Exception as e:
                logging.error(f"Error visualizing clusters with {dim_red_method}: {str(e)}")
                
            finally:
                # Khôi phục dữ liệu gốc
                data_loader.features = original_features
                data_loader.review_scores = original_scores
    
    # Lưu kết quả
    save_results(results, 'clustering', timestamp)
    
    # Log kết quả cuối cùng
    logging.info("\nFinal Results:")
    for algorithm in results:
        logging.info(f"\n{algorithm.upper()}:")
        logging.info(f"Best CV Silhouette Score: {results[algorithm]['best_score']:.4f}")
        logging.info("Test Metrics:")
        for metric, value in results[algorithm]['test_metrics'].items():
            logging.info(f"{metric}: {value:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Clustering for hotel reviews")
    
    # Thêm các arguments cần thiết
    parser.add_argument('--task_dir', type=str, 
                       default='D:\StatisticalMachineLearning\pythonProject1\Final_Project_StatisticalML\data', 
                       help='Directory to dataset')
    parser.add_argument('--dataset', type=str, default='booking_images',
                       help='Name of the dataset file')
    parser.add_argument('--num_features', nargs='+', 
                       default=['price', 'review_count', 'Comfort', 'Cleanliness', 'Facilities'],
                       help='List of numerical features')
    parser.add_argument('--cat_features', nargs='+',
                       default=['star', 'Pool', 'No_smoking_room', 'Families_room', 
                               'Room_service', 'Free_parking', 'Breakfast', 
                               '24h_front_desk', 'Airport_shuttle'],
                       help='List of categorical features')
    parser.add_argument('--load_model', type=str, default=None,
                       help='Path to pre-trained model to load')
    parser.add_argument('--n_splits', type=int, default=5,
                       help='Number of folds for cross-validation')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save output files')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--min_clusters', type=int, default=2,
                       help='Minimum number of clusters to try for KMeans')
    parser.add_argument('--max_clusters', type=int, default=11,
                       help='Maximum number of clusters to try for KMeans')
    parser.add_argument('--eps_range', type=float, nargs=3, default=[0.1, 2.1, 0.1],
                       help='Range for DBSCAN eps parameter (start, stop, step)')
    parser.add_argument('--min_samples_range', type=int, nargs=2, default=[3, 11],
                       help='Range for DBSCAN min_samples parameter (start, stop)')
    
    args = parser.parse_args()
    
    # Tạo các thư mục cần thiết
    for directory in ['results', 'models', 'logs']:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Run clustering
    run_clustering_with_cv(args)