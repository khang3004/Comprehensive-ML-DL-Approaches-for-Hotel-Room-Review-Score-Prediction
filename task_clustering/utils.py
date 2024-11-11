import os
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_squared_error, silhouette_score, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score
)
from sklearn.cluster import KMeans

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def convert_review_score_to_class(score):
    if score < 4:
        return 0  # phòng kém
    elif score < 7:
        return 1  # phòng bth
    elif score < 9:
        return 2  # phòng tốt
    else:
        return 3  # phòng vip

def get_default_model_path(args):
    return os.path.join('results', f"{args.model_type}_{args.task_type}_{args.dataset}_best_model.pt")

def calculate_rmse(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse

# Hàm để đánh giá phân cụm bằng Silhouette Score và Elbow Method
def evaluate_clustering(model, data_loader):
    embeddings = []
    for (inputs, _) in data_loader:
        with torch.no_grad():
            outputs = model.model(inputs[0].cuda()).squeeze() if args.model_type == 'dl' else model.predict(inputs[0])
            embeddings.extend(outputs.cpu().numpy() if args.model_type == 'dl' else outputs)
    embeddings = np.array(embeddings)
    
    # Silhouette Score
    kmeans = KMeans(n_clusters=3, random_state=args.seed)
    labels = kmeans.fit_predict(embeddings)
    silhouette_avg = silhouette_score(embeddings, labels)
    
    print(f"Silhouette Score for Clustering: {silhouette_avg:.4f}")
    with open(os.path.join('results', f"{args.dataset}_clustering_eval.txt"), 'a+') as f:
        f.write(f"Silhouette Score: {silhouette_avg:.4f}\n")
    
    # Elbow Method
    distortions = []
    K = range(1, 10)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=args.seed)
        kmeans.fit(embeddings)
        distortions.append(kmeans.inertia_)
    
    plt.figure(figsize=(8, 5))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Distortion')
    plt.title('Elbow Method For Optimal k')
    plt.savefig(os.path.join('results', f"{args.dataset}_elbow_plot.png"))
    plt.show()

class History:
    def __init__(self, metric_name):
        self.metric_name = metric_name
        self.train_metrics = []
        self.val_metrics = []
        self.test_metrics = []
    
    def update(self, train_metrics, val_metrics, test_metrics):
        self.train_metrics.append(train_metrics)
        self.val_metrics.append(val_metrics)
        self.test_metrics.append(test_metrics)
    
    def save_history(self, filepath):
        history_df = pd.DataFrame({
            f'train_{self.metric_name}': [m[self.metric_name] for m in self.train_metrics],
            f'val_{self.metric_name}': [m[self.metric_name] for m in self.val_metrics],
            f'test_{self.metric_name}': [m[self.metric_name] for m in self.test_metrics]
        })
        history_df.to_csv(filepath, index=False)
    
    def plot_metrics(self, save_dir):
        plt.figure(figsize=(10, 6))
        plt.plot([m[self.metric_name] for m in self.train_metrics], label='Train')
        plt.plot([m[self.metric_name] for m in self.val_metrics], label='Validation')
        plt.plot([m[self.metric_name] for m in self.test_metrics], label='Test')
        plt.title(f'Model {self.metric_name} over time')
        plt.xlabel('Epoch')
        plt.ylabel(self.metric_name)
        plt.legend()
        plt.savefig(os.path.join(save_dir, f'metrics_{self.metric_name}.png'))
        plt.close()

def calculate_classification_metrics(y_true, y_pred, y_pred_proba):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro'),
        'recall': recall_score(y_true, y_pred, average='macro'),
        'f1': f1_score(y_true, y_pred, average='macro')
    }
    
    # Calculate AUC-ROC and AUC-PR for each class
    n_classes = len(np.unique(y_true))
    for i in range(n_classes):
        metrics[f'auc_roc_class_{i}'] = roc_auc_score(
            (y_true == i).astype(int), 
            y_pred_proba[:, i],
            average='macro'
        )
        metrics[f'auc_pr_class_{i}'] = average_precision_score(
            (y_true == i).astype(int), 
            y_pred_proba[:, i],
            average='macro'
        )
    
    return metrics