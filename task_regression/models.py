import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models

class CombinedModel(nn.Module):
    def __init__(self, num_features, num_classes, task_type):
        super(CombinedModel, self).__init__()
        
        # Image model (ResNet18)
        self.image_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = self.image_model.fc.in_features
        
        # Thay đổi kiến trúc xử lý ảnh
        self.image_model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Feature model với kiến trúc phức tạp hơn
        self.feature_model = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Linear(num_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Combined layers với nhiều layer hơn
        self.fc = nn.Sequential(
            nn.Linear(192, 128),  # 128 (image) + 64 (features) = 192
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1 if task_type == 'regression' else num_classes)
        )
        
        self.task_type = task_type

    def forward(self, inputs):
        images, features = inputs
        
        # Xử lý ảnh
        image_features = self.image_model(images)  # [batch_size, 128]
        
        # Xử lý đặc trưng số
        feature_out = self.feature_model(features)  # [batch_size, 64]
        
        # Kết hợp
        combined = torch.cat((image_features, feature_out), dim=1)  # [batch_size, 192]
        
        # Đầu ra
        output = self.fc(combined)
        return output


