import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

#DataLoaders cho Machine Learning Models
class Supervised_ML_CustomDataLoader:
    def __init__(self, args):
        """
        Initialize CustomDataLoader with args containing all necessary parameters
        """
        self.task_dir = args.task_dir
        self.dataset = args.dataset


        self.csv_path = os.path.join(self.task_dir, f'{self.dataset}.csv')
        self.num_features = args.num_features
        self.cat_features = args.cat_features
        self.test_size = args.test_size
        self.random_state = args.random_state
        
        # Đọc dữ liệu
        self.data = pd.read_csv(self.csv_path)
        
        # Khởi tạo imputer cho numerical và categorical features
        self.num_imputer = SimpleImputer(strategy='mean')
        self.cat_imputer = SimpleImputer(strategy='most_frequent')
        self.scaler = StandardScaler()
        
    def get_loaders(self, batch_size):
        # Tách features
        X_num = self.data[self.num_features].copy()
        X_cat = self.data[self.cat_features].copy()
        y = self.data['review_score'].values  # Chuyển thành numpy array ngay từ đầu
        
        # Xử lý missing values
        X_num_imputed = self.num_imputer.fit_transform(X_num)
        X_cat_imputed = self.cat_imputer.fit_transform(X_cat)
        
        # Chuẩn hóa numerical features
        X_num_scaled = self.scaler.fit_transform(X_num_imputed)
        
        # Kết hợp features
        X = np.concatenate([X_num_scaled, X_cat_imputed], axis=1)
        
        # Chia train-val-test
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=self.random_state
        )
        
        # Tạo DataLoader
        train_loader = self._create_data_loader(X_train, y_train, batch_size)
        val_loader = self._create_data_loader(X_val, y_val, batch_size)
        test_loader = self._create_data_loader(X_test, y_test, batch_size)
        
        return train_loader, val_loader, test_loader
    
    def _create_data_loader(self, X, y, batch_size):
        # Chuyển đổi numpy array thành tensor trực tiếp
        X_tensor = torch.FloatTensor(X)  # Bỏ .values vì X đã là numpy array
        y_tensor = torch.FloatTensor(y)  # Bỏ .values vì y đã là numpy array
        
        # Tạo TensorDataset
        dataset = TensorDataset(X_tensor, y_tensor)
        
        # Tạo DataLoader
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)



#DataLoaders cho Deep Learning Models
class DL_CustomDataLoader:
    def __init__(self, args):
        """
        Khởi tạo CustomDataLoader với các tham số từ args
        """
        self.task_dir = args.task_dir
        self.dataset = args.dataset


        self.csv_path = os.path.join(self.task_dir, f'{self.dataset}.csv')
        self.args = args
        self.img_dir = args.img_dir
        self.device = args.device
        self.batch_size = args.batch_size
        
        # Đọc và tiền xử lý dữ liệu
        self.data = self._load_and_preprocess_data()
        
        # Lưu các đặc trưng đã được lọc
        self.filtered_features = self._get_filtered_features()
        
        
    def _load_and_preprocess_data(self):
        """
        Đọc và tiền xử lý dữ liệu từ CSV
        """
        # Đọc dữ liệu
        df = pd.read_csv(self.csv_path)
        
        # Xử lý missing values cho các cột quan trọng
        df = df.dropna(subset=['review_score'])
        
        # Reset index để lấy index làm tên file ảnh
        df = df.reset_index()
        
        # Tạo tên file ảnh theo format: index_review_score.jpg
        df['image'] = df.apply(lambda row: f"{row['index']}_{row['review_score']}.jpg", axis=1)
        
        # Kiểm tra và lọc các ảnh tồn tại
        valid_images = []
        for img_name in df['image'].unique():
            img_path = os.path.join(self.img_dir, img_name)
            if os.path.exists(img_path):
                try:
                    with Image.open(img_path) as img:
                        valid_images.append(img_name)
                except:
                    continue
        
        # Lọc DataFrame chỉ giữ lại các hàng có ảnh hợp lệ
        df = df[df['image'].isin(valid_images)]
        
        # Nếu DataFrame trống, raise error
        if len(df) == 0:
            raise ValueError(f"Không tìm thấy ảnh hợp lệ nào trong thư mục {self.img_dir}")
            
        # Drop cột index đã thêm vào
        df = df.drop('index', axis=1)
        
        # Fill NA cho các cột features
        numeric_cols = ['price', 'review_count', 'Comfort', 'Cleanliness', 'Facilities']
        categorical_cols = ['star', 'Pool', 'No_smoking_room', 'Families_room', 
                          'Room_service', 'Free_parking', 'Breakfast', 
                          '24h_front_desk', 'Airport_shuttle']
        
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        df[categorical_cols] = df[categorical_cols].fillna(0)
        
        return df

    def _get_filtered_features(self):
        """
        Lấy danh sách các features đã được lọc
        """
        return self.args.num_features + self.args.cat_features

    

    def get_data_loaders(self, val_split=0.1):
        """
        Tạo và trả về train_loader, val_loader và test_loader
        """
        # Chia dataset
        train_val_data, test_data = train_test_split(
            self.data,
            test_size=self.args.test_size,
            random_state=self.args.random_state
        )
        
        train_data, val_data = train_test_split(
            train_val_data,
            test_size=val_split,
            random_state=self.args.random_state
        )
        
        # Tạo datasets
        # Tạo datasets với transform=None để sử dụng _get_transforms() từ HotelDataset
        train_dataset = HotelDataset(train_data, self.img_dir, transform=None, 
                                    task_type=self.args.task_type, mode='train')
        val_dataset = HotelDataset(val_data, self.img_dir, transform=None, 
                                task_type=self.args.task_type, mode='test')
        test_dataset = HotelDataset(test_data, self.img_dir, transform=None, 
                                task_type=self.args.task_type, mode='test')
        
        # Tạo dataloaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

        return train_loader, val_loader, test_loader
    
class HotelDataset(Dataset):
    def __init__(self, data, img_dir, transform=None, task_type='classification', mode='train'):
        self.data = data
        self.img_dir = img_dir
        self.transform = transform
        self.task_type = task_type
        self.mode = mode
        
        # Sử dụng transform tương ứng với mode (train/test)
        if transform is None:
            self.transform = self._get_transforms()[mode]
        else:
            self.transform = transform[mode]
        # Chuẩn bị features và labels
        self.features = self._prepare_features()
        # Thêm một chiều mới cho labels để khớp với output
        self.labels = torch.tensor(self.data['review_score'].values, dtype=torch.float32).reshape(-1, 1)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # Chuyển review_score thành tensor với shape [1]
        review_score = torch.tensor([row['review_score']], dtype=torch.float32)
        
        # Load image
        img_filename = row['image']
        img_path = os.path.join(self.img_dir, img_filename)
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)

        # Tách riêng numerical và categorical features
        num_features = row[['price', 'review_count', 'Comfort', 'Cleanliness', 'Facilities']].values.astype(np.float32)
        cat_features = row[['star', 'Pool', 'No_smoking_room', 'Families_room', 
                        'Room_service', 'Free_parking', 'Breakfast', 
                        '24h_front_desk', 'Airport_shuttle']].values.astype(np.float32)
        
        # Chuyển đổi sang tensor
        num_features = torch.tensor(num_features, dtype=torch.float32)
        cat_features = torch.tensor(cat_features, dtype=torch.float32)
        
        # Kết hợp features
        features = torch.cat((num_features, cat_features))

        return (img, features), review_score

    def _prepare_features(self):
        """
        Chuẩn bị các đặc trưng số và categorical
        """
        # Chuẩn hóa đặc trưng số
        num_features = self.data[['price', 'review_count', 'Comfort', 'Cleanliness', 'Facilities']].values
        scaler = StandardScaler()
        num_features = scaler.fit_transform(num_features)
        
        # Đặc trưng categorical giữ nguyên vì đã là 0-1
        cat_features = self.data[['star', 'Pool', 'No_smoking_room', 'Families_room', 
                                 'Room_service', 'Free_parking', 'Breakfast', 
                                 '24h_front_desk', 'Airport_shuttle']].values
        
        # Chuyển đổi sang tensor
        num_features = torch.tensor(num_features, dtype=torch.float32)
        cat_features = torch.tensor(cat_features, dtype=torch.float32)
        
        return torch.cat((num_features, cat_features), dim=1)
    def _get_transforms(self):
        """
        Định nghĩa các transforms cho ảnh với data augmentation
        """
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
        ])
    
        return {'train': train_transform, 'test': test_transform}