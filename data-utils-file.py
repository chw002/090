import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
import config

class ISIC2020Dataset(Dataset):
    """ISIC 2020皮肤病变图像数据集"""
    
    def __init__(self, csv_file, img_dir, transform=None, is_test=False):
        """
        参数:
            csv_file (str): 包含图像信息的CSV文件路径
            img_dir (str): 图像目录路径
            transform (callable, optional): 应用于图像的转换
            is_test (bool): 是否为测试集
        """
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test
        
        # 读取重复图像列表
        if os.path.exists(config.DUPLICATES_CSV):
            self.duplicates_df = pd.read_csv(config.DUPLICATES_CSV)
            self.duplicate_images = set(self.duplicates_df['image_name_1']).union(
                set(self.duplicates_df['image_name_2']))
        else:
            self.duplicate_images = set()
            
        # 移除重复图像
        if not is_test:
            self.data_frame = self.data_frame[~self.data_frame['image_name'].isin(self.duplicate_images)]
            
        # 标准化列名 (train.csv和test.csv有不同的列名)
        if 'image_name' in self.data_frame.columns:
            self.img_col = 'image_name'
        else:
            self.img_col = 'image'
            
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        img_name = self.data_frame.iloc[idx][self.img_col]
        img_path = os.path.join(self.img_dir, img_name + '.jpg')
        
        # 读取图像
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            # 如果无法读取图像，返回一个黑色图像
            print(f"无法读取图像: {img_path}")
            image = Image.new('RGB', (config.IMAGE_SIZE, config.IMAGE_SIZE), (0, 0, 0))
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
            
        # 获取标签 (如果是训练集)
        if not self.is_test:
            label = self.data_frame.iloc[idx]['target']
            return image, label
        else:
            return image, img_name

def get_transforms(is_train=True):
    """
    获取图像变换
    
    参数:
        is_train (bool): 是否为训练集变换
        
    返回:
        transforms: torchvision变换
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def get_data_loaders():
    """
    创建训练、验证和测试数据加载器
    
    返回:
        train_loader, val_loader, test_loader: PyTorch数据加载器
    """
    # 加载训练数据集
    train_df = pd.read_csv(config.TRAIN_CSV)
    
    # 分割训练集和验证集
    train_df, val_df = train_test_split(
        train_df, test_size=0.2, stratify=train_df['target'], random_state=42
    )
    
    # 创建数据集
    train_dataset = ISIC2020Dataset(
        config.TRAIN_CSV, 
        config.TRAIN_IMAGES_DIR,
        transform=get_transforms(is_train=True),
        is_test=False
    )
    
    val_dataset = ISIC2020Dataset(
        config.TRAIN_CSV,
        config.TRAIN_IMAGES_DIR,
        transform=get_transforms(is_train=False),
        is_test=False
    )
    
    test_dataset = ISIC2020Dataset(
        config.TEST_CSV,
        config.TEST_IMAGES_DIR,
        transform=get_transforms(is_train=False),
        is_test=True
    )
    
    # 仅使用验证集的索引
    val_indices = val_df.index.tolist()
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
    
    # 仅使用训练集的索引
    train_indices = train_df.index.tolist()
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
