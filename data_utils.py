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
    """ISIC 2020 skin lesion image dataset"""
    
    def __init__(self, data_source, img_dir, transform=None, is_test=False):
        """
        Parameters:
            data_source: CSV file path or Pandas DataFrame
            img_dir: Image directory path
            transform: Transformations to apply to images
            is_test: Whether this is a test set
        """
        # Check input type and load data accordingly
        if isinstance(data_source, str):
            self.data_frame = pd.read_csv(data_source)
        else:
            self.data_frame = data_source  # Directly use the provided DataFrame
            
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test
        
        # Read duplicate images list
        if os.path.exists(config.DUPLICATES_CSV):
            self.duplicates_df = pd.read_csv(config.DUPLICATES_CSV)
            self.duplicate_images = set(self.duplicates_df['image_name_1']).union(
                set(self.duplicates_df['image_name_2']))
        else:
            self.duplicate_images = set()
            
        # Remove duplicate images
        if not is_test:
            self.data_frame = self.data_frame[~self.data_frame['image_name'].isin(self.duplicate_images)]
            
        # Standardize column names (train.csv and test.csv have different column names)
        if 'image_name' in self.data_frame.columns:
            self.img_col = 'image_name'
        else:
            self.img_col = 'image'
            
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        img_name = self.data_frame.iloc[idx][self.img_col]
        img_path = os.path.join(self.img_dir, img_name + '.jpg')
        
        # Read image
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            # If image cannot be read, return a black image
            print(f"Cannot read image: {img_path}")
            image = Image.new('RGB', (config.IMAGE_SIZE, config.IMAGE_SIZE), (0, 0, 0))
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
            
        # Get label (if training set)
        if not self.is_test:
            label = self.data_frame.iloc[idx]['target']
            return image, label
        else:
            return image, img_name

def get_transforms(is_train=True):
    """
    Get image transformations
    
    Parameters:
        is_train (bool): Whether transformations are for training set
        
    Returns:
        transforms: torchvision transformations
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
    Create train, validation, and test data loaders
    
    Returns:
        train_loader, val_loader, test_loader: PyTorch data loaders
    """
    # Load training dataset
    train_df = pd.read_csv(config.TRAIN_CSV)
    
    # Split into training and validation sets
    train_df, val_df = train_test_split(
        train_df, test_size=0.2, stratify=train_df['target'], random_state=42
    )
    
    # Create datasets - directly pass DataFrames
    train_dataset = ISIC2020Dataset(
        train_df,  # Directly pass DataFrame
        config.TRAIN_IMAGES_DIR,
        transform=get_transforms(is_train=True),
        is_test=False
    )
    
    val_dataset = ISIC2020Dataset(
        val_df,  # Directly pass DataFrame
        config.TRAIN_IMAGES_DIR,
        transform=get_transforms(is_train=False),
        is_test=False
    )
    
    # Test set still uses CSV file path
    test_dataset = ISIC2020Dataset(
        config.TEST_CSV,
        config.TEST_IMAGES_DIR,
        transform=get_transforms(is_train=False),
        is_test=True
    )
    
    # Create data loaders
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