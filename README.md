# Skin Lesion Classification: CNN Architecture Comparison

## Overview

This project provides a comprehensive framework for comparing different convolutional neural network (CNN) architectures on skin lesion classification using the ISIC 2020 dataset. The implementation allows researchers to evaluate and compare the performance of pure convolutional networks (AlexNet, VGG16), residual networks (ResNet50, EfficientNet-B0, MobileNetV2), and vision transformers (ViT) for melanoma detection.

## Dataset

The project uses the International Skin Imaging Collaboration (ISIC) 2020 Challenge dataset, which contains over 33,000 dermoscopic images of skin lesions with binary classification (benign/malignant). The dataset presents significant class imbalance (approximately 1.76% positive samples), making it a challenging but realistic medical image classification task.

## Features

- Complete data processing pipeline with image augmentation and normalization
- Implementation of six different model architectures for comparison
- Weighted loss function to address severe class imbalance
- Threshold optimization for improved classification performance
- Comprehensive evaluation metrics (accuracy, precision, recall, F1)
- Visualization tools for training history, confusion matrices, and model comparisons
- Efficient GPU acceleration with PyTorch
- Wandb integration for experiment tracking (optional)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/skin-lesion-classification.git
cd skin-lesion-classification

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install monai[all] pytorch-lightning wandb segmentation-models-pytorch opencv-python pandas matplotlib
```

## Project Structure
isic_cnn_comparison/
├── config.py           # Configuration parameters
├── data_utils.py       # Data loading and preprocessing
├── models.py           # Model definitions and factory
├── train.py            # Training functionality
├── evaluate.py         # Evaluation metrics and inference
├── visualize.py        # Results visualization
└── main.py             # Main execution script
