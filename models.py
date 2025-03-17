import torch
import torch.nn as nn
import torchvision.models as models
import timm
import segmentation_models_pytorch as smp
import config

def get_model(model_name, num_classes=1):
    """
    Get specified pre-trained model
    
    Parameters:
        model_name (str): Model name
        num_classes (int): Number of classes
        
    Returns:
        model: PyTorch model
    """
    # Pure convolutional network models
    if model_name == 'alexnet':
        model = models.alexnet(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        
    elif model_name == 'vgg16':
        model = models.vgg16_bn(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        
    # Residual network models
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
    # Vision Transformer
    elif model_name == 'vit':
        model = timm.create_model('vit_base_patch16_224', pretrained=True)
        model.head = nn.Linear(model.head.in_features, num_classes)
        
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # For binary classification
    if num_classes == 1:
        return model
    
    return model

class ModelFactory:
    """Model factory class for creating and initializing models"""
    
    @staticmethod
    def get_model_names():
        """Return list of all available models"""
        return [
            'alexnet',      # Pure convolutional network
            'vgg16',        # Pure convolutional network
            'resnet50',     # Residual network
            'efficientnet_b0', # Residual network
            'mobilenet_v2', # Residual network
            'vit'           # Vision Transformer
        ]
    
    @staticmethod
    def create_model(model_name=None):
        """
        Create specified model
        
        Parameters:
            model_name (str): Model name, if None uses MODEL_NAME from config
            
        Returns:
            model: PyTorch model
        """
        if model_name is None:
            model_name = config.MODEL_NAME
            
        print(f"Creating model: {model_name}")
        model = get_model(model_name)
        return model