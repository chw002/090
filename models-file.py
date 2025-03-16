import torch
import torch.nn as nn
import torchvision.models as models
import timm
import segmentation_models_pytorch as smp
import config

def get_model(model_name, num_classes=1):
    """
    获取指定的预训练模型
    
    参数:
        model_name (str): 模型名称
        num_classes (int): 分类数量
        
    返回:
        model: PyTorch模型
    """
    # 纯卷积网络模型
    if model_name == 'alexnet':
        model = models.alexnet(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        
    elif model_name == 'vgg16':
        model = models.vgg16_bn(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        
    # 残差结构模型
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
        raise ValueError(f"不支持的模型: {model_name}")
    
    # 如果是二分类问题，使用Sigmoid激活函数
    if num_classes == 1:
        return nn.Sequential(
            model,
            nn.Sigmoid()
        )
    
    return model

class ModelFactory:
    """模型工厂类，用于创建和初始化模型"""
    
    @staticmethod
    def get_model_names():
        """返回所有可用模型的列表"""
        return [
            'alexnet',      # 纯卷积网络
            'vgg16',        # 纯卷积网络
            'resnet50',     # 残差结构
            'efficientnet_b0', # 残差结构
            'mobilenet_v2', # 残差结构
            'vit'           # Vision Transformer
        ]
    
    @staticmethod
    def create_model(model_name=None):
        """
        创建指定的模型
        
        参数:
            model_name (str): 模型名称，如果为None则使用config中的MODEL_NAME
            
        返回:
            model: PyTorch模型
        """
        if model_name is None:
            model_name = config.MODEL_NAME
            
        print(f"正在创建模型: {model_name}")
        model = get_model(model_name)
        return model
