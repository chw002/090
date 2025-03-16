import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def evaluate_model(model, data_loader, criterion=None, device='cuda'):
    """
    评估模型性能
    
    参数:
        model: PyTorch模型
        data_loader: 数据加载器
        criterion: 损失函数 (如果为None则不计算损失)
        device: 设备 ('cuda' 或 'cpu')
        
    返回:
        avg_loss: 平均损失 (如果criterion为None则为0)
        metrics: 包含各种指标的字典
    """
    model.eval()
    all_preds = []
    all_targets = []
    losses = []
    
    with torch.no_grad():
        for data, target in tqdm(data_loader, desc="评估"):
            if isinstance(target, str):  # 测试集，target是图像名称
                continue
                
            data, target = data.to(device), target.float().to(device).unsqueeze(1)
            output = model(data)
            
            # 计算损失
            if criterion is not None:
                loss = criterion(output, target)
                losses.append(loss.item())
            
            # 获取预测和目标
            pred = (output > 0.5).float().cpu().numpy()
            target = target.cpu().numpy()
            
            all_preds.extend(pred)
            all_targets.extend(target)
    
    # 转换为numpy数组
    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()
    
    # 计算指标
    metrics = {
        'accuracy': accuracy_score(all_targets, all_preds),
        'precision': precision_score(all_targets, all_preds, zero_division=0),
        'recall': recall_score(all_targets, all_preds, zero_division=0),
        'f1': f1_score(all_targets, all_preds, zero_division=0)
    }
    
    # 计算平均损失
    avg_loss = np.mean(losses) if losses else 0
    
    return avg_loss, metrics

def predict(model, data_loader, device='cuda'):
    """
    使用模型进行预测
    
    参数:
        model: PyTorch模型
        data_loader: 数据加载器
        device: 设备 ('cuda' 或 'cpu')
        
    返回:
        predictions: 预测结果
        image_names: 图像名称 (如果可用)
    """
    model.eval()
    predictions = []
    image_names = []
    
    with torch.no_grad():
        for data, target in tqdm(data_loader, desc="预测"):
            # 检查target是否为字符串（测试集）
            if isinstance(target, torch.Tensor):
                batch_images = None
            else:
                batch_images = target
                
            data = data.to(device)
            outputs = model(data)
            
            # 获取预测
            preds = (outputs > 0.5).float().cpu().numpy()
            predictions.extend(preds)
            
            # 获取图像名称 (如果可用)
            if batch_images is not None:
                image_names.extend(batch_images)
    
    return np.array(predictions).flatten(), image_names
