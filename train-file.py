import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

import config
from models import ModelFactory
from evaluate import evaluate_model

def train_model(model, train_loader, val_loader, num_epochs=config.NUM_EPOCHS, 
                learning_rate=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY,
                device=config.DEVICE, model_name=None, use_wandb=config.USE_WANDB):
    """
    训练模型
    
    参数:
        model: PyTorch模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        num_epochs: 训练轮数
        learning_rate: 学习率
        weight_decay: 权重衰减
        device: 设备 ('cuda' 或 'cpu')
        model_name: 模型名称，用于保存
        use_wandb: 是否使用wandb记录
        
    返回:
        model: 训练后的模型
        history: 训练历史
    """
    if model_name is None:
        model_name = config.MODEL_NAME
        
    # 确保目录存在
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    # 移动模型到设备
    model = model.to(device)
    
    # 初始化优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCELoss()  # 二分类交叉熵损失
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=2, factor=0.5, verbose=True
    )
    
    # 记录训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': []
    }
    
    # 初始化wandb
    if use_wandb:
        wandb.init(project=config.WANDB_PROJECT, entity=config.WANDB_ENTITY, name=model_name)
        wandb.config.update({
            "learning_rate": learning_rate,
            "epochs": num_epochs,
            "batch_size": config.BATCH_SIZE,
            "model": model_name,
            "image_size": config.IMAGE_SIZE,
            "weight_decay": weight_decay
        })
    
    # 训练循环
    best_val_f1 = 0
    
    print(f"开始训练 {model_name}...")
    
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        train_losses = []
        
        # 训练一个epoch
        progress_bar = tqdm(train_loader, desc=f"第 {epoch+1}/{num_epochs} 轮")
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(device), target.float().to(device).unsqueeze(1)
            
            # 前向传播
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 记录损失
            train_losses.append(loss.item())
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # 计算平均训练损失
        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)
        
        # 验证
        val_loss, val_metrics = evaluate_model(model, val_loader, criterion, device)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1'])
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_DIR, f"{model_name}_best.pth"))
            print(f"保存最佳模型，F1: {best_val_f1:.4f}")
        
        # 记录到wandb
        if use_wandb:
            wandb_log = {
                'train/loss': avg_train_loss,
                'val/loss': val_loss,
                'val/accuracy': val_metrics['accuracy'],
                'val/precision': val_metrics['precision'],
                'val/recall': val_metrics['recall'],
                'val/f1': val_metrics['f1'],
                'learning_rate': optimizer.param_groups[0]['lr']
            }
            wandb.log(wandb_log, step=epoch)
        
        # 打印进度
        time_taken = time.time() - start_time
        print(f"第 {epoch+1}/{num_epochs} 轮完成 "
              f"[{time_taken:.1f}s] - "
              f"训练损失: {avg_train_loss:.4f}, "
              f"验证损失: {val_loss:.4f}, "
              f"准确率: {val_metrics['accuracy']:.4f}, "
              f"精确度: {val_metrics['precision']:.4f}, "
              f"召回率: {val_metrics['recall']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}")
    
    # 训练结束，保存最终模型
    torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_DIR, f"{model_name}_final.pth"))
    print(f"训练完成，最终F1: {history['val_f1'][-1]:.4f}, 最佳F1: {best_val_f1:.4f}")
    
    # 结束wandb
    if use_wandb:
        wandb.finish()
    
    return model, history
