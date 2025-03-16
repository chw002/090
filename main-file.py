import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm

import config
from data_utils import get_data_loaders
from models import ModelFactory
from train import train_model
from evaluate import evaluate_model, predict
from visualize import plot_training_history, plot_confusion_matrix, plot_model_comparison

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='ISIC 2020皮肤病变分类')
    
    # 训练参数
    parser.add_argument('--model', type=str, default=config.MODEL_NAME,
                        choices=ModelFactory.get_model_names(),
                        help='模型名称')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=config.NUM_EPOCHS,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=config.WEIGHT_DECAY,
                        help='权重衰减')
    parser.add_argument('--image_size', type=int, default=config.IMAGE_SIZE,
                        help='图像大小')
    
    # 模式参数
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'evaluate', 'predict', 'compare_all'],
                        help='运行模式')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='模型检查点路径')
    parser.add_argument('--use_wandb', action='store_true',
                        help='是否使用wandb')
    
    return parser.parse_args()

def train(args):
    """训练模型"""
    # 更新配置
    config.BATCH_SIZE = args.batch_size
    config.IMAGE_SIZE = args.image_size
    config.USE_WANDB = args.use_wandb
    
    # 获取数据加载器
    train_loader, val_loader, _ = get_data_loaders()
    
    # 创建模型
    model = ModelFactory.create_model(args.model)
    
    # 训练模型
    model, history = train_model(
        model, 
        train_loader, 
        val_loader, 
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        model_name=args.model,
        use_wandb=args.use_wandb
    )
    
    # 绘制训练历史
    plot_training_history(history, args.model)
    
    return model, history

def evaluate(args):
    """评估模型"""
    # 更新配置
    config.BATCH_SIZE = args.batch_size
    config.IMAGE_SIZE = args.image_size
    
    # 获取数据加载器
    _, val_loader, _ = get_data_loaders()
    
    # 创建模型
    model = ModelFactory.create_model(args.model)
    
    # 加载检查点
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"{args.model}_best.pth")
    
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"已加载检查点: {checkpoint_path}")
    else:
        print(f"未找到检查点: {checkpoint_path}")
        return
    
    # 评估模型
    criterion = nn.BCELoss()
    loss, metrics = evaluate_model(model, val_loader, criterion, config.DEVICE)
    
    # 打印结果
    print(f"评估结果:")
    print(f"损失: {loss:.4f}")
    print(f"准确率: {metrics['accuracy']:.4f}")
    print(f"精确度: {metrics['precision']:.4f}")
    print(f"召回率: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    
    return metrics

def compare_all_models(args):
    """比较所有模型"""
    # 更新配置
    config.BATCH_SIZE = args.batch_size
    config.IMAGE_SIZE = args.image_size
    
    # 获取模型列表
    model_names = ModelFactory.get_model_names()
    
    # 结果字典
    results = {}
    
    for model_name in model_names:
        print(f"\n正在评估模型: {model_name}")
        
        # 创建模型
        model = ModelFactory.create_model(model_name)
        
        # 加载检查点
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"{model_name}_best.pth")
        
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path))
            print(f"已加载检查点: {checkpoint_path}")
            
            # 获取数据加载器
            _, val_loader, _ = get_data_loaders()
            
            # 评估模型
            criterion = nn.BCELoss()
            loss, metrics = evaluate_model(model, val_loader, criterion, config.DEVICE)
            
            # 存储结果
            results[model_name] = metrics
            
            # 打印结果
            print(f"损失: {loss:.4f}")
            print(f"准确率: {metrics['accuracy']:.4f}")
            print(f"精确度: {metrics['precision']:.4f}")
            print(f"召回率: {metrics['recall']:.4f}")
            print(f"F1: {metrics['f1']:.4f}")
        else:
            print(f"未找到检查点: {checkpoint_path}")
    
    # 比较模型
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        plot_model_comparison(results, metric)
    
    return results

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 确保目录存在
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    
    # 设置设备
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    config.DEVICE = device
    print(f"使用设备: {device}")
    
    # 根据模式运行
    if args.mode == 'train':
        model, history = train(args)
    elif args.mode == 'evaluate':
        metrics = evaluate(args)
    elif args.mode == 'compare_all':
        results = compare_all_models(args)
    else:
        print(f"不支持的模式: {args.mode}")

if __name__ == '__main__':
    main()
