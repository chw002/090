import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import config

def plot_training_history(history, model_name=None, save_path=None):
    """
    绘制训练历史
    
    参数:
        history: 包含训练历史的字典
        model_name: 模型名称
        save_path: 保存路径 (如果为None则不保存)
    """
    if model_name is None:
        model_name = config.MODEL_NAME
        
    if save_path is None:
        save_path = os.path.join(config.RESULTS_DIR, f"{model_name}_history.png")
        
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 创建图形
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    
    # 绘制损失
    ax[0, 0].plot(history['train_loss'], label='Train')
    ax[0, 0].plot(history['val_loss'], label='Validation')
    ax[0, 0].set_title('Loss')
    ax[0, 0].set_xlabel('Epoch')
    ax[0, 0].set_ylabel('Loss')
    ax[0, 0].legend()
    ax[0, 0].grid(True)
    
    # 绘制准确率
    ax[0, 1].plot(history['val_accuracy'], label='Accuracy')
    ax[0, 1].set_title('Accuracy')
    ax[0, 1].set_xlabel('Epoch')
    ax[0, 1].set_ylabel('Accuracy')
    ax[0, 1].grid(True)
    
    # 绘制精确度和召回率
    ax[1, 0].plot(history['val_precision'], label='Precision')
    ax[1, 0].plot(history['val_recall'], label='Recall')
    ax[1, 0].set_title('Precision & Recall')
    ax[1, 0].set_xlabel('Epoch')
    ax[1, 0].set_ylabel('Score')
    ax[1, 0].legend()
    ax[1, 0].grid(True)
    
    # 绘制F1分数
    ax[1, 1].plot(history['val_f1'], label='F1')
    ax[1, 1].set_title('F1 Score')
    ax[1, 1].set_xlabel('Epoch')
    ax[1, 1].set_ylabel('F1')
    ax[1, 1].grid(True)
    
    plt.suptitle(f'Training History: {model_name}', fontsize=16)
    plt.tight_layout()
    
    # 保存图形
    if save_path:
        plt.savefig(save_path)
        print(f"训练历史已保存至: {save_path}")
    
    plt.show()

def plot_confusion_matrix(y_true, y_pred, model_name=None, save_path=None):
    """
    绘制混淆矩阵
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        model_name: 模型名称
        save_path: 保存路径 (如果为None则不保存)
    """
    if model_name is None:
        model_name = config.MODEL_NAME
        
    if save_path is None:
        save_path = os.path.join(config.RESULTS_DIR, f"{model_name}_confusion_matrix.png")
        
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # 保存图形
    if save_path:
        plt.savefig(save_path)
        print(f"混淆矩阵已保存至: {save_path}")
    
    plt.show()

def plot_model_comparison(model_results, metric='f1', save_path=None):
    """
    比较不同模型的性能
    
    参数:
        model_results: 字典，键为模型名称，值为指标值
        metric: 要比较的指标 ('accuracy', 'precision', 'recall', 'f1')
        save_path: 保存路径 (如果为None则不保存)
    """
    if save_path is None:
        save_path = os.path.join(config.RESULTS_DIR, f"model_comparison_{metric}.png")
        
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 获取模型名称和结果
    models = list(model_results.keys())
    results = [model_results[model][metric] for model in models]
    
    # 按结果排序
    sorted_indices = np.argsort(results)[::-1]  # 降序
    sorted_models = [models[i] for i in sorted_indices]
    sorted_results = [results[i] for i in sorted_indices]
    
    # 为模型类型添加颜色
    colors = []
    for model in sorted_models:
        if model in ['alexnet', 'vgg16']:
            colors.append('royalblue')  # 纯卷积网络
        elif model in ['resnet50', 'efficientnet_b0', 'mobilenet_v2']:
            colors.append('forestgreen')  # 残差结构
        else:
            colors.append('darkorange')  # ViT
    
    # 绘制条形图
    plt.figure(figsize=(12, 6))
    bars = plt.bar(sorted_models, sorted_results, color=colors)
    
    # 添加网格线
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom')
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='royalblue', label='纯卷积网络'),
        Patch(facecolor='forestgreen', label='残差结构'),
        Patch(facecolor='darkorange', label='Vision Transformer')
    ]
    plt.legend(handles=legend_elements)
    
    # 标题和标签
    plt.title(f'Model Comparison: {metric.capitalize()}')
    plt.xlabel('Model')
    plt.ylabel(metric.capitalize())
    plt.ylim(0, 1.05)  # 设置y轴范围
    
    # 旋转x轴标签
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # 保存图形
    if save_path:
        plt.savefig(save_path)
        print(f"模型比较图已保存至: {save_path}")
    
    plt.show()
