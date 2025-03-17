import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import config

def plot_training_history(history, model_name=None, save_path=None):
    """
    Plot training history
    
    Parameters:
        history: Dictionary containing training history
        model_name: Model name
        save_path: Save path (if None, not saved)
    """
    if model_name is None:
        model_name = config.MODEL_NAME
        
    if save_path is None:
        save_path = os.path.join(config.RESULTS_DIR, f"{model_name}_history.png")
        
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Create figure
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot loss
    ax[0, 0].plot(history['train_loss'], label='Train')
    ax[0, 0].plot(history['val_loss'], label='Validation')
    ax[0, 0].set_title('Loss')
    ax[0, 0].set_xlabel('Epoch')
    ax[0, 0].set_ylabel('Loss')
    ax[0, 0].legend()
    ax[0, 0].grid(True)
    
    # Plot accuracy
    ax[0, 1].plot(history['val_accuracy'], label='Accuracy')
    ax[0, 1].set_title('Accuracy')
    ax[0, 1].set_xlabel('Epoch')
    ax[0, 1].set_ylabel('Accuracy')
    ax[0, 1].grid(True)
    
    # Plot precision and recall
    ax[1, 0].plot(history['val_precision'], label='Precision')
    ax[1, 0].plot(history['val_recall'], label='Recall')
    ax[1, 0].set_title('Precision & Recall')
    ax[1, 0].set_xlabel('Epoch')
    ax[1, 0].set_ylabel('Score')
    ax[1, 0].legend()
    ax[1, 0].grid(True)
    
    # Plot F1 score
    ax[1, 1].plot(history['val_f1'], label='F1')
    ax[1, 1].set_title('F1 Score')
    ax[1, 1].set_xlabel('Epoch')
    ax[1, 1].set_ylabel('F1')
    ax[1, 1].grid(True)
    
    plt.suptitle(f'Training History: {model_name}', fontsize=16)
    plt.tight_layout()
    
    # Save figure
    if save_path:
        plt.savefig(save_path)
        print(f"Training history saved to: {save_path}")
    
    plt.show()

def plot_confusion_matrix(y_true, y_pred, model_name=None, save_path=None):
    """
    Plot confusion matrix
    
    Parameters:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Model name
        save_path: Save path (if None, not saved)
    """
    if model_name is None:
        model_name = config.MODEL_NAME
        
    if save_path is None:
        save_path = os.path.join(config.RESULTS_DIR, f"{model_name}_confusion_matrix.png")
        
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Save figure
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()

def plot_model_comparison(model_results, metric='f1', save_path=None):
    """
    Compare performance of different models
    
    Parameters:
        model_results: Dictionary, keys are model names, values are metric values
        metric: Metric to compare ('accuracy', 'precision', 'recall', 'f1')
        save_path: Save path (if None, not saved)
    """
    if save_path is None:
        save_path = os.path.join(config.RESULTS_DIR, f"model_comparison_{metric}.png")
        
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Get model names and results
    models = list(model_results.keys())
    results = [model_results[model][metric] for model in models]
    
    # Sort by results
    sorted_indices = np.argsort(results)[::-1]  # Descending
    sorted_models = [models[i] for i in sorted_indices]
    sorted_results = [results[i] for i in sorted_indices]
    
    # Add colors for model types
    colors = []
    for model in sorted_models:
        if model in ['alexnet', 'vgg16']:
            colors.append('royalblue')  # Pure convolutional networks
        elif model in ['resnet50', 'efficientnet_b0', 'mobilenet_v2']:
            colors.append('forestgreen')  # Residual networks
        else:
            colors.append('darkorange')  # Vision Transformer
    
    # Plot bar chart
    plt.figure(figsize=(12, 6))
    bars = plt.bar(sorted_models, sorted_results, color=colors)
    
    # Add grid lines
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='royalblue', label='Pure Convolutional Networks'),
        Patch(facecolor='forestgreen', label='Residual Networks'),
        Patch(facecolor='darkorange', label='Vision Transformer')
    ]
    plt.legend(handles=legend_elements)
    
    # Title and labels
    plt.title(f'Model Comparison: {metric.capitalize()}')
    plt.xlabel('Model')
    plt.ylabel(metric.capitalize())
    plt.ylim(0, 1.05)  # Set y-axis range
    
    # Rotate x-axis labels
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save figure
    if save_path:
        plt.savefig(save_path)
        print(f"Model comparison chart saved to: {save_path}")
    
    plt.show()