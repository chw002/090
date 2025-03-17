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
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='ISIC 2020 Skin Lesion Classification')
    
    # Training parameters
    parser.add_argument('--model', type=str, default=config.MODEL_NAME,
                        choices=ModelFactory.get_model_names(),
                        help='Model name')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=config.NUM_EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=config.WEIGHT_DECAY,
                        help='Weight decay')
    parser.add_argument('--image_size', type=int, default=config.IMAGE_SIZE,
                        help='Image size')
    
    # Mode parameters
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'evaluate', 'predict', 'compare_all'],
                        help='Running mode')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Model checkpoint path')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Whether to use wandb')
    
    return parser.parse_args()

def train(args):
    """Train model"""
    config.BATCH_SIZE = args.batch_size
    config.IMAGE_SIZE = args.image_size
    config.USE_WANDB = args.use_wandb
    
    # Get data loaders
    train_loader, val_loader, _ = get_data_loaders()
    
    # Create model
    model = ModelFactory.create_model(args.model)

    # model = model.to(device) 
    model = model.to(config.DEVICE)
    
    # Train model
    model, history = train_model(
        model, 
        train_loader, 
        val_loader, 
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        device=config.DEVICE,
        model_name=args.model,
        use_wandb=args.use_wandb
    )
    
    # Plot training history
    plot_training_history(history, args.model)
    
    return model, history

def evaluate(args):
    """Evaluate model"""
    # Update configuration
    config.BATCH_SIZE = args.batch_size
    config.IMAGE_SIZE = args.image_size
    
    # Get data loaders
    _, val_loader, _ = get_data_loaders()
    
    # Create model
    model = ModelFactory.create_model(args.model)
    
    # Load checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"{args.model}_best.pth")
    
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    
    # Evaluate model
    criterion = nn.BCELoss()
    loss, metrics = evaluate_model(model, val_loader, criterion, config.DEVICE)
    
    # Print results
    print(f"Evaluation results:")
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    
    return metrics

def compare_all_models(args):
    """Compare all models"""
    # Update configuration
    config.BATCH_SIZE = args.batch_size
    config.IMAGE_SIZE = args.image_size
    
    # Get model list
    model_names = ModelFactory.get_model_names()
    
    # Results dictionary
    results = {}
    
    for model_name in model_names:
        print(f"\nEvaluating model: {model_name}")
        
        # Create model
        model = ModelFactory.create_model(model_name)
        
        # Load checkpoint
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"{model_name}_best.pth")
        
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path))
            print(f"Loaded checkpoint: {checkpoint_path}")
            
            # Get data loaders
            _, val_loader, _ = get_data_loaders()
            
            # Evaluate model
            criterion = nn.BCELoss()
            loss, metrics = evaluate_model(model, val_loader, criterion, config.DEVICE)
            
            # Store results
            results[model_name] = metrics
            
            # Print results
            print(f"Loss: {loss:.4f}")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1: {metrics['f1']:.4f}")
        else:
            print(f"Checkpoint not found: {checkpoint_path}")
    
    # Compare models
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        plot_model_comparison(results, metric)
    
    return results

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Ensure directories exist
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    
    # Set device
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    config.DEVICE = device
    print(f"Using device: {device}")
    
    # Run according to mode
    if args.mode == 'train':
        model, history = train(args)
    elif args.mode == 'evaluate':
        metrics = evaluate(args)
    elif args.mode == 'compare_all':
        results = compare_all_models(args)
    else:
        print(f"Unsupported mode: {args.mode}")

if __name__ == '__main__':
    main()