import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def evaluate_model(model, data_loader, criterion=None, device='cuda'):
    """
    Evaluate model performance and search for optimal threshold
    
    Parameters:
        model: PyTorch model
        data_loader: Data loader
        criterion: Loss function (if None, loss is not calculated)
        device: Device ('cuda' or 'cpu')
        
    Returns:
        avg_loss: Average loss
        metrics: Dictionary containing various metrics
    """
    model.eval()
    all_outputs = []  # Store raw outputs (not binarized)
    all_targets = []
    losses = []
    
    with torch.no_grad():
        for data, target in tqdm(data_loader, desc="Evaluating"):
            if isinstance(target, str):  # Test set, target is image name
                continue
                
            data, target = data.to(device), target.float().to(device).unsqueeze(1)
            output = model(data)
            
            # Calculate loss
            if criterion is not None:
                loss = criterion(output, target)
                losses.append(loss.item())
            
            # Collect raw outputs and targets
            all_outputs.extend(output.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Convert to numpy arrays
    all_outputs = np.array(all_outputs).flatten()
    all_targets = np.array(all_targets).flatten()
    
    # Try different thresholds, find best F1 score
    best_f1 = 0
    best_threshold = 0.5
    best_metrics = None
    
    # Search for optimal threshold
    print("Searching for optimal classification threshold...")
    for threshold in np.arange(0.1, 0.9, 0.05):
        # Apply sigmoid to outputs (since we're using BCEWithLogitsLoss)
        probs = 1 / (1 + np.exp(-all_outputs))
            
        # Use current threshold for predictions
        preds = (probs > threshold).astype(float)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(all_targets, preds),
            'precision': precision_score(all_targets, preds, zero_division=0),
            'recall': recall_score(all_targets, preds, zero_division=0),
            'f1': f1_score(all_targets, preds, zero_division=0),
            'threshold': threshold
        }
        
        print(f"Threshold {threshold:.2f}: Accuracy={metrics['accuracy']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")
        
        # Update best F1 score
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_threshold = threshold
            best_metrics = metrics
    
    print(f"Best threshold: {best_threshold:.2f}, F1: {best_f1:.4f}")
    
    # Calculate and print confusion matrix
    probs = 1 / (1 + np.exp(-all_outputs))
    best_preds = (probs > best_threshold).astype(float)
    cm = confusion_matrix(all_targets, best_preds)
    print(f"Confusion matrix:\n{cm}")
    print(f"True Negative (TN): {cm[0,0]}, False Positive (FP): {cm[0,1]}")
    print(f"False Negative (FN): {cm[1,0]}, True Positive (TP): {cm[1,1]}")
    
    # Calculate average loss
    avg_loss = np.mean(losses) if losses else 0
    
    return avg_loss, best_metrics

def predict(model, data_loader, device='cuda'):
    """
    Use model to make predictions
    
    Parameters:
        model: PyTorch model
        data_loader: Data loader
        device: Device ('cuda' or 'cpu')
        
    Returns:
        predictions: Prediction results
        image_names: Image names (if available)
    """
    model.eval()
    predictions = []
    image_names = []
    
    with torch.no_grad():
        for data, target in tqdm(data_loader, desc="Predicting"):
            # Check if target is string (test set)
            if isinstance(target, torch.Tensor):
                batch_images = None
            else:
                batch_images = target
                
            data = data.to(device)
            outputs = model(data)
            
            # Get predictions
            preds = (outputs > 0.5).float().cpu().numpy()
            predictions.extend(preds)
            
            # Get image names (if available)
            if batch_images is not None:
                image_names.extend(batch_images)
    
    return np.array(predictions).flatten(), image_names