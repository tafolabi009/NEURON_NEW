"""
Metrics calculation utilities.
"""

from typing import Dict
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def calculate_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> float:
    """
    Calculate accuracy.
    
    Args:
        predictions: Predicted class indices or probabilities.
        targets: Ground truth class indices.
        
    Returns:
        Accuracy as a float between 0 and 1.
        
    Examples:
        >>> preds = torch.tensor([0, 1, 2, 1])
        >>> targets = torch.tensor([0, 1, 1, 1])
        >>> acc = calculate_accuracy(preds, targets)
    """
    # If predictions are probabilities, get class indices
    if predictions.dim() > 1 and predictions.size(1) > 1:
        predictions = torch.argmax(predictions, dim=1)
    
    correct = (predictions == targets).sum().item()
    total = targets.size(0)
    
    return correct / total


def calculate_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    average: str = 'weighted'
) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.
    
    Args:
        predictions: Predicted class indices or probabilities.
        targets: Ground truth class indices.
        average: Averaging strategy for multi-class metrics.
            Options: 'micro', 'macro', 'weighted', 'binary'. Default: 'weighted'
        
    Returns:
        Dictionary containing accuracy, precision, recall, and F1 score.
        
    Examples:
        >>> preds = torch.tensor([0, 1, 2, 1])
        >>> targets = torch.tensor([0, 1, 1, 1])
        >>> metrics = calculate_metrics(preds, targets)
        >>> print(f"Accuracy: {metrics['accuracy']:.4f}")
    """
    # Convert to numpy and get class indices
    if predictions.dim() > 1 and predictions.size(1) > 1:
        predictions = torch.argmax(predictions, dim=1)
    
    preds_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    # Handle binary vs multiclass
    if len(np.unique(targets_np)) == 2 and average == 'binary':
        avg = 'binary'
    else:
        avg = average
    
    metrics = {
        'accuracy': accuracy_score(targets_np, preds_np),
        'precision': precision_score(targets_np, preds_np, average=avg, zero_division=0),
        'recall': recall_score(targets_np, preds_np, average=avg, zero_division=0),
        'f1_score': f1_score(targets_np, preds_np, average=avg, zero_division=0),
    }
    
    return metrics
