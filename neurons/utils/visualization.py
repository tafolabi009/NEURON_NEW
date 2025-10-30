"""
Visualization utilities.
"""

from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: tuple = (12, 4)
) -> None:
    """
    Plot training history.
    
    Args:
        history: Dictionary containing training metrics over epochs.
        save_path: Path to save the figure. If None, displays the plot.
        figsize: Figure size as (width, height). Default: (12, 4)
        
    Examples:
        >>> history = trainer.fit(train_loader, val_loader, epochs=50)
        >>> plot_training_history(history, save_path='training_history.png')
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot loss
    if 'train_loss' in history:
        axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy (if available)
    if 'train_accuracy' in history:
        axes[1].plot(history['train_accuracy'], label='Train Acc', marker='o')
    if 'val_accuracy' in history:
        axes[1].plot(history['val_accuracy'], label='Val Acc', marker='s')
    if 'train_accuracy' in history or 'val_accuracy' in history:
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No accuracy data available',
                    ha='center', va='center', fontsize=12)
        axes[1].set_xticks([])
        axes[1].set_yticks([])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = False,
    save_path: Optional[str] = None,
    figsize: tuple = (8, 6)
) -> None:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        class_names: List of class names for display. Default: None
        normalize: Whether to normalize the confusion matrix. Default: False
        save_path: Path to save the figure. If None, displays the plot.
        figsize: Figure size as (width, height). Default: (8, 6)
        
    Examples:
        >>> from neurons.utils import plot_confusion_matrix
        >>> plot_confusion_matrix(
        ...     y_true, y_pred,
        ...     class_names=['Class A', 'Class B'],
        ...     save_path='confusion_matrix.png'
        ... )
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names if class_names else 'auto',
        yticklabels=class_names if class_names else 'auto',
        cbar=True
    )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
