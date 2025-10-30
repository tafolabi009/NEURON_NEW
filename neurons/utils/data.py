"""
Data handling utilities.
"""

from typing import Tuple, Optional, Union
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


def prepare_data(
    X: Union[np.ndarray, torch.Tensor],
    y: Union[np.ndarray, torch.Tensor],
    dtype: torch.dtype = torch.float32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare data for PyTorch training.
    
    Converts numpy arrays or lists to PyTorch tensors and ensures
    proper data types.
    
    Args:
        X: Input features.
        y: Target labels.
        dtype: Data type for X. Default: torch.float32
        
    Returns:
        Tuple of (X_tensor, y_tensor).
        
    Examples:
        >>> X = np.random.randn(100, 10)
        >>> y = np.random.randint(0, 2, 100)
        >>> X_tensor, y_tensor = prepare_data(X, y)
    """
    # Convert to tensors if needed
    if not isinstance(X, torch.Tensor):
        X = torch.from_numpy(np.array(X)).to(dtype)
    else:
        X = X.to(dtype)
    
    if not isinstance(y, torch.Tensor):
        y_array = np.array(y)
        # Determine appropriate dtype for y
        if np.issubdtype(y_array.dtype, np.integer):
            y = torch.from_numpy(y_array).long()
        else:
            y = torch.from_numpy(y_array).to(dtype)
    
    return X, y


def create_data_loader(
    X: Union[np.ndarray, torch.Tensor],
    y: Union[np.ndarray, torch.Tensor],
    batch_size: int = 32,
    shuffle: bool = True,
    drop_last: bool = False,
    num_workers: int = 0
) -> DataLoader:
    """
    Create a PyTorch DataLoader from data.
    
    Args:
        X: Input features.
        y: Target labels.
        batch_size: Batch size for training. Default: 32
        shuffle: Whether to shuffle the data. Default: True
        drop_last: Whether to drop the last incomplete batch. Default: False
        num_workers: Number of worker processes for data loading. Default: 0
        
    Returns:
        DataLoader instance.
        
    Examples:
        >>> X = np.random.randn(100, 10)
        >>> y = np.random.randint(0, 2, 100)
        >>> loader = create_data_loader(X, y, batch_size=16, shuffle=True)
        >>> for batch_X, batch_y in loader:
        ...     # Training code here
        ...     pass
    """
    X_tensor, y_tensor = prepare_data(X, y)
    dataset = TensorDataset(X_tensor, y_tensor)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return loader
