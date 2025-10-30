"""
Custom Triton/CUDA Kernels for NEURONSv2
High-performance kernels for critical operations
"""

from .triton_kernels import (
    dendritic_forward_kernel,
    spectral_compression_kernel,
    sparse_attention_kernel,
    has_triton
)

__all__ = [
    'dendritic_forward_kernel',
    'spectral_compression_kernel', 
    'sparse_attention_kernel',
    'has_triton'
]
