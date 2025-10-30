"""
Benchmark Scripts
"""

from neurons.benchmarks.train_mnist import train_mnist, evaluate_mnist
from neurons.benchmarks.train_cifar10 import train_cifar10, evaluate_cifar10
from neurons.benchmarks.train_fewshot import train_fewshot
from neurons.benchmarks.train_continual import train_continual

__all__ = [
    'train_mnist',
    'evaluate_mnist',
    'train_cifar10',
    'evaluate_cifar10',
    'train_fewshot',
    'train_continual',
]
