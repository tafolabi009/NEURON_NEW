# Resonance Neural Networks - Getting Started

## Installation

```bash
# Clone the repository
git clone https://github.com/tafolabi009/NEURON_NEW.git
cd NEURON_NEW

# Install in development mode
pip install -e .

# Or install from PyPI (once published)
pip install resonance-neural-networks
```

## Quick Start

### Basic Usage

```python
import torch
from resonance_nn import ResonanceNet

# Create model
model = ResonanceNet(
    input_dim=512,
    num_frequencies=64,
    hidden_dim=256,
    num_layers=4,
    holographic_capacity=1000,
    dropout=0.1,
)

# Process sequence
x = torch.randn(32, 128, 512)  # (batch, seq_len, dim)
output = model(x)
```

### Training

```python
from resonance_nn.training import ResonanceTrainer, create_criterion
from torch.utils.data import DataLoader

# Create trainer
trainer = ResonanceTrainer(
    model=model,
    learning_rate=1e-4,
    gradient_clip=1.0,
)

# Loss function
criterion = create_criterion('regression')

# Training loop
for epoch in range(num_epochs):
    loss = trainer.train_epoch(train_loader, criterion, epoch)
    val_metrics = trainer.validate(val_loader, criterion)
```

### Holographic Memory

```python
from resonance_nn.layers import HolographicMemory

# Create memory
memory = HolographicMemory(
    pattern_dim=512,
    hologram_dim=1024,
    capacity=1000,
)

# Encode patterns
pattern = torch.randn(512)
memory.encode(pattern)

# Reconstruct
reconstructed = memory.reconstruct()

# Check fidelity
fidelity = memory.get_reconstruction_fidelity(pattern)
print(f"Reconstruction fidelity: {fidelity:.4f}")
```

## Examples

### Run Quick Start
```bash
python examples/quickstart.py
```

### Verify Complexity Claims
```bash
python examples/verify_complexity.py
```

### Test Holographic Memory
```bash
python examples/holographic_demo.py
```

### Verify Gradient Stability
```bash
python examples/gradient_stability.py
```

### Train on Sequence Modeling
```bash
python examples/sequence_modeling.py
```

### Run All Examples
```bash
python run.py --mode all
```

## Architecture Components

### 1. Resonance Layer

The core frequency-domain processing layer:

```python
from resonance_nn.layers import ResonanceLayer

layer = ResonanceLayer(
    input_dim=512,
    num_frequencies=64,
    dropout=0.1,
)

# Process input
output = layer(input_sequence)

# Get gradient statistics
stats = layer.get_gradient_stats()
```

**Features:**
- O(n log n) complexity via FFT
- Stable gradient computation
- Complex-valued weights with magnitude/phase parameterization
- Cross-frequency interference modeling

### 2. Holographic Memory

Information storage through interference patterns:

```python
from resonance_nn.layers import HolographicMemory

memory = HolographicMemory(
    pattern_dim=512,
    hologram_dim=1024,
    capacity=1000,
)

# Store patterns
for pattern in patterns:
    memory.encode(pattern)

# Retrieve
retrieved = memory.reconstruct(query)

# Check capacity
utilization = memory.get_capacity_utilization()
theoretical_capacity = memory.get_theoretical_capacity()
```

**Features:**
- Holographic interference encoding
- Associative recall
- Theoretical capacity guarantees
- Multi-pattern superposition

### 3. Complete Models

#### ResonanceNet (General Purpose)
```python
from resonance_nn import ResonanceNet

model = ResonanceNet(
    input_dim=512,
    num_frequencies=64,
    hidden_dim=256,
    num_layers=4,
    holographic_capacity=1000,
)
```

#### ResonanceEncoder (Representation Learning)
```python
from resonance_nn.models import ResonanceEncoder

encoder = ResonanceEncoder(
    input_dim=512,
    num_frequencies=64,
    output_dim=128,
    num_layers=4,
)
```

#### ResonanceAutoencoder
```python
from resonance_nn.models import ResonanceAutoencoder

autoencoder = ResonanceAutoencoder(
    input_dim=512,
    latent_dim=128,
    num_frequencies=64,
    num_layers=4,
)
```

#### ResonanceClassifier
```python
from resonance_nn.models import ResonanceClassifier

classifier = ResonanceClassifier(
    input_dim=512,
    num_classes=10,
    num_frequencies=64,
    num_layers=4,
)
```

## Benchmarking

### Complexity Verification

```python
from resonance_nn.benchmark import ComplexityBenchmark

benchmark = ComplexityBenchmark()
results = benchmark.run(
    sequence_lengths=[64, 128, 256, 512, 1024],
    input_dim=512,
    num_frequencies=64,
)

# Analyze
analysis = benchmark.analyze_complexity()
print(f"R² with O(n log n): {analysis['r_squared_nlogn']:.4f}")

# Plot
benchmark.plot_results('complexity.png')
```

### Holographic Memory Testing

```python
from resonance_nn.benchmark import HolographicMemoryBenchmark

benchmark = HolographicMemoryBenchmark()
results = benchmark.test_reconstruction_fidelity(
    pattern_dim=512,
    num_patterns=[1, 10, 50, 100, 200],
)

benchmark.plot_results('memory.png')
```

### Gradient Stability

```python
from resonance_nn.benchmark import GradientStabilityTest

test = GradientStabilityTest()
results = test.test_gradient_stability(
    num_iterations=1000,
    input_dim=512,
)

print(f"Max gradient norm: {results['max_gradient_norm']:.4f}")
print(f"Gradient explosions: {results['gradient_explosions']}")
```

## Testing

Run the test suite:

```bash
# Install test dependencies
pip install pytest

# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_resonance.py -v

# Run specific test
pytest tests/test_resonance.py::TestResonanceLayer::test_forward_shape -v
```

## Performance Tips

1. **Use GPU**: The architecture benefits significantly from GPU acceleration
   ```python
   model = ResonanceNet(...).cuda()
   ```

2. **Batch Size**: Larger batch sizes improve GPU utilization
   ```python
   # Good for GPU
   batch_size = 64
   ```

3. **Frequency Count**: Balance between expressiveness and computation
   ```python
   # More frequencies = more expressive but slower
   num_frequencies = 64  # Good default
   ```

4. **Memory Capacity**: Set based on expected pattern count
   ```python
   # Set capacity to expected number of patterns
   holographic_capacity = 1000
   ```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{afolabi2025resonance,
  title={Resonance Neural Networks: Frequency-Domain Information Processing 
         with Holographic Memory and Provable Efficiency Guarantees},
  author={Afolabi, Oluwatosin A.},
  journal={Genovo Technologies},
  year={2025}
}
```

## License

MIT License - see LICENSE file for details.

## Support

For questions and issues:
- GitHub Issues: https://github.com/tafolabi009/NEURON_NEW/issues
- Email: afolabi@genovotech.com

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Acknowledgments

This implementation is based on the theoretical work:
- Stable frequency-domain gradient computation (Theorem 1)
- Holographic information capacity bounds (Theorem 2)
- O(n log n) complexity analysis (Theorem 3)
- Information conservation proofs (Theorem 4)
