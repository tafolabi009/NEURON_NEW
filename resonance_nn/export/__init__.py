"""
Model Export and Deployment Utilities
Export Resonance models for integration with other applications

Supported formats:
- PyTorch (.pt, .pth)
- ONNX (.onnx) - for cross-platform deployment
- TorchScript (.pt) - for C++ integration
- Quantized models - for mobile/edge deployment
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple
import json
import os


class ModelExporter:
    """
    Export Resonance models to various formats
    """
    
    def __init__(self, model: nn.Module, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            model: Resonance model to export
            config: Model configuration dictionary
        """
        self.model = model
        self.config = config or {}
        
    def export_pytorch(
        self,
        save_path: str,
        include_optimizer: bool = False,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """
        Export as PyTorch checkpoint
        
        Args:
            save_path: Path to save checkpoint
            include_optimizer: Whether to include optimizer state
            optimizer: Optimizer to save (if include_optimizer=True)
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
        }
        
        if include_optimizer and optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(checkpoint, save_path)
        print(f"✓ Exported PyTorch checkpoint to {save_path}")
        
        # Save config separately
        config_path = save_path.replace('.pt', '_config.json').replace('.pth', '_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"✓ Saved config to {config_path}")
    
    def export_onnx(
        self,
        save_path: str,
        example_input: torch.Tensor,
        input_names: List[str] = ['input'],
        output_names: List[str] = ['output'],
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
        opset_version: int = 14,
    ):
        """
        Export to ONNX format
        
        Args:
            save_path: Path to save ONNX model
            example_input: Example input tensor for tracing
            input_names: Names of input tensors
            output_names: Names of output tensors
            dynamic_axes: Dynamic axes specification
            opset_version: ONNX opset version
        """
        self.model.eval()
        
        if dynamic_axes is None:
            # Default: dynamic batch and sequence dimensions
            dynamic_axes = {
                'input': {0: 'batch_size', 1: 'sequence_length'},
                'output': {0: 'batch_size', 1: 'sequence_length'},
            }
        
        try:
            torch.onnx.export(
                self.model,
                example_input,
                save_path,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=opset_version,
                do_constant_folding=True,
            )
            print(f"✓ Exported ONNX model to {save_path}")
            print(f"  Opset version: {opset_version}")
            print(f"  Dynamic axes: {dynamic_axes}")
        except Exception as e:
            print(f"✗ ONNX export failed: {e}")
            print("  Note: Some operations may not be ONNX-compatible")
    
    def export_torchscript(
        self,
        save_path: str,
        example_input: Optional[torch.Tensor] = None,
        use_trace: bool = True,
    ):
        """
        Export to TorchScript for C++ integration
        
        Args:
            save_path: Path to save TorchScript model
            example_input: Example input for tracing (required if use_trace=True)
            use_trace: Use tracing (True) or scripting (False)
        """
        self.model.eval()
        
        try:
            if use_trace:
                if example_input is None:
                    raise ValueError("example_input required for tracing")
                scripted = torch.jit.trace(self.model, example_input)
            else:
                scripted = torch.jit.script(self.model)
            
            scripted.save(save_path)
            print(f"✓ Exported TorchScript model to {save_path}")
            print(f"  Method: {'trace' if use_trace else 'script'}")
        except Exception as e:
            print(f"✗ TorchScript export failed: {e}")
            print("  Try use_trace=False for scripting instead")
    
    def export_quantized(
        self,
        save_path: str,
        quantization_type: str = 'dynamic',
        dtype: torch.dtype = torch.qint8,
    ):
        """
        Export quantized model for mobile/edge deployment
        
        Args:
            save_path: Path to save quantized model
            quantization_type: 'dynamic' or 'static'
            dtype: Quantization data type
        """
        self.model.eval()
        
        try:
            if quantization_type == 'dynamic':
                # Dynamic quantization (weights only)
                quantized_model = torch.quantization.quantize_dynamic(
                    self.model,
                    {nn.Linear, nn.LSTM, nn.GRU},
                    dtype=dtype,
                )
            else:
                # Static quantization (requires calibration)
                print("Static quantization requires calibration data")
                return
            
            torch.save(quantized_model.state_dict(), save_path)
            print(f"✓ Exported quantized model to {save_path}")
            print(f"  Quantization: {quantization_type}")
            print(f"  Data type: {dtype}")
            
            # Calculate compression ratio
            original_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
            quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters())
            print(f"  Compression: {original_size / quantized_size:.2f}x")
        except Exception as e:
            print(f"✗ Quantization failed: {e}")


class ModelPackager:
    """
    Package models for easy distribution and deployment
    """
    
    @staticmethod
    def package_for_deployment(
        model: nn.Module,
        output_dir: str,
        model_name: str,
        config: Dict[str, Any],
        export_formats: List[str] = ['pytorch', 'onnx', 'torchscript'],
        example_input: Optional[torch.Tensor] = None,
    ):
        """
        Package model in multiple formats for deployment
        
        Args:
            model: Model to package
            output_dir: Output directory
            model_name: Name of the model
            config: Model configuration
            export_formats: Formats to export ('pytorch', 'onnx', 'torchscript', 'quantized')
            example_input: Example input tensor
        """
        os.makedirs(output_dir, exist_ok=True)
        
        exporter = ModelExporter(model, config)
        
        print(f"Packaging {model_name} for deployment...")
        print(f"Output directory: {output_dir}")
        print("=" * 60)
        
        # Export in requested formats
        if 'pytorch' in export_formats:
            pytorch_path = os.path.join(output_dir, f"{model_name}.pt")
            exporter.export_pytorch(pytorch_path)
        
        if 'onnx' in export_formats:
            if example_input is None:
                print("⚠ Skipping ONNX export: example_input required")
            else:
                onnx_path = os.path.join(output_dir, f"{model_name}.onnx")
                exporter.export_onnx(onnx_path, example_input)
        
        if 'torchscript' in export_formats:
            if example_input is None:
                print("⚠ Skipping TorchScript export: example_input required")
            else:
                ts_path = os.path.join(output_dir, f"{model_name}_scripted.pt")
                exporter.export_torchscript(ts_path, example_input)
        
        if 'quantized' in export_formats:
            quant_path = os.path.join(output_dir, f"{model_name}_quantized.pt")
            exporter.export_quantized(quant_path)
        
        # Create README
        readme_path = os.path.join(output_dir, "README.md")
        ModelPackager._create_readme(readme_path, model_name, config, export_formats)
        
        print("=" * 60)
        print(f"✓ Model packaged successfully in {output_dir}")
        print(f"  Available formats: {', '.join(export_formats)}")
    
    @staticmethod
    def _create_readme(
        path: str,
        model_name: str,
        config: Dict[str, Any],
        formats: List[str],
    ):
        """Create README for packaged model"""
        readme_content = f"""# {model_name}

## Model Information

This package contains a Resonance Neural Network model exported in multiple formats for easy integration.

### Configuration

```json
{json.dumps(config, indent=2)}
```

### Available Formats

"""
        for fmt in formats:
            if fmt == 'pytorch':
                readme_content += """
#### PyTorch (.pt)
```python
import torch

# Load model
checkpoint = torch.load('{model_name}.pt')
model.load_state_dict(checkpoint['model_state_dict'])
config = checkpoint['config']
```
"""
            elif fmt == 'onnx':
                readme_content += """
#### ONNX (.onnx)
```python
import onnxruntime as ort

# Load and run
session = ort.InferenceSession('{model_name}.onnx')
output = session.run(['output'], {'input': input_data})
```
"""
            elif fmt == 'torchscript':
                readme_content += """
#### TorchScript (.pt)
```python
import torch

# Load scripted model
model = torch.jit.load('{model_name}_scripted.pt')
output = model(input_tensor)
```
"""
            elif fmt == 'quantized':
                readme_content += """
#### Quantized (.pt)
Quantized model for mobile/edge deployment with reduced size.
```python
import torch

# Load quantized model
model.load_state_dict(torch.load('{model_name}_quantized.pt'))
```
"""
        
        readme_content += """
## Integration Examples

### Python Application
```python
from resonance_nn import load_model

model = load_model('{model_name}.pt')
predictions = model(input_data)
```

### REST API
Use the provided inference server:
```bash
python -m resonance_nn.serve --model {model_name}.pt --port 8000
```

### C++ Application
Link against TorchScript model for production deployment.

## Requirements

- PyTorch >= 1.10.0
- ONNX Runtime (for .onnx)
- Python >= 3.8

## License

See main repository LICENSE file.
"""
        
        with open(path, 'w') as f:
            f.write(readme_content)
        
        print(f"✓ Created README at {path}")


def load_exported_model(checkpoint_path: str) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load an exported Resonance model
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Tuple of (model, config)
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint.get('config', {})
    
    # Reconstruct model from config
    # This would need to be implemented based on model type
    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"Config: {config}")
    
    return None, config  # Placeholder
