# Web DiT: High-Performance Diffusion Transformers in the Browser 🚀

![Web DiT in Action](https://doggo.ninja/shZ3Uh.gif)

Web DiT is a cutting-edge implementation of Diffusion Transformers (DiT) that runs directly in your web browser using WebGPU. Our implementation achieves remarkable performance improvements over existing solutions, making real-time diffusion models accessible to anyone with a modern web browser.

## ⚡ Performance

Our implementation, dubbed "Favicon Diffusor", demonstrates exceptional performance:

| Implementation | Time (s) | vs Baseline | vs TensorFlow.js |
|----------------|-----------|-------------|------------------|
| Favicon Diffusor | 0.86 | 88.6% faster | 45.2% faster |
| TensorFlow.js | 1.57 | 79.3% faster | baseline |
| Baseline JS | 7.57 | baseline | 382% slower |

[![Performance Comparison](https://doggo.ninja/clucbV.png)](https://doggo.ninja/clucbV.png)

## 🌟 Features

- **WebGPU Acceleration**: Leverages modern WebGPU shaders for maximum performance
- **Optimized Architecture**: Implements the DiT (Diffusion Transformer) architecture with performance optimizations
- **Browser-Native**: Runs entirely in the browser - no server-side processing required
- **Custom WGSL Shaders**: Includes optimized shaders for:
  - Flash Attention
  - Layer Normalization
  - Patchification
  - Batched Matrix Multiplication
- **Real-time Processing**: Achieves sub-second inference times for image generation

## 🛠️ Technical Details

The implementation includes several key optimizations:
- Custom WGSL shaders for core operations
- Efficient memory management and tensor operations
- Optimized attention mechanisms
- Streamlined data pipelining

### Model Architecture
- Input Size: 64x64
- Patch Size: 4x4
- Hidden Dimension: 768
- Transformer Depth: 12
- Attention Head Dimension: 192
- MLP Expansion Factor: 4

## 🚀 Getting Started

### Prerequisites
- A browser with WebGPU support (Chrome Canary or other modern browsers with WebGPU flags enabled)
- Node.js and npm (for development)

### Installation
1. Clone the repository:
```bash
git clone https://github.com/neelr/favicon-diffusor.git
cd web_dit
```

1. Run a development server:
```bash
npx http-server
```

## 🔧 Development

The project structure includes:
- `dit.py` - PyTorch reference implementation
- `dit.js` - JavaScript implementation
- `shaders/` - WebGPU shader implementations
- `train.py` - Training scripts
- `compile.sh` - Compile the shaders into a single file
- Various utility and testing files

## 📚 Resources

- [WebGPU](https://webgpu.org/)
- [Stable Diffusion for Distillation!](https://github.com/CompVis/stable-diffusion)