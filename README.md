# favicon diffusor: a high-performance browser diffusion transformer 🚀

<img src="https://doggo.ninja/shZ3Uh.gif" alt="Web DiT in Action" width="500px">

![favicon](https://github.com/user-attachments/assets/0def1766-0467-4f07-a978-51782417906e)

Ever wanted fast diffusion on device? Struggled with compatibility and libraries? Worry no more—favicon diffusor is here! Using WebGPU, its supported on almost any device (that can run chrome) and can diffuse hippos anywhere (even as a favicon)!

A quick weekend project where I hacked on a bunch of WebGPU kernels from scratch and tried to optimize them. Building on my [last "from scratch DiT"](github.com/neelr/scratche-dit) this starts at the kernel level and rewrites diffusion transformers using WSGL. A subsecond 32-step diffusion inference time allows for an awesome demo of actually _diffusing the favicon of a website realtime_ in ~0.7s with a ~11M parameter model

## ⚡ Performance

Of course.... here are the approximate numbers on an M1 Pro! Currently faster than tf.js and (of course) baseline JS—transformers.js doesn't support custom layer building, so I didn't include it.

| Implementation | Time (s) | vs Baseline | vs TensorFlow.js |
|----------------|-----------|-------------|------------------|
| Favicon Diffusor | 0.86 | 88.6% faster | 45.2% faster |
| TensorFlow.js | 1.57 | 79.3% faster | baseline |
| Baseline JS | 7.57 | baseline | 382% slower |

<img src="https://doggo.ninja/clucbV.png" alt="Performance Comparison" width="400px">

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

## 🚀 Getting Started

### Prerequisites
- A browser with WebGPU support (Chrome Canary or other modern browsers with WebGPU flags enabled)
- Node.js and npm (for development)

### Installation
1. Clone the repository:
```bash
git clone https://github.com/neelr/favicon-diffusor.git
cd favicon-diffusor
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
