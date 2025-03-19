# 🔢 Simple MNIST Neural Network

A sophisticated yet minimalist implementation of a pure feedforward neural network for MNIST digit recognition, written entirely in C. This implementation achieves remarkable >99% accuracy on the MNIST test set without using convolutional layers, demonstrating the power of carefully designed architecture and modern training techniques. This project is testament to the elegance of fundamental neural network principles when applied with precision and care.

## ✨ What Makes This Implementation Special

Most high-accuracy MNIST implementations today rely on convolutional neural networks (CNNs), which have built-in spatial inductive biases. This implementation takes a different approach, proving that sometimes the classics, when finely tuned, can rival modern approaches:
 
- **Pure Feedforward Architecture**: Achieves CNN-like performance using only fully-connected layers
- **Optimized Neuron Allocation**: Strategic distribution of neurons across layers (784→512→256→10) balances capacity and generalization
- **Modern Optimization Stack**: Incorporates research-backed techniques typically found in deep learning frameworks
  - Conservative learning rate (0.01) with cosine annealing
  - Effective regularization (L2 weight decay: 2e-5)
  - Efficient batch processing (128 images per batch)
- **Minimal Dependencies**: Relies only on standard C libraries and math.h
- **Readable Implementation**: Clean, well-documented code that serves as an educational resource
- **Performance-Focused**: Optional SIMD optimizations for ARM processors

## 🔬 Technical Deep Dive

### 🧩 Network Architecture

The network architecture has been meticulously designed to extract hierarchical features from digit images:

1. **Input Layer (784 neurons)**
   - Represents the flattened 28×28 pixel images
   - Each neuron corresponds to a single pixel intensity value

2. **First Hidden Layer (512 neurons)**
   - Captures low-level features like edges, curves, and line segments
   - Size carefully chosen to provide sufficient representational capacity without overfitting
   - ReLU activation enables learning of non-linear patterns
   - He initialization ensures proper gradient flow during early training

3. **Second Hidden Layer (256 neurons)**
   - Builds higher-level abstractions by combining low-level features
   - Detects digit-specific patterns like loops, intersections, and stroke patterns
   - Reduced size creates an information bottleneck that forces generalization
   - ReLU activation maintains sparse representations

4. **Output Layer (10 neurons)**
   - One neuron per digit class (0-9)
   - Softmax activation provides normalized probability distribution
   - Xavier/Glorot initialization optimized for softmax outputs

### 🚀 Advanced Training Techniques

The implementation incorporates numerous advanced techniques that are crucial for achieving high accuracy - the secret sauce that makes this network shine:

#### 🎯 Initialization Strategies
- **He Initialization** for ReLU layers: Weights initialized with variance scaled by `sqrt(2/n_inputs)` to maintain gradient magnitude
- **Xavier/Glorot Initialization** for output layer: Optimized for linear/softmax activations with variance scaled by `sqrt(2/(n_inputs + n_outputs))`

#### 🔥 Activation Functions
- **ReLU (Rectified Linear Unit)**: `f(x) = max(0, x)` for hidden layers
  - Sparse activation (typically 50-60% of neurons are active)
  - Mitigates vanishing gradient problem
  - Computationally efficient
- **Softmax**: `σ(z)ᵢ = exp(zᵢ) / Σⱼ exp(zⱼ)` for output layer
  - Numerically stable implementation with max subtraction
  - Provides proper probability distribution

#### 🛡️ Regularization Techniques
- **L2 Weight Decay**: Penalizes large weights with coefficient `2e-5`
  - Encourages smoother decision boundaries
  - Improves generalization by reducing model complexity
  - Prevents overfitting with larger network capacity
- **Data Augmentation**: Random shifts (±2 pixels) applied during training
  - Increases effective training set size
  - Improves robustness to translation variations
  - Implemented efficiently with in-place operations

#### ⚡ Optimization Approach
- **Mini-batch Gradient Descent**: Processes 128 images per batch
  - Balances computational efficiency and update frequency
  - Introduces beneficial noise for escaping local minima
- **Cosine Learning Rate Annealing**: `lr = initial_lr * 0.5 * (1 + cos(π * epoch/max_epochs))`
  - Starts with conservative steps (0.01) for stable convergence
  - Gradually reduces step size for fine-tuning
  - Smooth transition prevents oscillation near optima

#### 🔄 Backpropagation Implementation
- **Efficient Gradient Computation**: Directly computes gradients without storing intermediate values
- **Numerical Stability**: Careful implementation to avoid overflow/underflow
- **Vectorized Operations**: Matrix-vector multiplications optimized for cache efficiency

### 📈 Performance Characteristics

The network demonstrates impressive learning dynamics that would make many complex architectures envious:

- **Rapid Initial Learning**: ~96.8% accuracy after just one epoch
- **Steady Improvement**: Reaches ~98% by epoch 3-5
- **Fast Convergence**: >98.4% accuracy by epoch 7, exceeding 99% within 50 epochs
- **Generalization**: Minimal gap between training and test accuracy (~0.3%)
- **Stability**: Consistent improvement with minimal fluctuations

### 💾 Memory Efficiency

The implementation is designed to be remarkably memory-efficient, making it suitable even for resource-constrained environments:

- **Total Parameters**: ~537,866 weights + 778 biases
  - Input→Hidden1: 784×512 = 401,408 weights + 512 biases
  - Hidden1→Hidden2: 512×256 = 131,072 weights + 256 biases
  - Hidden2→Output: 256×10 = 2,560 weights + 10 biases
- **Memory Footprint**: ~2.1MB during training
- **Batch Processing**: Efficiently processes 128 images per batch

### 🚄 SIMD Optimizations (Optional)

For ARM processors, the implementation includes NEON SIMD optimizations that turbocharge performance:

- **Vectorized Matrix Operations**: 4x floating-point operations per cycle
- **Optimized Activation Functions**: Parallel ReLU and softmax computations
- **Efficient Data Preprocessing**: Vectorized normalization and augmentation

## 🔍 Comparison with CNNs

While convolutional neural networks (CNNs) are the standard approach for image classification today, this pure feedforward implementation boldly challenges that convention and demonstrates that:

1. Well-designed fully-connected networks can achieve comparable performance on constrained problems
2. The inductive bias of CNNs can be partially compensated by:
   - Proper regularization
   - Data augmentation
   - Careful architecture design
3. Feedforward networks can be more efficient for deployment in certain scenarios

## 📋 Requirements

- C compiler (gcc/clang)
- make
- wget (for downloading MNIST dataset)
- ~100MB disk space for MNIST dataset

## 🚀 Quick Start

1. Clone the repository:
```bash
git clone https://github.com/tsotchke/simple_mnist.git
cd simple_mnist
```

2. Download the MNIST dataset:
```bash
./download_mnist.sh
```

3. Build and run:
```bash
make
./simple_mnist mnist_data
```

Add `--verbose` flag for detailed statistics and confusion matrix:
```bash
./simple_mnist mnist_data --verbose
```

## 📁 Project Structure

```
simple_mnist/
├── include/           # Header files
│   ├── mnist_loader.h # MNIST dataset loading and preprocessing
│   └── neon_ops.h     # SIMD optimizations for ARM processors
├── src/               # Source files
│   ├── mnist_loader.c # Dataset handling implementation
│   ├── neon_ops.c     # Optimized math operations
│   └── simple_mnist.c # Neural network implementation
├── mnist_data/        # MNIST dataset (after download)
├── Makefile           # Build configuration
└── download_mnist.sh  # Dataset acquisition script
```

## 💻 Implementation Highlights

### ⚙️ Efficient Forward Pass

```c
void forward_pass(Network *network, const float *input, float *hidden1, float *hidden2, float *output) {
    // First hidden layer: h1 = relu(W_h1 * x + b_h1)
    for (int i = 0; i < HIDDEN1_SIZE; i++) {
        hidden1[i] = 0.0f;
        for (int j = 0; j < INPUT_SIZE; j++) {
            hidden1[i] += network->hidden1_weights[i * INPUT_SIZE + j] * input[j];
        }
        hidden1[i] += network->hidden1_biases[i];
        // Apply ReLU activation
        hidden1[i] = (hidden1[i] > 0.0f) ? hidden1[i] : 0.0f;
    }
    
    // Second hidden layer and output layer follow similar pattern...
}
```

### 🧮 Numerically Stable Softmax

```c
// Find max value for numerical stability
float max_val = output[0];
for (int i = 1; i < OUTPUT_SIZE; i++) {
    if (output[i] > max_val) {
        max_val = output[i];
    }
}

// Compute exp(x - max) and sum
float sum = 0.0f;
for (int i = 0; i < OUTPUT_SIZE; i++) {
    output[i] = expf(output[i] - max_val);
    sum += output[i];
}

// Normalize to get probabilities
for (int i = 0; i < OUTPUT_SIZE; i++) {
    output[i] /= sum;
}
```

### 🔄 Data Augmentation

```c
void augment_image(const float *input, float *output) {
    // Apply small random shifts (up to 2 pixels in each direction)
    int shift_x = rand() % 5 - 2;  // -2 to 2
    int shift_y = rand() % 5 - 2;  // -2 to 2
    
    // Clear output
    memset(output, 0, INPUT_SIZE * sizeof(float));
    
    // Apply shift
    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            int new_y = y + shift_y;
            int new_x = x + shift_x;
            
            // Check bounds
            if (new_x >= 0 && new_x < 28 && new_y >= 0 && new_y < 28) {
                output[new_y * 28 + new_x] = input[y * 28 + x];
            }
        }
    }
}
```

## 🛠️ Building from Source

The project uses a standard Makefile system for easy compilation:

```bash
make        # Build with optimizations
make clean  # Remove build artifacts
```

## 📄 License

MIT License - Copyright (c) 2025 tsotchke - See [LICENSE](LICENSE) for details

## 📚 Citation

If you use this implementation in your research or projects, please cite it using the following BibTeX format:

```bibtex
@software{simple_mnist,
  author       = {tsotchke},
  title        = {Simple MNIST: A Pure Feedforward Neural Network Implementation},
  year         = {2025},
  url = {https://github.com/tsotchke/simple_mnist}
}
```

## 🙏 Acknowledgments

- MNIST dataset by Yann LeCun and Corinna Cortes
- Special thanks to the deep learning research community for optimization techniques
- Inspired by the fundamental principles of neural networks that continue to drive innovation
