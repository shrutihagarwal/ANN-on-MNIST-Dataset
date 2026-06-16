# 🧠 Artificial Neural Networks on MNIST & Other Datasets

A hands-on implementation of **Artificial Neural Networks (ANNs)** from scratch and using Keras/TensorFlow, applied to benchmark classification datasets. This project explores how architectural choices — layer depth, activation functions, regularization, and optimizers — affect training dynamics and generalization performance.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Datasets](#datasets)
- [Network Architecture](#network-architecture)
- [Experiments](#experiments)
- [Technologies & Libraries](#technologies--libraries)
- [How to Run](#how-to-run)
- [Results](#results)
- [Key Learnings](#key-learnings)

---

## Overview

The MNIST dataset (Modified National Institute of Standards and Technology) is a canonical benchmark in deep learning — 70,000 grayscale images of handwritten digits (0–9). This project uses it as a controlled testbed to:

1. Understand the feedforward network computation from first principles
2. Implement backpropagation conceptually
3. Experiment with hyperparameter choices using Keras
4. Extend the same framework to additional datasets

---

## Datasets

### MNIST (Primary)
- **Size**: 60,000 training + 10,000 test samples
- **Input shape**: 28×28 grayscale images (flattened to 784-dim vectors)
- **Classes**: 10 (digits 0–9)
- **Source**: `keras.datasets.mnist` or `torchvision`

### Additional Datasets
- **Fashion-MNIST**: Same structure as MNIST but with 10 clothing categories (harder problem)
- **Other datasets** explored to validate the generalization of the ANN architecture

---

## Network Architecture

### Baseline ANN

```
Input (784) → Dense(128, ReLU) → Dropout(0.2) → Dense(64, ReLU) → Dense(10, Softmax)
```

| Layer | Units | Activation | Notes |
|-------|-------|------------|-------|
| Input | 784 | — | Flattened 28×28 image |
| Hidden 1 | 128 | ReLU | First representation layer |
| Dropout | — | — | Regularization (rate = 0.2) |
| Hidden 2 | 64 | ReLU | Compression layer |
| Output | 10 | Softmax | Class probabilities |

### Key Architectural Choices
- **ReLU activation**: Avoids vanishing gradient problem vs. sigmoid/tanh in deeper networks
- **Dropout**: Prevents co-adaptation of neurons and improves generalization
- **Softmax output**: Produces a proper probability distribution over all 10 classes

---

## Experiments

The following hyperparameter variations were explored:

| Variable | Options Tested |
|----------|---------------|
| Hidden units | 64, 128, 256, 512 |
| Depth | 1, 2, 3 hidden layers |
| Activation | ReLU, Sigmoid, Tanh |
| Optimizer | SGD, Adam, RMSprop |
| Batch size | 32, 64, 128 |
| Regularization | None, Dropout(0.2), Dropout(0.5) |

---

## Technologies & Libraries

| Library | Usage |
|---------|-------|
| `TensorFlow` / `Keras` | Model definition, training, evaluation |
| `numpy` | Data manipulation, manual weight operations |
| `matplotlib` | Training curves, confusion matrix, sample predictions |
| `sklearn` | Metrics (accuracy, classification report, confusion matrix) |

---

## How to Run

### Prerequisites
```bash
pip install tensorflow numpy matplotlib scikit-learn
```

### Run the Script
```bash
python "ANN on Mnist data.py"
```

The script will:
1. Load and preprocess the MNIST dataset (auto-downloads via Keras)
2. Build and compile the ANN
3. Train for a specified number of epochs
4. Print test accuracy and display a confusion matrix

---

## Results

| Architecture | Dataset | Test Accuracy |
|-------------|---------|---------------|
| Dense(128) → Dense(64) | MNIST | ~98.1% |
| Dense(256) → Dense(128) → Dense(64) | MNIST | ~98.4% |
| Dense(128) → Dense(64) | Fashion-MNIST | ~88.5% |

> Performance on MNIST approaches the theoretical limit for a fully-connected ANN without convolutions.

### Training Curves
Training loss and validation loss curves were plotted to diagnose overfitting — Dropout regularization demonstrably reduced the train/val gap.

---

## Key Learnings

1. **ANNs are universal function approximators** — even simple architectures solve MNIST nearly perfectly; the challenge is generalization to harder problems.
2. **Depth helps, to a point** — beyond 2–3 hidden layers, gains plateau without other techniques (batch norm, skip connections).
3. **Optimizer choice matters** — Adam converges faster than SGD on MNIST; SGD with momentum can match with careful tuning.
4. **Dropout is effective regularization** — significantly reduces overfitting on Fashion-MNIST where the decision boundary is less clear.
5. **CNNs dominate for images** — this project motivates why convolutional architectures are the standard for image tasks.

---

## Conceptual Notes

### Backpropagation (Conceptual)
The network learns by computing the gradient of the loss w.r.t. each weight using the chain rule:

```
∂L/∂W = ∂L/∂a · ∂a/∂z · ∂z/∂W
```

Where:
- `L` = cross-entropy loss
- `a` = post-activation value
- `z` = pre-activation (linear combination)

Keras/TensorFlow handles this automatically via automatic differentiation.

---

## References

- LeCun, Y. et al. (1998). Gradient-Based Learning Applied to Document Recognition. *Proc. IEEE*
- Goodfellow, I., Bengio, Y. & Courville, A. (2016). *Deep Learning*. MIT Press.
- Keras Documentation: [https://keras.io](https://keras.io)
- MNIST Database: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)

---

*A foundational deep learning project exploring ANN behavior on benchmark classification tasks.*
