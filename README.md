# MobileNet V1 from Scratch on MNIST

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)

This repository contains a full implementation of the **MobileNet V1** architecture built from scratch using PyTorch. The model is specifically adapted to train on the **MNIST dataset** (grayscale handwritten digits).

## 📌 Project Overview

**MobileNet**, introduced by Google in 2017, is designed for efficient execution on mobile and embedded devices. Unlike standard CNNs (like VGG) which are computationally heavy, MobileNet uses **Depthwise Separable Convolutions** to reduce the number of parameters and computation cost by roughly **8-9 times** with minimal loss in accuracy.

### ⚙️ Adaptations for MNIST
The standard MobileNet is designed for ImageNet ($224 \times 224$). To make it work for MNIST, we applied the following changes:

1.  **Input Channels:** Modified the first convolution layer to accept **1 channel** (Grayscale) instead of 3 (RGB).
2.  **Image Resizing:** MNIST images ($28 \times 28$) are resized to **$32 \times 32$**.
    * *Reason:* MobileNet V1 downsamples the image 5 times (stride 2).
    * $32 \to 16 \to 8 \to 4 \to 2 \to 1$.
    * This ensures valid spatial dimensions throughout the network without the features vanishing to $0 \times 0$.
3.  **strides:** We adjusted the strides in the early layers to be less aggressive, preserving information for the small MNIST images.

## 🏗️ Architecture Details

### Depthwise Separable Convolution
The core innovation of MobileNet is splitting a standard convolution into two separate layers:

1.  **Depthwise Convolution:** Filters are applied to each input channel separately. (e.g., if input has 32 channels, we use 32 separate $3 \times 3$ filters).
2.  **Pointwise Convolution:** A $1 \times 1$ convolution combines the outputs of the depthwise layer into new features.

| Layer Type | Parameters (Approx) | Computation Cost |
| :--- | :--- | :--- |
| Standard Conv ($3 \times 3$) | $3 \cdot 3 \cdot C_{in} \cdot C_{out}$ | High |
| Depthwise Separable | $3 \cdot 3 \cdot C_{in} + 1 \cdot 1 \cdot C_{in} \cdot C_{out}$ | **~8-9x Lower** |

### Network Configuration (Custom)

| Layer | Type | Output Size (Input $32 \times 32$) |
| :--- | :--- | :--- |
| **Conv / s1** | Standard Conv | $32 \times 32$ |
| **Conv dw / s1** | Depthwise Separable | $32 \times 32$ |
| **Conv dw / s2** | Depthwise Separable | $16 \times 16$ |
| **Conv dw / s1** | Depthwise Separable | $16 \times 16$ |
| **Conv dw / s2** | Depthwise Separable | $8 \times 8$ |
| **Conv dw / s1** | Depthwise Separable | $8 \times 8$ |
| **Conv dw / s2** | Depthwise Separable | $4 \times 4$ |
| **...** | ... (5 blocks) | $4 \times 4$ |
| **Conv dw / s2** | Depthwise Separable | $2 \times 2$ |
| **Conv dw / s2** | Depthwise Separable | $1 \times 1$ |
| **FC** | Linear | 10 Classes |

## 🚀 Getting Started

### Prerequisites
* Python 3.x
* PyTorch
* Torchvision

### Installation
Clone the repository:
```bash
git clone [https://github.com/your-username/mobilenet-mnist-scratch.git](https://github.com/your-username/mobilenet-mnist-scratch.git)
cd mobilenet-mnist-scratch
pip install torch torchvision
