# CNNfromScratch

This project was developed as part of my **TIPE** (Travaux d'Initiative Personnelle Encadrés) during the academic year 2021/2022. A **TIPE** is a mandatory research-based project for students in the French preparatory class system, designed to explore scientific and engineering topics through personal initiative and research.

## Overview

The goal of this project was not to optimize for speed, but rather to build a **Convolutional Neural Network (CNN)** from scratch using **Python** and **Numpy**, with minimal reliance on external libraries. This project was my first step into the world of AI, focusing on core learning objectives instead of performance.

## Goals

- **Manual CNN Implementation**: Create all components of a CNN (convolutional layers, max-pooling, fully connected layers, etc.) manually using Python and Numpy.
- **Minimal Libraries**: Avoid common deep learning libraries (like TensorFlow or PyTorch) to deepen understanding. Only essential libraries (like `matplotlib`, `scipy`) are used for plotting and utility functions.
- **Learning Over Optimization**: This project focuses on understanding, not speed or large-scale performance.

## Project Structure

- **XOR Classification**: Basic neural network structure was tested with the XOR function as a sanity check.
- **MNIST Dataset**: The CNN was extended to handle image classification tasks, starting with two-digit classification from the MNIST dataset, later expanded to classify all 10 digits.
- **HAM10000 Dataset**: This dataset of skin lesion images was used to test the CNN's ability to classify medical images. Although my manual CNN worked, it was slow, so I compared the results using **Keras** for a faster version of the same network.

### Layers Built from Scratch:

- **Convolution Layer**
- **Max-Pooling Layer**
- **Fully Connected Layers**
- **Activation Functions**: ReLU, Sigmoid, Tanh
- **Softmax for Classification**
- **Stochastic Gradient Descent (SGD)** for Optimization


## Results

### Training on MNIST:

| Metric      | Value    |
|-------------|----------|
| Accuracy    | 98.2%    |
| Loss        | 0.057    |
| Epochs      | 50       |

### Training on HAM10000:

Due to the complexity and size of the dataset, the model trained using the manually coded CNN is very slow. Therefore, I later implemented the same network using **Keras** to speed up the process and compare results.

## Limitations

- **Speed**: Training large datasets like HAM10000 is very slow with the manual implementation. Optimizations such as GPU acceleration or parallel processing could significantly reduce training time.
- **Library Constraints**: Using only Numpy and minimal external packages means there are limitations on what can be achieved efficiently.

## Next Steps

- **Optimize for Performance**: Improve speed by using GPU and optimizing the Numpy code.
- **Extend to More Datasets**: Apply the same techniques to other datasets and test performance.

---

This project was a great learning experience and served as a foundation for future AI projects, giving me a deeper understanding of the core concepts behind CNNs and deep learning in general.
