# XOR Neural Network

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![NumPy](https://img.shields.io/badge/NumPy-1.20+-blue.svg)](https://numpy.org/)

A minimal multi-layer perceptron implementation in pure NumPy that learns the XOR function.

## Features

- **Manual Backpropagation**: Implements gradient computation from scratch
- **Binary Cross-Entropy Loss**: Proper loss function for binary classification
- **ASCII Decision Boundary**: Visual representation of learned decision boundary
- **Interactive Mode**: Test the trained model with custom inputs
- **Data Augmentation**: Noise injection for robust training

## Usage

```bash
python xor_min.py --epochs 5000 --lr 0.1 --grid
```

### Arguments
- `--hidden`: Hidden layer size (default: 4)
- `--epochs`: Training epochs (default: 5000)
- `--lr`: Learning rate (default: 0.1)
- `--noise`: Noise standard deviation (default: 0.05)
- `--grid`: Show ASCII decision boundary

## Requirements

- Python 3.8+
- NumPy
