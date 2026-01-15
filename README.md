# XOR Neural Network

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
