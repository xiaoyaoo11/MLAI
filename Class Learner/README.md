# Class Learner

A robust image classification project using PyTorch, featuring advanced model architectures, training optimizations, and easy-to-use prediction capabilities.

## Features

- Advanced model architectures (ResNet101) with improved classification heads
- Optimized training pipeline with mixup augmentation
- Flexible prediction system for both single images and batch processing
- Learning rate scheduling and regularization techniques
- Support for model checkpointing and best model saving

## Project Structure

```
Class Learner/
├── train.py             # Main training script
├── predict.py           # Prediction utilities for trained models
├── optimized_learner.py # Core learner implementation
├── checkpoints/         # Directory for saved models
└── dataset/            # Directory for training data
```

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- Other dependencies (install via requirements.txt)

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd Class-Learner
```

2. Set up a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Structure

Organize your dataset in the following structure:

```
dataset/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   └── ...
│   ├── class2/
│   │   ├── image2.jpg
│   │   └── ...
│   └── ...
└── val/
    ├── class1/
    │   ├── image1.jpg
    │   └── ...
    ├── class2/
    │   ├── image2.jpg
    │   └── ...
    └── ...
```

## Usage

### Training

To train a new model:

```bash
python train.py
```

Key parameters in `train.py`:

- `batch_size`: 16 (default)
- `epochs`: 40 (default)
- `learning_rate`: 0.0002
- `dataset_path`: Path to your dataset
- `work_dir`: Directory for saving checkpoints

### Prediction

For single image prediction:

```bash
python predict.py --image_path path/to/image.jpg --checkpoint_path checkpoints/best_model.pth
```

For batch prediction on a folder:

```bash
python predict.py --image_folder path/to/folder --checkpoint_path checkpoints/best_model.pth
```

## Model Architecture

The project uses an enhanced ResNet101 architecture with:

- Frozen early layers for transfer learning
- Advanced classification head with:
  - Batch normalization
  - Dropout layers
  - LeakyReLU activation
  - Multiple fully connected layers

## Training Features

- Mixup augmentation for better generalization
- OneCycleLR scheduler for optimal learning rate management
- AdamW optimizer with weight decay
- Automatic model checkpointing
- Best model saving based on validation performance

## Performance Optimization

The training pipeline includes several optimizations:

- Smaller batch size for better generalization
- Progressive layer unfreezing
- Learning rate warm-up
- Mixup data augmentation
- Dropout for regularization

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## License

[Specify your license here]

## Acknowledgments

- Thanks to the PyTorch team for the excellent framework
- The project structure is inspired by modern deep learning best practices

## Contact

[Your contact information or how to reach out with questions]
