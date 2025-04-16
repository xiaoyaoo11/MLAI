# Class Learner

An advanced image classification application using PyTorch, featuring state-of-the-art model architectures, comprehensive training optimizations, and an interactive web interface for performing image classification on custom datasets.

## Project Overview

Class Learner is a complete end-to-end solution for training custom image classification models and deploying them with a user-friendly web interface. This project combines powerful deep learning techniques with accessible interfaces, making it suitable for both research and practical applications.

## Features

- **State-of-the-Art Architecture**: Uses ResNet101 with customized classification head for superior accuracy
- **Advanced Training Pipeline**: Incorporates mixup augmentation, learning rate scheduling, gradient clipping, and early stopping
- **Comprehensive Data Augmentation**: Includes random crops, flips, rotations, color jitter, affine transforms, and random erasing
- **Flexible Prediction System**: Command-line tools for single image or batch folder processing
- **Interactive Web Interface**: Upload images through an intuitive browser UI for instant classification
- **Robust Model Optimization**: Progressive layer unfreezing, adaptive learning rates, and weight decay

## Project Structure

```
Class Learner/
├── train.py             # Main training script with customizable hyperparameters
├── predict.py           # Command-line prediction utility for trained models
├── optimized_learner.py # Core training engine with advanced techniques
├── app.py               # Flask web application for the browser interface
├── templates/           # HTML templates for the web interface
├── static/              # Static assets and uploaded images
├── run.sh               # Convenience script for starting the web server
├── checkpoints/         # Directory for saved models (created during training)
└── dataset/             # Directory for training data (user provided)
```

## Technical Implementation

### Model Architecture

The project implements an enhanced ResNet101 architecture with:

- Frozen early layers to prevent overfitting and leverage pre-trained feature extractors
- Progressive layer unfreezing during training for refined feature adaptation
- Multi-layer classification head with advanced components:
  - Batch normalization for training stability
  - Multiple dropout layers with varying rates (0.3-0.5) for regularization
  - LeakyReLU activations for improved gradient flow
  - Decreasing dimension fully connected layers (2048→1024→512→num_classes)

### Training Optimizations

- **Mixup Data Augmentation**: Combines pairs of examples and their labels using random interpolation
- **OneCycleLR Scheduler**: Implements the one-cycle learning rate policy for faster convergence
- **Extensive Data Augmentation**: Uses over 10 different augmentation techniques including:
  - Random resized crops with varying scales
  - Horizontal and vertical flips
  - Rotation and affine transforms
  - Color jittering and random grayscale conversion
  - Random erasing for occlusion robustness
- **Gradient Clipping**: Prevents exploding gradients for more stable training
- **Early Stopping**: Intelligently stops training when validation metrics plateau
- **Best Model Checkpointing**: Saves the model with highest validation accuracy

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/xiaoyaoo11/MLAI.git
   cd MLAI/Class Learner
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Preparation

Organize your dataset in the standard PyTorch ImageFolder structure:
```
dataset/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   └── ...
│   ├── class2/
│   │   ├── image1.jpg
│   │   └── ...
│   └── ...
└── val/
    ├── class1/
    │   ├── image1.jpg
    │   └── ...
    ├── class2/
    │   ├── image1.jpg
    │   └── ...
    └── ...
```

### Sample Datasets

If you need datasets to test the application, here are some recommended options:

1. **CIFAR-10**: A dataset of 60,000 32x32 color images in 10 classes (airplanes, cars, birds, cats, etc.)
   - [Download CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)

2. **Caltech-101**: Contains pictures of objects belonging to 101 categories with about 40 to 800 images per category
   - [Download Caltech-101](https://data.caltech.edu/records/mzrjq-6wc02)

3. **Flowers-102**: 102 flower categories commonly occurring in the United Kingdom
   - [Download Flowers-102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)

4. **Food-101**: 101 food categories with 101,000 images
   - [Download Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)

5. **Stanford Dogs**: Contains images of 120 breeds of dogs from around the world
   - [Download Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/)

After downloading, you'll need to organize the data according to the structure shown above.

## Usage Guide

### Training a Custom Model

To train a new model on your dataset:

```bash
python train.py
```

Advanced parameters that can be modified in the script:
- `batch_size`: 16 (default, smaller batch size improves generalization)
- `epochs`: 40 (default, can be increased for more complex datasets)
- `learning_rate`: 0.0002 (default, lower rate for stable training)
- `weight_decay`: 5e-4 (default, regularization parameter)
- `dataset_path`: Path to your organized dataset directory
- `work_dir`: Directory for saving model checkpoints

### Making Predictions via Command Line

For single image classification:

```bash
python predict.py --image path/to/your/image.jpg
```

For batch processing a folder of images:

```bash
python predict.py --image_folder path/to/image/folder
```

Optional parameters:
- `--checkpoint`: Custom path to a model checkpoint file (default: "checkpoints/best_model.pth")
- `--dataset_path`: Path to dataset for class names (default: "dataset")

### Using the Web Interface

Start the web interface with the convenience script:

```bash
./run.sh
```

Or with automatic requirements installation:

```bash
./run.sh --install
```

You can also manually start the Flask application:

```bash
export FLASK_APP=app.py
flask run --host=0.0.0.0 --port=5000
```

Then open your browser and navigate to `http://localhost:5000` to access the web interface where you can:
1. Upload an image using the file selector
2. View the model's prediction and confidence score
3. Try different images through the same interface

## Web Interface Features

- **Clean, Responsive Design**: Built with Bootstrap for a modern look on all devices
- **Simple Upload Flow**: Intuitive file selection and upload process
- **Visual Results**: Displays the uploaded image alongside predictions
- **Confidence Scores**: Shows model's confidence percentage for each prediction
- **Error Handling**: Graceful handling of unsupported file types and processing errors

## Performance Considerations

- Recommended minimum hardware: 8GB RAM, CUDA-capable GPU for training
- Training time varies by dataset size and complexity (1-8 hours on typical datasets)
- Inference time is typically <1 second per image on CPU, faster with GPU
- Web interface can handle common image formats (JPEG, PNG) up to 10MB

## Contributing

Contributions to Class Learner are welcome! Please feel free to submit pull requests or open issues for:
- Bug fixes
- Feature enhancements
- Documentation improvements
- Performance optimizations

## Acknowledgments

- PyTorch team for their excellent deep learning framework
- The academic community for advancing image classification techniques
- Contributors who have helped improve this project

## Contact

For questions, suggestions, or collaboration opportunities:
- GitHub Issues: [project issues page](https://github.com/xiaoyaoo11/MLAI/issues)
- Email: ebevutru@gmail.com
