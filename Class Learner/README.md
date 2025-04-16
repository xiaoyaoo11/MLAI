# Class Learner

An optimized PyTorch-based image classification training and inference framework.

## Features

- Flexible model training with configurable hyperparameters
- Automatic early stopping and model checkpointing
- Support for various pre-trained models (ResNet, VGG)
- GPU acceleration (when available)
- Comprehensive testing and inference capabilities

## Setup

1. Create and activate the virtual environment:

```bash
bash setup_venv.sh
source venv/bin/activate
```

2. Prepare your dataset in the following structure:
```
dataset/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── class2/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
└── val/
    ├── class1/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── class2/
        ├── image1.jpg
        ├── image2.jpg
        └── ...
```

3. Run the training script:

```bash
python train.py
```

## Usage Examples

```python
from optimized_learner import Learner, create_resnet_model, create_data_loaders
import torch.nn as nn
import torch.optim as optim

# Create data loaders
train_loader, val_loader, class_names = create_data_loaders("dataset", batch_size=32)

# Create model
model = create_resnet_model(num_classes=len(class_names))

# Define optimizer, loss function, and scheduler
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3)

# Create learner
learner = Learner(
    model=model,
    train_dataloader=train_loader,
    test_dataloader=val_loader,
    optimizer=optimizer,
    loss=criterion,
    scheduler=scheduler,
    work_dir="checkpoints"
)

# Train model
learner.train(epochs=10)

# Test model
loss, accuracy = learner.test()
print(f"Test accuracy: {accuracy:.2f}%")

# Make prediction
prediction = learner.inference("path/to/image.jpg")
print(f"Prediction: {prediction}")
```
