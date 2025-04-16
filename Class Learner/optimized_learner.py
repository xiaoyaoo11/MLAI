import os
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np  # Required for mixup implementation
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset


class Learner:
    """
    A class for training, testing, and inference with image classification models.
    Optimized for flexibility, performance, and accuracy with advanced techniques.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss: nn.Module,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        work_dir: str = "checkpoints",
        pre_train: bool = False,
        device: Optional[torch.device] = None,
        mixup_alpha: float = 0.2,  # Mixup regularization parameter
        use_mixup: bool = True,  # Whether to use mixup augmentation
    ):
        """
        Initialize the Learner with model, data, and training parameters.

        Args:
            model: PyTorch model (nn.Module)
            train_dataloader: DataLoader for training data
            test_dataloader: DataLoader for testing/validation data
            optimizer: PyTorch optimizer
            loss: Loss function
            scheduler: Learning rate scheduler
            work_dir: Directory to save model checkpoints
            pre_train: Whether to load pre-trained weights
            device: Device to use for training (default: auto-detect)
            mixup_alpha: Parameter for mixup regularization
            use_mixup: Whether to use mixup data augmentation
        """
        # Initialize parameters
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.criterion = loss
        self.scheduler = scheduler
        self.work_dir = work_dir
        self.mixup_alpha = mixup_alpha
        self.use_mixup = use_mixup

        # Get class names from dataloader
        self.class_names = (
            train_dataloader.dataset.classes
            if hasattr(train_dataloader.dataset, "classes")
            else None
        )

        # Auto-detect device (use GPU if available)
        self.device = (
            device
            if device
            else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        print(f"Using device: {self.device}")

        # Move model to device
        self.model.to(self.device)

        # Create directory for checkpoints
        os.makedirs(self.work_dir, exist_ok=True)

        # Training tracking variables
        self.best_acc = 0
        self.best_loss = float("inf")
        self.patience = 7  # Increased early stopping patience
        self.counter = 0  # Counter for early stopping

        # Load pre-trained weights if specified
        if pre_train:
            self._load_best_model()

    def mixup_data(self, x, y, alpha=0.2):
        """
        Applies mixup augmentation to batch data.

        Args:
            x: Input features
            y: Target labels
            alpha: Mixup interpolation strength parameter

        Returns:
            Mixed inputs, pairs of targets, and lambda value
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(self.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(self, pred, y_a, y_b, lam):
        """
        Mixup loss function
        """
        return lam * self.criterion(pred, y_a) + (1 - lam) * self.criterion(pred, y_b)

    def train(self, epochs: int = 10, verbose: bool = True) -> Dict[str, list]:
        """
        Train the model and save best model to checkpoint in work_dir.
        Implements advanced techniques like mixup for better generalization.

        Args:
            epochs: Number of training epochs
            verbose: Whether to print training progress

        Returns:
            Dictionary containing training history (losses and accuracies)
        """
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        if verbose:
            print(f"Training for {epochs} epochs on {self.device}...")

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            running_loss, correct, total = 0.0, 0, 0

            for inputs, targets in self.train_dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Apply mixup if enabled
                if self.use_mixup:
                    inputs, targets_a, targets_b, lam = self.mixup_data(
                        inputs, targets, self.mixup_alpha
                    )

                    # Forward pass
                    outputs = self.model(inputs)

                    # Mixup loss
                    loss = self.mixup_criterion(outputs, targets_a, targets_b, lam)

                    # For accuracy calculation with mixup
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += (
                        lam * predicted.eq(targets_a).sum().float()
                        + (1 - lam) * predicted.eq(targets_b).sum().float()
                    ).item()
                else:
                    # Regular forward pass
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                    # Statistics
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                # Backward + optimize
                loss.backward()

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)

            # Calculate epoch metrics
            train_loss = running_loss / len(self.train_dataloader.dataset)
            train_acc = 100.0 * correct / total

            # Save to history
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            # Evaluation phase
            val_loss, val_acc = self.test(self.test_dataloader)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            # Print progress
            if verbose:
                print(
                    f"Epoch [{epoch+1}/{epochs}] - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
                )

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Save best model and early stopping
            if val_acc > self.best_acc or (
                val_acc == self.best_acc and val_loss < self.best_loss
            ):
                self.best_acc = val_acc
                self.best_loss = val_loss
                self._save_model()
                if verbose:
                    print(
                        f"✅ Best model saved at epoch {epoch+1} with accuracy {val_acc:.2f}%"
                    )
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    if verbose:
                        print(f"⏳ Early stopping triggered after {epoch+1} epochs!")
                    break

        if verbose:
            print("Training completed!")
            print(f"Best validation accuracy: {self.best_acc:.2f}%")

        return history

    def test(self, test_dataloader: Optional[DataLoader] = None) -> Tuple[float, float]:
        """
        Test the model on a dataset and return loss and accuracy.

        Args:
            test_dataloader: DataLoader for test data. If None, use the instance's test_dataloader.

        Returns:
            Tuple of (loss, accuracy)
        """
        dataloader = (
            test_dataloader if test_dataloader is not None else self.test_dataloader
        )

        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        avg_loss = running_loss / len(dataloader.dataset)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def inference(self, image_path: str) -> str:
        """
        Perform inference on a single image and return the prediction.

        Args:
            image_path: Path to the image file

        Returns:
            Predicted class name
        """
        # Make sure the model is in evaluation mode
        self.model.eval()

        # Image preprocessing
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Load and preprocess the image
        image = Image.open(image_path)
        image_tensor = transform(image).unsqueeze(0).to(self.device)

        # Make prediction
        with torch.no_grad():
            output = self.model(image_tensor)
            _, predicted = output.max(1)
            prediction_idx = predicted.item()

        # Return class name if available, otherwise return the index
        if self.class_names and prediction_idx < len(self.class_names):
            return self.class_names[prediction_idx]
        return str(prediction_idx)

    def _save_model(self, path: Optional[str] = None) -> None:
        """Save the model to the specified path or default path."""
        save_path = path if path else os.path.join(self.work_dir, "best_model.pth")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_acc": self.best_acc,
                "best_loss": self.best_loss,
            },
            save_path,
        )

    def _load_best_model(self, path: Optional[str] = None) -> None:
        """Load the best model from the specified path or default path."""
        load_path = path if path else os.path.join(self.work_dir, "best_model.pth")
        if os.path.exists(load_path):
            checkpoint = torch.load(load_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.best_acc = checkpoint.get("best_acc", 0)
            self.best_loss = checkpoint.get("best_loss", float("inf"))
            self.model.to(self.device)
            print(f"Model loaded from {load_path}")
            return True
        print(f"No model found at {load_path}")
        return False


# Helper functions to create common model configurations
def create_resnet_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Create a ResNet-18 model with specified number of output classes."""
    model = models.resnet18(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def create_vgg_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Create a VGG-16 model with specified number of output classes."""
    model = models.vgg16(pretrained=pretrained)
    model.classifier[6] = nn.Linear(4096, num_classes)
    return model


def create_data_loaders(
    dataset_path: str, batch_size: int = 32, img_size: int = 224
) -> Tuple[DataLoader, DataLoader, list]:
    """
    Create train and test data loaders from a dataset directory.

    Args:
        dataset_path: Path to dataset (should have 'train' and 'val' subdirectories)
        batch_size: Batch size for data loaders
        img_size: Size to resize images to

    Returns:
        Tuple of (train_loader, val_loader, class_names)
    """
    # Data transformations with enhanced augmentation for better generalization
    transform_train = transforms.Compose(
        [
            transforms.Resize(
                (img_size + 32, img_size + 32)
            ),  # Resize larger then crop for more variations
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),  # Add vertical flips
            transforms.RandomRotation(20),  # Increase rotation range
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15
            ),  # Increase color jitter
            transforms.RandomAffine(
                degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10
            ),  # Add affine transforms
            transforms.RandomResizedCrop(
                img_size, scale=(0.7, 1.0)
            ),  # More crop variations
            transforms.RandomGrayscale(p=0.02),  # Occasional grayscale
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(
                p=0.2, scale=(0.02, 0.2)
            ),  # Random erasing for occlusion robustness
        ]
    )

    transform_val = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load datasets
    train_dataset = None
    val_dataset = None

    try:
        train_dataset = datasets.ImageFolder(
            root=os.path.join(dataset_path, "train"), transform=transform_train
        )

        val_dataset = datasets.ImageFolder(
            root=os.path.join(dataset_path, "val"), transform=transform_val
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None, []

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    print(
        f"Dataset loaded: {len(train_dataset)} train images, {len(val_dataset)} validation images"
    )
    print(f"Classes: {train_dataset.classes}")

    return train_loader, val_loader, train_dataset.classes
