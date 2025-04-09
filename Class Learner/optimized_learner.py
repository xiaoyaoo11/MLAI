# Standard library imports
import os
import time
from typing import List, Optional, Tuple, Union, Dict, Any

# PyTorch imports for deep learning functionality
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import _LRScheduler

# Optional imports for image processing
try:
    import PIL.Image as Image
    import torchvision.transforms as transforms
except ImportError:
    pass


class Learner:
    """
    A flexible and optimized deep learning model trainer class.
    
    This class provides a standardized interface for training, testing, and inference
    with PyTorch models. It handles model training loops, evaluation, checkpointing,
    and inference, with support for early stopping and learning rate scheduling.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        optimizer: optim.Optimizer,
        loss: nn.Module,
        scheduler: Optional[_LRScheduler] = None,
        work_dir: str = "checkpoint",
        pre_train: bool = False,
        device: Optional[torch.device] = None,
        patience: int = 5,
        class_names: Optional[List[str]] = None,
    ):
        """
        Initialize the Learner with model and training configurations.

        Args:
            model (nn.Module): PyTorch model to train
            train_dataloader (DataLoader): DataLoader for training data
            test_dataloader (DataLoader): DataLoader for validation/test data
            optimizer (optim.Optimizer): Optimizer for model parameter updates
            loss (nn.Module): Loss function for training
            scheduler (Optional[_LRScheduler]): Learning rate scheduler
            work_dir (str): Directory to save model checkpoints
            pre_train (bool): Whether to load pre-trained weights
            device (Optional[torch.device]): Device to run training on (CPU/GPU)
            patience (int): Number of epochs to wait before early stopping
            class_names (Optional[List[str]]): List of class names for classification
        """
        # Model and training components
        self.model = model
        self.train_loader = train_dataloader
        self.test_loader = test_dataloader
        self.optimizer = optimizer
        self.criterion = loss
        self.scheduler = scheduler
        self.work_dir = work_dir
        self.class_names = class_names
        
        # Set device (GPU if available, otherwise CPU)
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)
        
        # Early stopping parameters
        self.patience = patience
        self.best_acc = 0.0
        self.counter = 0
        
        # Create directory for saving checkpoints
        os.makedirs(self.work_dir, exist_ok=True)
        
        # Load pre-trained weights if specified
        if pre_train:
            self._load_pretrained()
            
        # Log initialization
        print(f"Learner initialized on {self.device}")
        print(f"Model: {type(self.model).__name__}")
        print(f"Optimizer: {type(self.optimizer).__name__}")
        print(f"Loss: {type(self.criterion).__name__}")
        if self.scheduler:
            print(f"Scheduler: {type(self.scheduler).__name__}")

    def _load_pretrained(self):
        """Load pre-trained weights from the checkpoint directory if available."""
        best_model_path = os.path.join(self.work_dir, "best_model.pth")
        if os.path.exists(best_model_path):
            self.model.load_state_dict(
                torch.load(best_model_path, map_location=self.device)
            )
            print(f"Pre-trained weights loaded from {best_model_path}")
        else:
            print("No pre-trained weights found, starting from scratch")

    def train(self, epochs: int = 10, verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the model for the specified number of epochs.
        
        Args:
            epochs (int): Number of training epochs
            verbose (bool): Whether to print progress during training
            
        Returns:
            Dict[str, List[float]]: Dictionary containing training history
                (train_loss, train_acc, val_loss, val_acc)
        """
        # Initialize history tracking
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }
        
        print(f"Starting training for {epochs} epochs on {self.device}")
        start_time = time.time()
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            running_loss, correct, total = 0.0, 0, 0
            
            for inputs, targets in self.train_loader:
                # Move data to device
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()
                
                # Track statistics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            # Calculate training metrics
            train_loss = running_loss / len(self.train_loader)
            train_acc = 100.0 * correct / total
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            
            # Validation phase
            val_loss, val_acc = self.test(self.test_loader)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            
            # Print progress
            if verbose:
                print(
                    f"Epoch [{epoch+1}/{epochs}] - "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
                )
            
            # Learning rate scheduling if available
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Save best model and check for early stopping
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self._save_checkpoint(epoch, val_acc)
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    if verbose:
                        print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Training summary
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f} seconds")
        print(f"Best validation accuracy: {self.best_acc:.2f}%")
        
        return history

    def _save_checkpoint(self, epoch: int, accuracy: float):
        """
        Save model checkpoint.
        
        Args:
            epoch (int): Current epoch number
            accuracy (float): Validation accuracy
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': accuracy,
        }
        
        # Save best model
        torch.save(self.model.state_dict(), f"{self.work_dir}/best_model.pth")
        
        # Save full checkpoint (optional)
        torch.save(checkpoint, f"{self.work_dir}/checkpoint.pth")
        print(f"Model checkpoint saved at epoch {epoch+1} with accuracy {accuracy:.2f}%")

    def test(self, dataloader: Optional[DataLoader] = None) -> Tuple[float, float]:
        """
        Evaluate the model on a dataset.
        
        Args:
            dataloader (Optional[DataLoader]): DataLoader for evaluation
                                              (uses test_loader if None)
                                              
        Returns:
            Tuple[float, float]: Loss and accuracy on the dataset
        """
        # Use test_loader if no dataloader is provided
        if dataloader is None:
            dataloader = self.test_loader
        
        self.model.eval()
        running_loss, correct, total = 0.0, 0, 0
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Track statistics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        # Calculate metrics
        test_loss = running_loss / len(dataloader)
        test_acc = 100.0 * correct / total
        
        return test_loss, test_acc

    def load_best_model(self):
        """Load the best model from checkpoint for inference."""
        best_model_path = os.path.join(self.work_dir, "best_model.pth")
        if os.path.exists(best_model_path):
            self.model.load_state_dict(
                torch.load(best_model_path, map_location=self.device)
            )
            self.model.to(self.device)
            self.model.eval()
            print("Best model loaded for inference")
        else:
            print("No best model found. Using current model state.")

    def inference(self, image_path: str) -> str:
        """
        Perform inference on a single image.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            str: Predicted class name
        """
        # Ensure model is in evaluation mode
        self.model.eval()
        
        # Check if PIL is available
        if 'Image' not in globals():
            raise ImportError("PIL.Image is required for inference but not available")
        
        # Define image transformation
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        # Load and preprocess image
        image = Image.open(image_path)
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Perform inference
        with torch.no_grad():
            output = self.model(image_tensor)
            _, predicted = output.max(1)
            
        # Get class name
        if self.class_names is not None:
            return self.class_names[predicted.item()]
        else:
            return str(predicted.item())

    def get_model_summary(self) -> str:
        """
        Generate a summary of the model architecture.
        
        Returns:
            str: Model summary
        """
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        summary = [
            f"Model: {type(self.model).__name__}",
            f"Total parameters: {total_params:,}",
            f"Trainable parameters: {trainable_params:,}",
            f"Non-trainable parameters: {total_params - trainable_params:,}",
            f"Device: {self.device}"
        ]
        
        return "\n".join(summary)


# Example usage:
"""
# 1. Define model, dataloaders, optimizer, and loss function
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# 2. Create Learner instance
learner = Learner(
    model=model,
    train_dataloader=train_loader,
    test_dataloader=test_loader,
    optimizer=optimizer,
    loss=criterion,
    scheduler=scheduler,
    work_dir="checkpoints",
    class_names=["class1", "class2", "class3"]
)

# 3. Train the model
history = learner.train(epochs=10)

# 4. Evaluate on test set
test_loss, test_acc = learner.test()
print(f"Test accuracy: {test_acc:.2f}%")

# 5. Inference on a single image
prediction = learner.inference("path/to/image.jpg")
print(f"Prediction: {prediction}")
"""
