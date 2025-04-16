import pathlib
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

# Set environment variables to optimize performance
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())

# Simplified device selection
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple MPS device")
else:
    device = torch.device("cpu")
    # Enable oneDNN optimizations for CPU
    if hasattr(torch.backends, 'mkldnn'):
        torch.backends.mkldnn.enabled = True
        print("Using CPU with oneDNN optimizations")
    else:
        print("Using CPU without oneDNN optimizations")

print(f"PyTorch version: {torch.__version__}")

# Set seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Data directory
training_dir = pathlib.Path("datasets/training_set")

# Data augmentation and normalization
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Increased resolution
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),  # More rotation
    transforms.RandomVerticalFlip(p=0.2),  # Added vertical flip
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),  # Enhanced color jitter
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Added random translation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Match training resolution
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
full_dataset = datasets.ImageFolder(root=str(training_dir), transform=train_transform)

# Split dataset
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

# Create datasets
train_dataset = datasets.ImageFolder(root=str(training_dir), transform=train_transform)
val_dataset = datasets.ImageFolder(root=str(training_dir), transform=val_transform)

# Random split
indices = torch.randperm(len(full_dataset)).tolist()
train_indices = indices[:train_size]
val_indices = indices[train_size:]

# Use subset of the dataset
from torch.utils.data import Subset
train_dataset = Subset(train_dataset, train_indices)
val_dataset = Subset(val_dataset, val_indices)

class_names = full_dataset.classes
num_classes = len(class_names)

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

print(f"Classes: {class_names}")
print(f"Training samples: {train_size}, Validation samples: {val_size}")

# Define model architecture with residual connections and attention mechanism
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = self.relu(out)
        return out

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x

class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=8):
        super(ImprovedCNN, self).__init__()
        
        # Initial convolution (reduced to 24 filters to save parameters)
        self.conv1 = nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(24)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual blocks (reduced channels)
        self.layer1 = nn.Sequential(
            ResidualBlock(24, 32),
            ChannelAttention(32)
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(32, 48, stride=2),
            ChannelAttention(48)
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(48, 64, stride=2),
            ChannelAttention(64)
        )
        
        # Global average pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(64, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Initial convolution
        x = self.relu(self.bn1(self.conv1(x)))
        
        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Global average pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classifier
        x = self.dropout(x)
        x = self.fc(x)
        
        return F.log_softmax(x, dim=1)

# Count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Create model
model = ImprovedCNN(num_classes=num_classes).to(device)
print(f"Model parameters: {count_parameters(model)}")

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# Training function
def train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=20):
    train_loss, val_loss, train_acc, val_acc = [], [], [], []
    best_acc = 0.0
    
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss.append(running_loss / len(train_loader))
        train_acc.append(correct / total)

        # Validation
        model.eval()
        val_running_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_loss.append(val_running_loss / len(val_loader))
        val_acc.append(correct / total)
        
        # Update scheduler
        scheduler.step(val_loss[-1])
        
        # Save best model
        if val_acc[-1] > best_acc:
            best_acc = val_acc[-1]
            model_save_path = f'best_model_acc_{best_acc:.4f}_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_acc,
                'train_acc': train_acc[-1],
                'val_loss': val_loss[-1],
                'train_loss': train_loss[-1],
            }, model_save_path)
            print(f"Saved best model to {model_save_path}")
        
        # Time elapsed
        elapsed_time = time.time() - start_time
        
        print(
            f"Epoch {epoch+1}/{epochs} [{elapsed_time:.1f}s] - "
            f"Train Loss: {train_loss[-1]:.4f}, Train Acc: {train_acc[-1]:.4f} - "
            f"Val Loss: {val_loss[-1]:.4f}, Val Acc: {val_acc[-1]:.4f}"
        )
        
    # Load best model with highest validation accuracy
    best_model_path = f'best_model_acc_{best_acc:.4f}_epoch_{epochs}.pth'
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from {best_model_path}")
    return model

# Train model
epochs = 20
model = train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs)

# Trace model for deployment
example_input = torch.randn(1, 3, 224, 224).to(device)
traced_model = torch.jit.trace(model, example_input)

# Save model with your student ID
traced_model.save("22111001.pt")
print("Model saved as 22111001.pt")

# Print final model parameters
print(f"Final model parameters: {count_parameters(model)}")
