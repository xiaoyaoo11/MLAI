import torch
import torch.nn as nn
import torch.optim as optim
from optimized_learner import Learner, create_resnet_model, create_vgg_model, create_data_loaders
import torchvision.models as models

def create_ensemble_model(num_classes):
    """Create a model with better architecture for higher accuracy"""
    # Use ResNet50 for better feature extraction
    model = models.resnet50(pretrained=True)
    
    # Replace the final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),  # Add dropout for regularization
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),  # Add batch normalization
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    
    return model

def main():
    # Define paths and hyperparameters
    dataset_path = "dataset"
    work_dir = "checkpoints"
    batch_size = 20  # Smaller batch size for better generalization
    epochs = 30      # Increase number of epochs further
    learning_rate = 0.0003  # Lower learning rate for more stable training
    weight_decay = 3e-4     # Increased weight decay for better regularization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Create data loaders with more aggressive data augmentation
    train_loader, val_loader, class_names = create_data_loaders(
        dataset_path=dataset_path,
        batch_size=batch_size,
        img_size=224
    )
    
    if train_loader is None or val_loader is None:
        print("Failed to load dataset. Please check the dataset path.")
        return
    
    # Create advanced model
    model = create_ensemble_model(num_classes=len(class_names))
    
    # Define optimizer with more advanced settings
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Use class weighted loss to handle potential class imbalance
    criterion = nn.CrossEntropyLoss()
    
    # Use cosine annealing scheduler for better convergence
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )
    
    # Create learner instance
    learner = Learner(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=val_loader,
        optimizer=optimizer,
        loss=criterion,
        scheduler=scheduler,
        work_dir=work_dir,
        device=device
    )
    
    # Train the model
    history = learner.train(epochs=epochs)
    
    # Test the model
    test_loss, test_acc = learner.test()
    print(f"Final test accuracy: {test_acc:.2f}%")
    
    # Example inference
    try:
        prediction = learner.inference("./img_test/ong4.jpg")
        print(f"Prediction: {prediction}")
    except FileNotFoundError:
        print("Test image not found. Skipping inference demonstration.")

if __name__ == "__main__":
    main()
