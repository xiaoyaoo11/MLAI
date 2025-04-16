import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from optimized_learner import (
    Learner,
    create_data_loaders,
    create_resnet_model,
    create_vgg_model,
)


def create_ensemble_model(num_classes):
    """Create a model with better architecture for higher accuracy"""
    # Use ResNet101 for better feature extraction
    model = models.resnet101(pretrained=True)
    
    # Freeze early layers to prevent overfitting
    for name, param in list(model.named_parameters())[:6*4]:  # Freeze first 6 blocks
        param.requires_grad = False
    
    # Add global average pooling and improved classification head
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.AdaptiveAvgPool1d(1) if num_features > 2048 else nn.Identity(),
        nn.Flatten(),
        nn.BatchNorm1d(num_features),
        nn.Dropout(0.4),
        nn.Linear(num_features, 1024),
        nn.LeakyReLU(0.2),
        nn.BatchNorm1d(1024),
        nn.Dropout(0.5),
        nn.Linear(1024, 512),
        nn.LeakyReLU(0.2),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )

    return model


def main():
    # Define paths and hyperparameters
    dataset_path = "dataset"
    work_dir = "checkpoints"
    batch_size = 16  # Smaller batch size for better generalization
    epochs = 40  # Increase number of epochs for deeper model
    learning_rate = 0.0002  # Lower learning rate for more stable training
    weight_decay = 5e-4  # Increased weight decay for better regularization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create data loaders with more aggressive data augmentation
    train_loader, val_loader, class_names = create_data_loaders(
        dataset_path=dataset_path, batch_size=batch_size, img_size=224
    )

    if train_loader is None or val_loader is None:
        print("Failed to load dataset. Please check the dataset path.")
        return

    # Create advanced model
    model = create_ensemble_model(num_classes=len(class_names))

    # Use different learning rates for different layers
    # Higher learning rate for new layers, lower for pre-trained layers
    params_to_update = []
    params_to_update_names = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
            params_to_update_names.append(name)
    
    # Define optimizer with more advanced settings
    optimizer = optim.AdamW(
        params_to_update,
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )

    # Use class weighted loss to handle potential class imbalance
    criterion = nn.CrossEntropyLoss()

    # One-cycle learning rate scheduler for better convergence
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=learning_rate * 10, 
        steps_per_epoch=len(train_loader), 
        epochs=epochs,
        pct_start=0.3,  # Spend 30% of time warming up
    )

    # Create learner instance with mixup data augmentation
    learner = Learner(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=val_loader,
        optimizer=optimizer,
        loss=criterion,
        scheduler=scheduler,
        work_dir=work_dir,
        device=device,
        mixup_alpha=0.4,  # Increased mixup for better robustness
        use_mixup=True,
    )

    # Train the model
    history = learner.train(epochs=epochs)

    # Load the best model for evaluation
    learner._load_best_model()
    
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
