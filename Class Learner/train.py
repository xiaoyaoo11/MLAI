import torch
import torch.nn as nn
import torch.optim as optim
from optimized_learner import Learner, create_resnet_model, create_data_loaders

def main():
    # Define paths and hyperparameters
    dataset_path = "dataset"
    work_dir = "checkpoints"
    batch_size = 32
    epochs = 10
    learning_rate = 0.001
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Create data loaders
    train_loader, val_loader, class_names = create_data_loaders(
        dataset_path=dataset_path,
        batch_size=batch_size
    )
    
    if train_loader is None or val_loader is None:
        print("Failed to load dataset. Please check the dataset path.")
        return
    
    # Create model
    model = create_resnet_model(num_classes=len(class_names))
    
    # Define optimizer, loss function and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
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
    learner.train(epochs=epochs)
    
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
