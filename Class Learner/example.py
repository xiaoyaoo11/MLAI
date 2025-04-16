import torch
import torch.nn as nn
import torch.optim as optim
from optimized_learner import Learner, create_resnet_model, create_vgg_model, create_data_loaders

def main():
    """
    Example demonstrating how to use the Learner class with different models
    """
    # Dataset path
    dataset_path = "dataset"
    
    # Create data loaders
    train_loader, val_loader, class_names = create_data_loaders(
        dataset_path=dataset_path,
        batch_size=32
    )
    
    if train_loader is None:
        print("Failed to load dataset!")
        return
    
    # Create a ResNet model
    print("\n=== Training with ResNet model ===")
    resnet_model = create_resnet_model(num_classes=len(class_names))
    
    # Define optimizer, loss, and scheduler
    optimizer = optim.Adam(resnet_model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    
    # Create and train the learner
    resnet_learner = Learner(
        model=resnet_model,
        train_dataloader=train_loader,
        test_dataloader=val_loader,
        optimizer=optimizer,
        loss=criterion,
        scheduler=scheduler,
        work_dir="checkpoints/resnet"
    )
    
    # Train for 5 epochs
    resnet_learner.train(epochs=5)
    
    # Test the model
    resnet_loss, resnet_acc = resnet_learner.test()
    print(f"ResNet Test Accuracy: {resnet_acc:.2f}%")
    
    # Create a VGG model
    print("\n=== Training with VGG model ===")
    vgg_model = create_vgg_model(num_classes=len(class_names))
    
    # Define optimizer, loss, and scheduler for VGG
    vgg_optimizer = optim.SGD(vgg_model.parameters(), lr=0.001, momentum=0.9)
    vgg_scheduler = optim.lr_scheduler.StepLR(vgg_optimizer, step_size=3, gamma=0.1)
    
    # Create and train the VGG learner
    vgg_learner = Learner(
        model=vgg_model,
        train_dataloader=train_loader,
        test_dataloader=val_loader,
        optimizer=vgg_optimizer,
        loss=criterion,
        scheduler=vgg_scheduler,
        work_dir="checkpoints/vgg"
    )
    
    # Train for 5 epochs
    vgg_learner.train(epochs=5)
    
    # Test the model
    vgg_loss, vgg_acc = vgg_learner.test()
    print(f"VGG Test Accuracy: {vgg_acc:.2f}%")
    
    # Compare models
    print("\n=== Model Comparison ===")
    print(f"ResNet Accuracy: {resnet_acc:.2f}%")
    print(f"VGG Accuracy: {vgg_acc:.2f}%")
    
    # Use the better model for inference
    if resnet_acc > vgg_acc:
        print("ResNet performed better, using for inference")
        best_learner = resnet_learner
    else:
        print("VGG performed better, using for inference")
        best_learner = vgg_learner
    
    # Example inference
    try:
        image_path = "./img_test/ong4.jpg"
        prediction = best_learner.inference(image_path)
        print(f"Prediction for {image_path}: {prediction}")
    except FileNotFoundError:
        print("Test image not found. Skipping inference demonstration.")

if __name__ == "__main__":
    main()
