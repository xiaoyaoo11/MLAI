import argparse
import os

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

from optimized_learner import Learner, create_data_loaders
from train import create_ensemble_model


def load_model(checkpoint_path, num_classes, device):
    """
    Load a model from checkpoint

    Args:
        checkpoint_path: Path to the checkpoint file
        num_classes: Number of output classes
        device: Device to load model to

    Returns:
        Loaded model
    """
    # Create model with same architecture as in training
    model = create_ensemble_model(num_classes=num_classes)

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Model loaded from {checkpoint_path}")
    print(f"Best accuracy: {checkpoint['best_acc']:.2f}%")

    return model


def predict_single_image(model, image_path, class_names, device):
    """
    Make a prediction on a single image

    Args:
        model: Loaded PyTorch model
        image_path: Path to image file
        class_names: List of class names
        device: Device to perform inference on

    Returns:
        Predicted class name and probability
    """
    # Image preprocessing - similar to validation transform in create_data_loaders
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load and preprocess the image
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None, None

    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_prob, top_class = torch.max(probabilities, 1)

    # Get class name and probability
    prediction_idx = top_class.item()
    probability = top_prob.item() * 100

    if class_names and prediction_idx < len(class_names):
        prediction = class_names[prediction_idx]
    else:
        prediction = str(prediction_idx)

    return prediction, probability


def batch_predict(model, image_folder, class_names, device):
    """
    Make predictions on all images in a folder

    Args:
        model: Loaded PyTorch model
        image_folder: Path to folder containing images
        class_names: List of class names
        device: Device to perform inference on
    """
    supported_extensions = [".jpg", ".jpeg", ".png", ".bmp"]

    for filename in os.listdir(image_folder):
        file_path = os.path.join(image_folder, filename)
        if os.path.isfile(file_path) and any(
            filename.lower().endswith(ext) for ext in supported_extensions
        ):
            prediction, probability = predict_single_image(
                model, file_path, class_names, device
            )
            if prediction:
                print(
                    f"Image: {filename}, Prediction: {prediction}, Confidence: {probability:.2f}%"
                )


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Predict images using trained model")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best_model.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--dataset_path", default="dataset", help="Path to dataset for class names"
    )
    parser.add_argument("--image", help="Path to single image for prediction")
    parser.add_argument(
        "--image_folder", help="Path to folder of images for batch prediction"
    )
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create data loaders to get class names (we don't need the loaders themselves)
    _, _, class_names = create_data_loaders(
        dataset_path=args.dataset_path, batch_size=1, img_size=224
    )

    if not class_names:
        print("Failed to load class names from dataset. Using numeric indices instead.")
        class_names = None
    else:
        print(f"Classes: {class_names}")

    # Load model from checkpoint
    num_classes = (
        len(class_names) if class_names else 1000
    )  # Default to ImageNet classes if no class_names
    model = load_model(args.checkpoint, num_classes, device)

    # Make predictions
    if args.image:
        # Single image prediction
        prediction, probability = predict_single_image(
            model, args.image, class_names, device
        )
        if prediction:
            print(f"Prediction: {prediction}")
            print(f"Confidence: {probability:.2f}%")
    elif args.image_folder:
        # Batch prediction on a folder of images
        print(f"Performing batch prediction on images in {args.image_folder}")
        batch_predict(model, args.image_folder, class_names, device)
    else:
        print("Please provide either --image or --image_folder argument")
        print("Example usage:")
        print("  python predict.py --image ./img_test/sample.jpg")
        print("  python predict.py --image_folder ./img_test")


if __name__ == "__main__":
    main()
