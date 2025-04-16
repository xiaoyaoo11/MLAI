import os

import torch
from flask import Flask, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from optimized_learner import create_data_loaders
from predict import load_model, predict_single_image

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Configure allowed extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Set checkpoint path and dataset path
CHECKPOINT_PATH = "checkpoints/best_model.pth"
DATASET_PATH = "dataset"

# Load model and class names
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_, _, class_names = create_data_loaders(
    dataset_path=DATASET_PATH, batch_size=1, img_size=224
)
num_classes = len(class_names) if class_names else 1000
model = None


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def upload_file():
    global model

    # Load model if not already loaded
    if model is None:
        try:
            model = load_model(CHECKPOINT_PATH, num_classes, device)
        except Exception as e:
            return render_template("error.html", error=f"Error loading model: {str(e)}")

    result = None
    if request.method == "POST":
        # Check if the post request has the file part
        if "file" not in request.files:
            return render_template("upload.html", error="No file part")

        file = request.files["file"]

        # If user does not select file, browser also submits an empty part
        if file.filename == "":
            return render_template("upload.html", error="No selected file")

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Make prediction
            try:
                prediction, probability = predict_single_image(
                    model, filepath, class_names, device
                )
                result = {
                    "image": os.path.join("uploads", filename),
                    "prediction": prediction,
                    "confidence": f"{probability:.2f}%",
                }
            except Exception as e:
                result = {
                    "image": os.path.join("uploads", filename),
                    "error": f"Error making prediction: {str(e)}",
                }

            return render_template("upload.html", result=result)

    return render_template("upload.html")


if __name__ == "__main__":
    app.run(debug=True)
