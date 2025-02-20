from flask import Flask, request, jsonify
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
from flask_cors import CORS
import gc

# Initialize the Flask app
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests for development/testing

# Define model loading as a function to optimize memory usage
def load_model():
    model_name = "Diginsa/Plant-Disease-Detection-Project"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    return processor, model

@app.route("/")
def home():
    return jsonify({"message": "Welcome to the Plant Disease Detection API!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if an image file is provided
        if "image" not in request.files:
            return jsonify({"error": "No image provided. Please upload an image."}), 400

        # Load the image from the request
        image_file = request.files["image"]
        image = Image.open(image_file).convert("RGB")

        # Load the model and processor (lazy loading)
        processor, model = load_model()

        # Preprocess the image
        inputs = processor(images=image, return_tensors="pt")

        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Extract predictions
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        predicted_class = model.config.id2label[predicted_class_idx]
        confidence = torch.nn.functional.softmax(logits, dim=-1)[0][predicted_class_idx].item()

        # Construct the response
        response = {
            "data": [
                "Model loaded successfully.",
                f"Predicted Class: {predicted_class}",
                f"Confidence Score: {round(confidence, 2)}"
            ],
            "statusCode": 200,
            "message": "Disease prediction data fetched successfully",
            "success": True
        }

        # Explicitly release model and processor memory
        del model, processor
        gc.collect()

        return jsonify(response)

    except Exception as e:
        # Error response format
        response = {
            "data": [],
            "statusCode": 500,
            "message": f"An error occurred: {str(e)}",
            "success": False
        }
        return jsonify(response), 500

if __name__ == "__main__":
    # Use production-ready Gunicorn with limited workers and threads if required
    app.run(debug=True, host="0.0.0.0", port=5000)
