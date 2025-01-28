from flask import Flask, request, jsonify
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
from flask_cors import CORS

# Initialize the Flask app
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests for development/testing

# Load the model and processor
model_name = "Diginsa/Plant-Disease-Detection-Project"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)

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

        # Return the result as JSON
        return jsonify({
            "predicted_class": predicted_class,
            "confidence": round(confidence, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
