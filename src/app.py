import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import io

app = Flask(__name__)
MODEL_PATH = 'model.h5'

# Load Model Safely
model = None
try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded.")
except:
    print("⚠️ Model not found. Run train.py first.")

def preprocess_image(image_bytes):
    """
    Standardize image to match training (224x224, RGB, 1/255)
    """
    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize((224, 224)) # ResNet Size
    if img.mode != "RGB":
        img = img.convert("RGB")
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file sent'}), 400
    
    file = request.files['file']
    img_array = preprocess_image(file.read())
    
    predictions = model.predict(img_array)
    class_id = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]))
    
    return jsonify({
        'class_id': class_id,
        'confidence': confidence
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
