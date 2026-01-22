import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import io

# Initialize Flask
app = Flask(__name__)

# Global Configuration
MODEL_PATH = 'model.h5'
IMG_SIZE = (224, 224)

# Load Model (Global scope to avoid reloading on every request)
print(" Loading Model...")
try:
    model = load_model(MODEL_PATH)
    print(" Model loaded successfully.")
except OSError:
    print("  WARNING: model.h5 not found. Please run src/train.py first!")
    model = None

def preprocess_image(image_bytes):
    """
    Converts raw bytes to a normalized numpy array for the model.
    """
    # 1. Open image
    img = Image.open(io.BytesIO(image_bytes))
    
    # 2. Resize to match Model Input (224x224)
    img = img.resize(IMG_SIZE)
    
    # 3. Ensure 3 Channels (RGB)
    if img.mode != "RGB":
        img = img.convert("RGB")
        
    # 4. Convert to Array & Normalize (0-1)
    img_array = np.array(img) / 255.0
    
    # 5. Add Batch Dimension (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check to see if API is running"""
    return jsonify({'status': 'active', 'model_loaded': model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint: /predict
    Method: POST
    Body: Form-data with key 'file' (Image)
    """
    if not model:
        return jsonify({'error': 'Model not trained yet. Run train.py.'}), 503

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Preprocess
        processed_img = preprocess_image(file.read())
        
        # Inference
        predictions = model.predict(processed_img)
        
        # Post-process
        class_index = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        
        return jsonify({
            'class_index': class_index,
            'confidence': round(confidence * 100, 2),
            'message': 'Success'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the server on Port 5000
    app.run(host='0.0.0.0', port=5000)
