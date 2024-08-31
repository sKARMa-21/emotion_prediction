from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io

# Initialize Flask application
application = Flask(__name__)

# Load the trained model
model = load_model('mood_detection_model.h5')

# Define a dictionary to map class indices to mood labels
class_labels = {0: 'happy', 1: 'sad', 2: 'neutral'}

# Root route to provide information about the API
@application.route('/')
def home():
    return 'Welcome to the Mood Detection API. Use POST /predict to get mood predictions.'

# Route to handle mood prediction requests
@application.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Load and preprocess the image
        img = image.load_img(io.BytesIO(file.read()), target_size=(128, 128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict the mood
        predictions = model.predict(img_array)
        mood_index = np.argmax(predictions[0])
        mood = class_labels.get(mood_index, 'unknown')

        return jsonify({'mood': mood})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the application
if __name__ == '__main__':
    application.run(debug=True)
