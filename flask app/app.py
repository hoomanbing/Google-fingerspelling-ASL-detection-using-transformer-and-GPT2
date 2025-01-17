from flask import Flask, render_template, request, jsonify
import cv2
import torch
from tensorflow.keras.models import load_model
import numpy as np
from utils.preprocess import preprocess_frame

app = Flask(__name__)

# Load models
model1 = torch.load('models/asl_model_1.h5')
model2 = load_model('models/asl_model_2.h5')

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to process video frames (live detection)
@app.route('/detect', methods=['POST'])
def detect():
    if 'video_frame' not in request.files:
        return jsonify({'error': 'No video frame uploaded'}), 400
    
    # Get the frame from the request
    file = request.files['video_frame']
    frame = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Preprocess frame
    processed_frame = preprocess_frame(frame)
    
    # Make predictions
    predictions1 = model1(processed_frame)
    predictions2 = model2.predict(processed_frame)
    
    # Combine predictions (logic depends on your use case)
    combined_predictions = combine_predictions(predictions1, predictions2)
    
    return jsonify({'predictions': combined_predictions})

def combine_predictions(pred1, pred2):
    return (pred1 + pred2) / 2  # Example: average of two models

if __name__ == '__main__':
    app.run(debug=True)
