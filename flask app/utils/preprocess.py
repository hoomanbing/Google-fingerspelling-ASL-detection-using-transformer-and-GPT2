import cv2
import numpy as np

def preprocess_frame(frame):
    # Resize frame to model input size (e.g., 224x224)
    frame = cv2.resize(frame, (224, 224))
    
    # Normalize pixel values
    frame = frame / 255.0
    
    # Expand dimensions for model input
    frame = np.expand_dims(frame, axis=0)
    
    return frame
