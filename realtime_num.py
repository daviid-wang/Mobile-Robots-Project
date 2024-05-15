import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import deque
import easyocr
import matplotlib.pyplot as plt

# Load the pre-trained model
model_path = os.path.join('TrainedModels', 'CNN-gnet-light-ultimate-data-15-epochs-128-batch-size.h5')
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

try:
    model = load_model(model_path)
except Exception as e:
    raise RuntimeError(f"Failed to load the model: {e}")

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Deque to hold recent predictions
recent_predictions = deque(maxlen=10)

def preprocess_roi(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    gray = cv2.resize(gray, (28, 28)).astype('float32') / 255.0
    gray_3ch = np.stack((gray,) * 3, axis=-1).reshape(1, 28, 28, 3)
    return gray_3ch

def show_frame(frame):
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.draw()
    plt.pause(0.001)

try:
    plt.ion()  # Turn on interactive mode
    fig = plt.figure()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply GaussianBlur to reduce noise and improve contour detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply edge detection
        edged = cv2.Canny(blurred, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)

            # Get the bounding box for the largest contour
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Extract the ROI
            roi = frame[y:y+h, x:x+w]

            # Preprocess the ROI
            processed_roi = preprocess_roi(roi)

            # Make a prediction using the CNN model
            prediction = model.predict(processed_roi)
            predicted_digit = np.argmax(prediction)
            recent_predictions.append(predicted_digit)

            # Smooth the prediction using the mode of recent predictions
            smoothed_prediction = max(set(recent_predictions), key=recent_predictions.count)

            # Draw the bounding box and prediction on the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'Digit: {smoothed_prediction}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Display the frame using matplotlib
        show_frame(frame)

        if plt.waitforbuttonpress(0.001):
            break
except Exception as e:
    print(f"Error: {e}")
finally:
    cap.release()
    plt.close()
