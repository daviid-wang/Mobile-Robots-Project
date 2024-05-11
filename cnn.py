import os
import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

model = tf.keras.models.load_model('cnn_digit_recon.keras')

def preprocess_image(image_path):
    """Load and preprocess the image."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not loaded properly")

    # Apply Gaussian Blur
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Adaptive Thresholding
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours and select the largest one
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in the image.")
    
    digit_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(digit_contour)
    img = img[y:y+h, x:x+w]  # Crop the digit region

    # Ensure the cropped region is not empty
    if img.size == 0:
        raise ValueError("Cropped image region is empty.")

    img = cv2.resize(img, (28, 28))  # Resize to network input size
    img = img.astype('float32') / 255.0  # Normalize the image
    img = img.reshape(1, 28, 28, 1)  # Reshape for model prediction

    return img

def predict_digit(img_path):
    try:
        img = preprocess_image(img_path)
        pred = model.predict(img)
        digit = np.argmax(pred)
        return digit
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

# Example usage
image_path = 'digit1.png'
predicted_digit = predict_digit(image_path)
if predicted_digit is not None:
    print("Predicted Digit:", predicted_digit)
    plt.imshow(cv2.imread(image_path), cmap='gray')
    plt.title(f"Predicted Digit: {predicted_digit}")
    plt.show()
else:
    print("Failed to process image.")
