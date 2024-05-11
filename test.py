import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

model = tf.keras.models.load_model('handwritten.keras')

def preprocess(img_pth):
    image = cv2.imread(img_pth)
    if image is None or image.size == 0:
        raise ValueError("Image not loaded properly or is empty")
    
    # Convert to HSV color space and threshold
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, thresh = cv2.threshold(hsv[:,:,2], 200, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found, possibly no paper detected.")
    
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    if w*h == 0:
        raise ValueError("Bounding rectangle has zero area.")
    
    cropped = image[y:y+h, x:x+w]
    cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    return cropped_gray

def process_image():
    img_num = 1
    while os.path.isfile(f"digit{img_num}.png"):
        try:
            crop_image = preprocess(f"digit{img_num}.png")
            _, processed_img = cv2.threshold(crop_image, 120, 255, cv2.THRESH_BINARY_INV)
            if np.mean(processed_img) > 127:
                processed_img = np.invert(processed_img)
            
            processed_img = cv2.resize(processed_img, (28, 28)).reshape(1, 28, 28, 1) / 255.0
            
            prediction = model.predict(processed_img)
            predicted_digit = np.argmax(prediction)
            
            print(f"Digit is probably a {predicted_digit}")
            plt.imshow(processed_img[0, :, :, 0], cmap=plt.cm.binary)
            plt.show()
        except Exception as e:
            print(f"Error processing image digit{img_num}.png: {e}")
        finally:
            img_num += 1

process_image()
