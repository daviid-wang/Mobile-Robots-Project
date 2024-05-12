import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

# model = tf.keras.models.load_model('handwritten.keras')

# def preprocess(img_pth):
#     image = cv2.imread(img_pth)
#     if image is None or image.size == 0:
#         raise ValueError("Image not loaded properly or is empty")
    
#     # Convert to HSV color space and threshold
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     _, thresh = cv2.threshold(hsv[:,:,2], 200, 255, cv2.THRESH_BINARY)
    
#     # Find contours
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not contours:
#         raise ValueError("No contours found, possibly no paper detected.")
    
#     largest_contour = max(contours, key=cv2.contourArea)
#     x, y, w, h = cv2.boundingRect(largest_contour)
#     if w*h == 0:
#         raise ValueError("Bounding rectangle has zero area.")
    
#     cropped = image[y:y+h, x:x+w]
#     cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
#     return cropped_gray

# def process_image():
#     img_num = 1
#     while os.path.isfile(f"digit{img_num}.png"):
#         try:
#             crop_image = preprocess(f"digit{img_num}.png")
#             _, processed_img = cv2.threshold(crop_image, 120, 255, cv2.THRESH_BINARY_INV)
#             if np.mean(processed_img) > 127:
#                 processed_img = np.invert(processed_img)
            
#             processed_img = cv2.resize(processed_img, (28, 28)).reshape(1, 28, 28, 1) / 255.0
            
#             prediction = model.predict(processed_img)
#             predicted_digit = np.argmax(prediction)
            
#             print(f"Digit is probably a {predicted_digit}")
#             plt.imshow(processed_img[0, :, :, 0], cmap=plt.cm.binary)
#             plt.show()
#         except Exception as e:
#             print(f"Error processing image digit{img_num}.png: {e}")
#         finally:
#             img_num += 1

# process_image()


# import cv2
# import matplotlib.pyplot as plt

# def visualize_preprocessing_steps(img_path):
#     image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#     if image is None:
#         print("Image not loaded.")
#         return
    
#     plt.figure(figsize=(10, 8))
    
#     # Original Image
#     plt.subplot(2, 3, 1)
#     plt.imshow(image, cmap='gray')
#     plt.title("Original Image")
    
#     # Adaptive Thresholding
#     thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                    cv2.THRESH_BINARY_INV, 11, 2)
#     plt.subplot(2, 3, 2)
#     plt.imshow(thresh, cmap='gray')
#     plt.title("Adaptive Thresholding")

#     # Morphological Opening
#     kernel = np.ones((3,3),np.uint8)
#     opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
#     plt.subplot(2, 3, 3)
#     plt.imshow(opening, cmap='gray')
#     plt.title("Morphological Opening")

#     # Find Contours and Bounding Box
#     contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if contours:
#         largest_contour = max(contours, key=cv2.contourArea)
#         x, y, w, h = cv2.boundingRect(largest_contour)
#         cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
#     plt.subplot(2, 3, 4)
#     plt.imshow(image, cmap='gray')
#     plt.title("Contour with Bounding Box")

#     # Cropped Image
#     cropped = image[y:y+h, x:x+w]
#     plt.subplot(2, 3, 5)
#     plt.imshow(cropped, cmap='gray')
#     plt.title("Cropped Image")

#     plt.tight_layout()
#     plt.show()

# # Replace 'path_to_image.jpg' with your actual image path
# visualize_preprocessing_steps('digit8.png')

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

# Load your model
model = tf.keras.models.load_model('handwritten.keras')

def preprocess_image(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not loaded.")

    # Less aggressive thresholding
    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 19, 5)
    
    # Smaller morphological operation
    kernel = np.ones((2, 2), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, kernel)

    # Find contours and crop loosely
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        margin = 5  # Reduced margin
        x, y = max(0, x-margin), max(0, y-margin)
        w, h = w + 2*margin, h + 2*margin
        cropped = image[y:y+h, x:x+w]
        cropped = cv2.resize(cropped, (28, 28))
        return cropped
    else:
        raise ValueError("No suitable contour found.")

# Testing the preprocessing
test_img_path = 'digit1.png'
processed_digit = preprocess_image(test_img_path)
plt.imshow(processed_digit, cmap='gray')
plt.title("Processed Digit Image")
plt.show()


def process_image():
    img_num = 1
    while os.path.isfile(f"digit{img_num}.png"):
        try:
            crop_image = preprocess_image(f"digit{img_num}.png")
            processed_img = crop_image.astype('float32') / 255.0
            processed_img = processed_img.reshape(1, 28, 28, 1)
            
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
