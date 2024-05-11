import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import tensorflow as tf
from keras import layers
from keras import models


# mnist = tf.keras.datasets.mnist

# def model_build():
#     model = models.Sequential([
#         layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
#         layers.MaxPooling2D((2, 2)),
#         layers.Conv2D(64, (3, 3), activation='relu'),
#         layers.MaxPooling2D((2, 2)),
#         layers.Conv2D(64, (3, 3), activation='relu'),
#         layers.Flatten(),
#         layers.Dense(64, activation='relu'),
#         layers.Dense(10, activation='softmax')
#     ])
#     model.compile(optimizer='adam',
#                   loss='sparse_categorical_crossentropy',
#                   metrics=['accuracy'])
#     return model

# model = model_build()
# model.summary()

# # Load and prepare the MNIST dataset
# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
# test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# # Train the model
# model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_split=0.1)

# model.save('cnn_digit_recon.keras')

model = tf.keras.models.load_model('cnn_digit_recon.keras')

# def preprocess_and_predict(image_path):
#     try:
#         # Read the image in grayscale
#         image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#         if image is None:
#             raise ValueError("Image not found")

#         # Apply binary threshold inversion
#         _, image = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY_INV)

#         # Invert the image if it is mostly white
#         if np.mean(image) > 127:
#             image = np.invert(image)

#         # Resize image to 28x28 pixels as required by the model
#         image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
#         image = image.reshape(1, 28, 28, 1) / 255.0  # Normalize and add batch dimension

#         # Predict the digit
#         prediction = model.predict(image)
#         predicted_digit = np.argmax(prediction)

#         # Display the image and prediction
#         plt.imshow(image.reshape(28, 28), cmap='gray')  # Reshape back to 2D for display
#         plt.title(f"Predicted digit: {predicted_digit}")
#         plt.show()

#     except Exception as e:
#         print(f"Error processing image {image_path}: {e}")

# # Loop through files assuming a naming convention "digit1.png", "digit2.png", etc.
# img_num = 1
# while os.path.isfile(f"digit{img_num}.png"):
#     preprocess_and_predict(f"digit{img_num}.png")
#     img_num += 1

        
# # def img_preprocess(img):
# #     image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
# #     image = cv2.resize(image, (28,28), interpolation=cv2.INTER_AREA)
# #     image = image.astype('float32') / 255
# #     image = np.expand_dims(image, axis=0)
# #     image = np.expand_dims(image, axis = -1)
# #     return image

# # def digit(imgpth, model):
# #     image = img_preprocess(imgpth)
# #     prediction = model.predict(image)
# #     predicted_digit = np.argmax(prediction)
    
# #     print(f"Digit is probably a {predicted_digit}")
# #     plt.imshow(image[0,:,:,0], cmap=plt.cm.binary)
# #     plt.show()
# #     return predicted_digit

# # predicted_digit = digit("digit8.png", model)
# # print(f"Predicted Digit: {predicted_digit}")

def preprocess_image(image_path):
    """ Load and preprocess the image. """
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
    digit_contour = max(contours, key=cv2.contourArea) if contours else None
    if not digit_contour:
        raise ValueError("No contours found in the image.")
    
    x, y, w, h = cv2.boundingRect(digit_contour)
    img = img[y:y+h, x:x+w]  # Crop the digit region
    
    img = cv2.resize(img, (28, 28))  # Resize to network input size
    img = img / 255.0  # Normalize the image
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

