import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers


# mnist = tf.keras.datasets.mnist
# (x_train, y_train) , (x_test,y_test) = mnist.load_data()
# x_train = tf.keras.utils.normalize(x_train, axis = 1)
# x_test = tf.keras.utils.normalize(x_test, axis = 1)

# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape = (28,28)))
# model.add(tf.keras.layers.Dense(128, activation = 'relu'))
# model.add(tf.keras.layers.Dense(128, activation = 'relu'))
# model.add(tf.keras.layers.Dense(10, activation = 'softmax'))
# model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics = ['accuracy'])

# model.fit(x_train, y_train, epochs = 3)
# model.save('handwritten.keras') 

# model = tf.keras.models.load_model('handwritten.keras') 

# img_num = 1
# while os.path.isfile(f"test_digits/digit{img_num}.png"):
#     try:
#         image = cv2.imread(f"test_digits/digit{img_num}.png")[:,:,0]
#         image - np.invert(np.array([image]))
#         prediction = model.predict(image)
#         print("digit is probably a {np.argmax(prediction)}")
#         plt.imshow(image[0], cmap=plt.cm.binary)
#         plt.show()
#     except:
#         print("Error!")
#     finally:
#         img_num +=1



model = tf.keras.models.load_model('handwritten.keras')

img_num = 1
while os.path.isfile(f"digit{img_num}.png"):
    try:
        image = cv2.imread(f"digit{img_num}.png", cv2.IMREAD_GRAYSCALE)
        _, image = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY_INV)
       
        if np.mean(image) > 127:
            image = np.invert(image)  # Inverting the image 
        image = cv2.resize(image, (28, 28)).reshape(1, 28, 28, 1) / 255.0  
        
        prediction = model.predict(image)
        predicted_digit = np.argmax(prediction)
        
        print(f"Digit is probably a {predicted_digit}")
        plt.imshow(image[0,:,:,0], cmap=plt.cm.binary)
        plt.show()
    except Exception as e:
        print(f"Error processing image: {e}")
    finally:
        img_num += 1

# model = tf.keras.models.load_model('handwritten.keras')

# def preprocess(img_pth):
#     image = cv2.imread(img_pth)
#     if image is None:
#         raise ValueError("Image not loaded properly")
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     _, thresh = cv2.threshold(hsv[:,:,2], 200, 255, cv2.THRESH_BINARY)
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not contours:
#         raise ValueError("No contours found, possibly no paper detected.")
#     largest_contour = max(contours, key=cv2.contourArea)
#     x,y,w,h = cv2.boundingRect(largest_contour)
#     cropped = image[y:y+h, x:x:w]
#     cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BAYER_BG2GRAY)
#     return cropped_gray

# def process_image():
#     img_num = 1
#     while os.path.isfile(f"digit{img_num}.png"):
#         try:
#             crop_image = preprocess(f"digit{img_num}.png")
#             _, processed_img = cv2.threshold(crop_image, 120,255, cv2.THRESH_BINARY_INV)
#             if np.mean(processed_img) > 127:
#                 processed_img = np.invert(processed_img)  # Inverting the image 
#             processed_img = cv2.resize(processed_img, (28, 28)).reshape(1, 28, 28, 1) / 255.0  
            
#             prediction = model.predict(processed_img)
#             predicted_digit = np.argmax(prediction)
            
#             print(f"Digit is probably a {predicted_digit}")
#             plt.imshow(processed_img[0,:,:,0], cmap=plt.cm.binary)
#             plt.show()
#         except Exception as e:
#             print(f"Error processing image: {e}")
#         finally:
#             img_num += 1
            
            
# process_image()

