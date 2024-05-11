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

# model.fit(x_train, y_train, epochs = 20)
# model.save('live_feed_handwritten.keras') 
cap = cv2.VideoCapture(0)
model = tf.keras.models.load_model('handwritten.keras')
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       # _, gray = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 11, 2)
    
        # if np.mean(gray) > 127:
        #     gray = np.invert(gray)
        gray = cv2.resize(gray, (28, 28)).reshape(1, 28, 28, 1) / 255.0
        prediction = model.predict(gray)
        predicted_digit = np.argmax(prediction)
        cv2.putText(frame, f'Digit: {predicted_digit}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as e:
    print(f"Error: {e}")
finally:

    cap.release()
    cv2.destroyAllWindows()