# import os
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf
# from keras import layers

# # Check if cv2.VideoCapture is available
# if not hasattr(cv2, 'VideoCapture'):
#     raise ImportError("OpenCV installation is incomplete. 'cv2.VideoCapture' is not available.")

# # Load and preprocess the MNIST dataset
# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train = tf.keras.utils.normalize(x_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)

# # Define the model
# model = tf.keras.Sequential([
#     tf.keras.layers.Flatten(input_shape=(28, 28)),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(10, activation='softmax')
# ])

# # Compile the model
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # Train the model
# model.fit(x_train, y_train, epochs=20)

# # Save the trained model
# model.save('live_feed_handwritten.keras')

# # Load the trained model
# model = tf.keras.models.load_model('live_feed_handwritten.keras')

# # Initialize the video capture
# cap = cv2.VideoCapture(0)

# def show_frame(frame):
#     plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     plt.axis('off')
#     plt.draw()
#     plt.pause(0.001)
#     plt.clf()

# try:
#     plt.ion()  # Turn on interactive mode
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to grab frame")
#             break

#         # Convert the frame to grayscale
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Apply adaptive thresholding
#         gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

#         # Resize and normalize the image
#         gray = cv2.resize(gray, (28, 28)).reshape(1, 28, 28, 1) / 255.0

#         # Make a prediction
#         prediction = model.predict(gray)
#         predicted_digit = np.argmax(prediction)

#         # Display the result
#         cv2.putText(frame, f'Digit: {predicted_digit}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

#         # Show the frame using Matplotlib
#         show_frame(frame)

#         if plt.waitforbuttonpress(0.001):
#             break
# except Exception as e:
#     print(f"Error: {e}")
# finally:
#     cap.release()
#     plt.ioff()
#     plt.close()

# import cv2
# import numpy as np
# import easyocr
# import matplotlib.pyplot as plt
# from collections import deque, Counter
# import time

# # Initialize EasyOCR reader
# reader = easyocr.Reader(['en'])

# # Initialize the video capture
# cap = cv2.VideoCapture(0)

# # Create a figure and axes for plotting
# fig, axes = plt.subplots(1, 4, figsize=(20, 5))
# axes[0].set_title("Original Frame")
# axes[0].axis('off')
# axes[1].set_title("Grayscale Frame")
# axes[1].axis('off')
# axes[2].set_title("Thresholded Frame")
# axes[2].axis('off')
# axes[3].set_title("OCR Result")
# axes[3].axis('off')

# def show_frame(orig_frame, gray_frame, thresh_frame, final_frame):
#     axes[0].imshow(cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB))
#     axes[1].imshow(gray_frame, cmap='gray')
#     axes[2].imshow(thresh_frame, cmap='gray')
#     axes[3].imshow(final_frame, cmap='gray')
#     plt.draw()
#     plt.pause(0.001)

# def preprocess_for_ocr(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     edged = cv2.Canny(blurred, 50, 150)
#     return edged

# def get_stable_prediction(predictions):
#     return Counter(predictions).most_common(1)[0][0]

# def capture_predictions_for_duration(duration=5):
#     start_time = time.time()
#     frame_buffer = deque()
    
#     while time.time() - start_time < duration:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to grab frame")
#             continue
        
#         # Original frame
#         orig_frame = frame.copy()

#         # Convert the frame to grayscale
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Apply adaptive thresholding
#         thresh_frame = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
#         # Find contours
#         contours, _ = cv2.findContours(thresh_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         if contours:
#             # Filter contours by area to avoid small noisy contours
#             filtered_contours = [c for c in contours if cv2.contourArea(c) > 1000]
#             if filtered_contours:
#                 # Find the largest contour
#                 largest_contour = max(filtered_contours, key=cv2.contourArea)
#                 x, y, w, h = cv2.boundingRect(largest_contour)
                
#                 # Extract ROI
#                 roi = frame[y:y+h, x:x+w]
                
#                 # Preprocess the ROI for OCR
#                 processed_roi = preprocess_for_ocr(roi)

#                 # Use EasyOCR to recognize text
#                 results = reader.readtext(processed_roi, detail=0, paragraph=False)

#                 if results:
#                     # Filter out non-digit characters
#                     recognized_digits = [r for r in results if r.isdigit()]
#                     if recognized_digits:
#                         predicted_digit = max(set(recognized_digits), key=recognized_digits.count)
#                         frame_buffer.append(predicted_digit)
                
#                 # Display frames for debugging
#                 final_frame = cv2.resize(processed_roi, (320, 320))  # Upscale for better visibility
#                 if results:
#                     final_frame = cv2.putText(final_frame, f'Digit: {predicted_digit}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
#                 else:
#                     final_frame = cv2.putText(final_frame, f'Digit: None', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

#                 show_frame(orig_frame, gray, thresh_frame, final_frame)
                
#         if plt.waitforbuttonpress(0.001):
#             break
    
#     return frame_buffer

# try:
#     plt.ion()  # Turn on interactive mode
#     stable_prediction = None  # To store the final stable prediction
    
#     while True:
#         frame_buffer = capture_predictions_for_duration(duration=5)
        
#         if frame_buffer:
#             # Get the most frequent prediction
#             stable_prediction = get_stable_prediction(frame_buffer)
            
#             # Display the final stable prediction on the terminal
#             print(f"Final Stable Prediction: {stable_prediction}")
            
#             # Show the final stable prediction continuously
#             while True:
#                 ret, frame = cap.read()
#                 if not ret:
#                     print("Failed to grab frame")
#                     break

#                 # Convert the frame to grayscale
#                 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
#                 # Apply adaptive thresholding
#                 thresh_frame = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
                
#                 # Preprocessed frame with prediction
#                 final_frame = cv2.resize(thresh_frame, (320, 320))  # Upscale for better visibility
#                 final_frame = cv2.putText(final_frame, f'Digit: {stable_prediction}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                
#                 show_frame(frame, gray, thresh_frame, final_frame)
                
#                 if plt.waitforbuttonpress(0.001):
#                     break

#         if plt.waitforbuttonpress(0.001):
#             break
# except Exception as e:
#     print(f"Error: {e}")
# finally:
#     cap.release()
#     plt.ioff()
#     plt.close()


# import cv2
# import numpy as np
# import easyocr
# import matplotlib.pyplot as plt
# from collections import deque, Counter
# import time

# # Initialize EasyOCR reader
# reader = easyocr.Reader(['en'])

# # Initialize the video capture
# cap = cv2.VideoCapture(0)

# # Create a figure and axes for plotting
# fig, axes = plt.subplots(1, 4, figsize=(20, 5))
# axes[0].set_title("Original Frame")
# axes[0].axis('off')
# axes[1].set_title("Grayscale Frame")
# axes[1].axis('off')
# axes[2].set_title("Thresholded Frame")
# axes[2].axis('off')
# axes[3].set_title("OCR Result")
# axes[3].axis('off')

# def show_frame(orig_frame, gray_frame, thresh_frame, final_frame):
#     axes[0].imshow(cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB))
#     axes[1].imshow(gray_frame, cmap='gray')
#     axes[2].imshow(thresh_frame, cmap='gray')
#     axes[3].imshow(final_frame, cmap='gray')
#     plt.draw()
#     plt.pause(0.001)

# def preprocess_for_ocr(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     edged = cv2.Canny(blurred, 50, 150)
#     return edged

# def get_stable_prediction(predictions):
#     return Counter(predictions).most_common(1)[0][0]

# def capture_predictions_for_duration(duration=5):
#     start_time = time.time()
#     frame_buffer = deque()
    
#     while time.time() - start_time < duration:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to grab frame")
#             continue
        
#         # Original frame
#         orig_frame = frame.copy()

#         # Convert the frame to grayscale
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Apply adaptive thresholding
#         thresh_frame = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
#         # Find contours
#         contours, _ = cv2.findContours(thresh_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         if contours:
#             # Filter contours by area to avoid small noisy contours
#             filtered_contours = [c for c in contours if cv2.contourArea(c) > 1000]
#             if filtered_contours:
#                 # Find the largest contour
#                 largest_contour = max(filtered_contours, key=cv2.contourArea)
#                 x, y, w, h = cv2.boundingRect(largest_contour)
                
#                 # Extract ROI
#                 roi = frame[y:y+h, x:x+w]
                
#                 # Preprocess the ROI for OCR
#                 processed_roi = preprocess_for_ocr(roi)

#                 # Use EasyOCR to recognize text
#                 results = reader.readtext(processed_roi, detail=0, paragraph=False)

#                 if results:
#                     # Filter out non-digit characters
#                     recognized_digits = [r for r in results if r.isdigit()]
#                     if recognized_digits:
#                         predicted_digit = max(set(recognized_digits), key=recognized_digits.count)
#                         frame_buffer.append(predicted_digit)
                
#                 # Display frames for debugging
#                 final_frame = cv2.resize(processed_roi, (320, 320))  # Upscale for better visibility
#                 if results:
#                     final_frame = cv2.putText(final_frame, f'Digit: {predicted_digit}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
#                 else:
#                     final_frame = cv2.putText(final_frame, f'Digit: None', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

#                 show_frame(orig_frame, gray, thresh_frame, final_frame)
                
#         if plt.waitforbuttonpress(0.001):
#             break
    
#     return frame_buffer

# try:
#     plt.ion()  # Turn on interactive mode
#     all_predictions = deque()  # To store all predictions
    
#     while True:
#         frame_buffer = capture_predictions_for_duration(duration=5)
        
#         if frame_buffer:
#             # Update all predictions
#             all_predictions.extend(frame_buffer)
            
#             # Get the most frequent prediction
#             stable_prediction = get_stable_prediction(all_predictions)
            
#             # Display the final stable prediction on the terminal
#             print(f"Current Stable Prediction: {stable_prediction}")
#             print(f"All Predictions: {list(all_predictions)}")
            
#             # Show the final stable prediction continuously
#             while True:
#                 ret, frame = cap.read()
#                 if not ret:
#                     print("Failed to grab frame")
#                     break

#                 # Convert the frame to grayscale
#                 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
#                 # Apply adaptive thresholding
#                 thresh_frame = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
                
#                 # Preprocessed frame with prediction
#                 final_frame = cv2.resize(thresh_frame, (320, 320))  # Upscale for better visibility
#                 final_frame = cv2.putText(final_frame, f'Digit: {stable_prediction}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                
#                 show_frame(frame, gray, thresh_frame, final_frame)
                
#                 if plt.waitforbuttonpress(0.001):
#                     break

# except Exception as e:
#     print(f"Error: {e}")
# finally:
#     cap.release()
#     plt.ioff()
#     plt.close()
#     print(f"Final List of Predictions: {list(all_predictions)}")


#TEST!








#TEST
# import cv2
# import numpy as np
# import easyocr
# import matplotlib.pyplot as plt
# from collections import deque, Counter
# import time

# # Initialize EasyOCR reader
# reader = easyocr.Reader(['en'])

# # Initialize the video capture
# cap = cv2.VideoCapture(0)

# # Create a figure and axes for plotting
# fig, axes = plt.subplots(1, 4, figsize=(20, 5))
# axes[0].set_title("Original Frame")
# axes[0].axis('off')
# axes[1].set_title("Grayscale Frame")
# axes[1].axis('off')
# axes[2].set_title("Thresholded Frame")
# axes[2].axis('off')
# axes[3].set_title("OCR Result")
# axes[3].axis('off')

# def show_frame(orig_frame, gray_frame, thresh_frame, final_frame):
#     axes[0].imshow(cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB))
#     axes[1].imshow(gray_frame, cmap='gray')
#     axes[2].imshow(thresh_frame, cmap='gray')
#     axes[3].imshow(final_frame, cmap='gray')
#     plt.draw()
#     plt.pause(0.001)

# def preprocess_for_ocr(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     edged = cv2.Canny(blurred, 50, 150)
#     return edged

# def get_stable_prediction(predictions):
#     return Counter(predictions).most_common(1)[0][0]

# def capture_predictions_for_duration(duration=5):
#     start_time = time.time()
#     frame_buffer = deque()
    
#     while time.time() - start_time < duration:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to grab frame")
#             continue
        
#         # Original frame
#         orig_frame = frame.copy()

#         # Convert the frame to grayscale
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Apply adaptive thresholding
#         thresh_frame = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
#         # Find contours
#         contours, _ = cv2.findContours(thresh_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         if contours:
#             # Filter contours by area to avoid small noisy contours
#             filtered_contours = [c for c in contours if cv2.contourArea(c) > 1000]
#             if filtered_contours:
#                 # Find the largest contour
#                 largest_contour = max(filtered_contours, key=cv2.contourArea)
#                 x, y, w, h = cv2.boundingRect(largest_contour)
                
#                 # Extract ROI
#                 roi = frame[y:y+h, x:x+w]
                
#                 # Preprocess the ROI for OCR
#                 processed_roi = preprocess_for_ocr(roi)

#                 # Use EasyOCR to recognize text
#                 results = reader.readtext(processed_roi, detail=0, paragraph=False)

#                 predicted_digit = None  # Initialize the variable
#                 if results:
#                     # Filter out non-digit characters
#                     recognized_digits = [r for r in results if r.isdigit()]
#                     if recognized_digits:
#                         predicted_digit = max(set(recognized_digits), key=recognized_digits.count)
#                         frame_buffer.append(predicted_digit)
                
#                 # Display frames for debugging
#                 final_frame = cv2.resize(processed_roi, (320, 320))  # Upscale for better visibility
#                 if predicted_digit is not None:
#                     final_frame = cv2.putText(final_frame, f'Digit: {predicted_digit}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
#                 else:
#                     final_frame = cv2.putText(final_frame, f'Digit: None', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

#                 show_frame(orig_frame, gray, thresh_frame, final_frame)
                
#         if plt.waitforbuttonpress(0.001):
#             break
    
#     return frame_buffer

# try:
#     plt.ion()  # Turn on interactive mode
#     all_predictions = []  # To store all unique predictions
#     stable_prediction = None  # To store the final stable prediction
    
#     while True:
#         frame_buffer = capture_predictions_for_duration(duration=5)
        
#         if frame_buffer:
#             # Update all predictions and ensure no consecutive duplicates
#             for pred in frame_buffer:
#                 if not all_predictions or pred != all_predictions[-1]:
#                     all_predictions.append(pred)
#                     print(f"All Predictions: {all_predictions}")
            
#             # Get the most frequent prediction
#             stable_prediction = get_stable_prediction(frame_buffer)
            
#             # Display the final stable prediction on the terminal
#             print(f"Current Stable Prediction: {stable_prediction}")
        
#         # Show the final stable prediction continuously
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to grab frame")
#             break

#         # Convert the frame to grayscale
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
#         # Apply adaptive thresholding
#         thresh_frame = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
#         # Preprocessed frame with prediction
#         final_frame = cv2.resize(thresh_frame, (320, 320))  # Upscale for better visibility
#         if stable_prediction is not None:
#             final_frame = cv2.putText(final_frame, f'Digit: {stable_prediction}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
#         else:
#             final_frame = cv2.putText(final_frame, f'Digit: None', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
#         show_frame(frame, gray, thresh_frame, final_frame)
        
#         if plt.waitforbuttonpress(0.001):
#             break

# except Exception as e:
#     print(f"Error: {e}")
# finally:
#     cap.release()
#     plt.ioff()
#     plt.close()
#     print(f"Final List of Predictions: {all_predictions}")

#test
# import cv2
# import numpy as np
# import easyocr
# import matplotlib.pyplot as plt
# from collections import deque, Counter
# import time

# # Initialize EasyOCR reader
# reader = easyocr.Reader(['en'])

# # Initialize the video capture
# cap = cv2.VideoCapture(0)

# # Create a figure and axes for plotting
# fig, axes = plt.subplots(1, 4, figsize=(20, 5))
# axes[0].set_title("Original Frame")
# axes[0].axis('off')
# axes[1].set_title("Grayscale Frame")
# axes[1].axis('off')
# axes[2].set_title("Thresholded Frame")
# axes[2].axis('off')
# axes[3].set_title("OCR Result")
# axes[3].axis('off')

# def show_frame(orig_frame, gray_frame, thresh_frame, final_frame):
#     axes[0].imshow(cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB))
#     axes[1].imshow(gray_frame, cmap='gray')
#     axes[2].imshow(thresh_frame, cmap='gray')
#     axes[3].imshow(final_frame, cmap='gray')
#     plt.draw()
#     plt.pause(0.001)

# def preprocess_for_ocr(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     edged = cv2.Canny(blurred, 50, 150)
#     return edged

# def get_stable_prediction(predictions):
#     return Counter(predictions).most_common(1)[0][0]

# def capture_predictions_for_duration(duration=5):
#     start_time = time.time()
#     frame_buffer = deque()
    
#     while time.time() - start_time < duration:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to grab frame")
#             continue
        
#         # Original frame
#         orig_frame = frame.copy()

#         # Convert the frame to grayscale
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Apply adaptive thresholding
#         thresh_frame = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
#         # Find contours
#         contours, _ = cv2.findContours(thresh_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         if contours:
#             # Filter contours by area to avoid small noisy contours
#             filtered_contours = [c for c in contours if cv2.contourArea(c) > 2000]  # Increase the area threshold to avoid small objects
#             if filtered_contours:
#                 # Find the largest contour
#                 largest_contour = max(filtered_contours, key=cv2.contourArea)
#                 x, y, w, h = cv2.boundingRect(largest_contour)
                
#                 # Extract ROI
#                 roi = frame[y:y+h, x:x+w]

#                 # Check if the ROI aspect ratio is similar to A4 paper
#                 aspect_ratio = w / h
#                 if 0.6 < aspect_ratio < 1.4:  # A4 paper aspect ratio is approximately 1.414 (but allowing some margin)
#                     # Focus on the central part of the ROI
#                     roi_center = roi[h//4:h*3//4, w//4:w*3//4]

#                     # Preprocess the ROI for OCR
#                     processed_roi = preprocess_for_ocr(roi_center)

#                     # Use EasyOCR to recognize text
#                     results = reader.readtext(processed_roi, detail=0, paragraph=False)

#                     predicted_digit = None  # Initialize the variable
#                     if results:
#                         # Filter out non-digit characters and keep digits between 0 and 9
#                         recognized_digits = [r for r in results if r.isdigit() and 0 <= int(r) <= 9]
#                         if recognized_digits:
#                             predicted_digit = max(set(recognized_digits), key=recognized_digits.count)
#                             frame_buffer.append(predicted_digit)
                    
#                     # Display frames for debugging
#                     final_frame = cv2.resize(processed_roi, (320, 320))  # Upscale for better visibility
#                     if predicted_digit is not None:
#                         final_frame = cv2.putText(final_frame, f'Digit: {predicted_digit}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
#                     else:
#                         final_frame = cv2.putText(final_frame, f'Digit: None', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

#                     show_frame(orig_frame, gray, thresh_frame, final_frame)
                
#         if plt.waitforbuttonpress(0.001):
#             break
    
#     return frame_buffer

# try:
#     plt.ion()  # Turn on interactive mode
#     all_predictions = []  # To store all unique predictions
#     stable_prediction = None  # To store the final stable prediction
    
#     while True:
#         frame_buffer = capture_predictions_for_duration(duration=5)
        
#         if frame_buffer:
#             # Update all predictions and ensure no consecutive duplicates
#             for pred in frame_buffer:
#                 if not all_predictions or pred != all_predictions[-1]:
#                     all_predictions.append(pred)
#                     print(f"All Predictions: {all_predictions}")
            
#             # Get the most frequent prediction
#             stable_prediction = get_stable_prediction(frame_buffer)
            
#             # Display the final stable prediction on the terminal
#             print(f"Current Stable Prediction: {stable_prediction}")
        
#         # Show the final stable prediction continuously
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 print("Failed to grab frame")
#                 break

#             # Convert the frame to grayscale
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
#             # Apply adaptive thresholding
#             thresh_frame = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            
#             # Preprocessed frame with prediction
#             final_frame = cv2.resize(thresh_frame, (320, 320))  # Upscale for better visibility
#             if stable_prediction is not None:
#                 final_frame = cv2.putText(final_frame, f'Digit: {stable_prediction}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
#             else:
#                 final_frame = cv2.putText(final_frame, f'Digit: None', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            
#             show_frame(frame, gray, thresh_frame, final_frame)
            
#             # Check for 'q' key press using Matplotlib
#             if plt.waitforbuttonpress(0.001):
#                 break

#             # If the frame does not contain the stable prediction, re-capture predictions
#             current_results = reader.readtext(preprocess_for_ocr(frame), detail=0, paragraph=False)
#             current_recognized_digits = [r for r in current_results if r.isdigit() and 0 <= int(r) <= 9]
#             if not current_recognized_digits or stable_prediction not in current_recognized_digits:
#                 break

# except Exception as e:
#     print(f"Error: {e}")
# finally:
#     cap.release()
#     plt.ioff()
#     plt.close()
#     print(f"Final List of Predictions: {all_predictions}")








#GGGG test

import cv2
import numpy as np
import easyocr
import matplotlib.pyplot as plt
from collections import deque, Counter
import time

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Create a figure and axes for plotting
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].set_title("Original Frame")
axes[0].axis('off')
axes[1].set_title("Grayscale Frame")
axes[1].axis('off')
axes[2].set_title("Thresholded Frame")
axes[2].axis('off')
axes[3].set_title("OCR Result")
axes[3].axis('off')

def show_frame(orig_frame, gray_frame, thresh_frame, final_frame):
    axes[0].imshow(cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB))
    axes[1].imshow(gray_frame, cmap='gray')
    axes[2].imshow(thresh_frame, cmap='gray')
    axes[3].imshow(final_frame, cmap='gray')
    plt.draw()
    plt.pause(0.001)

def preprocess_for_ocr(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    return edged

def get_stable_prediction(predictions):
    return Counter(predictions).most_common(1)[0][0]

def capture_predictions_for_duration(duration=5):
    start_time = time.time()
    frame_buffer = deque()
    
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            continue
        
        # Original frame
        orig_frame = frame.copy()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding
        thresh_frame = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Filter contours by area to avoid small noisy contours
            filtered_contours = [c for c in contours if cv2.contourArea(c) > 2000]  # Increase the area threshold to avoid small objects
            if filtered_contours:
                # Find the largest contour
                largest_contour = max(filtered_contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Extract ROI
                roi = frame[y:y+h, x:x+w]
                
                # Preprocess the ROI for OCR
                processed_roi = preprocess_for_ocr(roi)

                # Use EasyOCR to recognize text
                results = reader.readtext(processed_roi, detail=0, paragraph=False)

                predicted_digit = None  # Initialize the variable
                if results:
                    # Filter out non-digit characters
                    recognized_digits = [r for r in results if r.isdigit()]
                    if recognized_digits:
                        predicted_digit = max(set(recognized_digits), key=recognized_digits.count)
                        frame_buffer.append(predicted_digit)
                
                # Display frames for debugging
                final_frame = cv2.resize(processed_roi, (320, 320))  # Upscale for better visibility
                if predicted_digit is not None:
                    final_frame = cv2.putText(final_frame, f'Digit: {predicted_digit}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                else:
                    final_frame = cv2.putText(final_frame, f'Digit: None', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                show_frame(orig_frame, gray, thresh_frame, final_frame)
                
        if plt.waitforbuttonpress(0.001):
            break
    
    return frame_buffer

try:
    plt.ion()  # Turn on interactive mode
    all_predictions = []  # To store all unique predictions
    stable_prediction = None  # To store the final stable prediction
    
    while True:
        frame_buffer = capture_predictions_for_duration(duration=5)
        
        if frame_buffer:
            # Update all predictions and ensure no consecutive duplicates
            for pred in frame_buffer:
                if not all_predictions or pred != all_predictions[-1]:
                    all_predictions.append(pred)
                    print(f"All Predictions: {all_predictions}")
            
            # Get the most frequent prediction
            stable_prediction = get_stable_prediction(frame_buffer)
            
            # Display the final stable prediction on the terminal
            print(f"Current Stable Prediction: {stable_prediction}")
        
        # Show the final stable prediction continuously
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh_frame = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        # Preprocessed frame with prediction
        final_frame = cv2.resize(thresh_frame, (320, 320))  # Upscale for better visibility
        if stable_prediction is not None:
            final_frame = cv2.putText(final_frame, f'Digit: {stable_prediction}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        else:
            final_frame = cv2.putText(final_frame, f'Digit: None', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        show_frame(frame, gray, thresh_frame, final_frame)
        
        if plt.waitforbuttonpress(0.001):
            break

except Exception as e:
    print(f"Error: {e}")
finally:
    cap.release()
    plt.ioff()
    plt.close()
    print(f"Final List of Predictions: {all_predictions}")























#GOOD ONE


# import cv2
# import numpy as np
# import easyocr
# import matplotlib.pyplot as plt
# from collections import deque, Counter
# import time

# # Initialize EasyOCR reader
# reader = easyocr.Reader(['en'])

# # Initialize the video capture
# cap = cv2.VideoCapture(0)

# # Create a figure and axes for plotting
# fig, axes = plt.subplots(1, 4, figsize=(20, 5))
# axes[0].set_title("Original Frame")
# axes[0].axis('off')
# axes[1].set_title("Grayscale Frame")
# axes[1].axis('off')
# axes[2].set_title("Thresholded Frame")
# axes[2].axis('off')
# axes[3].set_title("OCR Result")
# axes[3].axis('off')

# def show_frame(orig_frame, gray_frame, thresh_frame, final_frame):
#     axes[0].imshow(cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB))
#     axes[1].imshow(gray_frame, cmap='gray')
#     axes[2].imshow(thresh_frame, cmap='gray')
#     axes[3].imshow(final_frame, cmap='gray')
#     plt.draw()
#     plt.pause(0.001)

# def preprocess_for_ocr(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     edged = cv2.Canny(blurred, 50, 150)
#     return edged

# def get_stable_prediction(predictions):
#     return Counter(predictions).most_common(1)[0][0]

# def capture_predictions_for_duration(duration=5):
#     start_time = time.time()
#     frame_buffer = deque()
    
#     while time.time() - start_time < duration:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to grab frame")
#             continue
        
#         # Original frame
#         orig_frame = frame.copy()

#         # Convert the frame to grayscale
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Apply adaptive thresholding
#         thresh_frame = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
#         # Find contours
#         contours, _ = cv2.findContours(thresh_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         if contours:
#             # Filter contours by area to avoid small noisy contours
#             filtered_contours = [c for c in contours if cv2.contourArea(c) > 1000]
#             if filtered_contours:
#                 # Find the largest contour
#                 largest_contour = max(filtered_contours, key=cv2.contourArea)
#                 x, y, w, h = cv2.boundingRect(largest_contour)
                
#                 # Extract ROI
#                 roi = frame[y:y+h, x:x+w]
                
#                 # Preprocess the ROI for OCR
#                 processed_roi = preprocess_for_ocr(roi)

#                 # Use EasyOCR to recognize text
#                 results = reader.readtext(processed_roi, detail=0, paragraph=False)

#                 predicted_digit = None  # Initialize the variable
#                 if results:
#                     # Filter out non-digit characters
#                     recognized_digits = [r for r in results if r.isdigit()]
#                     if recognized_digits:
#                         predicted_digit = max(set(recognized_digits), key=recognized_digits.count)
#                         frame_buffer.append(predicted_digit)
                
#                 # Display frames for debugging
#                 final_frame = cv2.resize(processed_roi, (320, 320))  # Upscale for better visibility
#                 if predicted_digit is not None:
#                     final_frame = cv2.putText(final_frame, f'Digit: {predicted_digit}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
#                 else:
#                     final_frame = cv2.putText(final_frame, f'Digit: None', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

#                 show_frame(orig_frame, gray, thresh_frame, final_frame)
                
#         if plt.waitforbuttonpress(0.001):
#             break
    
#     return frame_buffer

# try:
#     plt.ion()  # Turn on interactive mode
#     all_predictions = deque(maxlen=30)  # To store all predictions, limited to 30 to avoid excessive memory use
#     stable_prediction = None  # To store the final stable prediction
    
#     while True:
#         frame_buffer = capture_predictions_for_duration(duration=5)
        
#         if frame_buffer:
#             # Update all predictions
#             all_predictions.extend(frame_buffer)
            
#             # Get the most frequent prediction
#             stable_prediction = get_stable_prediction(all_predictions)
            
#             # Display the final stable prediction on the terminal
#             print(f"Current Stable Prediction: {stable_prediction}")
#             print(f"All Predictions: {list(all_predictions)}")
        
#         # Show the final stable prediction continuously
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to grab frame")
#             break

#         # Convert the frame to grayscale
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
#         # Apply adaptive thresholding
#         thresh_frame = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
#         # Preprocessed frame with prediction
#         final_frame = cv2.resize(thresh_frame, (320, 320))  # Upscale for better visibility
#         if stable_prediction is not None:
#             final_frame = cv2.putText(final_frame, f'Digit: {stable_prediction}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
#         else:
#             final_frame = cv2.putText(final_frame, f'Digit: None', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
#         show_frame(frame, gray, thresh_frame, final_frame)
        
#         if plt.waitforbuttonpress(0.001):
#             break

# except Exception as e:
#     print(f"Error: {e}")
# finally:
#     cap.release()
#     plt.ioff()
#     plt.close()
#     print(f"Final List of Predictions: {list(all_predictions)}")





#SOLID
# import cv2
# import numpy as np
# import easyocr
# import matplotlib.pyplot as plt
# from collections import deque, Counter
# import time

# # Initialize EasyOCR reader
# reader = easyocr.Reader(['en'])

# # Initialize the video capture
# cap = cv2.VideoCapture(0)

# # Create a figure and axes for plotting
# fig, axes = plt.subplots(1, 4, figsize=(20, 5))
# axes[0].set_title("Original Frame")
# axes[0].axis('off')
# axes[1].set_title("Grayscale Frame")
# axes[1].axis('off')
# axes[2].set_title("Thresholded Frame")
# axes[2].axis('off')
# axes[3].set_title("OCR Result")
# axes[3].axis('off')

# def show_frame(orig_frame, gray_frame, thresh_frame, final_frame):
#     axes[0].imshow(cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB))
#     axes[1].imshow(gray_frame, cmap='gray')
#     axes[2].imshow(thresh_frame, cmap='gray')
#     axes[3].imshow(final_frame, cmap='gray')
#     plt.draw()
#     plt.pause(0.001)

# def preprocess_for_ocr(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     edged = cv2.Canny(blurred, 50, 150)
#     return edged

# def get_stable_prediction(predictions):
#     return Counter(predictions).most_common(1)[0][0]

# def capture_predictions_for_duration(duration=5):
#     start_time = time.time()
#     frame_buffer = deque()
    
#     while time.time() - start_time < duration:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to grab frame")
#             continue
        
#         # Original frame
#         orig_frame = frame.copy()

#         # Convert the frame to grayscale
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Apply adaptive thresholding
#         thresh_frame = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
#         # Find contours
#         contours, _ = cv2.findContours(thresh_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         if contours:
#             # Filter contours by area to avoid small noisy contours
#             filtered_contours = [c for c in contours if cv2.contourArea(c) > 1000]
#             if filtered_contours:
#                 # Find the largest contour
#                 largest_contour = max(filtered_contours, key=cv2.contourArea)
#                 x, y, w, h = cv2.boundingRect(largest_contour)
                
#                 # Extract ROI
#                 roi = frame[y:y+h, x:x+w]
                
#                 # Preprocess the ROI for OCR
#                 processed_roi = preprocess_for_ocr(roi)

#                 # Use EasyOCR to recognize text
#                 results = reader.readtext(processed_roi, detail=0, paragraph=False)

#                 if results:
#                     # Filter out non-digit characters
#                     recognized_digits = [r for r in results if r.isdigit()]
#                     if recognized_digits:
#                         predicted_digit = max(set(recognized_digits), key=recognized_digits.count)
#                         frame_buffer.append(predicted_digit)
                
#                 # Display frames for debugging
#                 final_frame = cv2.resize(processed_roi, (320, 320))  # Upscale for better visibility
#                 if results:
#                     final_frame = cv2.putText(final_frame, f'Digit: {predicted_digit}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
#                 else:
#                     final_frame = cv2.putText(final_frame, f'Digit: None', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

#                 show_frame(orig_frame, gray, thresh_frame, final_frame)
                
#         if plt.waitforbuttonpress(0.001):
#             break
    
#     return frame_buffer

# try:
#     plt.ion()  # Turn on interactive mode
#     all_predictions = deque()  # To store all predictions
    
#     while True:
#         frame_buffer = capture_predictions_for_duration(duration=5)
        
#         if frame_buffer:
#             # Update all predictions
#             all_predictions.extend(frame_buffer)
            
#             # Get the most frequent prediction
#             stable_prediction = get_stable_prediction(all_predictions)
            
#             # Display the final stable prediction on the terminal
#             print(f"Current Stable Prediction: {stable_prediction}")
#             print(f"All Predictions: {list(all_predictions)}")
            
#             # Show the final stable prediction continuously
#             while True:
#                 ret, frame = cap.read()
#                 if not ret:
#                     print("Failed to grab frame")
#                     break

#                 # Convert the frame to grayscale
#                 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
#                 # Apply adaptive thresholding
#                 thresh_frame = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
                
#                 # Preprocessed frame with prediction
#                 final_frame = cv2.resize(thresh_frame, (320, 320))  # Upscale for better visibility
#                 final_frame = cv2.putText(final_frame, f'Digit: {stable_prediction}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                
#                 show_frame(frame, gray, thresh_frame, final_frame)
                
#                 if plt.waitforbuttonpress(0.001):
#                     break

# except Exception as e:
#     print(f"Error: {e}")
# finally:
#     cap.release()
#     plt.ioff()
#     plt.close()
#     print(f"Final List of Predictions: {list(all_predictions)}")
