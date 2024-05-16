import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from cv_bridge import CvBridge
import cv2
import numpy as np
import easyocr
import matplotlib.pyplot as plt
from collections import Counter

class OCRNode(Node):
    def __init__(self):
        super().__init__('ocr_node')
        self.publisher = self.create_publisher(Int32, 'recognized_digits', 10)
        self.bridge = CvBridge()
        self.reader = easyocr.Reader(['en'], gpu=False)

        self.subscription = self.create_subscription(
            Image,
            'oak/rgb/image_raw',
            self.listener_callback,
            10
        )

        self.fig, self.ax = plt.subplots(1, figsize=(10, 5))
        self.ax.axis('off')
        self.img_display = self.ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
        plt.ion()
        
        self.all_predictions = []
        self.stable_prediction = None
        self.last_printed_prediction = None

    def show_frame(self, frame):
        self.img_display.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.draw()
        plt.pause(0.001)

    def preprocess_for_ocr(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return morph

    def get_stable_prediction(self, predictions):
        if not predictions:
            return None, 0.0
        prediction_counter = Counter(predictions)
        most_common_prediction = prediction_counter.most_common(1)[0]
        prediction, count = most_common_prediction
        confidence = count / len(predictions)
        return prediction, confidence

    def listener_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        processed_frame = self.preprocess_for_ocr(frame)

        contours, _ = cv2.findContours(processed_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        recognized_digits = []
        digits_set = set
        if contours:
            filtered_contours = [c for c in contours if 500 < cv2.contourArea(c) < 50000]
            for contour in filtered_contours:
                x, y, w, h = cv2.boundingRect(contour)
                padding = 10
                x = max(x - padding, 0)
                y = max(y - padding, 0)
                w = min(w + 2 * padding, frame.shape[1] - x)
                h = min(h + 2 * padding, frame.shape[0] - y)

                roi = frame[y:y+h, x:x+w]
                aspect_ratio = w / h
                if 0.5 < aspect_ratio < 2.0:
                    processed_roi = self.preprocess_for_ocr(roi)
                    results = self.reader.readtext(processed_roi, detail=0, paragraph=False)
                    if results:
                        recognized_digits.extend([r for r in results if r.isdigit() and 0 <= int(r) <= 9])
                        for digit in recognized_digits:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv2.putText(frame, f'Digit: {digit}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        if recognized_digits:
            for digit in recognized_digits:
                if not self.all_predictions or digit != self.all_predictions[-1]:
                    self.all_predictions.append(digit)
                    #self.get_logger().info(f"All Predictions: {self.all_predictions}")
                    self.get_logger().info(f"All Predictions: {digit}")
                self.stable_prediction, confidence = self.get_stable_prediction(self.all_predictions)

            if self.stable_prediction != self.last_printed_prediction:
                self.get_logger().info(f"Current Stable Prediction: {self.stable_prediction} with confidence {confidence:.2f}")
                self.last_printed_prediction = self.stable_prediction

            self.publisher.publish(Int32(data=int(self.stable_prediction)))
            
        elif not recognized_digits:
            self.get_logger().info(f"NO DIGIT FOUND")

        self.show_frame(frame)

    def destroy_node(self):
        plt.ioff()
        plt.close()
        super().destroy_node()
        self.get_logger().info(f"Final List of Predictions: {self.all_predictions}")

def main(args=None):
    rclpy.init(args=args)
    node = OCRNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()



#Beta7
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from std_msgs.msg import Int32
# from cv_bridge import CvBridge
# import cv2
# import numpy as np
# import easyocr
# import matplotlib.pyplot as plt
# from collections import Counter

# class OCRNode(Node):
#     def __init__(self):
#         super().__init__('ocr_node')
#         self.publisher = self.create_publisher(Int32, 'recognized_digits', 10)
#         self.bridge = CvBridge()
#         self.reader = easyocr.Reader(['en'], gpu=False)

#         self.subscription = self.create_subscription(
#             Image,
#             'oak/rgb/image_raw',
#             self.listener_callback,
#             10
#         )

#         self.fig, self.ax = plt.subplots(1, figsize=(10, 5))
#         self.ax.axis('off')
#         self.img_display = self.ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
#         plt.ion()
        
#         self.all_predictions = []
#         self.stable_prediction = None
#         self.last_printed_prediction = None

#     def show_frame(self, frame):
#         self.img_display.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#         plt.draw()
#         plt.pause(0.001)

#     def preprocess_for_ocr(self, frame):
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#         _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#         kernel = np.ones((3, 3), np.uint8)
#         morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
#         return morph

#     def get_stable_prediction(self, predictions):
#         if not predictions:
#             return None, 0.0
#         prediction_counter = Counter(predictions)
#         most_common_prediction = prediction_counter.most_common(1)[0]
#         prediction, count = most_common_prediction
#         confidence = count / len(predictions)
#         return prediction, confidence

#     def listener_callback(self, msg):
#         frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

#         processed_frame = self.preprocess_for_ocr(frame)

#         contours, _ = cv2.findContours(processed_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         recognized_digits = []
#         if contours:
#             filtered_contours = [c for c in contours if 500 < cv2.contourArea(c) < 50000]
#             for contour in filtered_contours:
#                 x, y, w, h = cv2.boundingRect(contour)
#                 padding = 10
#                 x = max(x - padding, 0)
#                 y = max(y - padding, 0)
#                 w = min(w + 2 * padding, frame.shape[1] - x)
#                 h = min(h + 2 * padding, frame.shape[0] - y)

#                 roi = frame[y:y+h, x:x+w]
#                 aspect_ratio = w / h
#                 if 0.5 < aspect_ratio < 2.0:
#                     processed_roi = self.preprocess_for_ocr(roi)
#                     results = self.reader.readtext(processed_roi, detail=0, paragraph=False)
#                     if results:
#                         recognized_digits.extend([r for r in results if r.isdigit() and 0 <= int(r) <= 9])
#                         for digit in recognized_digits:
#                             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                             cv2.putText(frame, f'Digit: {digit}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

#         if recognized_digits:
#             for digit in recognized_digits:
#                 if not self.all_predictions or digit != self.all_predictions[-1]:
#                     self.all_predictions.append(digit)
#                     self.get_logger().info(f"All Predictions: {self.all_predictions}")

#                 self.stable_prediction, confidence = self.get_stable_prediction(self.all_predictions)

#             if self.stable_prediction != self.last_printed_prediction:
#                 self.get_logger().info(f"Current Stable Prediction: {self.stable_prediction} with confidence {confidence:.2f}")
#                 self.last_printed_prediction = self.stable_prediction

#             self.publisher.publish(Int32(data=int(self.stable_prediction)))

#         self.show_frame(frame)

#     def destroy_node(self):
#         plt.ioff()
#         plt.close()
#         super().destroy_node()
#         self.get_logger().info(f"Final List of Predictions: {self.all_predictions}")

# def main(args=None):
#     rclpy.init(args=args)
#     node = OCRNode()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()



# #BETA6(working version, touch wood not to jinx it)
# import cv2
# import numpy as np
# import easyocr
# import matplotlib.pyplot as plt
# from collections import Counter

# # Initialize EasyOCR reader
# reader = easyocr.Reader(['en'], gpu=False)

# # Initialize the video capture
# cap = cv2.VideoCapture(0)

# # Create a figure and axes for plotting
# fig, ax = plt.subplots(1, figsize=(10, 5))
# ax.axis('off')
# img_display = ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))

# def show_frame(frame):
#     img_display.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     plt.draw()
#     plt.pause(0.001)

# def preprocess_for_ocr(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#     kernel = np.ones((3, 3), np.uint8)
#     morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
#     return morph

# def get_stable_prediction(predictions):
#     if not predictions:
#         return None, 0.0
#     prediction_counter = Counter(predictions)
#     most_common_prediction = prediction_counter.most_common(1)[0]
#     prediction, count = most_common_prediction
#     confidence = count / len(predictions)
#     return prediction, confidence

# def capture_and_process_image():
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to grab frame")
#         return None, None

#     # Convert the frame to grayscale and preprocess it
#     processed_frame = preprocess_for_ocr(frame)

#     # Find contours
#     contours, _ = cv2.findContours(processed_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     recognized_digits = []  # Initialize the variable for multiple digits
#     if contours:
#         # Filter contours by area to avoid small noisy contours
#         filtered_contours = [c for c in contours if 500 < cv2.contourArea(c) < 50000]  # Adjusted the area threshold
#         for contour in filtered_contours:
#             x, y, w, h = cv2.boundingRect(contour)

#             # Increase the size of the bounding box slightly
#             padding = 10
#             x = max(x - padding, 0)
#             y = max(y - padding, 0)
#             w = min(w + 2 * padding, frame.shape[1] - x)
#             h = min(h + 2 * padding, frame.shape[0] - y)

#             # Extract ROI
#             roi = frame[y:y+h, x:x+w]

#             # Check if the ROI aspect ratio is similar to A4 paper
#             aspect_ratio = w / h
#             if 0.5 < aspect_ratio < 2.0:  # Adjusted aspect ratio range to be more inclusive
#                 # Preprocess the ROI for OCR
#                 processed_roi = preprocess_for_ocr(roi)

#                 # Use EasyOCR to recognize text
#                 results = reader.readtext(processed_roi, detail=0, paragraph=False)

#                 if results:
#                     # Filter out non-digit characters and keep digits between 0 and 9
#                     recognized_digits.extend([r for r in results if r.isdigit() and 0 <= int(r) <= 9])
#                     for digit in recognized_digits:
#                         # Draw rectangles and text for each recognized digit
#                         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                         cv2.putText(frame, f'Digit: {digit}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

#     return frame, recognized_digits

# try:
#     plt.ion()  # Turn on interactive mode
#     all_predictions = []  # To store all unique predictions
#     stable_prediction = None  # To store the final stable prediction
#     last_printed_prediction = None  # To store the last printed stable prediction

#     while True:
#         frame, recognized_digits = capture_and_process_image()

#         if recognized_digits:
#             for digit in recognized_digits:
#                 # Update all predictions and ensure no consecutive duplicates
#                 if not all_predictions or digit != all_predictions[-1]:
#                     all_predictions.append(digit)
#                     print(f"All Predictions: {all_predictions}")
#                 # Get the most frequent prediction
#                 stable_prediction, confidence = get_stable_prediction(all_predictions)

#             # Only print the stable prediction if it's different from the last printed one
#             if stable_prediction != last_printed_prediction:
#                 print(f"Current Stable Prediction: {stable_prediction} with confidence {confidence:.2f}")
#                 last_printed_prediction = stable_prediction

#         if frame is not None:
#             show_frame(frame)

#         # Exit the loop if 'q' is pressed
#         if plt.waitforbuttonpress(0.001):
#             break

# except Exception as e:
#     print(f"Error: {e}")
# finally:
#     cap.release()
#     plt.ioff()
#     plt.close()
#     print(f"Final List of Predictions: {all_predictions}")

# --------------------------------------------------------------------------------------------

#BETA 5
# import cv2
# import numpy as np
# import easyocr
# import matplotlib.pyplot as plt
# from collections import Counter

# # Initialize EasyOCR reader
# reader = easyocr.Reader(['en'], gpu=False)

# # Initialize the video capture
# cap = cv2.VideoCapture(0)

# # Create a figure and axes for plotting
# fig, ax = plt.subplots(1, figsize=(10, 5))
# ax.axis('off')
# img_display = ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))

# def show_frame(frame):
#     img_display.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     plt.draw()
#     plt.pause(0.001)

# def preprocess_for_ocr(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#     kernel = np.ones((3, 3), np.uint8)
#     morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
#     return morph

# def get_stable_prediction(predictions):
#     if not predictions:
#         return None, 0.0
#     prediction_counter = Counter(predictions)
#     most_common_prediction = prediction_counter.most_common(1)[0]
#     prediction, count = most_common_prediction
#     confidence = count / len(predictions)
#     return prediction, confidence

# def capture_and_process_image():
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to grab frame")
#         return None, None

#     # Convert the frame to grayscale and preprocess it
#     processed_frame = preprocess_for_ocr(frame)

#     # Find contours
#     contours, _ = cv2.findContours(processed_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     recognized_digits = []  # Initialize the variable for multiple digits
#     if contours:
#         # Filter contours by area to avoid small noisy contours
#         filtered_contours = [c for c in contours if 500 < cv2.contourArea(c) < 50000]  # Adjusted the area threshold
#         for contour in filtered_contours:
#             x, y, w, h = cv2.boundingRect(contour)

#             # Extract ROI
#             roi = frame[y:y+h, x:x+w]

#             # Check if the ROI aspect ratio is similar to A4 paper
#             aspect_ratio = w / h
#             if 0.5 < aspect_ratio < 2.0:  # Adjusted aspect ratio range to be more inclusive
#                 # Preprocess the ROI for OCR
#                 processed_roi = preprocess_for_ocr(roi)

#                 # Use EasyOCR to recognize text
#                 results = reader.readtext(processed_roi, detail=0, paragraph=False)

#                 if results:
#                     # Filter out non-digit characters and keep digits between 0 and 9
#                     recognized_digits.extend([r for r in results if r.isdigit() and 0 <= int(r) <= 9])
#                     for digit in recognized_digits:
#                         # Draw rectangles and text for each recognized digit
#                         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                         cv2.putText(frame, f'Digit: {digit}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

#     return frame, recognized_digits

# try:
#     plt.ion()  # Turn on interactive mode
#     all_predictions = []  # To store all unique predictions
#     stable_prediction = None  # To store the final stable prediction
#     last_printed_prediction = None  # To store the last printed stable prediction

#     while True:
#         frame, recognized_digits = capture_and_process_image()

#         if recognized_digits:
#             for digit in recognized_digits:
#                 # Update all predictions and ensure no consecutive duplicates
#                 if not all_predictions or digit != all_predictions[-1]:
#                     all_predictions.append(digit)
#                     print(f"All Predictions: {all_predictions}")
#                 # Get the most frequent prediction
#                 stable_prediction, confidence = get_stable_prediction(all_predictions)

#             # Only print the stable prediction if it's different from the last printed one
#             if stable_prediction != last_printed_prediction:
#                 print(f"Current Stable Prediction: {stable_prediction} with confidence {confidence:.2f}")
#                 last_printed_prediction = stable_prediction

#         if frame is not None:
#             show_frame(frame)

#         # Exit the loop if 'q' is pressed
#         if plt.waitforbuttonpress(0.001):
#             break

# except Exception as e:
#     print(f"Error: {e}")
# finally:
#     cap.release()
#     plt.ioff()
#     plt.close()
#     print(f"Final List of Predictions: {all_predictions}")

#BETA 4
# import cv2
# import numpy as np
# import easyocr
# import matplotlib.pyplot as plt
# from collections import Counter

# # Initialize EasyOCR reader
# reader = easyocr.Reader(['en'], gpu=False)

# # Initialize the video capture
# cap = cv2.VideoCapture(0)

# # Create a figure and axes for plotting
# fig, ax = plt.subplots(1, figsize=(10, 5))
# ax.axis('off')
# img_display = ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))

# def show_frame(frame):
#     img_display.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     plt.draw()
#     plt.pause(0.001)

# def preprocess_for_ocr(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#     kernel = np.ones((3, 3), np.uint8)
#     morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
#     return morph

# def get_stable_prediction(predictions):
#     if not predictions:
#         return None, 0.0
#     prediction_counter = Counter(predictions)
#     most_common_prediction = prediction_counter.most_common(1)[0]
#     prediction, count = most_common_prediction
#     confidence = count / len(predictions)
#     return prediction, confidence

# def capture_and_process_image():
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to grab frame")
#         return None, None

#     # Convert the frame to grayscale and preprocess it
#     processed_frame = preprocess_for_ocr(frame)

#     # Find contours
#     contours, _ = cv2.findContours(processed_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     recognized_digits = []  # Initialize the variable for multiple digits
#     if contours:
#         # Filter contours by area to avoid small noisy contours
#         filtered_contours = [c for c in contours if 1000 < cv2.contourArea(c) < 50000]  # Adjust the area threshold as needed
#         for contour in filtered_contours:
#             x, y, w, h = cv2.boundingRect(contour)

#             # Extract ROI
#             roi = frame[y:y+h, x:x+w]

#             # Check if the ROI aspect ratio is similar to A4 paper
#             aspect_ratio = w / h
#             if 0.5 < aspect_ratio < 2.0:  # Adjusted aspect ratio range to be more inclusive
#                 # Preprocess the ROI for OCR
#                 processed_roi = preprocess_for_ocr(roi)

#                 # Use EasyOCR to recognize text
#                 results = reader.readtext(processed_roi, detail=0, paragraph=False)

#                 if results:
#                     # Filter out non-digit characters and keep digits between 0 and 9
#                     recognized_digits.extend([r for r in results if r.isdigit() and 0 <= int(r) <= 9])
#                     for digit in recognized_digits:
#                         # Draw rectangles and text for each recognized digit
#                         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                         cv2.putText(frame, f'Digit: {digit}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

#     return frame, recognized_digits

# try:
#     plt.ion()  # Turn on interactive mode
#     all_predictions = []  # To store all unique predictions
#     stable_prediction = None  # To store the final stable prediction
#     last_printed_prediction = None  # To store the last printed stable prediction

#     while True:
#         frame, recognized_digits = capture_and_process_image()

#         if recognized_digits:
#             for digit in recognized_digits:
#                 # Update all predictions and ensure no consecutive duplicates
#                 if not all_predictions or digit != all_predictions[-1]:
#                     all_predictions.append(digit)
#                     print(f"All Predictions: {all_predictions}")
#                 # Get the most frequent prediction
#                 stable_prediction, confidence = get_stable_prediction(all_predictions)

#             # Only print the stable prediction if it's different from the last printed one
#             if stable_prediction != last_printed_prediction:
#                 print(f"Current Stable Prediction: {stable_prediction} with confidence {confidence:.2f}")
#                 last_printed_prediction = stable_prediction

#         if frame is not None:
#             show_frame(frame)

#         # Exit the loop if 'q' is pressed
#         if plt.waitforbuttonpress(0.001):
#             break

# except Exception as e:
#     print(f"Error: {e}")
# finally:
#     cap.release()
#     plt.ioff()
#     plt.close()
#     print(f"Final List of Predictions: {all_predictions}")

#BETA3
# import cv2
# import numpy as np
# import easyocr
# import matplotlib.pyplot as plt
# from collections import Counter

# # Initialize EasyOCR reader
# reader = easyocr.Reader(['en'])

# # Initialize the video capture
# cap = cv2.VideoCapture(0)

# # Create a figure and axes for plotting
# fig, ax = plt.subplots(1, figsize=(10, 5))
# ax.axis('off')
# img_display = ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))

# def show_frame(frame):
#     img_display.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     plt.draw()
#     plt.pause(0.001)

# def preprocess_for_ocr(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#     kernel = np.ones((3, 3), np.uint8)
#     morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
#     return morph

# def get_stable_prediction(predictions):
#     if not predictions:
#         return None, 0.0
#     prediction_counter = Counter(predictions)
#     most_common_prediction = prediction_counter.most_common(1)[0]
#     prediction, count = most_common_prediction
#     confidence = count / len(predictions)
#     return prediction, confidence

# def capture_and_process_image():
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to grab frame")
#         return None, None

#     # Convert the frame to grayscale and preprocess it
#     processed_frame = preprocess_for_ocr(frame)

#     # Find contours
#     contours, _ = cv2.findContours(processed_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     recognized_digits = []  # Initialize the variable for multiple digits
#     if contours:
#         # Filter contours by area to avoid small noisy contours
#         filtered_contours = [c for c in contours if 1000 < cv2.contourArea(c) < 50000]  # Adjust the area threshold as needed
#         for contour in filtered_contours:
#             x, y, w, h = cv2.boundingRect(contour)

#             # Extract ROI
#             roi = frame[y:y+h, x:x+w]

#             # Check if the ROI aspect ratio is similar to A4 paper
#             aspect_ratio = w / h
#             if 0.5 < aspect_ratio < 2.0:  # Adjusted aspect ratio range to be more inclusive
#                 # Preprocess the ROI for OCR
#                 processed_roi = preprocess_for_ocr(roi)

#                 # Use EasyOCR to recognize text
#                 results = reader.readtext(processed_roi, detail=0, paragraph=False)

#                 if results:
#                     # Filter out non-digit characters and keep digits between 0 and 9
#                     recognized_digits.extend([r for r in results if r.isdigit() and 0 <= int(r) <= 9])
#                     for digit in recognized_digits:
#                         # Draw rectangles and text for each recognized digit
#                         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                         cv2.putText(frame, f'Digit: {digit}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
#     return frame, recognized_digits

# try:
#     plt.ion()  # Turn on interactive mode
#     all_predictions = []  # To store all unique predictions
#     stable_prediction = None  # To store the final stable prediction
#     last_printed_prediction = None  # To store the last printed stable prediction

#     while True:
#         frame, recognized_digits = capture_and_process_image()

#         if recognized_digits:
#             for digit in recognized_digits:
#                 # Update all predictions and ensure no consecutive duplicates
#                 if not all_predictions or digit != all_predictions[-1]:
#                     all_predictions.append(digit)
#                     print(f"All Predictions: {all_predictions}")
#                 # Get the most frequent prediction
#                 stable_prediction, confidence = get_stable_prediction(all_predictions)

#             # Only print the stable prediction if it's different from the last printed one
#             if stable_prediction != last_printed_prediction:
#                 print(f"Current Stable Prediction: {stable_prediction} with confidence {confidence:.2f}")
#                 last_printed_prediction = stable_prediction

#         if frame is not None:
#             show_frame(frame)

#         # Exit the loop if 'q' is pressed
#         if plt.waitforbuttonpress(0.001):
#             break

# except Exception as e:
#     print(f"Error: {e}")
# finally:
#     cap.release()
#     plt.ioff()
#     plt.close()
#     print(f"Final List of Predictions: {all_predictions}")


#BETA3
# import cv2
# import numpy as np
# import easyocr
# import matplotlib.pyplot as plt
# from collections import deque, Counter

# # Initialize EasyOCR reader
# reader = easyocr.Reader(['en'])

# # Initialize the video capture
# cap = cv2.VideoCapture(0)

# # Create a figure and axes for plotting
# fig, ax = plt.subplots(1, figsize=(10, 5))
# ax.axis('off')
# img_display = ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))

# def show_frame(frame):
#     img_display.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     plt.draw()
#     plt.pause(0.001)

# def preprocess_for_ocr(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#     kernel = np.ones((3, 3), np.uint8)
#     morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
#     return morph

# def get_stable_prediction(predictions):
#     if not predictions:
#         return None, 0.0
#     prediction_counter = Counter(predictions)
#     most_common_prediction = prediction_counter.most_common(1)[0]
#     prediction, count = most_common_prediction
#     confidence = count / len(predictions)
#     return prediction, confidence

# try:
#     plt.ion()  # Turn on interactive mode
#     all_predictions = []  # To store all unique predictions
#     stable_prediction = None  # To store the final stable prediction
#     last_printed_prediction = None  # To store the last printed stable prediction
#     confidence = 0.0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to grab frame")
#             break

#         # Convert the frame to grayscale and preprocess it
#         processed_frame = preprocess_for_ocr(frame)

#         # Find contours
#         contours, _ = cv2.findContours(processed_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         predicted_digits = []  # Initialize the variable for multiple digits
#         recognized_digits = []  # Initialize the recognized_digits within the loop
#         if contours:
#             # Filter contours by area to avoid small noisy contours
#             filtered_contours = [c for c in contours if 1000 < cv2.contourArea(c) < 50000]  # Adjust the area threshold as needed
#             for contour in filtered_contours:
#                 x, y, w, h = cv2.boundingRect(contour)

#                 # Extract ROI
#                 roi = frame[y:y+h, x:x+w]

#                 # Check if the ROI aspect ratio is similar to A4 paper
#                 aspect_ratio = w / h
#                 if 0.5 < aspect_ratio < 2.0:  # Adjusted aspect ratio range to be more inclusive
#                     # Preprocess the ROI for OCR
#                     processed_roi = preprocess_for_ocr(roi)

#                     # Use EasyOCR to recognize text
#                     results = reader.readtext(processed_roi, detail=0, paragraph=False)

#                     if results:
#                         # Filter out non-digit characters and keep digits between 0 and 9
#                         recognized_digits.extend([r for r in results if r.isdigit() and 0 <= int(r) <= 9])
#                         predicted_digits.extend(recognized_digits)
#                         for digit in recognized_digits:
#                             # Update all predictions and ensure no consecutive duplicates
#                             if not all_predictions or digit != all_predictions[-1]:
#                                 all_predictions.append(digit)
#                                 print(f"All Predictions: {all_predictions}")

#                         # Get the most frequent prediction
#                         stable_prediction, confidence = get_stable_prediction(all_predictions)

#                     # Display frames for debugging
#                     if processed_roi is not None and recognized_digits:
#                         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                         for digit in recognized_digits:
#                             cv2.putText(frame, f'Digit: {digit} ({confidence*100:.2f}%)', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
#                     else:
#                         cv2.putText(frame, f'Digit: None', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

#         # Only print the stable prediction if it's different from the last printed one
#         if stable_prediction != last_printed_prediction:
#             print(f"Current Stable Prediction: {stable_prediction} with confidence {confidence:.2f}")
#             last_printed_prediction = stable_prediction

#         show_frame(frame)

#         # Exit the loop if 'q' is pressed
#         if plt.waitforbuttonpress(0.001):
#             break

# except Exception as e:
#     print(f"Error: {e}")
# finally:
#     cap.release()
#     plt.ioff()
#     plt.close()
#     print(f"Final List of Predictions: {all_predictions}")



#BETA 2
# import cv2
# import numpy as np
# import easyocr
# import matplotlib.pyplot as plt
# from collections import deque, Counter

# # Initialize EasyOCR reader
# reader = easyocr.Reader(['en'])

# # Initialize the video capture
# cap = cv2.VideoCapture(0)

# # Create a figure and axes for plotting
# fig, ax = plt.subplots(1, figsize=(10, 5))
# ax.axis('off')
# img_display = ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))

# def show_frame(frame):
#     img_display.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     plt.draw()
#     plt.pause(0.001)

# def preprocess_for_ocr(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     return blurred

# def get_stable_prediction(predictions):
#     if not predictions:
#         return None, 0.0
#     prediction_counter = Counter(predictions)
#     most_common_prediction = prediction_counter.most_common(1)[0]
#     prediction, count = most_common_prediction
#     confidence = count / len(predictions)
#     return prediction, confidence

# try:
#     plt.ion()  # Turn on interactive mode
#     all_predictions = []  # To store all unique predictions
#     stable_prediction = None  # To store the final stable prediction
#     last_printed_prediction = None  # To store the last printed stable prediction
#     confidence = 0.0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to grab frame")
#             break

#         # Convert the frame to grayscale
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Apply Gaussian blur to reduce noise
#         blurred = cv2.GaussianBlur(gray, (5, 5), 0)

#         # Apply adaptive thresholding
#         thresh_frame = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

#         # Apply morphological operations to clean up the image
#         kernel = np.ones((3, 3), np.uint8)
#         morphed_frame = cv2.morphologyEx(thresh_frame, cv2.MORPH_CLOSE, kernel)

#         # Find contours
#         contours, _ = cv2.findContours(morphed_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         processed_roi = None  # Initialize the variable
#         predicted_digit = None  # Initialize the variable
#         if contours:
#             # Filter contours by area to avoid small noisy contours
#             filtered_contours = [c for c in contours if cv2.contourArea(c) > 2000]  # Adjust the area threshold as needed
#             if filtered_contours:
#                 # Find the largest contour
#                 largest_contour = max(filtered_contours, key=cv2.contourArea)
#                 x, y, w, h = cv2.boundingRect(largest_contour)

#                 # Extract ROI
#                 roi = frame[y:y+h, x:x+w]

#                 # Check if the ROI aspect ratio is similar to A4 paper
#                 aspect_ratio = w / h
#                 if 0.6 < aspect_ratio < 1.4:  # A4 paper aspect ratio is approximately 1.414 (but allowing some margin)
#                     # Preprocess the ROI for OCR
#                     processed_roi = preprocess_for_ocr(roi)

#                     # Use EasyOCR to recognize text
#                     results = reader.readtext(processed_roi, detail=0, paragraph=False)

#                     if results:
#                         # Filter out non-digit characters and keep digits between 0 and 9
#                         recognized_digits = [r for r in results if r.isdigit() and 0 <= int(r) <= 9]
#                         if recognized_digits:
#                             predicted_digit = max(set(recognized_digits), key=recognized_digits.count)
#                             # Update all predictions and ensure no consecutive duplicates
#                             if not all_predictions or predicted_digit != all_predictions[-1]:
#                                 all_predictions.append(predicted_digit)
#                                 print(f"All Predictions: {all_predictions}")
#                             # Get the most frequent prediction
#                             stable_prediction, confidence = get_stable_prediction(all_predictions)

#                     # Display frames for debugging
#                     if processed_roi is not None and predicted_digit is not None:
#                         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                         cv2.putText(frame, f'Digit: {predicted_digit} ({confidence*100:.2f}%)', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
#                     else:
#                         cv2.putText(frame, f'Digit: None', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

#         # Only print the stable prediction if it's different from the last printed one
#         if stable_prediction != last_printed_prediction:
#             print(f"Current Stable Prediction: {stable_prediction} with confidence {confidence:.2f}")
#             last_printed_prediction = stable_prediction

#         show_frame(frame)

#         # Exit the loop if 'q' is pressed
#         if plt.waitforbuttonpress(0.001):
#             break

# except Exception as e:
#     print(f"Error: {e}")
# finally:
#     cap.release()
#     plt.ioff()
#     plt.close()
#     print(f"Final List of Predictions: {all_predictions}")



# import cv2
# import numpy as np
# import easyocr
# import matplotlib.pyplot as plt
# from collections import deque, Counter

# # Initialize EasyOCR reader
# reader = easyocr.Reader(['en'])

# # Initialize the video capture
# cap = cv2.VideoCapture(0)

# # Create a figure and axes for plotting
# fig, ax = plt.subplots(1, figsize=(10, 5))
# ax.axis('off')
# img_display = ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))

# def show_frame(frame):
#     img_display.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     plt.draw()
#     plt.pause(0.001)

# def preprocess_for_ocr(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     return blurred

# def get_stable_prediction(predictions):
#     if not predictions:
#         return None, 0.0
#     prediction_counter = Counter(predictions)
#     most_common_prediction = prediction_counter.most_common(1)[0]
#     prediction, count = most_common_prediction
#     confidence = count / len(predictions)
#     return prediction, confidence

# try:
#     plt.ion()  # Turn on interactive mode
#     all_predictions = []  # To store all unique predictions
#     stable_prediction = None  # To store the final stable prediction
#     last_printed_prediction = None  # To store the last printed stable prediction
#     confidence = 0.0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to grab frame")
#             break

#         # Convert the frame to grayscale
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Apply Gaussian blur to reduce noise
#         blurred = cv2.GaussianBlur(gray, (5, 5), 0)

#         # Apply adaptive thresholding
#         thresh_frame = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

#         # Find contours
#         contours, _ = cv2.findContours(thresh_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         processed_roi = None  # Initialize the variable
#         predicted_digit = None  # Initialize the variable
#         if contours:
#             # Filter contours by area to avoid small noisy contours
#             filtered_contours = [c for c in contours if cv2.contourArea(c) > 5000]  # Adjust the area threshold as needed
#             if filtered_contours:
#                 # Find the largest contour
#                 largest_contour = max(filtered_contours, key=cv2.contourArea)
#                 x, y, w, h = cv2.boundingRect(largest_contour)

#                 # Extract ROI
#                 roi = frame[y:y+h, x:x+w]

#                 # Check if the ROI aspect ratio is similar to A4 paper
#                 aspect_ratio = w / h
#                 if 0.6 < aspect_ratio < 1.4:  # A4 paper aspect ratio is approximately 1.414 (but allowing some margin)
#                     # Preprocess the ROI for OCR
#                     processed_roi = preprocess_for_ocr(roi)

#                     # Use EasyOCR to recognize text
#                     results = reader.readtext(processed_roi, detail=0, paragraph=False)

#                     if results:
#                         # Filter out non-digit characters and keep digits between 0 and 9
#                         recognized_digits = [r for r in results if r.isdigit() and 0 <= int(r) <= 9]
#                         if recognized_digits:
#                             predicted_digit = max(set(recognized_digits), key=recognized_digits.count)
#                             # Update all predictions and ensure no consecutive duplicates
#                             if not all_predictions or predicted_digit != all_predictions[-1]:
#                                 all_predictions.append(predicted_digit)
#                                 print(f"All Predictions: {all_predictions}")
#                             # Get the most frequent prediction
#                             stable_prediction, confidence = get_stable_prediction(all_predictions)

#                     # Display frames for debugging
#                     if processed_roi is not None and predicted_digit is not None:
#                         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                         cv2.putText(frame, f'Digit: {predicted_digit} ({confidence*100:.2f}%)', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
#                     else:
#                         cv2.putText(frame, f'Digit: None', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

#         # Only print the stable prediction if it's different from the last printed one
#         if stable_prediction != last_printed_prediction:
#             print(f"Current Stable Prediction: {stable_prediction} with confidence {confidence:.2f}")
#             last_printed_prediction = stable_prediction

#         show_frame(frame)

#         # Exit the loop if 'q' is pressed
#         if plt.waitforbuttonpress(0.001):
#             break

# except Exception as e:
#     print(f"Error: {e}")
# finally:
#     cap.release()
#     plt.ioff()
#     plt.close()
#     print(f"Final List of Predictions: {all_predictions}")


#BETA1.1
# import cv2
# import numpy as np
# import easyocr
# import matplotlib.pyplot as plt
# from collections import deque, Counter

# # Initialize EasyOCR reader
# reader = easyocr.Reader(['en'])

# # Initialize the video capture
# cap = cv2.VideoCapture(0)

# # Create a figure and axes for plotting
# fig, ax = plt.subplots(1, figsize=(10, 5))
# ax.axis('off')
# img_display = ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))

# def show_frame(frame):
#     img_display.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     plt.draw()
#     plt.pause(0.001)

# def preprocess_for_ocr(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     return blurred

# def get_stable_prediction(predictions):
#     if not predictions:
#         return None, 0.0
#     prediction_counter = Counter(predictions)
#     most_common_prediction = prediction_counter.most_common(1)[0]
#     prediction, count = most_common_prediction
#     confidence = count / len(predictions)
#     return prediction, confidence

# try:
#     plt.ion()  # Turn on interactive mode
#     all_predictions = []  # To store all unique predictions
#     stable_prediction = None  # To store the final stable prediction
#     last_printed_prediction = None  # To store the last printed stable prediction
#     confidence = 0.0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to grab frame")
#             break

#         # Convert the frame to grayscale
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Apply adaptive thresholding
#         thresh_frame = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

#         # Find contours
#         contours, _ = cv2.findContours(thresh_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         processed_roi = None  # Initialize the variable
#         predicted_digit = None  # Initialize the variable
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
#                     # Preprocess the ROI for OCR
#                     processed_roi = preprocess_for_ocr(roi)

#                     # Use EasyOCR to recognize text
#                     results = reader.readtext(processed_roi, detail=0, paragraph=False)

#                     if results:
#                         # Filter out non-digit characters and keep digits between 0 and 9
#                         recognized_digits = [r for r in results if r.isdigit() and 0 <= int(r) <= 9]
#                         if recognized_digits:
#                             predicted_digit = max(set(recognized_digits), key=recognized_digits.count)
#                             # Update all predictions and ensure no consecutive duplicates
#                             if not all_predictions or predicted_digit != all_predictions[-1]:
#                                 all_predictions.append(predicted_digit)
#                                 print(f"All Predictions: {all_predictions}")
#                             # Get the most frequent prediction
#                             stable_prediction, confidence = get_stable_prediction(all_predictions)

#                     # Display frames for debugging
#                     if processed_roi is not None and predicted_digit is not None:
#                         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                         cv2.putText(frame, f'Digit: {predicted_digit} ({confidence*100:.2f}%)', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
#                     else:
#                         cv2.putText(frame, f'Digit: None', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

#         # Only print the stable prediction if it's different from the last printed one
#         if stable_prediction != last_printed_prediction:
#             print(f"Current Stable Prediction: {stable_prediction} with confidence {confidence:.2f}")
#             last_printed_prediction = stable_prediction

#         show_frame(frame)

#         # Exit the loop if 'q' is pressed
#         if plt.waitforbuttonpress(0.001):
#             break

# except Exception as e:
#     print(f"Error: {e}")
# finally:
#     cap.release()
#     plt.ioff()
#     plt.close()
#     print(f"Final List of Predictions: {all_predictions}")




#WORKING1
# import cv2
# import numpy as np
# import easyocr
# import matplotlib.pyplot as plt
# from collections import deque, Counter

# # Initialize EasyOCR reader
# reader = easyocr.Reader(['en'])

# # Initialize the video capture
# cap = cv2.VideoCapture(0)

# # Create a figure and axes for plotting
# fig, ax = plt.subplots(1, figsize=(10, 5))
# ax.axis('off')
# img_display = ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))

# def show_frame(frame):
#     img_display.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     plt.draw()
#     plt.pause(0.001)

# def preprocess_for_ocr(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     return blurred

# def get_stable_prediction(predictions):
#     if not predictions:
#         return None, 0.0
#     prediction_counter = Counter(predictions)
#     most_common_prediction = prediction_counter.most_common(1)[0]
#     prediction, count = most_common_prediction
#     confidence = count / len(predictions)
#     return prediction, confidence

# try:
#     plt.ion()  # Turn on interactive mode
#     all_predictions = []  # To store all unique predictions
#     stable_prediction = None  # To store the final stable prediction
#     confidence = 0.0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to grab frame")
#             break

#         # Convert the frame to grayscale
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Apply adaptive thresholding
#         thresh_frame = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

#         # Find contours
#         contours, _ = cv2.findContours(thresh_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         processed_roi = None  # Initialize the variable
#         predicted_digit = None  # Initialize the variable
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
#                     # Preprocess the ROI for OCR
#                     processed_roi = preprocess_for_ocr(roi)

#                     # Use EasyOCR to recognize text
#                     results = reader.readtext(processed_roi, detail=0, paragraph=False)

#                     if results:
#                         # Filter out non-digit characters and keep digits between 0 and 9
#                         recognized_digits = [r for r in results if r.isdigit() and 0 <= int(r) <= 9]
#                         if recognized_digits:
#                             predicted_digit = max(set(recognized_digits), key=recognized_digits.count)
#                             # Update all predictions and ensure no consecutive duplicates
#                             if not all_predictions or predicted_digit != all_predictions[-1]:
#                                 all_predictions.append(predicted_digit)
#                                 print(f"All Predictions: {all_predictions}")
#                             # Get the most frequent prediction
#                             stable_prediction, confidence = get_stable_prediction(all_predictions)
#                             # Display the final stable prediction on the terminal
#                             print(f"Current Stable Prediction: {stable_prediction} with confidence {confidence:.2f}")

#                     # Display frames for debugging
#                     if processed_roi is not None and predicted_digit is not None:
#                         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                         cv2.putText(frame, f'Digit: {predicted_digit} ({confidence*100:.2f}%)', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
#                     else:
#                         cv2.putText(frame, f'Digit: None', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

#         show_frame(frame)

#         # Exit the loop if 'q' is pressed
#         if plt.waitforbuttonpress(0.001):
#             break

# except Exception as e:
#     print(f"Error: {e}")
# finally:
#     cap.release()
#     plt.ioff()
#     plt.close()
#     print(f"Final List of Predictions: {all_predictions}")






#PARTIAL
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
# fig, ax = plt.subplots(1, figsize=(10, 5))
# ax.axis('off')
# img_display = ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))

# def show_frame(frame):
#     img_display.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     plt.draw()
#     plt.pause(0.001)

# def preprocess_for_ocr(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     edged = cv2.Canny(blurred, 50, 150)
#     return edged

# def get_stable_prediction(predictions):
#     if not predictions:
#         return None, 0.0
#     prediction_counter = Counter(predictions)
#     most_common_prediction = prediction_counter.most_common(1)[0]
#     prediction, count = most_common_prediction
#     confidence = count / len(predictions)
#     return prediction, confidence

# try:
#     plt.ion()  # Turn on interactive mode
#     all_predictions = []  # To store all unique predictions
#     stable_prediction = None  # To store the final stable prediction
#     confidence = 0.0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to grab frame")
#             break

#         # Convert the frame to grayscale
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Apply adaptive thresholding
#         thresh_frame = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

#         # Find contours
#         contours, _ = cv2.findContours(thresh_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         processed_roi = None  # Initialize the variable
#         predicted_digit = None  # Initialize the variable
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

#                     if results:
#                         # Filter out non-digit characters and keep digits between 0 and 9
#                         recognized_digits = [r for r in results if r.isdigit() and 0 <= int(r) <= 9]
#                         if recognized_digits:
#                             predicted_digit = max(set(recognized_digits), key=recognized_digits.count)
#                             # Update all predictions and ensure no consecutive duplicates
#                             if not all_predictions or predicted_digit != all_predictions[-1]:
#                                 all_predictions.append(predicted_digit)
#                                 print(f"All Predictions: {all_predictions}")
#                             # Get the most frequent prediction
#                             stable_prediction, confidence = get_stable_prediction(all_predictions)
#                             # Display the final stable prediction on the terminal
#                             print(f"Current Stable Prediction: {stable_prediction} with confidence {confidence:.2f}")

#                     # Display frames for debugging
#                     if processed_roi is not None and predicted_digit is not None:
#                         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                         cv2.putText(frame, f'Digit: {predicted_digit} ({confidence*100:.2f}%)', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
#                     else:
#                         cv2.putText(frame, f'Digit: None', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

#         show_frame(frame)

#         # Exit the loop if 'q' is pressed
#         if plt.waitforbuttonpress(0.001):
#             break

# except Exception as e:
#     print(f"Error: {e}")
# finally:
#     cap.release()
#     plt.ioff()
#     plt.close()
#     print(f"Final List of Predictions: {all_predictions}")
