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
        self.cap = cv2.VideoCapture(0)

        self.fig, self.ax = plt.subplots(1, figsize=(10, 5))
        self.ax.axis('off')
        self.img_display = self.ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
        plt.ion()
        
        self.timer = self.create_timer(0.1, self.capture_and_process_image)

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

    def capture_and_process_image(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("Failed to grab frame")
            return

        processed_frame = self.preprocess_for_ocr(frame)

        contours, _ = cv2.findContours(processed_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        recognized_digits = []
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
                    self.get_logger().info(f"All Predictions: {self.all_predictions}")

                self.stable_prediction, confidence = self.get_stable_prediction(self.all_predictions)

            if self.stable_prediction != self.last_printed_prediction:
                self.get_logger().info(f"Current Stable Prediction: {self.stable_prediction} with confidence {confidence:.2f}")
                self.last_printed_prediction = self.stable_prediction

            self.publisher.publish(Int32(data=int(self.stable_prediction)))

        self.show_frame(frame)

    def destroy_node(self):
        self.cap.release()
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

