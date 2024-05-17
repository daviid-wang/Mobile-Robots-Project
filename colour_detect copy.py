import numpy as np
import cv2
from PIL import Image
from keras import layers
import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from cv_bridge import CvBridge
from collections import Counter
     
        
def get_limits(color):
    c = np.uint8([[color]])  # BGR values
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

    hue = hsvC[0][0][0]  # Get the hue value

    # Handle red hue wrap-around
    if hue >= 165:  # Upper limit for divided red hue
        lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upperLimit = np.array([180, 255, 255], dtype=np.uint8)
    elif hue <= 15:  # Lower limit for divided red hue
        lowerLimit = np.array([0, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)
    else:
        lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)

    return lowerLimit, upperLimit

yellow = [0, 255, 255]  # yellow in BGR colorspace
red    = [0,0,255]        # red BGR
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()

    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lowerLimit, upperLimit = get_limits(color=red)
    #lowerLimit1, upperLimit1 = get_limits(color=red)
    #new_lower_lim = np.minimum(lowerLimit, lowerLimit1)
    #new_upper_lim = np.maximum(upperLimit, upperLimit1)

    mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)

    mask_ = Image.fromarray(mask)

    bbox = mask_.getbbox()

    if bbox is not None:
        x1, y1, x2, y2 = bbox

        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()

# def calculate_histogram(image):
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     hue = hsv[:,:,0]
#     # Calculate the histogram of the hue channel
#     hist = cv2.calcHist([hue], [0], None, [180], [0, 180])
#     plt.plot(hist)
#     plt.show()
#     return hist

# def find_color_ranges(hist, threshold=0.1):
#     peaks = []
#     # Find peaks above a certain threshold
#     for i in range(1, len(hist) - 1):
#         if hist[i] > threshold * np.max(hist) and hist[i] > hist[i - 1] and hist[i] > hist[i + 1]:
#             peaks.append(i)
#     return peaks

# def apply_color_mask(hsv, peak):
#     # Define a reasonable range around the peak
#     lower_bound = max(0, peak - 10)
#     upper_bound = min(180, peak + 10)
#     mask = cv2.inRange(hsv, (lower_bound, 100, 100), (upper_bound, 255, 255))
#     return mask

# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     hist = calculate_histogram(frame)
#     peaks = find_color_ranges(hist)

#     combined_mask = np.zeros_like(hsv[:,:,0])
#     for peak in peaks:
#         mask = apply_color_mask(hsv, peak)
#         combined_mask = cv2.bitwise_or(combined_mask, mask)

#     # Display the result
#     result = cv2.bitwise_and(frame, frame, mask=combined_mask)
#     cv2.imshow('frame', result)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()



