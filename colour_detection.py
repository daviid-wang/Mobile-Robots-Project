import numpy as np
import cv2
import matplotlib.pyplot as plt

# Capturing video through webcam
webcam = cv2.VideoCapture(0)

# Create a figure to display the image
plt.ion()
fig, ax = plt.subplots()

# Variable to check if the 'q' key is pressed
exit_program = False

def on_key(event):
    global exit_program
    if event.key == 'q':
        exit_program = True

fig.canvas.mpl_connect('key_press_event', on_key)

# Start a while loop
while not exit_program:
    # Reading the video from the webcam in image frames
    _, imageFrame = webcam.read()

    # Convert the imageFrame in BGR (RGB color space) to HSV (hue-saturation-value) color space
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

    # Set range for red color and define mask
    red_lower = np.array([136, 87, 111], np.uint8)
    red_upper = np.array([180, 255, 255], np.uint8)
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

    # Set range for yellow color and define mask
    yellow_lower = np.array([20, 100, 100], np.uint8)
    yellow_upper = np.array([30, 255, 255], np.uint8)
    yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper)

    # Morphological Transform, Dilation for each color and bitwise_and operator between imageFrame and mask determines to detect only that particular color
    kernel = np.ones((5, 5), "uint8")

    # For yellow color
    yellow_mask = cv2.dilate(yellow_mask, kernel)
    res_yellow = cv2.bitwise_and(imageFrame, imageFrame, mask=yellow_mask)
    
    # For red color
    red_mask = cv2.dilate(red_mask, kernel)
    res_red = cv2.bitwise_and(imageFrame, imageFrame, mask=red_mask)

    # Creating contour to track red color
    contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(imageFrame, "Red Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))

    # Creating contour to track yellow color
    contours, hierarchy = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(imageFrame, "Yellow Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255))

    # Display the resulting frame using matplotlib
    ax.clear()
    ax.imshow(cv2.cvtColor(imageFrame, cv2.COLOR_BGR2RGB))
    plt.draw()
    plt.pause(0.001)

# Release the webcam and close the figure
webcam.release()
plt.close()
