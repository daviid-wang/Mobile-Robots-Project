import numpy as np
import cv2
import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge




class ColourDetect(Node):
    def __init__(self):
        super().__init__('col_detection_node')

        self.subscription = self.create_subscription(Image, 'oak/rgb/image_raw', self.callback_listener, 10)
        self.publisher = self.create_publisher(String, 'col_detect', 10)
        
        self.bridge = CvBridge()
        self.fig, self.ax = plt.subplots()
        self.ax.axis('off')
        plt.ion()
        
        self.exit_prog = False
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
    
    def on_key(self, evnt):
        if evnt.key == 'q':
            self.exit_prog = True
    
    def callback_listener(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding = 'bgr8')
        hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
        
        lower_red = np.array([136,87,111], np.uint8)
        upper_red = np.array([180,255,255], np.uint8)
        mask_red = cv2.inRange(hsvFrame, lower_red, upper_red)
        
        lower_yellow = np.array([20,100,100],np.uint8)
        upper_yellow = np.array([30,255,255],np.uint8)
        mask_yellow = cv2.inRange(hsvFrame, lower_yellow, upper_yellow)
        
        kernal = np.ones((5,5), np.uint8)
        
        mask_yellow = cv2.dilate(mask_yellow, kernal)
        yellow_res = cv2.bitwise_and(frame, frame, mask=mask_yellow)
        mask_red = cv2.dilate(mask_red, kernal)
        red_res = cv2.bitwise_and(frame, frame, mask=mask_red)     
        self.get_logger().info("TIME TO LOOK FOR COLOURED OBJECTS!")
        #make contours to track red
        conts, h = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for _, cont in enumerate(conts):
            area = cv2.contourArea(cont)
            if area > 300:
                x,y,w,h = cv2.boundingRect(cont)
                frame = cv2.rectangle(frame, (x,y), (x + w, y + h), (0,0,255), 2)
                cv2.putText(frame, "RED", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255))
                self.publisher.publish(String(data = "RED"))
                self.get_logger().info("RED")

        #make contours to track yellow
        conts, h = cv2.findContours(mask_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for pic, cont in enumerate(conts):
            area = cv2.contourArea(cont)
            if area > 300:
                x,y,w,h = cv2.boundingRect(cont)
                frame = cv2.rectangle(frame, (x,y), (x + w, y + h), (0,255,255), 2)
                cv2.putText(frame, "YELLOW", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,255))
                self.publisher.publish(String(data = "YELLOW"))
                self.get_logger().info("YELLOW")
        
        self.ax.clear()
        self.ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.draw()
        plt.pause(0.001)
        
        if self.exit_prog:
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args = args)
    node = ColourDetect()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        plt.close()

if __name__ == '__main__':
    main()
