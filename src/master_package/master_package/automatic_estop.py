import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from sensor_msgs.msg import Joy
from cv_bridge import CvBridge
import cv2
import numpy as np


class MotionDetect(Node):
    def __init__(self):
        super().__init__('motion_detect')
        self.publisher = self.create_publisher(Bool, '/motion_detected', 10)
        self.subscription  = self.create_subscription(Image, '/oak/rgb/image_raw', self.image_callback, 10)
        self.bridge = CvBridge()
        self.prev_frame = None

    def image_callback(self, msg):
        curr_frme = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        gray = cv2.cvtColor(curr_frme, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21,21), 0)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return
        
        change_in_frame = cv2.absdiff(self.prev_frame, gray)
        self.prev_frame = gray
        
        _, threshold = cv2.threshold(change_in_frame, 25,255, cv2.THRESH_BINARY)
        threshold = cv2.dilate(threshold, None, iterations = 2)
        contours, _ = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_detect = False
        for contour in contours:
            if cv2.contourArea(contour) > 2500:
                motion_detect = True
                #self.get_logger().info("ROBOT STOPPED)")    
                break
        self.publisher.publish(Bool(data = motion_detect))

def main(args=None):
    rclpy.init(args=args)
    motion_detect = MotionDetect()
    rclpy.spin(motion_detect)
    motion_detect.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()



