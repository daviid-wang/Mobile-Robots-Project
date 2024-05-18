import numpy as np
import cv2
import matplotlib.pyplot as plt

class COLNode(node):
    def __init__(self):
        super().__init__('col_detection_node')
        self.subscription = self.create_subscription(Image, 'camera/image_raw', self.listener_callback, 10)
        self.publisher = self.create_publisher(Str, 'col_detection', 10)
        self.bridge = CvBridge()
        self.fig, self.ax = plt.subplots()
        self.ax.axis('off')
        plt.ion()
        
        self.exit_program = False
        self.fig.canvas.mpl_connect('button_press_event', self.on_key)
        

    def on_key(self, event):
        if event.key == 'q':
            exit_program = True
            
    def listener_callback(self, message):
        frame = self.bridge.imgmsg_to_cv2(message, ideal_encoding = 'bgr8')
        hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red = np.array([136,87,11], np.uint8)
        upper_red = np.array([180,255,255], np.uint8)
        mask_red = cv2.inRange(hsvFrame, lower_red, upper_red)
        
        lower_yellow = np.array([20,100,100], np.uint8)
        upper_yellow = np.array([30,255,255], np.uint8)
        mask_yellow = cv2.inRange(hsvFrame, lower_yellow, upper_yellow)  
        
        kernel = np.ones((5,5), "uint8")
        
        mask_yellow = cv2.dilate(mask_yellow, kernel) 
        yellow_res = cv2.bitwise_and(frame, frame, mask = mask_yellow) 
        
        mask_red = cv2.dilate(mask_red, kernel)    
        red_res = cv2.bitwise_and(frame, frame, mask = mask_red)
        
        contours, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for _, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 300:
                x,y,w,h = cv2.boundingRect(contour)
                frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)
                cv2.putText(frame, "RED", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255))
                self.publisher.publish(Str(data = "Red"))
                self.get_logger().info("RED COLOUR!!!!!")                
        
        contours, _ cv2.findContours(mask_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for _, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 300:
                x,y,w,h = cv2.boundingRect(contour)
                frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,255), 2)
                cv2.putText(frame, "YELLOW", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255))
                self.publisher.publish(Str(data = "Yellow"))
                self.get_logger().info("YELLOW COLOUR!!!!!")   
        
        self.ax.clear()
        self.ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.draw()
        plt.pause(0.001)
        if self.exit_program:
            rclpy.shutdown()
    
    def main(args=None):
        rclpy.init(args=args)
        node = col_detection_node()
        try:
            rcply.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            plt.close()
    
    if __name__ == '__main__':
        main()
        
    
        
