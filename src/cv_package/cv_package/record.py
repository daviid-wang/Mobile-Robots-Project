import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
import cv2
import numpy as np
from depthai_sdk import OakCamera

class CircularVideoBuffer:
    def __init__(self, buffer_time=5, fps=30):
        self.buffer_time = buffer_time
        self.fps = fps
        self.frames = []
        self.frame_width = 1920
        self.frame_height = 1080

    def add_frame(self, frame):
        if len(self.frames) >= self.buffer_time * self.fps:
            self.frames.pop(0)
        self.frames.append(frame)

    def save(self, filename):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, self.fps, (self.frame_width, self.frame_height))
        for frame in self.frames:
            out.write(frame)
        out.release()

class EmergencyStopHandler(Node):
    def __init__(self):
        super().__init__('emergency_stop_handler')
        self.subscription = self.create_subscription(
            Bool,
            'emergency_stop',
            self.listener_callback,
            10)
        self.camera = OakCamera()
        self.color = self.camera.create_camera('color', resolution='1080P', fps=30, encode=False)
        self.camera.start(blocking=False)
        self.buffer = CircularVideoBuffer(buffer_time=5, fps=30)
        self.get_logger().info('Emergency Stop Handler Initialized and Running')

    def listener_callback(self, msg):
        if msg.data:  # True means emergency stop
            self.get_logger().info('Emergency stop detected, saving last 5 seconds of video.')
            self.buffer.save('/home/username/Desktop/last_5_seconds.mp4')
            rclpy.shutdown()

    def capture_video(self):
        while rclpy.ok():
            frame = self.color.get_frame()
            if frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.buffer.add_frame(frame_rgb)
            rclpy.spin_once(self, timeout_sec=0.1)

def main(args=None):
    rclpy.init(args=args)
    emergency_stop_handler = EmergencyStopHandler()
    try:
        emergency_stop_handler.capture_video()
    except Exception as e:
        emergency_stop_handler.get_logger().error(f'An error occurred: {str(e)}')
    finally:
        emergency_stop_handler.camera.stop()
        emergency_stop_handler.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
