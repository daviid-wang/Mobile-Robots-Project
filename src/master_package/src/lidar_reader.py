import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class LidarReader(Node):

    def __init__(self):
        super().__init__('lidar_reader')
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10  # QoS profile depth
        )
        self.scan_data = []  # Array to store lidar scan data

    def lidar_callback(self, msg):
        self.scan_data = msg.ranges  # Store the scan ranges in the array

    def get_lidar_scan(self):
        return self.scan_data


def main(args=None):
    rclpy.init(args=args)
    lidar_reader = LidarReader()
    try:
        rclpy.spin(lidar_reader)
    except KeyboardInterrupt:
        pass
    finally:
        lidar_scan = lidar_reader.get_lidar_scan()
        lidar_reader.destroy_node()
        rclpy.shutdown()
        return lidar_scan


if __name__ == '__main__':
    lidar_scan = main()
    print(f"Lidar Scan Data (360 values): {lidar_scan}")
