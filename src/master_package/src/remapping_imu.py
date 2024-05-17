import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu

class ImuRemapNode(Node):
    def __init__(self):
        super().__init__('imu_remap_node')
        self.subscription = self.create_subscription(
            Imu,
            'imu/data_raw',
            self.imu_callback,
            10)
        self.publisher = self.create_publisher(Imu, 'imu/data_remapped', 10)
        self.get_logger().info('IMU Remap Node has been started.')

    def imu_callback(self, msg):
        # Negate the yaw velocity
        msg.angular_velocity.z = -msg.angular_velocity.z

        # Publish the modified message
        self.publisher.publish(msg)
        self.get_logger().info('Published remapped IMU data.')

def main(args=None):
    rclpy.init(args=args)
    node = ImuRemapNode()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
