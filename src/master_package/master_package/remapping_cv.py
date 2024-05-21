
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu

class ImuRemapNode(Node):
    def __init__(self):
        super().__init__('imu_remap_node')
        self.subscription = self.create_subscription(
            Imu,
            '/imu/data_raw',
            self.imu_callback,
            10)
        self.publisher = self.create_publisher(Imu, '/imu_remapped', 10)
        # self.get_logger().info('IMU Remap Node has been started.')

    def imu_callback(self, msg):
        # adjusted_msg = Imu()
        # adjusted_msg.header = msg.header
        # adjusted_msg.orientation = msg.orientation
        # adjusted_msg.angular_velocity = msg.angular_velocity
        # adjusted_msg.linear_acceleration = msg.linear_acceleration

        # Negate the yaw velocity
        msg.angular_velocity.z = -1*msg.angular_velocity.z

        # Publish the modified message
        self.publisher.publish(msg)
        # self.get_logger().info('Published remapped IMU data.')

def main(args=None):
    rclpy.init(args=args)
    imu_fix_node = ImuRemapNode()
    rclpy.spin(imu_fix_node)

    imu_fix_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
