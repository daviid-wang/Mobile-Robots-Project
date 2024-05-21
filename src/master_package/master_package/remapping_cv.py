import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TwistWithCovarianceStamped

class CmdVelRemapNode(Node):
    def __init__(self):
        super().__init__('cmd_vel_remap_node')
        self.subscription = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10)
        self.publisher = self.create_publisher(TwistWithCovarianceStamped, '/cmd_vel_remapped', 10)
        #self.get_logger().info('CmdVel Remap Node has been started.')

    def cmd_vel_callback(self, msg):
        # Create a new TwistWithCovarianceStamped message
        twist_cov_msg = TwistWithCovarianceStamped()
        
        # Set the header
        twist_cov_msg.header.stamp = self.get_clock().now().to_msg()
        twist_cov_msg.header.frame_id = 'base_link'
        
        # Copy Twist data
        twist_cov_msg.twist.twist = msg
        
        # Optionally, you can set the covariance matrix if needed
        # twist_cov_msg.twist.covariance = [0]*36
        
        # Publish the new message
        self.publisher.publish(twist_cov_msg)
        #self.get_logger().info('Published remapped cmd_vel data.')

def main(args=None):
    rclpy.init(args=args)
    node = CmdVelRemapNode()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()