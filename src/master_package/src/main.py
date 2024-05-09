import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid

class ExploreAndMap(Node):
    def __init__(self):
        super().__init__('explore_and_map')

        self.create_subscription(
            OccupancyGrid,
            'map',
            self.map_callback,
            10
        )

        self.create_subscription(
            PoseWithCovarianceStamped,
            'pose',
            self.pose_callback,
            10
        )

        self.publisher = self.create_publisher(
            Twist,
            'cmd_vel',
            10
        )

        self.goal_reached = False

    def map_callback(self, msg):
        # Process map updates if needed
        pass

    def pose_callback(self, msg):
        # Process pose updates and implement logic for exploration
        if not self.goal_reached:
            # Calculate distance to center for navigation logic
            distance_to_center = self.calculate_distance_to_center(msg.pose.pose.position.x, msg.pose.pose.position.y)

            # Simple logic to navigate towards center if not already there
            if distance_to_center > 0.1:  # Example threshold
                # Adjust robot motion to move towards center
                cmd_vel_msg = self.calculate_velocity_cmd(msg.pose.pose)
                self.publisher.publish(cmd_vel_msg)
            else:
                # Stop motion and set goal_reached to True
                self.publisher.publish(Twist())
                self.goal_reached = True
                self.get_logger().info('Exploration complete. Returning to center.')

    def calculate_distance_to_center(self, x, y):
        # Calculate Euclidean distance to the center of the map
        center_x = 7.5  # Assuming center of a 15x15 grid
        center_y = 7.5  # Assuming center of a 15x15 grid
        distance = ((center_x - x)**2 + (center_y - y)**2)**0.5
        return distance

    def calculate_velocity_cmd(self, current_pose):
        # Implement logic to calculate velocity command based on current pose
        cmd_vel_msg = Twist()
        # Example: Move straight towards the center using linear velocity
        cmd_vel_msg.linear.x = 0.2  # Adjust speed as needed
        return cmd_vel_msg

def main(args=None):
    rclpy.init(args=args)
    explorer = ExploreAndMap()
    rclpy.spin(explorer)
    explorer.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
