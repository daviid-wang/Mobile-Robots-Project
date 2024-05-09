import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped, PoseStamped
from sensor_msgs.msg import LaserScan
import numpy as np
from math import atan2, pi

class ExploreAndMap(Node):
    def __init__(self):
        super().__init__('explore_and_map')

        # Initialize the exploration array (30x30 grid with 0.5m spacing)
        self.exploreArray = np.zeros((30, 30), dtype=bool)

        self.create_subscription(
            PoseWithCovarianceStamped,
            'pose',
            self.pose_callback,
            10
        )

        self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            10
        )

        self.publisher = self.create_publisher(
            Twist,
            'cmd_vel',
            10
        )

        self.current_pose = None
        self.origin_pose = PoseStamped()
        self.origin_pose.pose.position.x = 0.0  # Top-left corner
        self.origin_pose.pose.position.y = 0.0  # Top-left corner
        self.waypoint = self.origin_pose
        self.goal_reached = False

    def pose_callback(self, msg):
        # Process pose updates and implement navigation logic
        self.current_pose = msg.pose.pose

        if not self.goal_reached and self.waypoint is not None:
            distance_to_target = self.calculate_distance_to_target(
                self.current_pose.position, self.waypoint.pose.position
            )

            if distance_to_target > 0.1:  # Example threshold
                # Calculate velocity command to move towards target
                cmd_vel_msg = self.calculate_velocity_cmd(
                    self.current_pose, self.waypoint.pose
                )
                self.publisher.publish(cmd_vel_msg)
            else:
                # Reached current waypoint, update exploration status
                self.update_exploration_status()

    def scan_callback(self, msg):
        # Process laser scan data to detect obstacles
        if self.current_pose is not None:
            # Check if there's an obstacle in front of the robot
            min_distance = min(msg.ranges)
            if min_distance < 0.5:  # Example threshold for obstacle detection
                # Obstacle detected, calculate a new path to avoid it
                self.plan_obstacle_avoidance()

    def update_exploration_status(self):
        # Update the exploration array based on the robot's current position
        x_index = int(self.current_pose.position.x / 0.5)  # Convert to grid index (0.5m spacing)
        y_index = int(self.current_pose.position.y / 0.5)  # Convert to grid index (0.5m spacing)

        if 0 <= x_index < 30 and 0 <= y_index < 30:
            self.exploreArray[x_index, y_index] = True

        self.choose_next_waypoint()

    def choose_next_waypoint(self):
        # Determine the nearest unexplored node as the next waypoint
        unexplored_indices = np.where(self.exploreArray == False)
        
        if len(unexplored_indices[0]) > 0:
            # Find the closest unexplored node
            closest_index = np.argmin(
                np.linalg.norm(np.array(unexplored_indices) - np.array([[self.current_pose.position.x / 0.5], [self.current_pose.position.y / 0.5]]), axis=0)
            )
            closest_cell = (unexplored_indices[0][closest_index], unexplored_indices[1][closest_index])
            
            # Set the next waypoint to the closest unexplored node
            self.waypoint = PoseStamped()
            self.waypoint.pose.position.x = closest_cell[0] * 0.5  # Convert grid index back to meters
            self.waypoint.pose.position.y = closest_cell[1] * 0.5  # Convert grid index back to meters
            self.waypoint.header.frame_id = 'map'  # Assuming map frame
        else:
            # All nodes explored, return to origin
            self.waypoint = self.origin_pose
            self.goal_reached = True
            self.get_logger().info('All nodes explored. Returning to origin.')

    def plan_obstacle_avoidance(self, scan_msg):
        # Stop linear motion
        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = 0.0

        # Determine the direction of rotation (left by default)
        rotation_direction = 1  # Rotate left
        scan_range = len(scan_msg.ranges)

        # Get index ranges for the right and left sides of the laser scan
        left_start = int(scan_range * 5 / 8)  # 112.5 degrees
        left_end = int(scan_range * 7 / 8)  # 157.5 degrees
        right_start = int(scan_range * 1 / 8)  # 22.5 degrees
        right_end = int(scan_range * 3 / 8)  # 67.5 degrees

        # Check if there's an obstacle in the front-right sector
        if min(scan_msg.ranges[right_start:right_end]) < 0.5:
            # Rotate left until the right side is clear
            cmd_vel_msg.angular.z = 0.5 * rotation_direction
        else:
            # Continue forward until an obstacle is detected
            cmd_vel_msg.linear.x = 0.2

        # Publish the new velocity command
        self.publisher.publish(cmd_vel_msg)



    def calculate_distance_to_target(self, current_position, target_position):
        # Calculate Euclidean distance to target position
        distance = np.linalg.norm(np.array([target_position.x, target_position.y]) - np.array([current_position.x, current_position.y]))
        return distance

    def calculate_velocity_cmd(self, current_pose, target_pose):
        # Implement logic to calculate velocity command towards target pose
        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = 0.2  # Adjust linear velocity as needed
        # Calculate desired heading angle
        desired_heading = atan2(target_pose.position.y - current_pose.position.y,
                                target_pose.position.x - current_pose.position.x)
        cmd_vel_msg.angular.z = 0.5 * (desired_heading - current_pose.orientation.z)
        return cmd_vel_msg

def main(args=None):
    rclpy.init(args=args)
    explorer = ExploreAndMap()
    rclpy.spin(explorer)
    explorer.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
