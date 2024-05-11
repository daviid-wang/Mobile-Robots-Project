import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped, PoseStamped
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
import numpy as np
from math import atan2

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

        self.map_received = False
        self.current_pose = None
        self.origin_pose = PoseStamped()
        self.origin_pose.pose.position.x = 7.5  # Assuming center of a 15x15 grid
        self.origin_pose.pose.position.y = 7.5  # Assuming center of a 15x15 grid
        self.waypoint = None
        self.goal_reached = False

    def map_callback(self, msg):
        # Process map updates
        if not self.map_received:
            self.map_data = msg
            self.map_received = True
            self.explore_map()

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
                # Reached current waypoint, update waypoint
                self.update_waypoint()

    def explore_map(self):
        # Explore the occupancy grid by finding unexplored cells
        unexplored_indices = np.where(np.array(self.map_data.data) == -1)[0]

        #TODO: Search the array within 15m by 15m for unsearched grid points

        if len(unexplored_indices) > 0:
            # Convert index to (row, col) coordinates
            unexplored_cells = [
                (index // self.map_data.info.width, index % self.map_data.info.width)
                for index in unexplored_indices
            ]

            # Choose the closest unexplored cell as the next waypoint
            closest_cell = min(
                unexplored_cells,
                key=lambda cell: self.calculate_distance_to_target(
                    self.current_pose.position, self.cell_to_position(cell)
                )
            )

            self.waypoint = PoseStamped()
            self.waypoint.pose.position = self.cell_to_position(closest_cell)
            self.waypoint.header = self.map_data.header

            self.get_logger().info(f'Navigating to unexplored cell: {closest_cell}')
        else:
            # No unexplored cells remaining, return to origin
            self.waypoint = self.origin_pose
            self.get_logger().info('All cells explored. Returning to origin.')

    def update_waypoint(self):
        # Update waypoint to the next unexplored cell or return to origin
        if self.waypoint == self.origin_pose:
            self.goal_reached = True
            self.publisher.publish(Twist())
        else:
            self.explore_map()

    def scan_callback(self, msg):
        # Process laser scan data to detect obstacles
        if self.current_pose is not None:
            # Check if there's an obstacle in front of the robot
            min_distance = min(msg.ranges)
            if min_distance < 0.5:  # Example threshold for obstacle detection
                # Obstacle detected, plan obstacle avoidance
                self.plan_obstacle_avoidance(msg)

    def plan_obstacle_avoidance(self, scan_msg):
        # Plan a new path to avoid the detected obstacle
        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = 0.0  # Stop linear motion

        # Determine the direction of rotation (left by default)
        rotation_direction = 1  # Rotate left
        scan_range = len(scan_msg.ranges)

        # Get index ranges for the right and left sides of the laser scan
        left_start = int(scan_range * 5 / 8)  # 112.5 degrees
        left_end = int(scan_range * 7 / 8)  # 157.5 degrees
        right_start = int(scan_range * 1 / 8)  # 22.5 degrees
        right_end = int(scan_range * 3 / 8)  # 45 degrees

        # Check if there's an obstacle in the front-right sector
        if min(scan_msg.ranges[right_start:right_end]) < 0.5:
            # Rotate left until the right side is clear
            cmd_vel_msg.angular.z = 0.5 * rotation_direction
        else:
            # Continue forward until an obstacle is detected
            cmd_vel_msg.linear.x = 0.2

        # Publish the new velocity command to avoid the obstacle
        self.publisher.publish(cmd_vel_msg)

    def calculate_distance_to_target(self, current_position, target_position):
        # Calculate Euclidean distance to target position
        distance = ((target_position.x - current_position.x) ** 2 +
                    (target_position.y - current_position.y) ** 2) ** 0.5
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

    def cell_to_position(self, cell):
        # Convert (row, col) cell coordinates to world position
        map_info = self.map_data.info
        x = map_info.origin.position.x + (cell[1] + 0.5) * map_info.resolution
        y = map_info.origin.position.y + (cell[0] + 0.5) * map_info.resolution
        return (x, y, map_info.origin.position.z)

def main(args=None):
    rclpy.init(args=args)
    explorer = ExploreAndMap()
    rclpy.spin(explorer)
    explorer.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
