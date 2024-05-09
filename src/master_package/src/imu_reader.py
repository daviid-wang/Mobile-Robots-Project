import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3
import math


class IMUListener(Node):

    def __init__(self):
        super().__init__('imu_listener')
        self.subscription = self.create_subscription(
            Imu,
            '/imu/data_raw',
            self.imu_callback,
            10  # QoS profile depth
        )
        self.last_time = None
        self.orientation = 0.0  # Initial orientation (radians)
        self.angular_velocity = 0.0  # Initial angular velocity (radians/second)
        self.position = Vector3()  # Robot's position (x, y, z)

    def imu_callback(self, msg):
        # Calculate time difference since last callback
        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_time is not None:
            dt = current_time - self.last_time

            # Extract angular velocity around z-axis from the message
            angular_vel_z = msg.angular_velocity.z
            self.angular_velocity += angular_vel_z * dt  # Integrate angular velocity
            self.orientation += self.angular_velocity * dt  # Integrate orientation

            # Normalize orientation to between 0 and 2*pi
            self.orientation = self.orientation % (2 * math.pi)

            # Update robot's position based on orientation (assuming flat surface)
            self.position.x += math.cos(self.orientation) * msg.linear_acceleration.x * dt
            self.position.y += math.sin(self.orientation) * msg.linear_acceleration.x * dt

        self.last_time = current_time

    def get_current_location(self):
        return self.position.x, self.position.y


def main(args=None):
    rclpy.init(args=args)
    imu_listener = IMUListener()
    try:
        rclpy.spin(imu_listener)
    except KeyboardInterrupt:
        pass
    finally:
        location = imu_listener.get_current_location()
        imu_listener.destroy_node()
        rclpy.shutdown()
        return location


if __name__ == '__main__':
    location = main()
    print(f"Current Location (x, y): {location}")
