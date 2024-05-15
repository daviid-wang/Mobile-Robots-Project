import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    
    # lidar = ExecuteProcess(
    #     cmd=["rviz2"]
    # )
    
    lidar = Node(
            package='sick_scan_xd',
            executable='sick_generic_caller',
            output='screen',
            arguments=["launch/sick_tim_7xx.launch.py"]
        )

    return LaunchDescription([
            lidar,
    ])