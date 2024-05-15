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

    # rviz_launch_arg = DeclareLaunchArgument(
    #     'rviz', default_value='true',
    #     description='Open RViz.'
    # )
    
    rviz = ExecuteProcess(
        cmd=["rviz2"]
    )

    return LaunchDescription([
            # rviz_launch_arg,
            rviz,
    ])