from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, ExecuteProcess

import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')

    joy_params = os.path.join(get_package_share_directory('master_package'),'config','joystick.yaml')

    # lidar_node = Node(
    #         package='sick_scan_xd',
    #         executable='sick_lidar',
    #         parameters=[joy_params, {'use_sim_time': use_sim_time}],
    #      )
    lidar = ExecuteProcess(
        cmd=["ros2", "launch", "sick_scan_xd", "sick_tim_7xx.launch.py"]
    )
    # twist_stamper = Node(
    #         package='twist_stamper',
    #         executable='twist_stamper',
    #         parameters=[{'use_sim_time': use_sim_time}],
    #         remappings=[('/cmd_vel_in','/diff_cont/cmd_vel_unstamped'),
    #                     ('/cmd_vel_out','/diff_cont/cmd_vel')]
    #      )


    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use sim time if true'),
        lidar
        # teleop_node,
        # twist_stamper       
    ])