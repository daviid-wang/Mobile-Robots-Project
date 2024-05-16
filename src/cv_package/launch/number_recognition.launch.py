from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument

import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # use_sim_time = LaunchConfiguration('use_sim_time')

    # joy_params = os.path.join(get_package_share_directory('master_package'),'config','joystick.yaml')

    number_recognition_node = Node(
            package='cv_package',
            executable='number_contour.py:main',
            output='screen'
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
        number_recognition_node
        # twist_stamper       
    ])