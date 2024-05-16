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

    # pkg_ros_gz_sim_demos = get_package_share_directory('roslab')
    # pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    # sdf_file = os.path.join(pkg_ros_gz_sim_demos, 'worlds', 'basic_urdf.sdf')
    # robot_file = os.path.join(pkg_ros_gz_sim_demos, 'robots', 'pioneer.urdf')

    # with open(sdf_file, 'r') as infp:
    #     world_desc = infp.read()

    # with open(robot_file, 'r') as infp:
    #     robot_desc = infp.read()

    rviz_launch_arg = DeclareLaunchArgument(
        'rviz', default_value='true',
        description='Open RViz.'
    )

    # gazebo = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(
    #         os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py'),
    #     ),
    #     launch_arguments={'gz_args': PathJoinSubstitution([
    #         pkg_ros_gz_sim_demos,
    #         'worlds',
    #         'basic_urdf.sdf'
    #     ])}.items(),
    # )


    # Get the parser plugin convert sdf to urdf using robot_description topic
    # robot_state_publisher = Node(
    #     package='robot_state_publisher',
    #     executable='robot_state_publisher',
    #     name='robot_state_publisher',
    #     output='both',
    #     parameters=[
    #         {'use_sim_time': False},
    #         {'robot_description': robot_desc},
    #     ]
    # )

    # Launch rviz
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        # arguments=['-d', os.path.join(pkg_ros_gz_sim_demos, 'rviz', 'vehicle.rviz')],
        condition=IfCondition(LaunchConfiguration('rviz')),
        parameters=[
            {'use_sim_time': False},
        ]
    )
    
    # Launch number recogniton
    # number_recognition = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(os.path.join('master_package', 'number_contour.py')),
    #     )
    
    number_recognition = Node( 
        package='cv_package', 
        executable='number_contour'
    )

    # robot = ExecuteProcess(
    #     cmd=["ros2", "run", "ros_gz_sim", "create", "-topic", "robot_description", "-z", "0.2"],
    #     name="spawn robot",
    #     output="both"
    # )

    # bridge = Node(
    #     package='ros_gz_bridge',
    #     executable='parameter_bridge',
    #     arguments=['/lidar@sensor_msgs/msg/LaserScan@ignition.msgs.LaserScan',
    #                 '/imu@sensor_msgs/msg/Imu@ignition.msgs.IMU',
    #                 '/model/pioneer3at_body/odometry@nav_msgs/msg/Odometry@ignition.msgs.Odometry',
    #                 '/cmd_vel@geometry_msgs/msg/Twist@ignition.msgs.Twist',
    #                 '/camera@sensor_msgs/msg/Image@ignition.msgs.Image',
    #                 '/model/pioneer3at_body/tf@tf2_msgs/msg/TFMessage@ignition.msgs.Pose_V',
    #                 '/clock@rosgraph_msgs/msg/Clock@ignition.msgs.Clock',],
    #     output='screen',
    #     remappings=[('/cmd_vel','/cmd_vel'),
    #                 ('/model/pioneer3at_body/odometry','/odom'),
    #                 ('/model/pioneer3at_body/tf','/tf')
    #     ]
    # )

    slam_toolbox = Node( 
        package='slam_toolbox', 
        executable='async_slam_toolbox_node', 
        parameters=[
                get_package_share_directory('master_package') + '/config/mapping.yaml'
        ], 
        output='screen',
    )

    phidgets = ExecuteProcess(
        cmd=["ros2", "launch", "phidgets_spatial", "spatial-launch.py"]
    )
    
    robot_localization = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[
            get_package_share_directory('master_package') + 'config/ekf.yaml', 
            # {'use_sim_time': use_sim_time}
    ])

    pioneer_base_fp_link_tf = Node(package='tf2_ros', 
        executable='static_transform_publisher', 
        name='base_fp_linkTF', 
        output='log', 
        arguments=['0.0', '0.0', '0.0', '0.0', '0.0', '0.0',  'pioneer3at_body/base_footprint', 'base_footprint']
    )

    joint_state_pub = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher'
    )

    # A gui tool for easy tele-operation.
    # robot_steering = Node(
    #     package="rqt_robot_steering",
    #     executable="rqt_robot_steering",
    # )
    # aria_node = ExecuteProcess(
    #     cmd=[[
    #         'ros2 run ariaNode ariaNode -rp /dev/ttyUSB0'
    #     ]],
    #     shell=True
    # )
    
    # test = ExecuteProcess(
    #     cmd=[[
    #         'ros2 topic echo /joy'
    #     ]],
    #     shell=True
    # )

    joystick = IncludeLaunchDescription(
                PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('master_package'),
                    'launch',
                    'joystick.launch.py'
                ])
            ]),
    )
    
    lidar = IncludeLaunchDescription(
                PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('master_package'),
                    'launch',
                    'lidar.launch.py'
                ])
            ]),
    )

    robot = IncludeLaunchDescription(
                PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('master_package'),
                    'launch',
                    'robot.launch.py'
                ])
            ]),
    )
    

    # Node(
    #         package='v4l2_camera',
    #         executable='v4l2_camera_node',
    #         output='screen',
    #         parameters=[{
    #             'image_size': [640,480],
    #             'camera_frame_id': 'camera_link_optical'
    #             }]
    # )

    return LaunchDescription([
        # rviz_launch_arg,
        # gazebo,
        # robot,
        # robot_state_publisher,
        # joint_state_pub,
        # rviz,
        # robot_steering,
        # bridge,
        number_recognition,
        # slam_toolbox,
        phidgets,
        # pioneer_base_fp_link_tf,
        # aria_node,
        # test,
        lidar,
        joystick
        # robot
    ])