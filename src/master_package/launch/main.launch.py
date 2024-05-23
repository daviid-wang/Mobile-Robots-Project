import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():

    # pkg_ros_gz_sim_demos = get_package_share_directory('roslab')
    master_package = get_package_share_directory('master_package')

    # sdf_file = os.path.join(pkg_ros_gz_sim_demos, 'worlds', 'basic_urdf.sdf')
    robot_file = os.path.join(master_package, 'robots', 'pioneer2.urdf')
    rosbag2_snapshot_config = os.path.join(
        get_package_share_directory('master_package'),
        'config',
        'multiple_topics.params.yaml'
        )


    # with open(sdf_file, 'r') as infp:
    #     world_desc = infp.read()

    with open(robot_file, 'r') as infp:
        robot_desc = infp.read()

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
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='both',
        parameters=[
            {'use_sim_time': False},
            {'robot_description': robot_desc},
        ]
    )

    # Launch rviz
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', os.path.join(master_package, 'rviz', 'rviz2.rviz')],
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
    
    
    colour_tracking = Node( 
        package='cv_package', 
        executable='col_detection'
    )

    manual_estop = Node(
        package = 'master_package',
        executable = 'estop'
    )
    
    auto_estop = Node(
        package = 'master_package',
        executable = 'automatic_estop'
    )
    
    imu_remapping = Node(
        package='master_package',
        executable='remapping_imu',
        # parameters=[
        #     {'input_topic': 'imu/data_raw'},
        #     {'output_topic': 'imu/data_remapped'},
        # ]
        output='screen',
    )

    auto_switch = Node(
        package='master_package',
        executable='main',
        # parameters=[
        #     {'input_topic': 'imu/data_raw'},
        #     {'output_topic': 'imu/data_remapped'},
        # ]
        output='screen',
    )

    remapping_cv = Node(
        package='master_package',
        executable='remapping_cv',
        # parameters=[
        #     {'input_topic': 'imu/data_raw'},
        #     {'output_topic': 'imu/data_remapped'},
        # ]
        output='screen',
    )

    start_stage2 = Node(
        package='master_package',
        executable='start_stage2',
        output='screen',
    )

    task_bridge = Node(
        package='master_package',
        executable='task_bridge2',
        output='screen',
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
        remappings=[
            ('/sick_tim_7xx/scan', '/scan'),
        ],
    )

    phidgets = ExecuteProcess(
        cmd=["ros2", "launch", "phidgets_spatial", "spatial-launch.py"]
    )

    phidgets2 = ComposableNodeContainer(
        name = 'imu_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            ComposableNode(
                package = 'phidget_spatial',
                plugin='phidgets::SpatialRosI',
                name='phidgets_spatial',
                parameters=[
                    os.path.join(get_package_share_directory("master_package")+'/config'+'/imu.yaml')
                ],
            )
        ],
        output='both',
    )
    
    robot_localization = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[get_package_share_directory('master_package') + '/config/ekf4.yaml'], 
        remappings = [('/odomotry/filtered', '/odom')]
            # {'use_sim_time': use_sim_time}
)

    nav2_launch = IncludeLaunchDescription(
                PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('nav2_bringup'),
                    'launch',
                    'navigation_launch.py'
                ])]),
                launch_arguments={
                    'map_subscribe_transcient_local': 'true',
                    'use_sim_time': 'false',
                    'params_file': os.path.join(get_package_share_directory('master_package'), 'config', 'nav2_params3.yaml'),
                }.items(),
    )

    pioneer_base_fp_link_tf = Node(
        package='tf2_ros', 
        executable='static_transform_publisher', 
        name='base_fp_linkTF', 
        output='log', 
        arguments=['0.0', '0.0', '0.0', '0.0', '0.0', '0.0',  'base_footprint', 'pioneer3at_body/base_footprint']
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
    aria_node = ExecuteProcess(
        cmd=[[
            'ros2 run ariaNode ariaNode -rp /dev/ttyUSB0'
        ]],
        shell=True
    )

    rosbag2_snapshotter = Node(
                package='rosbag2_snapshot',
                executable='snapshotter',
                name='snapshotter',
                output='screen',
                parameters=[rosbag2_snapshot_config]
    )
    
    trigger_snapshot = Node(
                package='master_package',
                executable='trigger_snapshot',
    )
    
    
    # rosbag2_snapshotter = IncludeLaunchDescription(
    #             PythonLaunchDescriptionSource([
    #             PathJoinSubstitution([
    #                 FindPackageShare('rosbag2_snapshot'),
    #                 'launch',
    #                 'snapshotter.launch.py'
    #             ])]),
    #             launch_arguments={
    #                 'params_file': os.path.join(get_package_share_directory('master_package'), 'config', 'multiple_topics.params.yaml'),
    #             }.items(),
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

    rvizLaunch = IncludeLaunchDescription(
                PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('master_package'),
                    'launch',
                    'rviz.launch.py'
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
        # auto_switch,
        rviz_launch_arg,
        # gazebo,
        # robot,
        robot_state_publisher,
        joint_state_pub,
        # rvizLaunch,
        rviz,
        # robot_steering,
        # bridge,
        robot_localization,
        imu_remapping,
        remapping_cv,
        #number_recognition,
        colour_tracking,
        slam_toolbox,
        # phidgets2,
        # pioneer_base_fp_link_tf,
        # aria_node,
        # test,
        #lidar,
        manual_estop,
        auto_estop,
        joystick,
        nav2_launch,
        start_stage2,
        task_bridge,
        # robot
        rosbag2_snapshotter,
        trigger_snapshot,
    ])