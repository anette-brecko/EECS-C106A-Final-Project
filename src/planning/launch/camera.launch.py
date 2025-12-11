from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.events import Shutdown
from launch.actions import IncludeLaunchDescription  
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler, EmitEvent
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Logitech launch
    logitech_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('usb_cam'),
                'launch',
                'camera.launch.py'
            )
        ),
        # launch_arguments={ # TODO: IDk if you need these
        #     'rgb_camera.color_profile': '1920x1080x30',
        # }.items(),
    )
    # Perception node for logitech
    perception_node = Node(
        package='ball_sense',
        executable='ball_sense',
        name='ball_sense',
        output='screen',
    )

    # ArUco recognition
    aruco_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('ros2_aruco'),
                'launch',
                'aruco_recognition.launch.py'
            )
        )
    )

    ar_marker_launch_arg = DeclareLaunchArgument(
        'ar_marker',
        default_value='ar_marker_6' # TODO: Change to our desried ar_marker
    )
    ar_marker = LaunchConfiguration('ar_marker')

    # Planning TF node
    aruco_tf_node = Node(
        package='planning',
        executable='aruco_tf',
        name='aruco_tf_node',
        output='screen',
        parameters=[{
            'ar_marker': ar_marker,
        }]
    )
    
    kinect_tf_node = Node(
        package='planning',
        executable='kinect_tf',
        name='kinect_tf_node',
        output='screen',
    )

    transform_ball_pose_node = Node(
        package='planning',
        executable='transform_ball_pose',
        name='transform_ball_pose_node',
        output='screen',
    )

    transform_wrist_pose_node = Node(
        package='planning',
        executable='transform_wrist_pose',
        name='transform_wrist_pose_node',
        output='screen',
    )

    # Static TF: base_link -> world
    # -------------------------------------------------
    # This TF is static because the "world" frame does not move.
    # It is necessary to define the "world" frame for MoveIt to work properly as this is the defualt planning frame.
    # -------------------------------------------------
    static_base_world = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_base_world',
        arguments=['0','0','0','0','0','0','1','base_link','world'],
        output='screen',
    )


    # -------------------------
    # Global shutdown on any process exit
    # -------------------------
    shutdown_on_any_exit = RegisterEventHandler(
        OnProcessExit(
            on_exit=[EmitEvent(event=Shutdown(reason='SOMETHING BONKED'))]
        )
    )
    
    return LaunchDescription([
        ar_marker_launch_arg,
        logitech_launch,
        aruco_launch,
        perception_node,
        kinect_tf_node,
        aruco_tf_node,
        static_base_world,
        transform_ball_pose_node,
        transform_wrist_pose_node,
        shutdown_on_any_exit
    ])
