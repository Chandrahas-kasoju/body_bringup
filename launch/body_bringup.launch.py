from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    hospibot_teleop_pkg = FindPackageShare('hospibot_teleop').find('hospibot_teleop')
    
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(hospibot_teleop_pkg, 'launch', 'teleop.launch.py')
            )
        ),
        Node(
            package='hospibot_led',
            executable='hospibot_led_node',
            name='hospibot_led_node',
            output='screen'
        )
    ])
