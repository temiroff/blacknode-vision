from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="perception_camera",
            executable="usb_camera",
            name="perception_camera",
            output="screen",
            parameters=[{
                "device": 0,
                "image_topic": "/camera/image_raw",
                "frame_id": "camera",
                "hz": 30.0,
                "width": 640,
                "height": 480,
                "rotation": 0,
            }],
        )
    ])
