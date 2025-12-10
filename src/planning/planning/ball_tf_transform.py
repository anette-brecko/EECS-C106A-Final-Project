#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster
from scipy.spatial.transform import Rotation as R
import numpy as np

class LogitechConstantTransformPublisher(Node):
    def __init__(self):
        super().__init__('logitech_constant_tf_publisher')
        self.br = StaticTransformBroadcaster(self)

        # Calculate a dynamic transform matrix for logitech->aruco tag
        

        # Create TransformStamped
        self.transform = TransformStamped()
        
        self.transform.child_frame_id = "ar_marker_6"
        self.transform.header.frame_id = "camera1"

        quat = R.from_matrix(G[0:3,0:3]).as_quat()
        
        self.transform.transform.translation.x = G[0][3]
        self.transform.transform.translation.y = G[1][3]
        self.transform.transform.translation.z = G[2][3]

        self.transform.transform.rotation.x = quat[0]
        self.transform.transform.rotation.y = quat[1]
        self.transform.transform.rotation.z = quat[2]
        self.transform.transform.rotation.w = quat[3]


        self.timer = self.create_timer(0.05, self.broadcast_tf)

    def broadcast_tf(self):
        self.transform.header.stamp = self.get_clock().now().to_msg()
        self.br.sendTransform(self.transform)

def main():
    rclpy.init()
    node = LogitechConstantTransformPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
