import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import PointStamped 

from scipy.spatial.transform import Rotation

import numpy as np

class TransformWristPose(Node):
    def __init__(self):
        super().__init__('transform_wrist_pose')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.wrist_pose_sub = self.create_subscription(
            PointStamped,
            '/target_point',
            self.wrist_pose_callback,
            10
        )

        self.wrist_pose_pub = self.create_publisher(PointStamped, '/wrist_pose', 1)

        rclpy.spin_once(self, timeout_sec=2)
        self.wrist_pose = None

    def wrist_pose_callback(self, msg: PointStamped):
        if self.wrist_pose is None:
            self.wrist_pose = self.transform_wrist_pose(msg)
        else:
            self.wristl_pose_pub.publish(self.wrist_pose)


    def transform_wrist_pose(self, msg: PointStamped):
        """ 
        Transform point into base_link frame
        Args: 
            - msg: PointStamped - The message from /wrist_pose, of the position of the wrist in camera_depth_optical_frame
        Returns:
            Point: point in base_link_frame in form [x, y, z]
        """
        try:
            trans = self.tf_buffer.lookup_transform(
                'base_link', 
                'camera1',  
                rclpy.time.Time()
            )

            translation = trans.transform.translation
            quat = trans.transform.rotation

            quat = np.array([quat.x, quat.y, quat.z, quat.w])
            R = Rotation.from_quat(quat).as_matrix()

            g = np.eye(4)
            g[0:3, 0:3] = R
            g[:3, 3] = np.array([translation.x, translation.y, translation.z])

            wrist_hom = np.array([msg.point.x, msg.point.y, msg.point.z, 1])
            wrist_base = g @ wrist_hom.T

            wrist_pose = PointStamped()            
            # Fill in message
            wrist_pose.header.stamp = self.get_clock().now().to_msg()
            wrist_pose.header.frame_id = "base_link"
            
            wrist_pose.point.x, wrist_pose.point.y, wrist_pose.point.z, _ = [float(x_i) for x_i in wrist_base]

            return wrist_pose
        except:
            return None
        
def main(args=None):
    rclpy.init(args=args)
    node = TransformWristPose()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
