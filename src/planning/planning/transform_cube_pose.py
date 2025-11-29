import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import PointStamped 

from scipy.spatial.transform import Rotation

import numpy as np

class TransformCubePose(Node):
    def __init__(self):
        super().__init__('transform_cube_pose')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.cube_pose_sub = self.create_subscription(
            PointStamped,
            '/cube_pose',
            self.cube_pose_callback,
            10
        )

        self.cube_pose_pub = self.create_publisher(PointStamped, '/cube_pose_base', 1)

        rclpy.spin_once(self, timeout_sec=2)
        self.cube_pose = None

    def cube_pose_callback(self, msg: PointStamped):
        if self.cube_pose is None:
            self.cube_pose = self.transform_cube_pose(msg)
        else:
            self.cube_pose_pub.publish(self.cube_pose)


    def transform_cube_pose(self, msg: PointStamped):
        """ 
        Transform point into base_link frame
        Args: 
            - msg: PointStamped - The message from /cube_pose, of the position of the cube in camera_depth_optical_frame
        Returns:
            Point: point in base_link_frame in form [x, y, z]
        """
        try:
            trans = self.tf_buffer.lookup_transform('base_link', 'camera_depth_optical_frame', rclpy.time.Time())

            translation = trans.transform.translation
            quat = trans.transform.rotation

            quat = np.array([quat.x, quat.y, quat.z, quat.w])
            R = Rotation.from_quat(quat).as_matrix()

            g = np.eye(4)
            g[0:3, 0:3] = R
            g[:3, 3] = np.array([translation.x, translation.y, translation.z])

            cube_hom = np.array([msg.point.x, msg.point.y, msg.point.z, 1])
            cube_base = g @ cube_hom.T
            print(cube_base)

            cube_pose = PointStamped()            
            # Fill in message
            cube_pose.header.stamp = self.get_clock().now().to_msg()
            cube_pose.header.frame_id = "base_link"
            
            cube_pose.point.x, cube_pose.point.y, cube_pose.point.z = float(cube_base[0]), float(cube_base[1]), float(cube_base[2])

            return cube_pose
        except:
            return None
        
def main(args=None):
    rclpy.init(args=args)
    node = TransformCubePose()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
