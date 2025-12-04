import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import PointStamped 

from scipy.spatial.transform import Rotation

import numpy as np

class TransformBallPose(Node):
    def __init__(self):
        super().__init__('transform_ball_pose')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.ball_pose_sub = self.create_subscription(
            PointStamped,
            '/ball_pose',
            self.ball_pose_callback,
            10
        )

        self.ball_pose_pub = self.create_publisher(PointStamped, '/ball_pose_base', 1)

        rclpy.spin_once(self, timeout_sec=2)
        self.ball_pose = None

    def ball_pose_callback(self, msg: PointStamped):
        if self.ball_pose is None:
            self.ball_pose = self.transform_ball_pose(msg)
        else:
            self.balll_pose_pub.publish(self.ball_pose)


    def transform_ball_pose(self, msg: PointStamped):
        """ 
        Transform point into base_link frame
        Args: 
            - msg: PointStamped - The message from /ball_pose, of the position of the ball in camera_depth_optical_frame
        Returns:
            Point: point in base_link_frame in form [x, y, z]
        """
        try:
            trans = self.tf_buffer.lookup_transform(
                'base_link', 
                'camera_depth_optical_frame', #TODO: change based on camera 
                rclpy.time.Time()
            )

            translation = trans.transform.translation
            quat = trans.transform.rotation

            quat = np.array([quat.x, quat.y, quat.z, quat.w])
            R = Rotation.from_quat(quat).as_matrix()

            g = np.eye(4)
            g[0:3, 0:3] = R
            g[:3, 3] = np.array([translation.x, translation.y, translation.z])

            ball_hom = np.array([msg.point.x, msg.point.y, msg.point.z, 1])
            ball_base = g @ ball_hom.T

            ball_pose = PointStamped()            
            # Fill in message
            ball_pose.header.stamp = self.get_clock().now().to_msg()
            ball_pose.header.frame_id = "base_link"
            
            ball_pose.point.x, ball_pose.point.y, ball_pose.point.z, _ = [float(x_i) for x_i in ball_base]

            return ball_pose
        except:
            return None
        
def main(args=None):
    rclpy.init(args=args)
    node = TransformBallPose()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
