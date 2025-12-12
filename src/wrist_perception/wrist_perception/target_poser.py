import cv2
from cv_bridge import CvBridge
import numpy as np
from mmpose.apis import init_model, inference_topdown
from rtmpose3d import RTMPose3D
#from mmpose.datasets import Dataset
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Header

class VideoPoseEstimator(Node):

    def __init__(self):
        super().__init__('video_pose_estimator')

        self.last_inference_time = 0.0
        self.inference_interval = 0.33  # 5 FPS

        self.bridge = CvBridge()

        self.ready_count = 0

        # Kinect camera paramters
        self.fx = 388.198
        self.fy = 389.033

        # Subscribe to the RGB image topic
        self.create_subscription(Image, '/kinect2/rgbd/depth_rect', self.depth_callback, 10)
        self.create_subscription(Image, '/kinect2/rgbd/rgb_rect', self.image_callback, 10)

        # Initialize ROS2 publisher for wrist positions
        self.target_point_publisher = self.create_publisher(PointStamped, '/target_point', 10)

        # Load the pose estimation model
        self.model = RTMPose3D.from_pretrained('rbarac/rtmpose3d', device='cuda:0')

    def image_callback(self, msg):
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self.last_inference_time < self.inference_interval:
            return
        self.last_inference_time = now

        try:
            # Convert the ROS image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgra8')
            self.rgb_image = cv_image[:, :, :3]
            # X center point = 212.0 
            centerX = self.rgb_image.shape[0] / 2
            # Y center point = 256.0
            centerY = self.rgb_image.shape[1] / 2
            # print(self.centerX, self.centerY)

        except Exception as e:
            self.get_logger().error(f'Failed to convert RGB image: {e}')
            return

        # Run inference on the current frame
        results = self.model(self.rgb_image, return_tensors='np')

        if not results:
            return

        # Extract ring finger positions from pose results
        keypoints_2d = results['keypoints_2d']
        left_index = 129
        right_index = 108
        try:
            left_wrist_position = keypoints_2d[0][left_index]
            right_wrist_position = keypoints_2d[0][right_index]

            if (results['scores'][0][left_index] > 0.7 and results['scores'][0][right_index] > 0.7):
                left_depth = self.depth_image[int(left_wrist_position[1]), int(left_wrist_position[0])]
                left_depth_measurement = (left_depth - 109) / 21.7
                right_depth = self.depth_image[int(right_wrist_position[1]), int(right_wrist_position[0])]
                right_depth_measurement = (right_depth - 109) / 21.7
                left_x = ((left_wrist_position[0] - centerX) * left_depth_measurement) / self.fx
                left_y = ((left_wrist_position[1]- centerY) * left_depth_measurement) / self.fy
                right_x = ((right_wrist_position[0]- centerX) * right_depth_measurement) / self.fx
                right_y = ((right_wrist_position[1]- centerY) * right_depth_measurement) / self.fy
                #print("left: ", round(left_x, 2), round(left_y, 2), round(left_depth_measurement, 2), 
                #"right: ", round(right_x, 2), round(right_y, 2), round(right_depth_measurement, 2))
                dist = np.sqrt((left_x - right_x) ** 2 + (left_y - right_y) ** 2 + (left_depth_measurement - right_depth_measurement) ** 2)
                if dist < 5:
                    # If Euclidean distance is less than 2.5 inches, convert to meters and send to publisher
                    target_x = -((left_x + right_x) / 2) / 39.37
                    target_y = ((left_y + right_y) / 2) / 39.37
                    target_z = ((left_depth_measurement + right_depth_measurement) / 2) / 39.37
                    target_point = [target_x, target_y, target_z]
                    print("Ready to toss! #", (self.ready_count + 1))
                    self.ready_count += 1
                    self.publish_points(target_point)
        except IndexError:
            print("No person detected")

    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, '16UC1')

    def publish_points(self, target_pos):
        # Publish target positions
        target_point_msg = PointStamped()
        target_point_msg.header = Header()
        target_point_msg.header.stamp = self.get_clock().now().to_msg()
        target_point_msg.header.frame_id = "kinect"
        target_point_msg.point.x, target_point_msg.point.y, target_point_msg.point.z = target_pos
        self.target_point_publisher.publish(target_point_msg)

def main(args=None):
    rclpy.init(args=args)
    node = VideoPoseEstimator()
    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
