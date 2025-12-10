import cv2
import numpy as np
from mmpose.apis import init_model, inference_topdown
#from mmpose.datasets import Dataset
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Header

class VideoPoseEstimator(Node):

    def __init__(self):
        super().__init__('video_pose_estimator')

        # Initialize ROS2 publisher for wrist positions
        self.publisher_left_wrist = self.create_publisher(PointStamped, '/left_wrist_position', 10)
        self.publisher_right_wrist = self.create_publisher(PointStamped, '/right_wrist_position', 10)

        # Initialize OpenCV for video capture
        self.cap = cv2.VideoCapture(0)  # Use 0 for webcam or specify a video file path

        # Load the pose estimation model
        config_file = 'demo/rtmdet_m_640-8xb32_coco-person.py'  # Modify with the actual config path
        checkpoint_file = 'rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth'  # Modify with the actual model path
        self.model = init_model(config_file, checkpoint_file, device='cuda')  # or 'cpu'

        if not self.cap.isOpened():
            self.get_logger().error('Failed to open video stream.')
            exit(1)

    def extract_wrist_positions(self, pose_results):
        """
        Extract wrist positions (left and right wrists) from pose results.
        Modify the indices depending on the model's output keypoints structure.
        """
        left_wrist_position = pose_results['keypoints'][11]  # Example index for left wrist
        right_wrist_position = pose_results['keypoints'][12]  # Example index for right wrist
        return left_wrist_position, right_wrist_position

    def process_video(self):
        while rclpy.ok():
            ret, frame = self.cap.read()
            if not ret:
                break

            # Preprocess the frame if needed (e.g., resize, normalization, etc.)
            # Assuming 'frame' is in BGR format, OpenCV default format
            # If needed, convert to RGB as expected by your model
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run inference on the current frame
            pose_results = inference_topdown(self.model, frame_rgb)

            # Extract wrist positions from pose results
            left_wrist_position, right_wrist_position = self.extract_wrist_positions(pose_results)

            # Create PointStamped messages for left and right wrists
            left_wrist_msg = PointStamped()
            left_wrist_msg.header = Header()
            left_wrist_msg.header.stamp = self.get_clock().now().to_msg()
            left_wrist_msg.header.frame_id = "world"
            left_wrist_msg.point.x, left_wrist_msg.point.y, left_wrist_msg.point.z = left_wrist_position

            right_wrist_msg = PointStamped()
            right_wrist_msg.header = Header()
            right_wrist_msg.header.stamp = self.get_clock().now().to_msg()
            right_wrist_msg.header.frame_id = "world"
            right_wrist_msg.point.x, right_wrist_msg.point.y, right_wrist_msg.point.z = right_wrist_position

            # Publish wrist positions
            self.publisher_left_wrist.publish(left_wrist_msg)
            self.publisher_right_wrist.publish(right_wrist_msg)

            # Display the frame with pose keypoints drawn (optional)
            cv2.imshow('Pose Estimation', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                break

        # Clean up
        self.cap.release()
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    video_pose_estimator = VideoPoseEstimator()
    video_pose_estimator.process_video()

    rclpy.spin(video_pose_estimator)

    video_pose_estimator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
