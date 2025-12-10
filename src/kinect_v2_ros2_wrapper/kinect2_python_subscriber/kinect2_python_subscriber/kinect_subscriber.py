import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class KinectSubscriber(Node):
    def __init__(self):
        super().__init__('kinect_subscriber')
        
        # Initialize CvBridge to convert ROS image messages to OpenCV images
        self.bridge = CvBridge()

        # Subscribe to the RGB + Depth topics
        self.create_subscription(Image, '/kinect2/rgb/raw', self.rgb_callback, 10)
        self.create_subscription(Image, '/kinect2/ir/raw', self.depth_callback, 10)

    def rgb_callback(self, msg):
        try:
            # Convert the ROS image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

            # Display the RGB image
            cv2.imshow("RGB Image", cv_image)
            cv2.waitKey(1)  # Add a small delay for OpenCV to update the window
            #print("do you like pina coladas 😛")
        except Exception as e:
            self.get_logger().error(f'Failed to convert RGB image: {e}')

    def depth_callback(self, msg):
        try:
            # Convert the depth image message to OpenCV format (16UC1 for depth images)
            depth_image = self.bridge.imgmsg_to_cv2(msg, '16UC1')
            # Display the depth image
            cv2.imshow("Depth Image", depth_image)
            cv2.waitKey(1)
            #print("and getting caught in the rain 🤨")
        except Exception as e:
            self.get_logger().error(f'Failed to convert depth image: {e}')

def main(args=None):
    rclpy.init(args=args)
    kinect_subscriber = KinectSubscriber()
    rclpy.spin(kinect_subscriber)
    kinect_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

