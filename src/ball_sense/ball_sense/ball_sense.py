#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
import os
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
import cv2
import numpy as np
from ament_index_python.packages import get_package_share_directory
from scipy import ndimage


class HSVFilterNode(Node):
    def __init__(self):
        super().__init__("hsv_filter_node")

        # Low light: H = 138.9 S = 90 V = 8
        # Better light: H = 167.4 S = 41 V = 36

        # Adjusted camera specs: H = 161.5, S = 21, V = 24
        #                        H = 170,   S = 54, V = 47
        # Brightness = 150, Contrast = 160, Saturation = 175, Gain = Max? 255?

        # Declare HSV threshold parameters
        # self.declare_parameter("lower_h", 138.9)
        # self.declare_parameter("lower_s", 90)
        # self.declare_parameter("lower_v", 8)
        # self.declare_parameter("upper_h", 167.4)
        # self.declare_parameter("upper_s", 41)
        # self.declare_parameter("upper_v", 36)
        # mask1 = cv2.inRange(hsv, (30,16,37), (80,147,96))
        self.declare_parameter("lower_h", 30)
        self.declare_parameter("lower_s", 16)
        self.declare_parameter("lower_v", 37)
        self.declare_parameter("upper_h", 80)
        self.declare_parameter("upper_s", 147)
        self.declare_parameter("upper_v", 96)
        self.surface_area = 0.001551791655

        # Subscriber
        self.subscription = self.create_subscription(Image, "/camera1/image_raw", self.image_callback, 10)
        self.camera_info_sub = self.create_subscription(CameraInfo, '/camera1/camera_info', self.camera_info_callback, 1)

        # Publishers
        self.mask_pub = self.create_publisher(Image, "hsv_mask", 10)
        self.filtered_pub = self.create_publisher(Image, "hsv_filtered", 10)
        self.ball_position_pub = self.create_publisher(PointStamped, '/ball_pose', 1)

        self.camera_intrinsics = None

        self.bridge = CvBridge()
        self.get_logger().info("HSV Filter Node started!")

    def camera_info_callback(self, msg):
        if self.camera_intrinsics is None:
            self.get_logger().info("Recieved Camera Info")
            fx = msg.k[0]
            fy = msg.k[4]
            # cx = msg.k[2]
            # cy = msg.k[5]
            cx = 1280 / 2
            cy = 720 / 2
            self.camera_intrinsics = [fx, fy, cx, cy]

    def image_callback(self, msg):
        if self.camera_intrinsics is None:
            self.get_logger().info('Camera is none.')
            return

        # Convert ROS2 image → OpenCV BGR image
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Read HSV bounds
        lower = np.array([
            self.get_parameter("lower_h").value,
            self.get_parameter("lower_s").value,
            self.get_parameter("lower_v").value,
        ])
        upper = np.array([
            self.get_parameter("upper_h").value,
            self.get_parameter("upper_s").value,
            self.get_parameter("upper_v").value,
        ])

        # Create mask
        mask = cv2.inRange(hsv, lower, upper)

        kernel = np.ones((7,7),np.uint8)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Apply mask to original BGR image
        filtered = cv2.bitwise_and(frame, frame, mask=mask)

        # Convert back to ROS2 Image messages
        mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding="mono8")
        filtered_msg = self.bridge.cv2_to_imgmsg(filtered, encoding="bgr8")

        # Publish
        self.mask_pub.publish(mask_msg)
        self.filtered_pub.publish(filtered_msg)

        pixel_count = np.count_nonzero(mask)
        numerator = self.camera_intrinsics[0] * self.camera_intrinsics[1] * self.surface_area
        depth = 0
            
        if pixel_count > 75:
            depth = np.sqrt(numerator / pixel_count)

            # self.get_logger().info(f'Ball {i+1}: depth={depth:.3f}m')

            # TODO: Get u, and v of ball in image coordinates
            non_zero_mask_x, non_zero_mask_y  = np.nonzero(mask)
            # u = np.mean(non_zero_mask_x)
            # v = np.mean(non_zero_mask_y)
            u, v = ndimage.center_of_mass(mask)

            # TODO: Find X , Y , Z of ball
            # switched u and v
            # if using line 121: x=v, y=u
            X = ((v - self.camera_intrinsics[2]) * depth) / self.camera_intrinsics[0]
            Y = ((u - self.camera_intrinsics[3]) * depth) / self.camera_intrinsics[1]
            Z = depth

            # Here we use Nathan's hacky desmos LSRL solution:
            correction = (0.14645 * Z + 0.63488) * .01
            X += correction

            point_cam = PointStamped()
            point_cam.header.stamp = msg.header.stamp
            point_cam.header.frame_id = 'camera1'
            point_cam.point.x = X
            point_cam.point.y = Y
            point_cam.point.z = Z
            #self.get_logger().info(f'Ball at: {X}, {Y}, {Z}')
            print(f'Ball at: {X}, {Y}, {Z}')
            self.ball_position_pub.publish(point_cam)
        else:
            self.get_logger().info('No balls spotted')



def main(args=None):
    rclpy.init(args=args)
    node = HSVFilterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()