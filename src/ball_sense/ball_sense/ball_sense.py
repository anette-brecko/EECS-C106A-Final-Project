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
        self.declare_parameter("lower_h", 30)
        self.declare_parameter("lower_s", 16)
        self.declare_parameter("lower_v", 37)
        self.declare_parameter("upper_h", 80)
        self.declare_parameter("upper_s", 147)
        self.declare_parameter("upper_v", 96)
        self.surface_area = 0.001551791655
        self.BALL_RADIUS = 0.022225

        # Subscriber
        self.subscription = self.create_subscription(Image, "/camera1/image_raw", self.image_callback, 10)
        self.camera_info_sub = self.create_subscription(CameraInfo, '/camera1/camera_info', self.camera_info_callback, 1)

        # Publishers
        self.mask_pub = self.create_publisher(Image, "hsv_mask", 10)
        self.filtered_pub = self.create_publisher(Image, "hsv_filtered", 10)
        self.contour_pub = self.create_publisher(Image, "circ_contour", 10)
        self.ball_position_pub = self.create_publisher(PointStamped, '/ball_pose', 1)

        self.camera_intrinsics_received = False

        self.bridge = CvBridge()
        self.get_logger().info("HSV Filter Node started!")

    def camera_info_callback(self, msg):
        if not self.camera_intrinsics_received:
            self.get_logger().info("Recieved Camera Info")
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.get_logger().info(f'Camera Intrinsics: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}')
            self.camera_intrinsics_received = True
   
    def image_callback(self, msg):
        if self.camera_intrinsics_received is None:
            self.get_logger().info('Camera is none.')
            return

        # Convert ROS2 image → OpenCV BGR image
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # Blur to denoise and get smoother results
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)

        # Convert to HSV
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

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

        # Clean up mask
        kernel = np.ones((3,3),np.uint8)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)

        #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Apply mask to original BGR image
        filtered = cv2.bitwise_and(frame, frame, mask=mask)

        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            # Find the largest contour (assuming it's the sphere)
            c = max(contours, key=cv2.contourArea)
        
            # Fit a circle around the contour
            ((u, v), radius_pix) = cv2.minEnclosingCircle(c)
            
            # Only proceed if the object is big enough (filter noise)
            if radius_pix > 10:
                # Draw the circle and centroid on the frame
                cv2.circle(frame, (int(u), int(v)), int(radius_pix), (0, 255, 255), 2)
                cv2.circle(frame, (int(u), int(v)), 5, (0, 0, 255), -1)
          
                # Use geometric mean of focal lengths to get depth
                depth = np.sqrt(self.fx * self.fy) * self.BALL_RADIUS / radius_pix

                # Find X , Y , Z of ball
                X = ((u - self.cx) * depth) / self.fx
                Y = ((v - self.cy) * depth) / self.fy
                Z = depth

                point_cam = PointStamped()
                point_cam.header.stamp = msg.header.stamp
                point_cam.header.frame_id = 'camera1'
                point_cam.point.x = X
                point_cam.point.y = Y
                point_cam.point.z = Z
                
                self.get_logger().info(f'Ball at: {X}, {Y}, {Z}')
                self.ball_position_pub.publish(point_cam)
            else:
                self.get_logger().info('No balls spotted')

        # Convert back to ROS2 Image messages
        mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding="mono8")
        filtered_msg = self.bridge.cv2_to_imgmsg(filtered, encoding="bgr8")
        contour_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")

        # Publish
        self.mask_pub.publish(mask_msg)
        self.filtered_pub.publish(filtered_msg)
        self.contour_pub.publish(contour_msg)


        # Here we use Nathan's hacky desmos LSRL solution:
        #correction = (0.14645 * Z + 0.63488) * .01
       
def main(args=None):
    rclpy.init(args=args)
    node = HSVFilterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
