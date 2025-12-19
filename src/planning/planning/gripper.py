#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger  # Simple request/response: empty request, bool+string response
import socket
import os
import time

class GripperService(Node):

    """

    A ROS2 service that toggles the Robotiq gripper open/close via URCap socket.

    Uses std_srvs/Trigger — call once to toggle state.

    """

    def __init__(self):
        super().__init__('gripper_toggle_service')
        # Load robot IP from environment

        try:
            self.robot_ip = os.environ['EE106_UR_ROBOT_IP']

        except KeyError:
            self.get_logger().error("Environment variable EE106_UR_ROBOT_IP not set!")

            raise
        self.port = 63352
        self.gripper_closed = False  # Track gripper state
        # Create the service

        self.srv = self.create_service(Trigger, 'toggle_gripper', self.toggle_callback)
        self.reset_srv = self.create_service(Trigger, 'reset_gripper', self.reset_gripper_callback)
        self.get_logger().info("Gripper reset service ready: /reset_gripper")
        self.get_logger().info("Gripper toggle service ready: /toggle_gripper")
        # Initialize the gripper
        self.init_gripper()



    def send_cmd(self, cmd: str):
        """Send a command string to the gripper socket."""

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect((self.robot_ip, self.port))
                sock.sendall(cmd.encode('ascii'))

        except Exception as e:
            self.get_logger().error(f"Socket error: {e}")



    def init_gripper(self):
        """Initialize and activate the gripper."""
        self.get_logger().info("Initializing Robotiq gripper...")
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect((self.robot_ip, self.port))
                sock.sendall(b"SET ACT 1\n")
                sock.sendall(b"SET GTO 0\n")
                sock.sendall(b"SET SPE 255\n")
                sock.sendall(b"SET GTO 1\n")
                time.sleep(0.5)
                sock.setblocking(False)

                try:
                    output = sock.recv(4096).decode('ascii')
                    self.get_logger().debug(f"Init response: {output}")
                except BlockingIOError:
                    pass

        except Exception as e:
            self.get_logger().error(f"Failed to initialize gripper: {e}")



    def toggle_callback(self, request, response):
        """Callback for /toggle_gripper service."""

        target_pos = 255 if not self.gripper_closed else 0  # 255 = close, 0 = open
        cmd = f"SET POS {target_pos}\n"
        self.send_cmd(cmd)

        self.gripper_closed = not self.gripper_closed
        state_str = "closed" if self.gripper_closed else "opened"
        response.success = True
        response.message = f"Gripper {state_str}"
        self.get_logger().info(response.message)
        return response

    def reset_gripper_callback(self, request, response):
        """Reset the gripper to a known state (open)."""
        self.init_gripper()
        self.get_logger().info("Resetting gripper to open state...")
        self.send_cmd("SET POS 0\n")
        self.gripper_closed = False
        response.success = True
        response.message = "Gripper reset to open state"
        return response

def main(args=None):
    rclpy.init(args=args)
    node = GripperService()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()





if __name__ == '__main__':
    main()
