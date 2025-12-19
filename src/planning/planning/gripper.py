#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from gripper_msgs.srv import SetInteger  # Simple request/response: empty request, bool+string response
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
        # Create the service

        self.srv = self.create_service(SetInteger, 'set_gripper', self.set_callback)
        self.get_logger().info("Gripper reset service ready: /set_gripper")
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
                sock.sendall(b"SET FOR 0\n")
                time.sleep(0.5)
                sock.setblocking(False)

                try:
                    output = sock.recv(4096).decode('ascii')
                    self.get_logger().debug(f"Init response: {output}")
                except BlockingIOError:
                    pass

        except Exception as e:
            self.get_logger().error(f"Failed to initialize gripper: {e}")

    def set_callback(self, request, response):
        """Callback for /toggle_gripper service."""
        cmd = f"SET POS {request.data}\n"
        self.send_cmd(cmd)
        

        response.success = True
        response.message = f"Gripper at {request.data}"
        self.get_logger().info(response.message)
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
