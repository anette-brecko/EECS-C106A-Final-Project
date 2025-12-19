#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
# Replace 'planning.srv' with your actual package name if different
from gripper_msgs.srv import SetInteger 
import sys
import tty
import termios
import threading
import select

class GripperForceTeleop(Node):
    def __init__(self):
        super().__init__('gripper_force_teleop')
        
        # --- Settings ---
        self.current_force = 50   # Start with a low, safe force
        self.force_step = 10      # How much to change per keypress
        
        # --- Service Client ---
        self.cli = self.create_client(SetInteger, '/set_gripper')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /set_gripper service...')
        
        self.get_logger().info("Connected to gripper service.")

        # --- Threading for Keyboard ---
        self.running = True
        self.input_thread = threading.Thread(target=self.keyboard_loop, daemon=True)
        self.input_thread.start()

    def keyboard_loop(self):
        """Monitors keyboard input in a non-blocking way."""
        self.print_usage()
        
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        
        try:
            tty.setraw(sys.stdin.fileno())
            
            while self.running:
                # Check for input with a timeout so we can exit cleanly
                if sys.stdin in select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)
                    if key == '\x03': # Ctrl+C
                        self.running = False
                        break
                    self.handle_key(key)
                    
        except Exception as e:
            print(f"\nError in input loop: {e}")
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            # Signal the ROS node to stop
            rclpy.shutdown()

    def handle_key(self, key):
        """Process keypresses to update force or send commands."""
        updated = False

        if key == 'a':
            self.current_force = max(0, self.current_force - 1)
            updated = True
        elif key == 'd':
            self.current_force = min(255, self.current_force + 1)
            updated = True

        
        if key == 'w': # Increase Force
            self.current_force = min(255, self.current_force + self.force_step)
            updated = True
        
        elif key == 's': # Decrease Force
            self.current_force = max(0, self.current_force - self.force_step)
            updated = True
            
        elif key == ' ': # Spacebar to Execute
            self.send_force_command()
            return # Don't reprint the status bar, the log will show result

        if updated:
            self.print_status()

    def send_force_command(self):
        """Calls the service with the currently selected force."""
        print(f"\r\nSending Trigger with Force: {self.current_force}...", end='')
        
        req = SetInteger.Request()
        req.data = int(self.current_force)
        
        future = self.cli.call_async(req)
        future.add_done_callback(self.service_response_callback)

    def service_response_callback(self, future):
        """Async callback when the gripper finishes toggling."""
        try:
            response = future.result()
            print(f"\r\n[RESULT] Success: {response.success} | Msg: {response.message}")
        except Exception as e:
            print(f"\r\n[ERROR] Service call failed: {e}")
        
        # Reprint the status bar so the user knows they can continue
        self.print_status()

    def print_usage(self):
        print("\n" + "="*40)
        print("  GRIPPER FORCE TELEOP")
        print("="*40)
        print("  w      : Increase Force (+10)")
        print("  s      : Decrease Force (-10)")
        print("  SPACE  : EXECUTE (Toggle Gripper)")
        print("  Ctrl+C : Quit")
        print("="*40)
        self.print_status()

    def print_status(self):
        """Visual bar for force 0-255"""
        # Create a visual bar [###.......]
        bar_len = 20
        filled = int((self.current_force / 255.0) * bar_len)
        bar = '#' * filled + '.' * (bar_len - filled)
        
        # \r overwrites the current line
        sys.stdout.write(f"\rTarget Force: [{bar}] {self.current_force:3d} / 255")
        sys.stdout.flush()

def main(args=None):
    rclpy.init(args=args)
    node = GripperForceTeleop()
    
    try:
        # Just spin efficiently, the threaded loop handles the logic
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except ExternalShutdownException:
        pass
    finally:
        node.destroy_node()
        # rclpy.shutdown is handled in the thread or exception

if __name__ == "__main__":
    main()