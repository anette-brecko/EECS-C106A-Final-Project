import rclpy
from sensor_msgs.msg import JointState
from rclpy.utilities import remove_ros_args
from .state_machine import UR7e_StateMachine
import sys

class UR7e_ReplayTestLaunch(UR7e_StateMachine):
    def __init__(self):
        super().__init__('replay_test')
        clean_args = remove_ros_args(args=sys.argv)
        self.traj_save_filename = clean_args[1]
        self.full_speed = float(clean_args[2])

    def joint_state_callback(self, msg: JointState):
        if self.joint_state is not None:
            return

        self.get_logger().info("Getting ready!")

        self.joint_state = msg

        self.get_logger().info("Loading trajectory")     
        throwing_trajectory, t_release, start_cfg = self.trajectory_planner.play_loaded_trajectory(self.traj_save_filename)

        self.job_queue.append('close_grip')
        self.job_queue.append(start_cfg)
        
        # Launch Ball
        self.job_queue.append((throwing_trajectory, t_release))

        # Reset
        self.job_queue.append(start_cfg)

        self.execute_jobs()


def main(args=None):
    rclpy.init(args=args)
    node = UR7e_ReplayTestLaunch()
    rclpy.spin(node)
    node.destroy_node()

if __name__ == '__main__':
    main()
