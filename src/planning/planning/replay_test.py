import rclpy
from sensor_msgs.msg import JointState
from rclpy.utilities import remove_ros_args
from .state_machine import UR7e_StateMachine
import sys

class UR7e_DemoLaunch(UR7e_StateMachine):
    def __init__(self):
        super().__init__('replay_test')
        clean_args = remove_ros_args(args=sys.argv)
        self.traj_save_filename = clean_args[1]
        self.full_speed = 1.0
        self.trajectory_planner.speed = float(clean_args[2])

    def joint_state_callback(self, msg: JointState):
        if self.joint_state is not None:
            return

        self.get_logger().info("Getting ready!")

        self.joint_state = msg

        self.get_logger().info("Loading trajectory")     
        throwing_trajectory, t_release, start_cfg = self.trajectory_planner.play_loaded_trajectory(self.traj_save_filename)

        y = .074
        pre_grasp_state = self.ik_planner.compute_ik(self.joint_state, 0.0, y,  0.1)
        self.job_queue.append(pre_grasp_state)

        # 2) Move to Grasp Position (lower the gripper to the ball)
        # theoretical max z offset is 6 cm but that's dangerous
        # need to get the gripper a cm or lower during grab
        grasp_state = self.ik_planner.compute_ik(pre_grasp_state, 0.0, y, - 0.032)
        self.job_queue.append(grasp_state)

        self.job_queue.append('toggle_grip')
        self.job_queue.append(start_cfg)
        
        # Launch Ball
        self.job_queue.append((throwing_trajectory, t_release))

        # Reset
        self.job_queue.append(start_cfg)

        self.execute_jobs()


def main(args=None):
    rclpy.init(args=args)
    node = UR7e_DemoLaunch()
    rclpy.spin(node)
    node.destroy_node()

if __name__ == '__main__':
    main()
