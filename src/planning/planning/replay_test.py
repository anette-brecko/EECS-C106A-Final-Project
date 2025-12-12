import rclpy
from sensor_msgs.msg import JointState
from rclpy.utilities import remove_ros_args
from .trajectory_planner import UR7e_TrajectoryPlanner
import sys

class UR7e_ReplayTestLaunch(UR7e_TrajectoryPlanner):
    def __init__(self):
        super().__init__('replay_test', 50)
        clean_args = remove_ros_args(args=sys.argv)
        self.traj_save_filename = clean_args[1]
        self.ik_planner.speed = float(clean_args[2])



    def joint_state_callback(self, msg: JointState):
        if self.joint_state is not None:
            #self.get_logger().info("Already moved")
            return

        self.get_logger().info("Getting ready!")

        self.joint_state = msg

        self.job_queue.append(0.2)

        self.get_logger().info("Loading trajectory")     
        throwing_trajectory, t_release, start_cfg = self.ik_planner.play_loaded_trajectory(self.traj_save_filename)

        self.job_queue.append('toggle_grip')
        self.job_queue.append(start_cfg)
        
        # Launch Ball
        self.job_queue.append(1.0)      
        self.job_queue.append((throwing_trajectory, t_release))
        self.job_queue.append(0.2)   

        # Release the gripper
        self.job_queue.append('toggle_grip')

        # Reset
        self.job_queue.append(start_cfg)

        self.execute_jobs()
        self.reset()

    def reset(self):
        self.ball_pose = None
        self.target_pose = None
        self.ball_loaded = False



def main(args=None):
    rclpy.init(args=args)
    node = UR7e_ReplayTestLaunch()
    rclpy.spin(node)
    node.destroy_node()

if __name__ == '__main__':
    main()
