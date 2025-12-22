# ROS Libraries
import rclpy
from geometry_msgs.msg import PointStamped 
from .state_machine import UR7e_StateMachine

class UR7e_BallGraspAndLaunch(UR7e_StateMachine):
    def __init__(self):
        super().__init__('ball_grasp')

        self.ball_pub = self.create_subscription(PointStamped, '/ball_pose_base', self.ball_callback, 1) 

        self.ball_pose = None
        self.current_plan = None
        self.joint_state = None
        self.ball_loaded = False
        self.trajectory_planner._warmup(50)


    def ball_callback(self, ball_pose):
        if self.ball_pose is not None:
            return

        if self.joint_state is None:
            self.get_logger().info("No joint state yet, cannot proceed")
            return

        self.ball_pose = ball_pose
        

        # 1) Move to Pre-Grasp Position (gripper above the ball)
        self.job_queue.append('open_grip')
        pre_grasp_state = self.ik_planner.compute_ik(self.joint_state, ball_pose.point.x, ball_pose.point.y, ball_pose.point.z + 0.5)
        self.job_queue.append(pre_grasp_state)

        # 2) Move to Grasp Position (lower the gripper to the ball)
        # theoretical max z offset is 6 cm but that's dangerous
        # need to get the gripper a cm or lower during grab
        grasp_state = self.ik_planner.compute_ik(pre_grasp_state, ball_pose.point.x, ball_pose.point.y, ball_pose.point.z + 0.16)
        self.job_queue.append(grasp_state)
        self.job_queue.append('close_grip')

        # 4) Move back to Pre-Grasp Position
        self.launch_state = self.joint_state # TODO: Different state?
        self.job_queue.append(self.launch_state)
        
        self.execute_jobs()
        self.ball_loaded = True


    def target_callback(self, target_pose):
        if not self.ball_loaded: # Make sure ball is loaded and ready to launch
            return

        # 5) Launch Ball
        self.job_queue.append(1.0)
        throwing_trajectory, t_release = self.ik_planner.plan_to_target(self.joint_state, target_pose, 1.5)
        self.job_queue.append((throwing_trajectory, t_release))

        # 6) Reset to launch position
        self.job_queue.append(0.2)   
        self.job_queue.append(self.launch_state)
        
        self.execute_jobs()
        self.reset()

    def reset(self):
        self.ball_pose = None
        self.target_pose = None
        self.ball_loaded = False


def main(args=None):
    rclpy.init(args=args)
    node = UR7e_BallGraspAndLaunch()
    rclpy.spin(node)
    node.destroy_node()

if __name__ == '__main__':
    main()
