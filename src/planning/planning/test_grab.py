# ROS Libraries
from std_srvs.srv import Trigger
import rclpy
from geometry_msgs.msg import PointStamped 
from .trajectory_planner import UR7e_TrajectoryPlanner

class UR7e_BallGrasp(UR7e_TrajectoryPlanner):
    def __init__(self):
        super().__init__('ball_grasp')

        self.ball_pub = self.create_subscription(PointStamped, '/ball_pose_base', self.ball_callback, 1) # TODO: CHECK IF TOPIC ALIGNS WITH YOURS
        
        self.ball_pose = None
        self.joint_state = None


    def ball_callback(self, ball_pose):
        if self.ball_pose is not None:
            return

        if self.joint_state is None:
            self.get_logger().info("No joint state yet, cannot proceed")
            return
        
        self.ball_pose = ball_pose

        # 1) Move to Pre-Grasp Position (gripper above the ball)
        # TODO: Ball offsets!!!
        pre_grasp_state = self.ik_planner.compute_ik(self.joint_state, ball_pose.point.x, ball_pose.point.y - 0.0, ball_pose.point.z + 0)
        self.job_queue.append(pre_grasp_state)

        # 2) Move to Grasp Position (lower the gripper to the ball)
        # TODO: Ball offsets!!!
        # theoretical max z offset is 6 cm but that's dangerous
        grasp_state = self.ik_planner.compute_ik(pre_grasp_state, ball_pose.point.x, ball_pose.point.y - 0.0, ball_pose.point.z - 0.03)
        self.job_queue.append(grasp_state)

        # 3) Close the gripper. See job_queue entries defined in init above for how to add this action.
        self.job_queue.append('toggle_grip')
        
        # 4) Move back to Pre-Grasp Position
        self.launch_state = pre_grasp_state
        self.job_queue.append(self.launch_state)
        self.execute_jobs()


def main(args=None):
    rclpy.init(args=args)
    node = UR7e_BallGrasp()
    rclpy.spin(node)
    node.destroy_node()

if __name__ == '__main__':
    main()
