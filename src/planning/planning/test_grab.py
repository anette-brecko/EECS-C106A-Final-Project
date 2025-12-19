# ROS Libraries
import rclpy
from geometry_msgs.msg import PointStamped 
from .state_machine import UR7e_StateMachine

class UR7e_BallGrasp(UR7e_StateMachine):
    def __init__(self):
        super().__init__('ball_grasp')

        self.ball_pub = self.create_subscription(PointStamped, '/ball_pose_base', self.ball_callback, 1) 
        
        self.ball_pose = None
        self.joint_state = None


    def ball_callback(self, ball_pose):
        if self.ball_pose is not None:
            return

        if self.joint_state is None:
            self.get_logger().info("No joint state yet, cannot proceed")
            return
        
        self.ball_pose = ball_pose

        # # 1) Move to Pre-Grasp Position (gripper above the ball)
        # self.job_queue.append('open_grip')
        # pre_grasp_state = self.ik_planner.compute_ik(self.joint_state, ball_pose.point.x + 0.10, ball_pose.point.y - 0.05,  0.1)
        # self.job_queue.append(pre_grasp_state)

        # # 2) Move to Grasp Position (lower the gripper to the ball)
        # # theoretical max z offset is 6 cm but that's dangerous
        # # need to get the gripper a cm or lower during grab
        # grasp_state = self.ik_planner.compute_ik(pre_grasp_state, ball_pose.point.x + 0.10, ball_pose.point.y - 0.05, - 0.032)

    # 1) Move to Pre-Grasp Position (gripper above the ball)
<<<<<<< HEAD
<<<<<<< HEAD
        self.job_queue.append('open_grip')
        pre_grasp_state = self.ik_planner.compute_ik(self.joint_state, ball_pose.point.x, ball_pose.point.y, ball_pose.point.z + 0.5)

=======
        self.job_queue.append('toggle_grip')
        pre_grasp_state = self.ik_planner.compute_ik(self.joint_state, ball_pose.point.x, ball_pose.point.y,  0.1)
>>>>>>> parent of 06d6a5c (Update pyroki... Wack results.)
=======
        self.job_queue.append('toggle_grip')
        pre_grasp_state = self.ik_planner.compute_ik(self.joint_state, ball_pose.point.x, ball_pose.point.y,  0.1)
>>>>>>> parent of 06d6a5c (Update pyroki... Wack results.)
        self.job_queue.append(pre_grasp_state)

        # 2) Move to Grasp Position (lower the gripper to the ball)
        # theoretical max z offset is 6 cm but that's dangerous
        # need to get the gripper a cm or lower during grab
<<<<<<< HEAD
<<<<<<< HEAD
        grasp_state = self.ik_planner.compute_ik(pre_grasp_state, ball_pose.point.x, ball_pose.point.y, ball_pose.point.z + 0.165)
=======
        grasp_state = self.ik_planner.compute_ik(pre_grasp_state, ball_pose.point.x, ball_pose.point.y, - 0.032)
>>>>>>> parent of 06d6a5c (Update pyroki... Wack results.)
=======
        grasp_state = self.ik_planner.compute_ik(pre_grasp_state, ball_pose.point.x, ball_pose.point.y, - 0.032)
>>>>>>> parent of 06d6a5c (Update pyroki... Wack results.)
        self.job_queue.append(grasp_state)

        self.job_queue.append('toggle_grip')
        
        # 4) Move back to Pre-Grasp Position
        self.launch_state = pre_grasp_state
        self.job_queue.append(self.launch_state)
<<<<<<< HEAD
<<<<<<< HEAD
        self.job_queue.append('open_grip')
=======
>>>>>>> parent of 06d6a5c (Update pyroki... Wack results.)
=======
>>>>>>> parent of 06d6a5c (Update pyroki... Wack results.)
        self.execute_jobs()


def main(args=None):
    rclpy.init(args=args)
    node = UR7e_BallGrasp()
    rclpy.spin(node)
    node.destroy_node()

if __name__ == '__main__':
    main()
