# ROS Libraries
from std_srvs.srv import Trigger
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PointStamped 
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
from scipy.spatial.transform import Rotation as R
import numpy as np

from planning.ik import IKPlanner

class UR7e_BallGraspAndLaunch(Node):
    def __init__(self):
        super().__init__('ball_grasp')

        self.ball_pub = self.create_subscription(PointStamped, '/ball_pose_base', self.ball_callback, 1) # TODO: CHECK IF TOPIC ALIGNS WITH YOURS
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 1)

        self.exec_ac = ActionClient(
            self, FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )

        self.gripper_cli = self.create_client(Trigger, '/toggle_gripper')

        self.ball_pose = None
        self.current_plan = None
        self.joint_state = None
        self.ball_loaded = False

        self.ik_planner = IKPlanner()

        self.job_queue = [] # Entries should be of type either JointState, RobotTrajectory, or String('toggle_grip')

    def joint_state_callback(self, msg: JointState):
        self.joint_state = msg

    def ball_callback(self, ball_pose):
        if self.ball_pose is not None:
            return

        if self.joint_state is None:
            self.get_logger().info("No joint state yet, cannot proceed")
            return

        # 1) Move to Pre-Grasp Position (gripper above the ball)
        # TODO: Ball offsets!!!
        pre_grasp_state = self.ik_planner.compute_ik(self.joint_state, ball_pose.point.x, ball_pose.point.y - 0.035, ball_pose.point.z + 0.185)
        self.job_queue.append(pre_grasp_state)

        # 2) Move to Grasp Position (lower the gripper to the ball)
        # TODO: Ball offsets!!!
        grasp_state = self.ik_planner.compute_ik(pre_grasp_state, ball_pose.point.x, ball_pose.point.y - 0.035, ball_pose.point.z + 0.158)
        self.job_queue.append(grasp_state)

        # 3) Close the gripper. See job_queue entries defined in init above for how to add this action.
        self.job_queue.append('toggle_grip')
        
        # 4) Move back to Pre-Grasp Position
        self.launch_state = pre_grasp_state
        self.job_queue.append(self.launch_state)
        self.execute_jobs()
        self.ball_loaded = True


    def target_callback(self, target_pose):
        if self.job_queue or not self.ball_loaded: # Make sure ball is loaded and ready to launch
            return

        if self.joint_state is None:
            self.get_logger().info("No joint state yet, cannot proceed")
            return

        # 5) Launch Ball
        throwing_trajectory, t_release = self.ik_planner.plan_to_target(self.launch_state, target_pose, 50, 1.5)
        self.job_queue.append((throwing_trajectory, t_release))

        # 6) Release the gripper
        self.job_queue.append('toggle_grip')

        self.job_queue.append(self.launch_state)
        self.execute_jobs()
        self.reset()

    def reset(self):
        self.ball_pose = None
        self.target_pose = None
        self.ball_loaded = False

    def execute_jobs(self):
        if not self.job_queue:
            self.get_logger().info("All jobs completed.")
            #rclpy.shutdown()
            return

        self.get_logger().info(f"Executing job queue, {len(self.job_queue)} jobs remaining.")
        next_job = self.job_queue.pop(0)

        if isinstance(next_job, JointState):
            traj = self.ik_planner.plan_to_joints(next_job)
            if traj is None:
                self.get_logger().error("Failed to plan to position")
                return

            self.get_logger().info("Planned to position")

            self._execute_joint_trajectory(traj.joint_trajectory)
        elif isinstance(next_job, tuple):
            self.get_logger().info("Planned to launch")
            self._execute_joint_trajectory(next_job[0], release_time=next_job[1])
        elif next_job == 'toggle_grip':
            self.get_logger().info("Toggling gripper")
            self._toggle_gripper()
        else:
            self.get_logger().error("Unknown job type.")
            self.execute_jobs()  # Proceed to next job

    def _toggle_gripper(self):
        if not self.gripper_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Gripper service not available')
            rclpy.shutdown()
            return

        req = Trigger.Request()
        future = self.gripper_cli.call_async(req)
        # wait for 2 seconds
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

        self.get_logger().info('Gripper toggled.')
        self.execute_jobs()  # Proceed to next job

            
    def _execute_joint_trajectory(self, joint_traj, release_time=None):
        self.get_logger().info('Waiting for controller action server...')
        self.exec_ac.wait_for_server()

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = joint_traj

        self.get_logger().info('Sending trajectory to controller...')

        self._current_release_time = release_time
        self._release_triggered = False

        send_future = self.exec_ac.send_goal_async(goal, self._feedback_callback)
        print(send_future)
        send_future.add_done_callback(self._on_goal_sent)

    def _feedback_callback(self, feedback_msg):
        if self._release_triggered or self._current_release_time is None:
            return
        
        current_time = feedback_msg.feedback.actual.time_from_start.sec + \
                       feedback_msg.feedback.actual.time_from_start.nanosec * 1e-9

        # If we passed the release time, FIRE!
        if current_time >= self._current_release_time:
            self.get_logger().info(f"RELEASE TRIGGERED at {current_time:.3f}s (Target: {self._current_release_time:.3f}s)")
            
            # Open gripper
            req = Trigger.Request()
            self.gripper_cli.call_async(req)

            self._release_triggered = True

    def _on_goal_sent(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('bonk')
            rclpy.shutdown()
            return

        self.get_logger().info('Executing...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_exec_done)

    def _on_exec_done(self, future):
        try:
            result = future.result().result
            self.get_logger().info('Execution complete.')
            self.execute_jobs()  # Proceed to next job
        except Exception as e:
            self.get_logger().error(f'Execution failed: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = UR7e_BallGraspAndLaunch()
    rclpy.spin(node)
    node.destroy_node()

if __name__ == '__main__':
    main()
