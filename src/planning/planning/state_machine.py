# ROS Libraries
from std_srvs.srv import Trigger
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
import numpy as np
from ur_msgs.srv import SetSpeedSliderFraction

from planning.ik import IKPlanner
from traj_gen.traj_planner import TrajectoryPlanner

class UR7e_StateMachine(Node):
    def __init__(self, name='traj_planner'):
        super().__init__(name)

        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 1)

        # Create a temporary client just for this call (or init it in __init__)
        self.speed_client = self.create_client(SetSpeedSliderFraction, '/io_and_status_controller/set_speed_slider')

        # Wait for service to be available (with timeout to avoid hanging forever)
        if not self.speed_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("Service /io_and_status_controller/set_speed_slider not available.")
            return

        self.exec_ac = ActionClient(
            self, FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )

        self.gripper_cli = self.create_client(Trigger, '/toggle_gripper')

        self.ik_planner = IKPlanner()
        self.trajectory_planner = TrajectoryPlanner()

        self.joint_state = None
<<<<<<< HEAD
        self.moving = False

        self.gripper_loose = 60
        self.gripper_tight = 80
        self.gripper_open = 0
        self.loose_delay = 0.1
=======
>>>>>>> parent of 06d6a5c (Update pyroki... Wack results.)

        self.job_queue = [] # Entries should be of type either JointState, RobotTrajectory, or String('toggle_grip')

    def joint_state_callback(self, msg: JointState):
        self.joint_state = msg

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
            self._set_speed_slider(0.2)
            self._execute_joint_trajectory(traj.joint_trajectory)
        elif isinstance(next_job, tuple):
            self.get_logger().info("Planned to launch")
            self._set_speed_slider(1.0)
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

        self._release_time = release_time

        send_future = self.exec_ac.send_goal_async(goal)
        send_future.add_done_callback(lambda future: self._on_goal_sent(future, release_time))

    def _on_goal_sent(self, future, release_time):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('bonk')
            rclpy.shutdown()
            return

        self.get_logger().info('Executing...')

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_exec_done)
    
        if release_time is not None:
            self._release_timer = self.create_timer(
                release_time / self.full_speed, # Scale to speed 
                self._timer_release_callback
            )

    def _timer_release_callback(self):
        self.get_logger().info("TIMER FIRED: Releasing Gripper!")
        self._toggle_gripper()
        
        # Destroy timer so it doesn't fire again
        self._release_timer.destroy()

    def _on_exec_done(self, future):
        try:
            result = future.result().result
            self.get_logger().info('Execution complete.')
            self.execute_jobs()  # Proceed to next job
        except Exception as e:
            self.get_logger().error(f'Execution failed: {e}')

    def _set_speed_slider(self, fraction: float):
        """
        Sets the speed slider fraction (0.0 to 1.0).
        """
        # Create request
        req = SetSpeedSliderFraction.Request()
        req.speed_slider_fraction = fraction

        # Call async
        future = self.speed_client.call_async(req)
        
        # Optional: Add a callback to check if it worked
        future.add_done_callback(self._on_speed_set_done)

    def _on_speed_set_done(self, future):
        try:
            response = future.result()
            self.get_logger().info("Speed slider updated successfully.")
        except Exception as e:
            self.get_logger().error(f"Service call failed HERE: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = UR7e_StateMachine()
    rclpy.spin(node)
    node.destroy_node()

if __name__ == '__main__':
    main()