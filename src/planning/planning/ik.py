import rclpy
from rclpy.node import Node
from moveit_msgs.srv import GetPositionIK, GetMotionPlan
from moveit_msgs.msg import PositionIKRequest, Constraints, JointConstraint, RobotTrajectory
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

import sys
import numpy as np
import pyroki as pk

from .trajectory_generation.generate_samples import solve_by_sampling
from .world import World
from .load_urdf import load_ur7e_with_gripper, UR7eJointVar
import jax_dataclasses as jdc
from .trajectory_generation.save_and_load import save_trajectory, load_trajectory

import jax.numpy as jnp


# Example usage:
# -------------------------------------------------
# current joint state (replace with your robot's)
# current_state = JointState()
# current_state.name = [
#     'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
#     'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
# ]
# current_state.position = [0.0, -1.57, 1.57, 0.0, 0.0, 0.0]

# Compute IK for target point
# ik_solution = node.compute_ik(current_state, 0.4, 0.1, 0.3)

# if ik_solution:
#     # Plan motion to the found joint configuration
#     trajectory = node.plan_to_joints(ik_solution)
#     if trajectory:
#         node.get_logger().info('Trajectory ready to execute.')


class IKPlanner(Node):
    def __init__(self):
        super().__init__('ik_planner')

        # ---- Clients ----
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        self.plan_client = self.create_client(GetMotionPlan, '/plan_kinematic_path')
        
        for srv, name in [(self.ik_client, 'compute_ik'),
                          (self.plan_client, 'plan_kinematic_path')]:
            while not srv.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f'Waiting for /{name} service...')

        # ----- PyRoki setup -----
        #urdf_path = "/opt/ros/humble/share/ur_description/urdf/ur.urdf.xacro"

        urdf = load_ur7e_with_gripper()    
        self.robot_coll = pk.collision.RobotCollision.from_urdf(urdf)

        # For UR5 it's important to initialize the robot in a safe configuration;
        default_cfg = np.array([4.712, -1.850, -1.425, -1.405, 1.593, -3.141])
        self.robot = pk.Robot.from_urdf(urdf, default_joint_cfg=default_cfg)
        UR7eJointVar.default_factory = staticmethod(lambda: jnp.array(default_cfg))

        self.robot = jdc.replace(self.robot, joint_var_cls=UR7eJointVar)
        self.target_link_name = "robotiq_hande_end"

        self.world = World(self.robot, urdf, self.target_link_name) 

        self.speed = 1.0


    # -----------------------------------------------------------
    # Compute IK for a given (x, y, z) + quat and current robot joint state
    # -----------------------------------------------------------
    def compute_ik(self, current_joint_state, x, y, z,
                   qx=0.0, qy=1.0, qz=0.0, qw=0.0): # Think about why the default quaternion is like this. Why is qy=1?
        pose = PoseStamped()
        pose.header.frame_id = 'base_link'
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        ik_req = GetPositionIK.Request()
        # Lookup the format for ik request and build ik_req by filling in necessary parameters. 
        ik_req.ik_request.pose_stamped = pose
        ik_req.ik_request.robot_state.joint_state = current_joint_state
        ik_req.ik_request.ik_link_name = 'wrist_3_link'
        ik_req.ik_request.avoid_collisions = True
        ik_req.ik_request.timeout = Duration(sec=2)
        ik_req.ik_request.group_name = 'ur_manipulator'
        

        future = self.ik_client.call_async(ik_req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is None:
            self.get_logger().error('IK service failed.')
            return None

        result = future.result()
        if result.error_code.val != result.error_code.SUCCESS:
            self.get_logger().error(f'IK failed, code: {result.error_code.val}')
            return None

        self.get_logger().info('IK solution found.')
        return result.solution.joint_state

    # -----------------------------------------------------------
    # Plan motion given a desired joint configuration
    # -----------------------------------------------------------
    def plan_to_joints(self, target_joint_state):
        req = GetMotionPlan.Request()
        req.motion_plan_request.group_name = 'ur_manipulator'
        req.motion_plan_request.allowed_planning_time = 5.0
        req.motion_plan_request.planner_id = "RRTConnectkConfigDefault"

        goal_constraints = Constraints()
        for name, pos in zip(target_joint_state.name, target_joint_state.position):
            goal_constraints.joint_constraints.append(
                JointConstraint(
                    joint_name=name,
                    position=pos,
                    tolerance_above=0.01,
                    tolerance_below=0.01,
                    weight=1.0
                )
            )

        req.motion_plan_request.goal_constraints.append(goal_constraints)
        future = self.plan_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is None:
            self.get_logger().error('Planning service failed.')
            return None

        result = future.result()
        if result.motion_plan_response.error_code.val != 1:
            self.get_logger().error('Planning failed.')
            return None

        self.get_logger().info('Motion plan computed successfully.')
        return result.motion_plan_response.trajectory

    def _solve_to_target(self, start_cfg, target_pos, timesteps, dt):
        """ Solve the trajectory problem """
        return solve_by_sampling(
            self.robot,
            self.robot_coll,
            self.world.gen_world_coll(),
            self.target_link_name,
            start_cfg,
            target_pos,
            timesteps,
            dt,
            7,
            0.85 * 0.8, 
            300,
            10,
        )
    
    def _trajectory_points_to_msg(self, start_cfg, traj_points, dt) -> JointTrajectory:
        """ Convert trajectory points to trajectory_msgs/JointTrajectory message. """
        
        # 2. Initialize the JointTrajectory (the core of the message)
        joint_traj = JointTrajectory()
        joint_traj.joint_names = [
                'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
            ]


        time_from_start = 0.0

        start_point = JointTrajectoryPoint()
        start_point.positions = start_cfg.tolist()
        start_point.velocities = [0.0] * len(start_cfg)
        start_point.accelerations = [0.0] * len(start_cfg)
        start_point.time_from_start.sec = 0
        start_point.time_from_start.nanosec = 0

        velocities = self._estimate_gradients(traj_points, dt) * self.speed

        
        # 3. Iterate through trajectory points
        for i, q in enumerate(traj_points):
            point = JointTrajectoryPoint()
            
            # Position: The core joint configuration (Q)
            point.positions = q.tolist()
            point.velocities = velocities[i].tolist()
            #point.accelerations = [0.0] * len(q)
            
            # Time when this point should be reached
            time_from_start += dt / self.speed
            point.time_from_start.sec = int(time_from_start)
            point.time_from_start.nanosec = int((time_from_start - int(time_from_start)) * 1e9)
            
            joint_traj.points.append(point)

        joint_traj.points[0] = start_point
        joint_traj.points.pop(1)

        return joint_traj
        
    def plan_to_target(self, start_joint_state, target_pos, timesteps, time_horizon,filename = None):
        """ Return message of planned trajectory and visualize before execution"""
        dt = time_horizon / timesteps
        start_cfg = self._joint_state_to_cfg(start_joint_state)

        status = "regenerate"
        traj, t_release, t_target = None, None, None
        solutions = None
        while True:
            # Check if we need to solve again
            match status:
                case "regenerate" | "next" if not solutions:
                    solutions = solve_by_sampling(
                        self.robot,
                        self.robot_coll,
                        self.world.gen_world_coll(),
                        self.target_link_name,
                        start_cfg,
                        target_pos,
                        timesteps,
                        dt,
                        robot_max_reach=0.85 * 0.8, # max 
                        max_vel=7, 
                        num_samples=50,
                        num_samples_iterated=10,
                    )
                    traj, t_release, t_target = solutions.pop(0)
                case "next" if solutions:
                    traj, t_release, t_target = solutions.pop(0)
                case "execute":
                    if filename:
                        save_trajectory(
                            filename,
                            start_cfg,
                            target_pos,
                            traj,
                            t_release,
                            t_target,
                            timesteps,
                            dt
                        )
                    return self._trajectory_points_to_msg(start_cfg, traj, dt), t_release

            status = self.world.visualize_all(
                    start_cfg, 
                    target_pos, 
                    traj, 
                    t_release, 
                    t_target, 
                    timesteps, 
                    dt
                )

                

    def play_loaded_trajectory(self, filename: str):
        start_cfg, target_pos, traj, t_release, t_target, timesteps, dt = load_trajectory(filename)

        # Visualize
        self.world.visualize_all(start_cfg, target_pos, traj, t_release, t_target, timesteps, dt)

        return self._trajectory_points_to_msg(start_cfg, np.array(traj), dt), t_release

    def _joint_state_to_cfg(self, joint_state: JointState) -> np.ndarray:
        """ Convert JointState to configuration vector """
        data_dict = dict(zip(joint_state.name, joint_state.position))
        target_joint_names = [
                'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]
        return np.array([data_dict[name] for name in target_joint_names])
   
    def _estimate_gradients(self, data: np.ndarray, dt: float) -> np.ndarray:
        """
        Generic finite difference function. 
        Works for calculating Velocity (from Pos) or Acceleration (from Vel).
        """
        gradients = np.zeros_like(data)
        
        # Central Difference (Interior)
        gradients[1:-1] = (data[2:] - data[:-2]) / (2 * dt)
        
        # Forward Difference (Start)
        gradients[0] = (-3*data[0] + 4*data[1] - data[2]) / (2 * dt)
        
        # Backward Difference (End)
        gradients[-1] = (3*data[-1] - 4*data[-2] + data[-3]) / (2 * dt)
        
        return gradients

        
def main(args=None):
    rclpy.init(args=args)
    node = IKPlanner()

    # ---------- Test setup ----------
    current_state = JointState()
    current_state.name = [
        'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
        'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
    ]

    # 4.722274303436279
    # -1.8504554233946742
    # -1.4257320165634155
    # -1.4052301210216065
    # 1.5935229063034058
    # -3.14103871980776

    current_state.position = [4.722, -1.850, -1.425, -1.405, 1.593, -3.141]

    # ---------- Run IK ----------
    node.get_logger().info("Testing IK computation...")
    ik_result = node.compute_ik(current_state, 0.125, 0.611, 0.423)

    # ---------- Check correctness ----------
    if ik_result is None:
        node.get_logger().error("IK computation returned None.")
        sys.exit(1)

    if not hasattr(ik_result, "name") or not hasattr(ik_result, "position"):
        node.get_logger().error("IK result missing required fields (name, position).")
        sys.exit(1)

    if len(ik_result.name) != len(ik_result.position):
        node.get_logger().error("IK joint names and positions length mismatch.")
        sys.exit(1)

    if len(ik_result.name) < 6:
        node.get_logger().error("IK returned fewer than 6 joints — likely incorrect.")
        sys.exit(1)

    node.get_logger().info("IK check passed.")
    
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
