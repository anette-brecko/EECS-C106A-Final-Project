import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from optimal_planner_msgs.srv import OptimalTrajectoryPlan

import sys
import numpy as np
import pyroki as pk

from ._trajectory_generation.generate_samples import solve_by_sampling
from .world import World
from .load_urdf import load_ur7e_with_gripper, UR7eJointVar
import jax_dataclasses as jdc
from ._trajectory_generation.save_and_load import save_trajectory, load_trajectory

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


class TrajectoryServer(Node):
    timesteps = 50
    def __init__(self):
        super().__init__('trajectory_planner')

        urdf = load_ur7e_with_gripper()    
        self.robot_coll = pk.collision.RobotCollision.from_urdf(urdf)

        # For UR5 it's important to initialize the robot in a safe configuration;
        self.default_cfg = np.array([4.712, -1.850, -1.425, -1.405, 1.593, -3.141])
        self.robot = pk.Robot.from_urdf(urdf, default_joint_cfg=self.default_cfg)
        UR7eJointVar.default_factory = staticmethod(lambda: jnp.array(self.default_cfg))

        self.robot = jdc.replace(self.robot, joint_var_cls=UR7eJointVar)
        self.target_link_name = "robotiq_hande_end"

        self.world = World(self.robot, urdf, self.target_link_name) 
        self.speed = 1.0

        self.srv = self.create_service(OptimalTrajectoryPlan, '/optimal_trajectory_plan', self.plan_callback)
        self.get_logger().info("Gripper reset service ready: /set_gripper")
        self._warmup()

    def _warmup(self):
        self.get_logger().info(f'Warmup jax')
        solve_by_sampling(
            self.robot,
            self.robot_coll,
            self.world.gen_world_coll(),
            self.target_link_name,
            self.default_cfg,
            np.array([0.3, 2, 0]),
            self.timesteps,
            0.02,
            robot_max_reach=0.85 * 0.8, # max 
            max_vel=7, 
            num_samples=1,
            num_samples_iterated=1,
        )

    def plan_callback(self, request, response):
        response.trajectory, response.time_release, response.time_target = self._plan_to_target(
            request.start_joint_state,
            request.target_pos,
            self.timesteps,
            request.time_horizon / self.timesteps,
            speed = request.speed,
            num_samples = request.num_samples,
            num_samples_iterated = request.num_samples_iterated
        )
        return response

    def _solve_to_target(
            self, 
            start_cfg, 
            target_pos, 
            timesteps, 
            dt, 
            num_samples=300, 
            num_samples_iterated=5
            ):
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
            robot_max_reach=0.85 * 0.8, # max 
            max_vel=7, 
            num_samples=num_samples,
            num_samples_iterated=num_samples_iterated,
        )
    
      
    def _plan_to_target(
            self, 
            start_joint_state, 
            target_pos, 
            timesteps, 
            time_horizon,
            speed = 1.0,
            num_samples = 100,
            num_samples_iterated=10,
            ):
        """ Return message of planned trajectory and visualize before execution"""
        dt = time_horizon / timesteps
        start_cfg = self._joint_state_to_cfg(start_joint_state)

        status = "regenerate"
        traj, t_release, t_target = None, None, None
        solutions = None
        idx = 0
        while True:
            # Check if we need to solve again
            match status:
                case "regenerate":
                    solutions = self._solve_to_target(
                        start_cfg,
                        target_pos,
                        timesteps,
                        dt,
                        num_samples,
                        num_samples_iterated
                    )
                    traj, t_release, t_target = solutions[0]
                    idx = 0
                case "next":
                    idx = (idx + 1) % len(solutions)
                    traj, t_release, t_target = solutions[idx]
                case "execute":
                    return self._trajectory_points_to_msg(start_cfg, traj, dt, speed), t_release / speed, t_target
            status = self.world.visualize_all(
                    start_cfg, 
                    target_pos, 
                    traj, 
                    t_release, 
                    t_target, 
                    timesteps, 
                    dt
                )

def _joint_state_to_cfg(joint_state: JointState) -> np.ndarray:
    """ Convert JointState to configuration vector """
    data_dict = dict(zip(joint_state.name, joint_state.position))
    target_joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
    ]
    return np.array([data_dict[name] for name in target_joint_names])

def _cfg_to_joint_state(cfg: np.ndarray) -> JointState:
    """ Convert JointState to configuration vector """
    joint_state = JointState()
    joint_state.name = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
    ]
    joint_state.position = cfg.tolist()
    return joint_state
   
def _estimate_gradients(data: np.ndarray, dt: float) -> np.ndarray:
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

def _trajectory_points_to_msg(start_cfg, traj_points, dt, speed) -> JointTrajectory:
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

    velocities = _estimate_gradients(traj_points, dt) * speed
    
    # 3. Iterate through trajectory points
    for i, q in enumerate(traj_points):
        point = JointTrajectoryPoint()
        
        # Position: The core joint configuration (Q)
        point.positions = q.tolist() 
        point.velocities = velocities[i].tolist()
        #point.accelerations = [0.0] * len(q)
        
        # Time when this point should be reached
        time_from_start += dt / speed
        point.time_from_start.sec = int(time_from_start)
        point.time_from_start.nanosec = int((time_from_start - int(time_from_start)) * 1e9)
        
        joint_traj.points.append(point)

    joint_traj.points[0] = start_point
    #joint_traj.points.pop(1)

    return joint_traj
            
def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryServer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
