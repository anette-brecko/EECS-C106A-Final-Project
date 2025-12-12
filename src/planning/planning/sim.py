"""Trajectory Optimization

Basic Trajectory Optimization using PyRoKi.

Robot going over a wall, while avoiding world-collisions.
"""

from .world import World
import numpy as np
import jax.numpy as jnp
import pyroki as pk
from .trajectory_generation.generate_samples import solve_by_sampling
from .trajectory_generation.save_and_load import save_trajectory
import tyro
from .load_urdf import load_ur7e_with_gripper, UR7eJointVar
import jax_dataclasses as jdc


def main(filename: str, timesteps: int, time_horizon: float):

    # Initialize robot
    urdf = load_ur7e_with_gripper()
    robot_coll = pk.collision.RobotCollision.from_urdf(urdf)

    # For UR5 it's important to initialize the robot in a safe configuration;
    default_cfg = np.array([4.712, -1.850, -1.425, -1.405, 1.593, -3.141]) 
    robot = pk.Robot.from_urdf(urdf, default_joint_cfg=default_cfg)
    UR7eJointVar.default_factory = staticmethod(lambda: jnp.array(default_cfg))
    robot = jdc.replace(robot, joint_var_cls=UR7eJointVar)
    target_link_name = "robotiq_hande_end"

    # Initialize world
    world = World(robot, urdf, target_link_name)

    # Generate example trajectory
    start_cfg = default_cfg
    target_pos = np.array([-0.3, 2.0, .2])
    dt = time_horizon / timesteps

    status = "regenerate"
    traj, t_release, t_target = None, None, None
    solutions = None
    idx = 0
    while True:
        # Check if we need to solve again
        match status:
            case "regenerate":
                solutions = solve_by_sampling(
                    robot,
                    robot_coll,
                    world.gen_world_coll(),
                    target_link_name,
                    start_cfg,
                    target_pos,
                    timesteps,
                    dt,
                    robot_max_reach=0.85 * 0.8, # max 
                    max_vel=7, 
                    num_samples=300,
                    num_samples_iterated=10,
                )
                traj, t_release, t_target = solutions[0]
                idx = 0
            case "next":
                idx = (idx + 1) % len(solutions)
                traj, t_release, t_target = solutions[idx]
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
                print("Saving")
        status = world.visualize_all(
                start_cfg, 
                target_pos, 
                traj, 
                t_release, 
                t_target, 
                timesteps, 
                dt
            )
    
if __name__ == "__main__":
    tyro.cli(main)
