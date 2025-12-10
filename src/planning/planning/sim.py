"""Trajectory Optimization

Basic Trajectory Optimization using PyRoKi.

Robot going over a wall, while avoiding world-collisions.
"""

from .world import World
import numpy as np
import jax.numpy as jnp
import pyroki as pk
from .trajectory_generation.generate_samples import solve_by_sampling
import tyro
from .load_urdf import load_ur7e_with_gripper, UR7eJointVar
from ament_index_python.packages import get_package_share_directory
import os
import jax_dataclasses as jdc

def main():
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
    target_pos = np.array([-0.3, 2.0, .7])
    time_horizon = 1.19
    timesteps = 50
    dt = time_horizon / timesteps

    status = "regenerate"
    traj, t_release, t_target = None, None, None
    solutions = None
    while True:
        # Check if we need to solve again
        match status:
            case "regenerate":
                print("STARTING AGAIN!")
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
                    num_samples=10,
                )
                traj, t_release, t_target = solutions.pop(0)
            case "next" if solutions:
                traj, t_release, t_target = solutions.pop(0)

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
