"""Trajectory Optimization

Basic Trajectory Optimization using PyRoKi.

Robot going over a wall, while avoiding world-collisions.
"""

import time
from typing import Callable
from world import World
import numpy as np
import pyroki as pk
from robot_descriptions.loaders.yourdfpy import load_robot_description
from oneshot_gen_traj import solve_static_trajopt
import tyro

def main():
    # Initialize robot
    urdf = load_robot_description("ur5_description") # TODO: Change to ur7e
    robot_coll = pk.collision.RobotCollision.from_urdf(urdf)

    # For UR5 it's important to initialize the robot in a safe configuration;
    default_cfg = np.array([3.141, -1.850, -1.425, -1.405, 1.593, -3.141]) # TODO: Check
    robot = pk.Robot.from_urdf(urdf, default_joint_cfg=default_cfg)
    target_link_name = "ee_link"

    # Initialize world
    world = World(robot, urdf, target_link_name)

    world._visualize_joints(default_cfg)

    # Generate example trajectory
    start_cfg = default_cfg
    target_pos = np.array([2.0, -0.3, .7])
    time_horizon = 1.2
    timesteps = 40
    dt = time_horizon / timesteps

    traj, t_release, t_target = solve_static_trajopt(
            robot,
            robot_coll,
            world.gen_world_coll(),
            target_link_name,
            start_cfg,
            target_pos,
            timesteps,
            dt,
            0.85 * 0.8, # TODO: MAX REACH! Make parameter
            7 # TODO: MAX VEL! Make parameter
        )
    
    # Visualize!
    world.visualize_all(start_cfg, target_pos, traj, t_release, t_target, timesteps, dt)

    
if __name__ == "__main__":
    tyro.cli(main)
