"""Trajectory Optimization

Basic Trajectory Optimization using PyRoKi.

Robot going over a wall, while avoiding world-collisions.
"""

from .world import World
import numpy as np
import pyroki as pk
from .trajectory_generation.generate_samples import solve_by_sampling
import tyro
import os
from robot_descriptions.loaders.yourdfpy import load_robot_description


def main():
    # Initialize robot
    urdf = load_robot_description("ur5_description")
    robot_coll = pk.collision.RobotCollision.from_urdf(urdf)

    # For UR5 it's important to initialize the robot in a safe configuration;
    default_cfg = np.array([4.712, -1.850, -1.425, -1.405, 1.593, -3.141]) 
    robot = pk.Robot.from_urdf(urdf, default_joint_cfg=default_cfg)
    target_link_name = "ee_link"

    # Initialize world
    world = World(robot, urdf, target_link_name)

    # Generate example trajectory
    start_cfg = default_cfg
    target_pos = np.array([-0.3, 2.0, .7])
    time_horizon = 1.19
    timesteps = 40
    dt = time_horizon / timesteps

    
    while True:
        traj, t_release, t_target = solve_by_sampling(
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
            cache_dir=os.path.join('/Users/jhaertel/code/catchv22/src/planning', 'saved_problems'),
            num_samples=50,
        )
        status = world.visualize_all(
                start_cfg, 
                target_pos, 
                traj, 
                t_release, 
                t_target, 
                timesteps, 
                dt
            )

        # Check if we need to solve again
        if "regenerate" == status: 
            print("STARTING AGAIN!")
            continue

    
if __name__ == "__main__":
    tyro.cli(main)
