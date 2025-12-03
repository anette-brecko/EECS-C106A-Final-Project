"""Trajectory Optimization

Basic Trajectory Optimization using PyRoKi.

Robot going over a wall, while avoiding world-collisions.
"""

import time
from typing import Callable

import numpy as np
import jax.numpy as jnp
import pyroki as pk
import trimesh
import tyro
import viser
from viser.extras import ViserUrdf
from robot_descriptions.loaders.yourdfpy import load_robot_description

from gen_traj import solve_static_trajopt
from gen_traj import compute_ee_spatial_jacobian
from gen_traj import solve_single_ik_with_collision

def main():
    urdf = load_robot_description("ur5_description") # TODO: Change to ur7e
    down_wxyz = np.array([0.707, 0, 0.707, 0])
    up_wxyz = np.array([0.707, 0, -0.707, 0])
    target_link_name = "ee_link"

    # For UR5 it's important to initialize the robot in a safe configuration;
    # the zero-configuration puts the robot aligned with the wall obstacle.
    default_cfg = np.zeros(6)
    default_cfg[1] = -1.308
    robot = pk.Robot.from_urdf(urdf, default_joint_cfg=default_cfg)

    robot_coll = pk.collision.RobotCollision.from_urdf(urdf)

    # Define the trajectory problem:
    # - number of timesteps, timestep size
    timesteps, dt = 100, 0.02
    # - the start and end poses.
    start_pos, target_pos = np.array([0.4, 0.15, 0.15]), np.array([2.0, -0.3, .7])
    start_wxyz = down_wxyz

    # Define the obstacles:
    # - Ground
    ground_coll = pk.collision.HalfSpace.from_point_and_normal(
        np.array([0.0, 0.0, -0.25]), np.array([0.0, 0.0, 1.0])
    )
    # - Wall
    wall_coll = pk.collision.HalfSpace.from_point_and_normal(
        np.array([-2, 0, 0]), np.array([1.0, 0.0, 0.0])
    )

    # - Pillar

    # - Table
    table_height = 0.4
    table_width = 0.3
    table_length = 0.2
    table_offset = np.array([0.5, 0.0, 0]) # position of the center of the top face relative to robot
    table_intervals = np.arange(start=0, stop=table_length, step=0.05) 
    translation = np.concatenate(
        [
            table_intervals.reshape(-1, 1),
            np.full((table_intervals.shape[0], 1), 0.0),
            np.full((table_intervals.shape[0], 1), table_height / 2),
        ],
        axis=1,
    ) + table_offset + np.array([0.0, 0.0, -table_height / 2])

    table_coll_1 = pk.collision.Capsule.from_radius_height(
        position=translation + np.array([0.0, -table_width / 4, 0.0]),
        radius=np.full((translation.shape[0], 1), table_width / 4),
        height=np.full((translation.shape[0], 1), table_height),
    )

    table_coll_2 = pk.collision.Capsule.from_radius_height(
        position=translation + np.array([0.0, table_width / 4, 0.0]),
        radius=np.full((translation.shape[0], 1), table_width / 4),
        height=np.full((translation.shape[0], 1), table_height),
    )

    table_coll = pk.collision.Capsule.from_radius_height(
        position=translation,
        radius=np.full((translation.shape[0], 1), table_width / 2),
        height=np.full((translation.shape[0], 1), table_height),
    )

    # - Monitor
    

    world_coll = [ground_coll, table_coll]
    

    start_cfg = solve_single_ik_with_collision(
            robot,
            robot_coll,
            world_coll,
            robot.links.names.index(target_link_name),
            start_pos,
            start_wxyz
    )





    traj, t_release, t_target = solve_static_trajopt(
        robot,
        robot_coll,
        world_coll,
        target_link_name,
        start_cfg,
        target_pos,
        timesteps,
        dt,
        0.85 * 0.8,
        7
    )


    traj = np.array(traj)

    # Visualize!
    server = viser.ViserServer()
    urdf_vis = ViserUrdf(server, urdf)
    server.scene.add_grid("/grid", width=2, height=2, cell_size=0.1)
    server.scene.add_mesh_trimesh(
        "wall_box",
        trimesh.creation.box(
            extents=(table_length, table_width, table_height),
            transform=trimesh.transformations.translation_matrix(
                np.array([0, 0.0, -table_height / 2]) + table_offset
            ),
        ),
    )
    server.scene.add_frame(
        "/start",
        position=start_pos,
        wxyz=start_wxyz,
        axes_length=0.05,
        axes_radius=0.01,
    )
    ee_axis = server.scene.add_frame(
        "/ee",
        position=start_pos,
        wxyz=start_wxyz,
        axes_length=0.05,
        axes_radius=0.01,
    )

    server.scene.add_icosphere(
        "/target",
        position=target_pos,
        radius=0.025,
        color=(0, 0, 255)
    )

    #t_release = 0.2

    timestep_release = int(np.ceil(t_release / dt))
    times_idx = np.arange(timestep_release, int(np.ceil(t_target/dt)) + 15) # TODO t_target / dt)
    times = dt * times_idx 
    
    #test_ball_trajectory = lambda t: np.array([1.0, 0.0, 5.0]) * (t - t_release) - 0.5 * np.array([0.0, 0.0, 9.81]) * ((t - t_release) ** 2)
    #ball_traj = [test_ball_trajectory(t) for t in times]

    ball_traj = [ball_trajectory(robot, target_link_name, traj, t_release, dt)(t) for t in times]
    server.scene.add_spline_catmull_rom(
        "/ball_traj",
        points=np.array(ball_traj),
        line_width=2.0,
        segments = 30,
    )

    ball = server.scene.add_icosphere(
            "/ball",
            position=start_pos,
            radius = 0.02,
            color=(255, 0, 0),
    )
               
    slider = server.gui.add_slider(
        "Timestep", min=0, max=timesteps - 1, step=1, initial_value=0
    )
    playing = server.gui.add_checkbox("Playing", initial_value=True)

    while True:
        if playing.value:
            slider.value = (slider.value + 1) % timesteps

        urdf_vis.update_cfg(traj[slider.value])

        q = robot.forward_kinematics(traj[slider.value])
        twist = jnp.take(q, robot.links.names.index(target_link_name), axis=-2)
        ee_axis.wxyz = twist[...,:4]
        ee_axis.position = twist[...,4:]
        
        if slider.value in times_idx:
            ball.visible = True
            ball.position = ball_traj[slider.value - timestep_release]
        else: 
            ball.visible = False

        time.sleep(dt)

def ball_trajectory(robot, target_link_name, traj, time_release, dt) -> Callable[[float], np.ndarray]:
    target_link_index = robot.links.names.index(target_link_name)

    idx_float = time_release / dt
        
    # Get the integer bounds for interpolation
    # Clamp to ensure we don't go out of bounds [0, timesteps-2]
    # We use -2 because we need a "next" neighbor for velocity calculation
    max_idx = traj.shape[0] - 2
    idx_floor = np.clip(np.floor(idx_float).astype(int), 0, max_idx)
    idx_ceil = idx_floor + 1
    
    alpha = idx_float - idx_floor

    q_curr = traj[idx_floor]
    q_next = traj[idx_ceil]
    
    # 4. Interpolate Joint Position and Velocity
    # Linear Interpolation for q
    q_release = (1.0 - alpha) * q_curr + alpha * q_next
    
    # Joint Velocity is the slope between the two points
    q_dot_release = (q_next - q_curr) / dt

    # Get starting position of launch
    q = robot.forward_kinematics(q_release)
    x0 = jnp.take(q, target_link_index, axis=-2)[..., 4:]
    
    # Get launch velocity
    jacobian = compute_ee_spatial_jacobian(robot, q_release, jnp.array([target_link_index]))
    twist = jacobian @ q_dot_release.squeeze()
    v0 = twist[:3][None]

    gravity_vec = np.array([0.0, 0.0, -9.81]) 
    pred_pos = lambda t: (x0 + v0 * (t - time_release) + 0.5 * gravity_vec * ((t - time_release) ** 2))[0]
    return pred_pos

if __name__ == "__main__":
    tyro.cli(main)
