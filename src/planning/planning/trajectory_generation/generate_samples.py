from typing import Sequence

import jax
import jax.numpy as jnp
import jaxlie
import numpy as onp
import pyroki as pk
from jax.typing import ArrayLike 

from .solve_ik import solve_ik_with_collision
from .jacobian import compute_ee_spatial_jacobian
from .gen_traj import solve_static_trajopt, choose_best_samples, _build_problem

import os

def quadratic_bezier_trajectory(p0, p1, p2):
    return lambda t: p0 * (1 - t) ** 2 + 2 * p1 * t * (1 - t) + p2 * t ** 2

def cubic_bezier_trajectory(p0, p1, p2, p3):
    return lambda t: p0 * (1 - t) ** 3 + 3 * p1 * t * (1 - t) ** 2 + 3 * p2 * t ** 2 * (1 - t) + p3 * t ** 3

def solve_by_sampling(
    robot: pk.Robot,
    robot_coll: pk.collision.RobotCollision,
    world_coll: Sequence[pk.collision.CollGeom],
    target_link_name: str,
    start_cfg: ArrayLike,
    target_position: ArrayLike,
    timesteps: int,
    dt: float,
    max_vel: float,
    robot_max_reach: float,
    num_samples: int = 300,
    num_samples_iterated: int = 10,
    g: float = 9.81,
) -> list[tuple[onp.ndarray, float, float]]:
    samples = generate_samples(robot, robot_coll, world_coll, target_link_name, start_cfg, target_position, timesteps, dt, g, max_vel, robot_max_reach, num_samples)
    
    problem = _build_problem(
            robot, 
            robot_coll, 
            world_coll, 
            robot.links.names.index(target_link_name), 
            jnp.array(start_cfg),
            jnp.array(target_position),
            timesteps, 
            dt, 
            g
        )

    best_samples = choose_best_samples(samples, num_samples_iterated, robot, problem, start_cfg, target_position, timesteps)
    return solve_static_trajopt(robot, robot_coll, world_coll, target_link_name, start_cfg, target_position, timesteps, dt, best_samples, g)


def generate_samples(
    robot: pk.Robot,
    robot_coll: pk.collision.RobotCollision,
    world_coll: Sequence[pk.collision.CollGeom],
    target_link_name: str,
    q_0: ArrayLike,
    target_position: ArrayLike,
    timesteps: int,
    dt: float,
    g: float,
    max_vel: float,
    robot_max_reach: float,
    num_samples: int = 100,
) -> list[tuple[onp.ndarray, float, float]]:
    """Generates a list of samples for the trajectory optimization problem."""
    samples = []
    target_link_index = robot.links.names.index(target_link_name)
    
    def gen_pos_sample():
        """Generates a position samples from a uniform distribution in a user defined box"""
        # Should have positive x-axis and z-axis bias
        while True:
            x = onp.random.uniform(-.3, .8)
            y = onp.random.uniform(-.8, .8)
            z = onp.random.uniform(-.3, .8)
            x_rel = onp.array([x, y, z])
            if onp.linalg.norm(x_rel) > robot_max_reach: continue
            return x_rel

    def gen_t_rel_sample():
        """Generates a time of launch uniformly in [.25, .75] * timesteps * dt"""
        return onp.random.uniform(0.25, 0.75) * timesteps * dt

    def gen_orientation_sample(v_rel, x_rel):
        """Generates a random orientation for the end effector"""
        # neg y-axis should be pointing in v_rel direction
        y_axis = -v_rel / onp.linalg.norm(v_rel)

        # z-axis should be somewhat away from origin
        away = x_rel / onp.linalg.norm(x_rel)
                
        # x-axis as cross product of the two
        x_axis = onp.cross(y_axis, away)
        z_axis = onp.cross(x_axis, y_axis)

        rot = jaxlie.SO3.from_matrix(onp.column_stack((x_axis, y_axis, z_axis)))
        return rot.wxyz

    while len(samples) < num_samples:
        x_rel = jnp.array(gen_pos_sample())
        t_rel = gen_t_rel_sample()

        dx = target_position - x_rel

        # Air time which minimizes the magnitude of the necessary velocity to reach the target
        dt_air = onp.sqrt(2.0 / g * onp.linalg.norm(dx))

        # Velocity given by air time and different in launch position and target
        vel_rel = ((target_position - x_rel) + 0.5 * onp.array([0.0, 0.0, g]) * dt_air ** 2) / dt_air
        
        # Toss out samples if too fast above user defined max velocity
        if onp.linalg.norm(vel_rel) > max_vel:
            print("Too fast")
            continue

        orientation_rel_wxyz = gen_orientation_sample(vel_rel, x_rel)
        
        # Solve IK
        q_rel = solve_ik_with_collision(
            robot, 
            robot_coll, 
            world_coll, 
            target_link_index,
            q_0,
            x_rel, 
            orientation_rel_wxyz
        )

        if not check_ik_convergence(robot, q_rel, target_link_index, x_rel, orientation_rel_wxyz): 
            continue

        J_rel = compute_ee_spatial_jacobian(robot, q_rel, jnp.array(target_link_index))
        target_spacial_vel = jnp.concatenate([jnp.array(vel_rel), jnp.zeros(3)])
        q_dot_rel = jnp.linalg.pinv(J_rel) @ target_spacial_vel


        x_start = robot.forward_kinematics(jnp.array(q_0))[target_link_index][..., 4:]
        
        # Midpoint with noise (allows "exploring" around obstacles)
        x_mid = (x_start + x_rel) * 0.5 
        x_mid += onp.random.uniform(-0.3, 0.3, size=3) # +/- 30cm noise

        q_mid = solve_ik_with_collision(
            robot, robot_coll, world_coll, target_link_index, 
            q_0, x_mid, orientation_rel_wxyz, 5, 0.001, 10.0  # Orientation matters less here
        )

        t_mid = t_rel * onp.random.uniform(0.2, 0.4)

    
        waypoint_trajectory = cubic_bezier_trajectory(q_0, q_0, q_mid, q_mid)

        # Generate a bezier curve tranjectory between joint configurations
        t_speed_up = t_rel - t_mid
        release_trajectory = cubic_bezier_trajectory(q_mid, q_mid, q_rel - t_speed_up / 3.0 * q_dot_rel,  q_rel)

        t_deaccel = timesteps * dt - t_rel
        deaccel_trajectory = quadratic_bezier_trajectory(q_rel, q_rel + t_deaccel / 2.0 * q_dot_rel, q_rel + t_deaccel / 2.0 * q_dot_rel)

        def traj(t):
            if t <= t_mid:
                return waypoint_trajectory(t / t_mid)
            elif t > t_mid and t <= t_rel:
                return release_trajectory( (t - t_mid) / t_speed_up )
            else:
                return deaccel_trajectory( (t - t_rel) / t_deaccel )

        traj_points = [traj(t) for t in dt * onp.arange(0, timesteps)]
        traj_points = onp.array(jnp.stack(traj_points))
        samples.append((traj_points, t_rel, t_rel + dt_air))
        
        if (len(samples) % 10) == 0 and samples: print(f'Generated {len(samples)} samples')
    
    return samples

#@jax.jit
def check_ik_convergence(
    robot: pk.Robot, 
    q_sol: jax.Array, 
    target_link_index: int,
    target_pos: jax.Array, 
    target_wxyz: jax.Array,
    pos_tol: float = 0.02, # 1 cm tolerance
    rot_tol: float = 0.1,  # ~5.7 degrees tolerance
) -> bool:
    """
    Returns True if the solution q_sol puts the EE within tolerance of the target.
    """
    # Robust extraction (handles batch dims if present, though likely not here)
    ee_pose_vec = robot.forward_kinematics(q_sol)[target_link_index]
    ee_pose = jaxlie.SE3(ee_pose_vec)
    
    # Construct Target Pose
    target_pose = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(target_wxyz), target_pos
    )
    
    # This gives us the transformation required to move FROM target TO current
    diff = target_pose.inverse() @ ee_pose
    
    # Measure magnitudes
    pos_error = jnp.linalg.norm(diff.translation())
    rot_error = jnp.linalg.norm(diff.rotation().log())
    
    return (pos_error < pos_tol) and (rot_error < rot_tol)
