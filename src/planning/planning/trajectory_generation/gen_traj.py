from typing import Sequence

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp
import pyroki as pk
from jax import Array
from jax.typing import ArrayLike 
# import os
# jax.config.update("jax_logging_level", "WARNING")
# jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/jax"))
# jax.config.update("jax_persistent_cache_enable_xla_caches", "all")
#jax.config.update("jax_explain_cache_misses", True)

from .jacobian import compute_ee_spatial_jacobian
from .generate_samples import generate_samples


class TimeVar(
    jaxls.Var[Array],
    default_factory=lambda: jnp.array([1.0]),  # Start with a factor of 1.0
): ...

def solve_static_trajopt(
    robot: pk.Robot,
    robot_coll: pk.collision.RobotCollision,
    world_coll: Sequence[pk.collision.CollGeom],
    target_link_name: str,
    start_cfg: ArrayLike,
    target_position: ArrayLike,
    timesteps: int,
    dt: float,
    initial_trajectories: list[jnp.ndarray],
    g: float = 9.81,

) -> tuple[onp.ndarray, float, float]:
    if not isinstance(start_cfg, onp.ndarray) and not isinstance(start_cfg, jnp.ndarray):
        raise ValueError(f"Invalid type for `ArrayLike`: {type(start_cfg)}")

    target_link_index = robot.links.names.index(target_link_name)

    traj_vars = robot.joint_var_cls(jnp.arange(timesteps))
    time_release = TimeVar(jnp.array([0])) #TimeVar(0)   
    time_target = TimeVar(jnp.array([1]))

    robot_unbatched = robot
    robot_coll_unbatched = robot_coll
    
    robot = jax.tree.map(lambda x: x[None], robot)  # Add batch dimension.
    robot_coll = jax.tree.map(lambda x: x[None], robot_coll)  # Add batch dimension.

    # Basic regularization / limit costs.
    factors: list[jaxls.Cost] = [
        pk.costs.rest_cost(
            traj_vars,
            traj_vars.default_factory()[None],
            jnp.array([0.0001])[None],
        ),
        pk.costs.limit_cost(
            robot,
            traj_vars,
            jnp.array([100.0])[None],
        ),
        pk.costs.manipulability_cost(
            robot,
            traj_vars,
            jnp.array([target_link_index]),
            jnp.array([1.0 / timesteps])[None]
        )
    ]

    # Collision avoidance.
    def compute_world_coll_residual(
        vals: jaxls.VarValues,
        robot: pk.Robot,
        robot_coll: pk.collision.RobotCollision,
        world_coll_obj: pk.collision.CollGeom,
        prev_traj_vars: jaxls.Var[jax.Array],
        curr_traj_vars: jaxls.Var[jax.Array],
    ):
        coll = robot_coll.get_swept_capsules(
            robot, vals[prev_traj_vars], vals[curr_traj_vars]
        )
        dist = pk.collision.collide(
            coll.reshape((-1, 1)), world_coll_obj.reshape((1, -1))
        )
        colldist = pk.collision.colldist_from_sdf(dist, 0.1)
        return (colldist * 20.0).flatten()

    # for world_coll_obj in world_coll:
    #     factors.append(
    #         jaxls.Cost(
    #             compute_world_coll_residual,
    #             (
    #                 robot,
    #                 robot_coll,
    #                 jax.tree.map(lambda x: x[None], world_coll_obj),
    #                 robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
    #                 robot.joint_var_cls(jnp.arange(1, timesteps)),
    #             ),
    #             name="World Collision (sweep)",
    #         )
    #     )

    # factors.append(
    #     pk.costs.self_collision_cost(
    #         robot,
    #         robot_coll, traj_vars,
    #         0.005,
    #         2000.0 / timesteps,
    #     )
    # )

    @jaxls.Cost.create_factory(name="terminal_velocity_limit_cost")
    def terminal_velocity_limit_cost(
        vals: jaxls.VarValues,
        var_t: jaxls.Var[Array],
        var_t_minus_1: jaxls.Var[Array],
        var_t_minus_2: jaxls.Var[Array],
        var_t_minus_3: jaxls.Var[Array],
        var_t_minus_4: jaxls.Var[Array],
    ) -> Array:
        """Computes the residual penalizing velocity limit violations (5-point stencil)."""
        q_t   = vals[var_t]
        q_tm1 = vals[var_t_minus_1]
        q_tm2 = vals[var_t_minus_2]
        q_tm3 = vals[var_t_minus_3]
        q_tm4 = vals[var_t_minus_4]
         
        velocity = (25 * q_t - 48 * q_tm1 + 36 * q_tm2 - 16 * q_tm3 + 3 * q_tm4) / (12 * dt)
        return (velocity * 20.0).flatten()
    
    # Start / end velocity constraints.
    factors.extend(
        [
            jaxls.Cost(
                lambda vals, var: ((vals[var] - start_cfg) * 200.0).flatten(),
                (robot.joint_var_cls(jnp.arange(0, 2)),),
                name="start_pose_constraint",
            ),
            terminal_velocity_limit_cost(
                robot.joint_var_cls(jnp.arange(timesteps-6, timesteps)),
                robot.joint_var_cls(jnp.arange(timesteps-7, timesteps - 1)),
                robot.joint_var_cls(jnp.arange(timesteps-8, timesteps - 2)),
                robot.joint_var_cls(jnp.arange(timesteps-9, timesteps - 3)),
                robot.joint_var_cls(jnp.arange(timesteps-10, timesteps - 4)),
            ),

        ]
    )

    # Velocity / acceleration / jerk minimization.
    factors.extend(
        [
            pk.costs.smoothness_cost(
                robot.joint_var_cls(jnp.arange(1, timesteps)),
                robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
                jnp.array([2.5 / timesteps])[None],
            ),
            pk.costs.five_point_velocity_cost(
                robot,
                robot.joint_var_cls(jnp.arange(4, timesteps)),
                robot.joint_var_cls(jnp.arange(3, timesteps - 1)),
                robot.joint_var_cls(jnp.arange(1, timesteps - 3)),
                robot.joint_var_cls(jnp.arange(0, timesteps - 4)),
                dt,
                jnp.array([50.0 / timesteps])[None],
            ),
            pk.costs.five_point_acceleration_cost(
                robot.joint_var_cls(jnp.arange(2, timesteps - 2)),
                robot.joint_var_cls(jnp.arange(4, timesteps)),
                robot.joint_var_cls(jnp.arange(3, timesteps - 1)),
                robot.joint_var_cls(jnp.arange(1, timesteps - 3)),
                robot.joint_var_cls(jnp.arange(0, timesteps - 4)),
                dt,
                jnp.array([2.0 / timesteps])[None],
            ),
            pk.costs.five_point_jerk_cost(
                robot.joint_var_cls(jnp.arange(6, timesteps)),
                robot.joint_var_cls(jnp.arange(5, timesteps - 1)),
                robot.joint_var_cls(jnp.arange(4, timesteps - 2)),
                robot.joint_var_cls(jnp.arange(2, timesteps - 4)),
                robot.joint_var_cls(jnp.arange(1, timesteps - 5)),
                robot.joint_var_cls(jnp.arange(0, timesteps - 6)),
                dt,
                jnp.array([2.0 / timesteps])[None],
            ),
        ]
    )

     # Trajectory tossing
    @jaxls.Cost.create_factory(name="toss_target_cost")
    def toss_target_cost(
        vals: jaxls.VarValues,
        robot: pk.Robot,
        traj_vars_tuple: tuple[jaxls.Var[jax.Array], ...],
        time_release: TimeVar,
        time_target: TimeVar,
    ):
        t_rel = vals[time_release]
        
        idx_float = t_rel / dt
        
        # Get the integer bounds for interpolation
        # Clamp to ensure we don't go out of bounds [0, timesteps-2]
        # We use -2 because we need a "next" neighbor for velocity calculation
        max_idx = vals[traj_vars].shape[0] - 2
        idx_floor = jnp.clip(jnp.floor(idx_float).astype(int), 0, max_idx)
        idx_ceil = idx_floor + 1
        
        alpha = idx_float - idx_floor
        
        # 3. Retrieve discrete joint configurations
        qs_list = [vals[v] for v in traj_vars_tuple]
        qs_full = jnp.stack(qs_list)

        q_prev = qs_full[idx_floor - 1]
        q_curr = qs_full[idx_floor]     # q at t_floor
        q_next = qs_full[idx_ceil]      # q at t_ceil
        q_next_next = qs_full[idx_ceil + 1]

        # --- 3. Interpolate Position (Still Linear) ---
        q_release = (1.0 - alpha) * q_curr + alpha * q_next

        # --- 4. Interpolate Central Difference Velocity ---
        # Velocity at q_curr (t_floor) using central difference: (q_next - q_prev) / 2*dt
        q_dot_floor = (q_next - q_prev) / (2.0 * dt)

        # Velocity at q_next (t_ceil) using central difference: (q_next_next - q_curr) / 2*dt
        q_dot_ceil = (q_next_next - q_curr) / (2.0 * dt) 

        # Linear interpolation of the two velocity estimates
        q_dot_release = (1.0 - alpha) * q_dot_floor + alpha * q_dot_ceil

        # Get starting position of launch
        q = robot.forward_kinematics(q_release)
        x0 = jnp.take(q, target_link_index, axis=-2)[..., 4:]
        
        # Get launch velocity
        jacobian = compute_ee_spatial_jacobian(robot, q_release, jnp.array([target_link_index]))
        twist = jacobian @ q_dot_release.squeeze()
        v0 = twist[:3][None]

        # Time duration of flight
        delta_t = vals[time_target] - t_rel
        
        # Projectile motion: x(t) = x0 + v0*t + 0.5*g*t^2
        gravity_vec = jnp.array([0.0, 0.0, -g]) 
        
        pred_pos = x0 + v0 * delta_t + 0.5 * gravity_vec * (delta_t ** 2)
        
        # Target position (passed from outside, you might need to add it to args or closure)
        # For now assuming target_position is available in closure as `target_position`
        target_pos_arr = jnp.array(target_position)
        return ((pred_pos - target_pos_arr) * 300.0).flatten()
        
    @jaxls.Cost.create_factory(name="positive_time_cost")
    def positive_time_cost(
        vals: jaxls.VarValues,
        time_release: TimeVar,
        time_target: TimeVar,
    ):
        t_rel = vals[time_release]
        t_tgt = vals[time_target]
        total_duration = timesteps*dt

        # Enforce a minimum flight time to prevent infinite velocity spikes
        # (e.g., ball must fly for at least 0.1 seconds)
        min_flight_time = 0.2 

        # 1. Release Time Lower Bound: t_rel >= 0
        # If t_rel is -0.1, error is 0.1
        err_rel_low = jnp.maximum(0.0, -t_rel - 3 * dt)

        # 2. Release Time Upper Bound: t_rel <= total_duration
        # If t_rel is 5.1 and max is 5.0, error is 0.1
        err_rel_high = jnp.maximum(0.0, t_rel - total_duration - 3 * dt)

        # 3. Flight Time Constraint: t_tgt >= t_rel + min_flight
        # If t_tgt is too early, this error grows
        err_flight = jnp.maximum(0.0, (t_rel + min_flight_time) - t_tgt)
        
        err = jnp.array([err_rel_low, err_rel_high, err_flight])

        err_from_intended = (t_rel - total_duration * 0.6) 

        # Weighting: 100.0 is a strong weight to enforce these strictly.
        return  jnp.append(err * 100.0,  err_from_intended * 0.1)


    @jaxls.Cost.create_factory(name="toss_alignment_cost")
    def toss_alignment_cost(
        vals: jaxls.VarValues,
        robot: pk.Robot,
        traj_vars_tuple: tuple[jaxls.Var[jax.Array], ...],
        time_release: TimeVar,
    ):
        """Want the z-axis to lie tangent to launch trajectory for smooth flight."""
        t_rel = vals[time_release]
        
        idx_float = t_rel / dt
        
        # Get the integer bounds for interpolation
        # Clamp to ensure we don't go out of bounds [0, timesteps-2]
        # We use -2 because we need a "next" neighbor for velocity calculation
        max_idx = vals[traj_vars].shape[0] - 2
        idx_floor = jnp.clip(jnp.floor(idx_float).astype(int), 0, max_idx)
        idx_ceil = idx_floor + 1
        
        alpha = idx_float - idx_floor
        
        # 3. Retrieve discrete joint configurations
        qs_list = [vals[v] for v in traj_vars_tuple]
        qs_full = jnp.stack(qs_list)

        q_prev = qs_full[idx_floor - 1]
        q_curr = qs_full[idx_floor]     # q at t_floor
        q_next = qs_full[idx_ceil]      # q at t_ceil
        q_next_next = qs_full[idx_ceil + 1]

        # --- 3. Interpolate Position (Still Linear) ---
        q_release = (1.0 - alpha) * q_curr + alpha * q_next

        # --- 4. Interpolate Central Difference Velocity ---
        # Velocity at q_curr (t_floor) using central difference: (q_next - q_prev) / 2*dt
        q_dot_floor = (q_next - q_prev) / (2.0 * dt)

        # Velocity at q_next (t_ceil) using central difference: (q_next_next - q_curr) / 2*dt
        q_dot_ceil = (q_next_next - q_curr) / (2.0 * dt) 

        # Linear interpolation of the two velocity estimates
        q_dot_release = (1.0 - alpha) * q_dot_floor + alpha * q_dot_ceil

        # Get starting position of launch
        q = robot.forward_kinematics(q_release)
        R_ee = jaxlie.SE3(jnp.take(q, target_link_index, axis=-2)[..., :4])
        y_ee = R_ee.rotation() @ jnp.array([0.0, 1.0, 0.0])
         
        # Get launch velocity
        jacobian = compute_ee_spatial_jacobian(robot, q_release, jnp.array([target_link_index]))
        twist = jacobian @ q_dot_release.squeeze()
        v0 = twist[:3][None]

        v_norm = jnp.linalg.norm(v0) + 1e-6
        v_dir = v0 / v_norm

        return ((jnp.cross(-y_ee, v_dir)) * 40.0).flatten()
 

    # factors.extend(
    #     [
    #         positive_time_cost(time_release, time_target),
    #         toss_target_cost(robot, tuple(traj_vars[i] for i in range(timesteps)), time_release, time_target),
    #         toss_alignment_cost(robot, tuple(traj_vars[i] for i in range(timesteps)), time_release)
    #     ]
    # )
   
    # print(f'Generating {num_samples} samples')


    problem = jaxls.LeastSquaresProblem(factors, [traj_vars, time_release, time_target]).analyze()
    
    def cost(sample):
        vals = jaxls.VarValues.make((traj_vars.with_value(sample[0]),
                                     time_release.with_value(jnp.array([sample[1]])),
                                    time_target.with_value(jnp.array([sample[2]]))))
        residual = problem.compute_residual_vector(vals)
        return jnp.dot(residual, residual)

    def refine_trajectories(sample):
       init_traj, init_t_rel, init_t_target = sample
       solution = (
           problem
           .solve(
               initial_vals=
               jaxls.VarValues.make((traj_vars.with_value(init_traj), 
                                     time_release.with_value(jnp.array([init_t_rel])), 
                                     time_target.with_value(jnp.array([init_t_target])))),
           )
        )
       return solution[traj_vars], solution[time_release], solution[time_target], cost(solution)
    traj, t_release, t_target, _ = min(initial_trajectories, key=lambda s: s[-1])

    # 4. Solve the optimization problem.
    return onp.array(traj), t_release.item(), t_target.item()


