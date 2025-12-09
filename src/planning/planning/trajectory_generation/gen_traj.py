from typing import Sequence, Callable

import jax
import jax.numpy as jnp
import jaxlie
import jaxls
import numpy as onp
import pyroki as pk
from jax import Array
from jax.typing import ArrayLike 
from .jacobian import compute_ee_spatial_jacobian

import os

from .save_and_load import save_problem, load_problem, TimeVar, StartConfigVar, TargetPosVar


# import os
# jax.config.update("jax_logging_level", "WARNING")
# jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/jax"))
# jax.config.update("jax_persistent_cache_enable_xla_caches", "all")
#jax.config.update("jax_explain_cache_misses", True)
def _param_to_var_vals(
    robot: pk.Robot,
    start_cfg: ArrayLike,
    target_position: ArrayLike,
    timesteps: int,
    traj: onp.ndarray,
    t_release: float,
    t_target: float
) -> jaxls.VarValues:
    traj_vars = robot.joint_var_cls(jnp.arange(timesteps))
    time_release_var = TimeVar(0)
    time_target_var = TimeVar(1)
    start_var = StartConfigVar(0)
    target_pos_var = TargetPosVar(0)

    return jaxls.VarValues.make((
        traj_vars.with_value(jnp.array(traj)), 
        time_release_var.with_value(jnp.array([t_release])), 
        time_target_var.with_value(jnp.array([t_target])),
        start_var.with_value(jnp.array(start_cfg)),
        target_pos_var.with_value(jnp.array(target_position))
    ))

def solve_static_trajopt(
    robot: pk.Robot,
    robot_coll: pk.collision.RobotCollision,
    world_coll: Sequence[pk.collision.CollGeom],
    target_link_name: str,
    start_cfg: ArrayLike,
    target_position: ArrayLike,
    timesteps: int,
    dt: float,
    initial_trajectories: list[tuple[onp.ndarray, float, float]],
    g: float = 9.81,
    cache_dir: str | os.PathLike = "/tmp",
) -> tuple[onp.ndarray, float, float]:
    traj_vars = robot.joint_var_cls(jnp.arange(timesteps))
    time_release_var = TimeVar(0)
    time_target_var = TimeVar(1)

    problem = analyze_problem(robot, robot_coll, world_coll, target_link_name, timesteps, dt, g, cache_dir) 
    sample_to_var_vals = lambda sample: _param_to_var_vals(robot, start_cfg, target_position, timesteps, sample[0], sample[1], sample[2])
    samples_var_vals = [sample_to_var_vals(sample) for sample in initial_trajectories]

    def solve_trajectory(sample):
        solution = problem.solve(
               initial_vals=sample
            )
        return solution[traj_vars], solution[time_release_var], solution[time_target_var]

    if len(initial_trajectories) == 0:
        traj, t_release, t_target = solve_trajectory(samples_var_vals[0])
    else:
        __cost = _cost(problem)
        cost = lambda sample: __cost(sample_to_var_vals(sample))
        traj, t_release, t_target = min([solve_trajectory(sample) for sample in samples_var_vals], key=cost)

    return onp.array(traj), t_release.item(), t_target.item()

def _cost(problem: jaxls.AnalyzedLeastSquaresProblem) -> Callable[..., float]:
    def cost(sample: jaxls.VarValues) -> float:
        residual = problem.compute_residual_vector(sample)
        residual = jnp.array(residual)
        return jnp.dot(residual, residual).item()
    return cost

def choose_best_samples(
    samples: list[tuple[onp.ndarray, float, float]],
    num_samples_selected: int,
    robot: pk.Robot,
    problem: jaxls.AnalyzedLeastSquaresProblem,
    start_cfg: ArrayLike,
    target_position: ArrayLike,
    timesteps: int
) -> list[tuple[onp.ndarray, float, float]]:
    sample_to_var_vals = lambda sample: _param_to_var_vals(robot, start_cfg, target_position, timesteps, sample[0], sample[1], sample[2])
    cost = _cost(problem)
    return sorted(samples, key=lambda sample: cost(sample_to_var_vals(sample)))[:num_samples_selected]

def analyze_problem(
    robot: pk.Robot,
    robot_coll: pk.collision.RobotCollision,
    world_coll: Sequence[pk.collision.CollGeom],
    target_link_name: str,
    timesteps: int,
    dt: float,
    g: float = 9.81,
    cache_dir: str | os.PathLike = "/tmp"
):
    cache_dir = os.path.expanduser(cache_dir)
    filename = os.path.join(cache_dir, f"dt_{dt:.6f}_timesteps_{timesteps}.pkl")
    
    try:
        return load_problem(filename)
    except Exception as e:
        print(f"Could not load problem from {filename}. Generating from scratch.")
        problem = _analyze_problem(robot, robot_coll, world_coll, target_link_name, timesteps, dt, g)
        save_problem(problem, filename)
        raise e
        return problem

def _analyze_problem(
    robot: pk.Robot,
    robot_coll: pk.collision.RobotCollision,
    world_coll: Sequence[pk.collision.CollGeom],
    target_link_name: str,
    timesteps: int,
    dt: float,
    g: float = 9.81
): 
    target_link_index = robot.links.names.index(target_link_name)

    traj_vars = robot.joint_var_cls(jnp.arange(timesteps))
    time_release_var = TimeVar(0)     
    time_target_var = TimeVar(1)
    start_var = StartConfigVar(0)
    target_pos_var = TargetPosVar(0)

    robot_batched = jax.tree.map(lambda x: x[None], robot)
    robot_coll_batched = jax.tree.map(lambda x: x[None], robot_coll)

    # Basic regularization / limit costs.
    factors: list[jaxls.Cost] = [
        pk.costs.rest_cost(
            traj_vars,
            traj_vars.default_factory()[None],
            jnp.array([0.0001])[None],
        ),
        pk.costs.limit_cost(
            robot_batched,
            traj_vars,
            jnp.array([100.0])[None],
        ),
        pk.costs.manipulability_cost(
            robot_batched,
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

    for world_coll_obj in world_coll:
        factors.append(
            jaxls.Cost(
                compute_world_coll_residual,
                (
                    robot_batched,
                    robot_coll_batched,
                    jax.tree.map(lambda x: x[None], world_coll_obj),
                    robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
                    robot.joint_var_cls(jnp.arange(1, timesteps)),
                ),
                name="World Collision (sweep)",
            )
        )

    factors.append(
        pk.costs.self_collision_cost(
            robot_batched,
            robot_coll_batched,
            traj_vars,
            0.005,
            2000.0 / timesteps,
        )
    )

    @jaxls.Cost.create_factory(name="terminal_velocity_limit_cost")
    def terminal_velocity_limit_cost(
        vals: jaxls.VarValues,
        var_t: jaxls.Var[Array],
        var_t_minus_1: jaxls.Var[Array],
        var_t_minus_2: jaxls.Var[Array],
        var_t_minus_3: jaxls.Var[Array],
        var_t_minus_4: jaxls.Var[Array],
        dt: float,
    ) -> Array:
        """Computes the residual penalizing velocity limit violations (5-point stencil)."""
        q_t   = vals[var_t]
        q_tm1 = vals[var_t_minus_1]
        q_tm2 = vals[var_t_minus_2]
        q_tm3 = vals[var_t_minus_3]
        q_tm4 = vals[var_t_minus_4]
         
        velocity = (25 * q_t - 48 * q_tm1 + 36 * q_tm2 - 16 * q_tm3 + 3 * q_tm4) / (12 * dt)
        return (velocity * 20.0).flatten()
    
    # Start / end velocity constraints
    @jaxls.Cost.create_factory(name="start_pose_constraint")
    def start_pose_cost(vals: jaxls.VarValues, var: jaxls.Var, s_var: StartConfigVar):
        s_val_fixed = jax.lax.stop_gradient(vals[s_var])
        return ((vals[var] - s_val_fixed) * 200.0).flatten()

    factors.extend(
        [
            start_pose_cost(robot.joint_var_cls(jnp.arange(0, 2)), start_var),
            terminal_velocity_limit_cost(
                robot.joint_var_cls(jnp.arange(timesteps-6, timesteps)),
                robot.joint_var_cls(jnp.arange(timesteps-7, timesteps - 1)),
                robot.joint_var_cls(jnp.arange(timesteps-8, timesteps - 2)),
                robot.joint_var_cls(jnp.arange(timesteps-9, timesteps - 3)),
                robot.joint_var_cls(jnp.arange(timesteps-10, timesteps - 4)),
                dt
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
                robot_batched,
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
                jnp.array([4.0 / timesteps])[None],
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
        target_pos_v: TargetPosVar,
    ):
        t_rel = vals[time_release].squeeze()
        target_pos = jax.lax.stop_gradient(vals[target_pos_v])
        
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
        q_release = q_release.reshape(-1)

        # --- 4. Interpolate Central Difference Velocity ---
        # Velocity at q_curr (t_floor) using central difference: (q_next - q_prev) / 2*dt
        q_dot_floor = (q_next - q_prev) / (2.0 * dt)

        # Velocity at q_next (t_ceil) using central difference: (q_next_next - q_curr) / 2*dt
        q_dot_ceil = (q_next_next - q_curr) / (2.0 * dt) 

        # Linear interpolation of the two velocity estimates
        q_dot_release = (1.0 - alpha) * q_dot_floor + alpha * q_dot_ceil
        q_dot_release = q_dot_release.reshape(-1)

        # Get starting position of launch
        q = robot.forward_kinematics(q_release)
        x0 = jnp.take(q, target_link_index, axis=-2)[..., 4:]
        
        # Get launch velocity
        jacobian = compute_ee_spatial_jacobian(robot, q_release, jnp.array([target_link_index]))
        twist = jacobian @ q_dot_release.squeeze()
        v0 = twist[:3][None]

        # Time duration of flight
        delta_t = vals[time_target].squeeze() - t_rel
        
        # Projectile motion: x(t) = x0 + v0*t + 0.5*g*t^2
        gravity_vec = jnp.array([0.0, 0.0, -g]) 
        
        pred_pos = x0 + v0 * delta_t + 0.5 * gravity_vec * (delta_t ** 2)
        
        # Target position (passed from outside, you might need to add it to args or closure)
        # For now assuming target_position is available in closure as `target_position`
        return ((pred_pos - target_pos) * 300.0).flatten()
        
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
        t_rel = vals[time_release].squeeze()
        
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
 

    factors.extend(
        [
            positive_time_cost(time_release_var, time_target_var),
            toss_target_cost(
                robot, 
                tuple(traj_vars[i] for i in range(timesteps)), 
                time_release_var, 
                time_target_var, 
                target_pos_var
            ),
            toss_alignment_cost(
                robot, 
                tuple(traj_vars[i] for i in range(timesteps)), 
                time_release_var
            )
        ]
    ) 
   
    return jaxls.LeastSquaresProblem(factors, [traj_vars, time_release_var, time_target_var, start_var, target_pos_var]).analyze()
