from typing import Sequence, Callable, Optional

import jax
import jax.numpy as jnp
import jaxlie
import jaxls
import numpy as onp
import pyroki as pk
from jax import Array
from jax.typing import ArrayLike 
import jax_dataclasses as jdc
from .jacobian import compute_ee_spatial_jacobian

import os

#os.environ["JAX_PLATFORM_NAME"] = "gpu"
#jax.config.update('jax_num_cpu_devices', 24)
# jax.config.update("jax_logging_level", "WARNING")
# jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/jax"))
# jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")
# jax.config.update("jax_explain_cache_misses", True)

class TimeVar(
    jaxls.Var[Array],
    default_factory=lambda: jnp.array([1.0])
): ...


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
    timesteps: int
) -> list[tuple[onp.ndarray, float, float]]:
    traj_vars = robot.joint_var_cls(jnp.arange(timesteps))
    time_release_var = TimeVar(0)     
    time_target_var = TimeVar(1)

    __cost = _cost(problem)
    cost = lambda sample: __cost(
            jaxls.VarValues.make((
                traj_vars.with_value(jnp.array(sample[0])),
                time_release_var.with_value(jnp.array(sample[1])),
                time_target_var.with_value(jnp.array(sample[2]))
            ))
        )
    return sorted(samples, key=cost)[:num_samples_selected]

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
    problem: Optional[jaxls.AnalyzedLeastSquaresProblem] = None
) -> list[tuple[onp.ndarray, float, float]]:
    """Gives the optimized trajectories from best to worst"""
    target_link_idx = robot.links.names.index(target_link_name)
    start_cfg_jax = jnp.array(start_cfg)
    target_pos_jax = jnp.array(target_position)

    init_trajs, init_t_rels, init_t_tgts = zip(*initial_trajectories)

    # 2. Convert to JAX arrays
    #    Use [:, None] to ensure the time variables are shape (N, 1) instead of (N,)
    init_trajs = jnp.array(init_trajs)
    init_t_rels = jnp.array(init_t_rels)[:, None]
    init_t_tgts = jnp.array(init_t_tgts)[:, None]

    if problem is None:
        problem = _build_problem(
                robot, 
                robot_coll, 
                world_coll, 
                target_link_idx, 
                start_cfg_jax,
                target_pos_jax, 
                timesteps, 
                dt, 
                g
            ) 

    trajs, t_rels, t_tgts = _solve_static_trajopt(
        robot,
        robot_coll,
        world_coll,
        target_link_idx,
        start_cfg_jax,
        target_pos_jax,
        timesteps,
        dt,
        init_trajs,
        init_t_rels,
        init_t_tgts,
        g
    )

    trajs, t_rels, t_tgts = onp.array(trajs), onp.array(t_rels), onp.array(t_tgts)
    
    solved_samples = list(zip(trajs, t_rels.flatten(), t_tgts.flatten()))

    return choose_best_samples(
        solved_samples,
        len(solved_samples),
        robot,
        problem,
        timesteps
    )

def _build_problem(
    robot: pk.Robot,
    robot_coll: pk.collision.RobotCollision,
    world_coll: Sequence[pk.collision.CollGeom],
    target_link_index: jdc.Static[int],
    start_cfg: jnp.ndarray,
    target_pos: jnp.ndarray,
    timesteps: jdc.Static[int],
    dt: float,
    g: float = 9.81
) -> jaxls.AnalyzedLeastSquaresProblem:
    traj_vars = robot.joint_var_cls(jnp.arange(timesteps))
    time_release_var = TimeVar(0)     
    time_target_var = TimeVar(1)

    robot_batched = jax.tree.map(lambda x: x[None], robot)
    robot_coll_batched = jax.tree.map(lambda x: x[None], robot_coll)

    # Basic regularization / limit costs.
    factors: list[jaxls.Cost] = [
        pk.costs.limit_cost(
            robot_batched,
            traj_vars,
            jnp.array([100.0])[None],
        )
    ]

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

    # Start / end pose constraints.
    factors.append(
        pk.costs.self_collision_cost(
            robot_batched,
            robot_coll_batched,
            traj_vars,
            0.003,
            100.0 / timesteps,
        )
    )

    @jaxls.Cost.factory(name="terminal_velocity_limit_cost")
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
    
    @jaxls.Cost.factory(name="velocity_limit_cost_forward")
    def velocity_limit_cost_forward(
        vals: jaxls.VarValues,
        var_t: jaxls.Var[Array],
        var_t_plus_1: jaxls.Var[Array],
        var_t_plus_2: jaxls.Var[Array],
        dt: float,
    ) -> Array:
        """Computes the residual penalizing velocity limit violations (3-point forward stencil)."""
        q_t   = vals[var_t]
        q_tp1 = vals[var_t_plus_1]
        q_tp2 = vals[var_t_plus_2]
        
        velocity = (-3 * q_t + 4 * q_tp1 - 1 * q_tp2) / (2 * dt)
        
        return (velocity * 20.0).flatten()
    factors.extend(
        [
            jaxls.Cost(
                lambda vals, var: ((vals[var] - start_cfg) * 100.0).flatten(),
                (robot.joint_var_cls(jnp.arange(0, 1)),),
                name="start_pose_constraint",
            ),
            terminal_velocity_limit_cost(
                robot.joint_var_cls(jnp.arange(timesteps-5, timesteps)),
                robot.joint_var_cls(jnp.arange(timesteps-6, timesteps - 1)),
                robot.joint_var_cls(jnp.arange(timesteps-7, timesteps - 2)),
                robot.joint_var_cls(jnp.arange(timesteps-8, timesteps - 3)),
                robot.joint_var_cls(jnp.arange(timesteps-9, timesteps - 4)),
                dt
            ),
            velocity_limit_cost_forward(
                robot.joint_var_cls(jnp.arange(0, 3)),
                robot.joint_var_cls(jnp.arange(1, 4)),
                robot.joint_var_cls(jnp.arange(2, 5)),
                dt
            ),
        ]
    )

    # Velocity / acceleration / jerk minimization.
    factors.extend(
        [
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
                jnp.array([1.0 / timesteps])[None],
            ),
        ]
    )

     # Trajectory tossing
    @jaxls.Cost.factory(name="toss_target_cost")
    def toss_target_cost(
        vals: jaxls.VarValues,
        traj_vars_tuple: tuple[jaxls.Var[jax.Array], ...],
        time_release: TimeVar,
        time_target: TimeVar,
    ):
        t_rel = vals[time_release].squeeze()
        t_tgt = vals[time_target].squeeze()

        qs_full = jnp.stack([vals[v] for v in traj_vars_tuple])

        delta_t = t_tgt - t_rel
        
        max_idx = qs_full.shape[0] - 2
        idx_float = t_rel / dt
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
        q_dot_ceil = (q_next_next - q_curr) / (2.0 * dt) 
        q_dot_release = (1.0 - alpha) * q_dot_floor + alpha * q_dot_ceil
        
        q = robot.forward_kinematics(q_release)
        x0 = q[..., target_link_index, 4:]
        
        jacobian = compute_ee_spatial_jacobian(robot, q_release, jnp.array([target_link_index]))
        twist = jacobian @ q_dot_release
        v0 = twist[:3][None]
        
        gravity_vec = jnp.array([0.0, 0.0, -g]) 
        delta_t = t_tgt - t_rel
        pred_pos = x0 + v0 * delta_t + 0.5 * gravity_vec * (delta_t ** 2)

        pos_error = ((pred_pos - target_pos) * 300.0).flatten()

        R_ee = jaxlie.SE3(jnp.take(q, target_link_index, axis=-2)[..., :4])
        y_ee = R_ee.rotation() @ jnp.array([0.0, 1.0, 0.0])
        
        v_norm = jnp.linalg.norm(v0) + 1e-6
        v_dir = v0 / v_norm

        align_error = ((jnp.cross(-y_ee, v_dir)) * 40.0).flatten()
        return jnp.concatenate([pos_error, align_error]) 
           
    @jaxls.Cost.factory(name="positive_time_cost")
    def positive_time_cost(
        vals: jaxls.VarValues,
        time_release: TimeVar,
        time_target: TimeVar,
    ):
        t_rel = vals[time_release]
        t_tgt = vals[time_target]
        total_duration = timesteps * dt

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
        return  jnp.append(err * 100.0,  err_from_intended * 0.001)
       
    @jaxls.Cost.factory(name="ensure_acceleration_at_release")
    def ensure_acceleration_at_release(
        vals: jaxls.VarValues,
        var_traj: jaxls.Var[Array], # The full joint trajectory variable
        var_time_release: jaxls.Var[Array], # The scalar release time
        dt: float
    ) -> Array:
        # 1. Get the trajectory and release time
        traj = vals[var_traj] # Shape (T, 6)
        t_rel = vals[var_time_release].squeeze()
        
        # 2. Convert continuous time t_rel to an integer index
        # (We use a soft approximation or just floor for simplicity in logic)
        # For a robust cost, we usually pick the index closest to t_rel
        idx = (t_rel / dt).astype(int)
        
        # Safety clip
        idx = jnp.clip(idx, 1, traj.shape[0] - 2)
        
        # 3. Calculate Velocity and Acceleration at that index
        q_prev = traj[idx - 1]
        q_curr = traj[idx]
        q_next = traj[idx + 1]
        
        vel_prev = (q_curr - q_prev)
        vel_next = (q_next - q_curr)
        
        # 4. Acceleration proxy: difference in velocities
        # We want vel_next >= vel_prev (speeding up)
        # So we punish if vel_next < vel_prev
        # Cost = ReLU(vel_prev_mag - vel_next_mag)
        
        speed_prev = jnp.linalg.norm(vel_prev)
        speed_next = jnp.linalg.norm(vel_next)
        
        # If we are slowing down, this value becomes positive (a cost).
        # If we are speeding up, it is zero.
        deceleration_penalty = jax.nn.softplus(speed_prev - speed_next)
        
        return (deceleration_penalty * 30.0).flatten()
    
    @jaxls.Cost.factory(name="maximize_release_velocity")
    def maximize_release_velocity(vals, var_traj, var_t_rel, dt):
        traj = vals[var_traj]
        t_rel = vals[var_t_rel].squeeze()
        
        # 2. Calculate Index from time (robustly)
        idx_float = t_rel / dt
        max_idx = traj.shape[0] - 2
        
        # Use simple rounding or floor to pick the step. 
        # (We use clip to ensure we don't access out-of-bounds memory)
        idx = jnp.clip(jnp.floor(idx_float).astype(int), 1, max_idx)
        
        # 3. Calculate Velocity at that index
        # v = (q_current - q_prev) / dt
        q_curr = traj[idx]
        q_prev = traj[idx-1]
        v_release = (q_curr - q_prev) / dt
        
        # 4. Compute speed (scalar)
        speed = jnp.linalg.norm(v_release)
        
        # 5. Return as 1D Array
        # We want to MAXIMIZE speed, so we MINIMIZE negative speed.
        # .flatten() turns the scalar into a shape (1,) array.
        return (-1.0 * speed * 0.001).flatten()
    
    factors.extend(
        [
            ensure_acceleration_at_release(traj_vars, time_release_var, dt),
            maximize_release_velocity(traj_vars, time_release_var, dt),
            positive_time_cost(time_release_var, time_target_var),
            toss_target_cost(
                tuple(traj_vars[i] for i in range(timesteps)), 
                time_release_var, 
                time_target_var
            )
        ]
    ) 
   
    return jaxls.LeastSquaresProblem(factors, [traj_vars, time_release_var, time_target_var]).analyze()

@jdc.jit
def _solve_static_trajopt(
    robot: pk.Robot,
    robot_coll: pk.collision.RobotCollision,
    world_coll: Sequence[pk.collision.CollGeom],
    target_link_idx: jdc.Static[int],
    start_cfg: jnp.ndarray,
    target_pos: jnp.ndarray,
    timesteps: jdc.Static[int],
    dt: float,
    init_trajs: jnp.ndarray, # (N, T, D)
    init_t_rels: jnp.ndarray, # (N, 1)
    init_t_tgts: jnp.ndarray, # (N, 1)
    g: float = 9.81
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: 
    traj_vars = robot.joint_var_cls(jnp.arange(timesteps))
    time_release_var = TimeVar(0)     
    time_target_var = TimeVar(1)

    problem = _build_problem(
        robot,
        robot_coll,
        world_coll,
        target_link_idx,
        start_cfg,
        target_pos,
        timesteps,
        dt,
        g
    )

    def solve_single(init_traj, init_t_rel, init_t_tgt):
        # Ensure scalar inputs are shaped correctly for variables (1D array)
        init_t_rel_arr = jnp.atleast_1d(init_t_rel)
        init_t_tgt_arr = jnp.atleast_1d(init_t_tgt)

        solution = problem.solve(
            initial_vals=jaxls.VarValues.make((
                traj_vars.with_value(init_traj),
                time_release_var.with_value(init_t_rel_arr),
                time_target_var.with_value(init_t_tgt_arr)
            ))
        )
        return solution[traj_vars], solution[time_release_var], solution[time_target_var]

    # 3. Vectorize the solver over the batch dimension (N)
    # init_trajs: (N, T, D) -> maps over dim 0
    # init_t_rels: (N, 1)   -> maps over dim 0
    # init_t_tgts: (N, 1)   -> maps over dim 0
    solved_trajs, solved_t_rels, solved_t_tgts = jax.vmap(solve_single)(
        init_trajs, 
        init_t_rels, 
        init_t_tgts
    )

    return solved_trajs, solved_t_rels, solved_t_tgts
