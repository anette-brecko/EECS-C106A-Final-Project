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
    robot_max_reach: float,
    max_vel: float,
    num_samples: int = 100,
    g: float = 9.81,
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
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

    for world_coll_obj in world_coll:
        factors.append(
            jaxls.Cost(
                compute_world_coll_residual,
                (
                    robot,
                    robot_coll,
                    jax.tree.map(lambda x: x[None], world_coll_obj),
                    robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
                    robot.joint_var_cls(jnp.arange(1, timesteps)),
                ),
                name="World Collision (sweep)",
            )
        )

    factors.append(
        pk.costs.self_collision_cost(
            robot,
            robot_coll, traj_vars,
            0.005,
            50.0,
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

        q_curr = qs_full[idx_floor]
        q_next = qs_full[idx_ceil]
        
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
        min_flight_time = 0.3 

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

        q_curr = qs_full[idx_floor]
        q_next = qs_full[idx_ceil]
        
        # 4. Interpolate Joint Position and Velocity
        # Linear Interpolation for q
        q_release = (1.0 - alpha) * q_curr + alpha * q_next
        
        # Joint Velocity is the slope between the two points
        q_dot_release = (q_next - q_curr) / dt

        # Get starting position of launch
        q = robot.forward_kinematics(q_release)
        R_ee = jaxlie.SE3(jnp.take(q, target_link_index, axis=-2)[..., :4])
        z_ee = R_ee.rotation() @ jnp.array([0.0, 0.0, 1.0])
         
        # Get launch velocity
        jacobian = compute_ee_spatial_jacobian(robot, q_release, jnp.array([target_link_index]))
        twist = jacobian @ q_dot_release.squeeze()
        v0 = twist[:3][None]

        v_norm = jnp.linalg.norm(v0) + 1e-6
        v_dir = v0 / v_norm

        return ((z_ee - v_dir) * 40.0).flatten()
 

    factors.extend(
        [
            positive_time_cost(time_release, time_target),
            toss_target_cost(robot, tuple(traj_vars[i] for i in range(timesteps)), time_release, time_target),
            toss_alignment_cost(robot, tuple(traj_vars[i] for i in range(timesteps)), time_release)
        ]
    )
   
    print(f'Generating {num_samples} samples')
    samples = generate_samples(
        robot_unbatched, 
        robot_coll_unbatched,
        world_coll,
        target_link_index,
        start_cfg, 
        target_position, 
        timesteps,
        dt,
        g,
        max_vel,
        robot_max_reach,
        num_samples
    )

    problem = jaxls.LeastSquaresProblem(factors, [traj_vars, time_release, time_target]).analyze()
    
    def cost(sample):
        vals = jaxls.VarValues.make((traj_vars.with_value(sample[0]),
                                     time_release.with_value(jnp.array([sample[1]])),
                                    time_target.with_value(jnp.array([sample[2]]))))
        residual = problem.compute_residual_vector(vals)
        return jnp.dot(residual, residual)

   # print("Sorting samples")
   # best_samples = sorted(samples, key=cost)[:5]

   # print("Refining samples")
   # def refine_sample(sample):
   #     init_traj, init_t_rel, init_t_target = sample
   #     solution = (
   #         problem
   #         .solve(
   #             initial_vals=
   #             jaxls.VarValues.make((traj_vars.with_value(init_traj), 
   #                                   time_release.with_value(jnp.array([init_t_rel])), 
   #                                   time_target.with_value(jnp.array([init_t_target])))),
   #         )
   #      )
   #     return (solution[traj_vars], solution[time_release], solution[time_target])


   # refined_samples = [refine_sample(sample) for sample in best_samples]

   # traj, t_rel, t_target = min(refined_samples, key=cost)
   # return onp.array(traj), onp.array(t_rel), onp.array(t_target)


    init_traj, init_t_rel, init_t_target = min(samples, key=cost)

    # return onp.array(init_traj), onp.array(init_t_rel), onp.array(init_t_target)

    # 4. Solve the optimization problem.
    solution = (
        problem
        .solve(
            initial_vals=
            jaxls.VarValues.make((traj_vars.with_value(init_traj), 
                                  time_release.with_value(jnp.array([init_t_rel])), 
                                  time_target.with_value(jnp.array([init_t_target])))),
        )
    )
    return onp.array(solution[traj_vars]), onp.array(solution[time_release]), onp.array(solution[time_target])

def quadratic_bezier_trajectory(p0, p1, p2):
    return lambda t: p0 * (1 - t) ** 2 + 2 * p1 * t * (1 - t) + p2 * t ** 2

def cubic_bezier_trajectory(p0, p1, p2, p3):
    return lambda t: p0 * (1 - t) ** 3 + 3 * p1 * t * (1 - t) ** 2 + 3 * p2 * t ** 2 * (1 - t) + p3 * t ** 3

def generate_samples(
    robot: pk.Robot,
    robot_coll: pk.collision.RobotCollision,
    world_coll: Sequence[pk.collision.CollGeom],
    target_link_index: int,
    q_0: ArrayLike,
    target_position: ArrayLike,
    timesteps: int,
    dt: float,
    g: float,
    max_vel: float,
    robot_max_reach: float,
    num_samples: int = 100,
) -> list[tuple[Array, float, float]]:
    """Generates a list of samples for the trajectory optimization problem."""
    samples = []

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
        # z-axis should be pointing in v_rel direction
        z_axis = v_rel / onp.linalg.norm(v_rel)

        # x-axis away from origin
        x_axis = x_rel / onp.linalg.norm(x_rel)
        
        # x-axis as cross product of the two
        y_axis = onp.cross(z_axis, x_axis)

        rot = jaxlie.SO3.from_matrix(onp.column_stack((x_axis, y_axis, z_axis)))

        # Generate random orientation pointing in the z-direction
        z_rot = jaxlie.SO3.from_z_radians(onp.random.uniform(-0.3 * onp.pi, 0.3 * onp.pi))

        random_rot = rot #@ z_rot
        return random_rot.wxyz

    while len(samples) < num_samples:
        if (len(samples) % 50) == 0: print(len(samples))

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
        traj_points = jnp.stack(traj_points)
        samples.append((traj_points, t_rel, t_rel + dt_air))
    
    return samples

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

def compute_ee_spatial_jacobian(
    robot: pk.Robot, 
    q: jax.Array,
    target_link_index: jax.Array,
) -> jax.Array:
    """
    Computes the 6xN Linear Jacobian (World Frame) assuming a serial chain
    where all actuated joints contribute to the target motion (ie the end effector).
    """
    # Get transforms for every joint frame and link frame
    Ts_world_joint = robot._forward_kinematics_joints(q)
    Ts_world_link = robot._link_poses_from_joint_poses(Ts_world_joint)
    
    # Cast to SE3 for convenient operations
    target_pose_vec = jnp.take(Ts_world_link, target_link_index, axis=-2)
    
    # Cast to SE3 objects
    Ts_world_joint = jaxlie.SE3(Ts_world_joint)
    T_world_ee = jaxlie.SE3(target_pose_vec)
    joint_twists = robot.joints.twists 

    # Rotation from local joint frame to world frame
    R_world_joint = Ts_world_joint.rotation()

    omega_local = joint_twists[:, 3:] 
    vel_local = joint_twists[:, :3]

    # Rotate axes into World Frame
    omega_wrt_world = R_world_joint @ omega_local
    vel_wrt_world = R_world_joint @ vel_local

    # Calculate Linear Velocity Contribution at Target Point
    # formula: v_i = omega_i x (p_target - p_joint_i) + v_init_i
    p_diff = T_world_ee.translation() - Ts_world_joint.translation()
    
    # Cross product broadcast over all joints
    J_linear = jnp.cross(omega_wrt_world, p_diff) + vel_wrt_world
    J_angular = omega_wrt_world
    jacobian = jnp.concatenate([J_linear, J_angular], axis=-1).squeeze()

    # 6. Filter for Actuated Joints
    dest_indices = robot.joints.actuated_indices 
    
    # Create Mask (1, 10) for broadcasting
    valid_mask = (dest_indices != -1)[None, :]
    
    # Apply mask (6, 10) * (1, 10) -> (6, 10)
    masked_J = jacobian.T * valid_mask
    
    # 7. Scatter Add
    # Map from 10 columns -> 6 columns
    safe_indices = jnp.maximum(dest_indices, 0)
    num_act = robot.joints.num_actuated_joints
    
    # Initialize (6, N)
    jac_actuated = jnp.zeros((6, num_act)).at[:, safe_indices].add(masked_J)
    return jac_actuated

@jdc.jit
def solve_ik_with_collision(
    robot: pk.Robot,
    coll: pk.collision.RobotCollision,
    world_coll_list: Sequence[pk.collision.CollGeom],
    target_link_index: int,
    start_cfg: ArrayLike,
    target_position: jax.Array,
    target_wxyz: jax.Array,
    pos_weight: float = 5.0,
    rot_weight: float = 1.0,
    self_coll_weight: float = 0.02
) -> jax.Array:
    """Solves the basic IK problem with collision avoidance. Returns joint configuration."""
    joint_var = robot.joint_var_cls(0)

    # Weights and margins defined directly in factors.
    factors = [
        pk.costs.pose_cost(
            robot,
            joint_var,
            jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(jnp.array(target_wxyz)), jnp.array(target_position)
            ),
            jnp.array(target_link_index),
            jnp.array([pos_weight] * 3),
            jnp.array([rot_weight] * 3)
        ),
    ]
    factors.extend(
        [
            pk.costs.limit_cost(
                robot,
                joint_var,
                jnp.array(100.0),
            ),
            pk.costs.rest_cost(
                joint_var,
                jnp.array(joint_var.default_factory()),
                jnp.array(0.001),
            ),
            pk.costs.self_collision_cost(
                robot,
                coll,
                joint_var,
                0.02,
                self_coll_weight,
            ),
            pk.costs.manipulability_cost(
                robot,
                joint_var,
                jnp.array([target_link_index]),
                jnp.array([0.001])
             )
,
        ]
    )
    factors.extend(
        [
            pk.costs.world_collision_cost(
                robot,
                coll,
                joint_var,
                world_coll,
                0.05,
                10.0,
            )
            for world_coll in world_coll_list
        ]
    )

    # Small cost to encourage the start + end configs to be close to each other.
    @jaxls.Cost.create_factory(name="JointSimilarityCost")
    def joint_similarity_cost(vals, var):
        return ((start_cfg - vals[var]) * 0.10).flatten()

    factors.append(joint_similarity_cost(joint_var))

    sol = jaxls.LeastSquaresProblem(factors, [joint_var]).analyze().solve(verbose=False)
    return sol[joint_var]

@jdc.jit
def solve_single_ik_with_collision(
    robot: pk.Robot,
    coll: pk.collision.RobotCollision,
    world_coll_list: Sequence[pk.collision.CollGeom],
    target_link_index: int,
    target_position: ArrayLike,
    target_wxyz: ArrayLike,
) -> jax.Array:
    """Solves the basic IK problem with collision avoidance. Returns joint configuration."""
    joint_var = robot.joint_var_cls(0)

    # Weights and margins defined directly in factors.
    factors = [
        pk.costs.pose_cost(
            robot,
            joint_var,
            jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(jnp.array(target_wxyz)), jnp.array(target_position)
            ),
            jnp.array(target_link_index),
            jnp.array([5.0] * 3),
            jnp.array([1.0] * 3),
        ),
    ]
    factors.extend(
        [
            pk.costs.limit_cost(
                robot,
                joint_var,
                jnp.array(100.0),
            ),
            pk.costs.rest_cost(
                joint_var,
                jnp.array(joint_var.default_factory()),
                jnp.array(0.001),
            ),
            pk.costs.self_collision_cost(
                robot,
                coll,
                joint_var,
                0.02,
                5.0,
            ),
            pk.costs.manipulability_cost(
                robot,
                joint_var,
                jnp.array([target_link_index]),
                jnp.array([0.001])
            )
        ]
    )
    factors.extend(
        [
            pk.costs.world_collision_cost(
                robot,
                coll,
                joint_var,
                world_coll,
                0.05,
                10.0,
            )
            for world_coll in world_coll_list
        ]
    )

    sol = jaxls.LeastSquaresProblem(factors, [joint_var]).analyze().solve(verbose=False)
    return sol[joint_var]
