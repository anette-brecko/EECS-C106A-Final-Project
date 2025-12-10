from typing import Sequence

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import pyroki as pk
from jax.typing import ArrayLike 

@jdc.jit
def solve_ik_with_collision(
    robot: pk.Robot,
    coll: pk.collision.RobotCollision,
    world_coll_list: Sequence[pk.collision.CollGeom],
    target_link_index: int,
    start_cfg: jnp.array,
    target_position: jnp.array,
    target_wxyz: jnp.array,
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
