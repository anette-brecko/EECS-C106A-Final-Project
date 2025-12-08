import jax
import jax.numpy as jnp
import jaxlie
import pyroki as pk
from jax.typing import ArrayLike 

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
