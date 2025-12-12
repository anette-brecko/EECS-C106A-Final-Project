import os
import xacro
import yourdfpy
import pyroki as pk
import tempfile
from ament_index_python.packages import get_package_share_directory, PackageNotFoundError
import subprocess

import jaxls
from jax import Array
import jax.numpy as jnp

class UR7eJointVar(
    jaxls.Var[Array],
    default_factory=lambda: jnp.zeros(6)
): ...

def load_xacro_robot(xacro_path, mappings) -> yourdfpy.URDF:
    """
    Parses XACRO and loads it into yourdfpy
    
    Returns:
        tuple: (pk.Robot, pk.collision.RobotCollision, yourdfpy.URDF)
    """
    if not os.path.exists(xacro_path):
        raise FileNotFoundError(f"XACRO file not found: {xacro_path}")

    try:
        doc = xacro.process_file(xacro_path, mappings=mappings)
        urdf_string = doc.toxml()
    except Exception as e:
        raise ValueError(f"XACRO processing failed: {e}")
    
    # 2. Save to a temp file
    # We save it in the same folder as the XACRO to keep relative paths valid.
    temp_dir = tempfile.gettempdir()
    temp_urdf_path = os.path.join(temp_dir, "temp_processed_robot.urdf")
    
    with open(temp_urdf_path, 'w') as f:
        f.write(urdf_string)

    try:
        urdf_model = yourdfpy.URDF.load(temp_urdf_path, filename_handler=ros_package_handler)
    finally:
        # 6. Cleanup
        if os.path.exists(temp_urdf_path):
            os.remove(temp_urdf_path)

    return urdf_model

def ros_package_handler(fname):
    """
    Resolves 'package://' paths using the ROS 2 ament index.
    """
    if fname.startswith("package://"):
        # Remove the 'package://' prefix (length 10)
        path_without_prefix = fname[10:]
        
        # Split the path into the package name and the relative file path
        # We limit the split to 1 to separate the package name from the rest
        try:
            package_name, relative_path = path_without_prefix.split('/', 1)
            
            # Resolve the package share directory
            package_path = get_package_share_directory(package_name)
            
            # return the absolute path
            return os.path.join(package_path, relative_path)
            
        except ValueError:
            print(f"Warning: Invalid package format: {fname}")
            return fname
        except PackageNotFoundError:
            print(f"Warning: ROS Package '{package_name}' not found.")
            return fname
            
    # Return the original filename if it doesn't start with package://
    return fname

def load_ur7e_with_gripper() -> yourdfpy.URDF:
    planning_dir = get_package_share_directory("planning")
    xacro_path = os.path.join(planning_dir, 'urdf', 'ur7e_with_gripper.urdf.xacro')

    urdf = load_xacro_robot(
        xacro_path, 
        mappings = {
            'ur_type': 'ur7e',
            'name': 'ur'
        }
    )
    
    # Make finger a fixed joint as we are not optimizing over it
    gripper_joint = urdf.actuated_joints[6] # Get the 7th joint
    print(f"Modifying joint in-memory: '{gripper_joint.name}' (Type: {gripper_joint.type} -> fixed)")
    gripper_joint.type = "fixed"

    # Remove and mimic joints 
    for j in urdf.robot.joints:
        if j.mimic is not None and j.mimic.joint == gripper_joint.name:
            print(f"  - Found dependent mimic joint: '{j.name}'")
            print(f"    -> Setting type to 'fixed' and removing mimic relationship.")
            j.type = "fixed"
            j.mimic = None  # Crucial: stop it from looking for the driver

    urdf._update_actuated_joints()
    return urdf

def load_ur7e() -> yourdfpy.URDF:
    ur_desc_path = get_package_share_directory('ur_description')
    xacro_path = os.path.join(ur_desc_path, 'urdf', 'ur.urdf.xacro')
    return load_xacro_robot(
            xacro_path,   
            mappings = {
                'ur_type': 'ur7e',
                'name': 'ur'
            }
        )

