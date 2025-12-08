import os
import xacro
import yourdfpy
import pyroki as pk
import tempfile
from ament_index_python.packages import get_package_share_directory, PackageNotFoundError
import subprocess


def load_xacro_robot():
    """
    Parses XACRO, loads it into yourdfpy, and creates PyROKI objects.
    
    Returns:
        tuple: (pk.Robot, pk.collision.RobotCollision, yourdfpy.URDF)
    """
    ur_desc_path = get_package_share_directory('ur_description')
    # robotiq_desc_path = get_package_share_directory('robotiq_hande_description')
    planning_dir = get_package_share_directory("planning")

    xacro_path = os.path.join(ur_desc_path, 'urdf', 'ur.urdf.xacro')
    #xacro_path = os.path.join(planning_dir, 'urdf', 'ur7e_with_pedestal_and_gripper.urdf.xacro')
    xacro_path = os.path.join(planning_dir, 'urdf', 'ur7e_with_gripper.urdf.xacro')


    if not os.path.exists(xacro_path):
        raise FileNotFoundError(f"XACRO file not found: {xacro_path}")


    try:
        mappings = {
            'ur_type': 'ur7e',
            'name': 'ur',
            'tool_length_m': '0.10'
        }
        doc = xacro.process_file(xacro_path, mappings=mappings)
        urdf_string = doc.toxml()
    except Exception as e:
        raise ValueError(f"XACRO processing failed: {e}")
    
    # 2. Save to a temp file (needed for yourdfpy to resolve mesh paths)
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

    # Make finger a fixed joint
    gripper_joint = urdf_model.actuated_joints[6] # Get the 7th joint
    print(f"Modifying joint in-memory: '{gripper_joint.name}' (Type: {gripper_joint.type} -> fixed)")
    
    # Setting this to 'fixed' effectively removes it from the actuated list
    # that Pyroki generates in _robot_urdf_parser.py
    gripper_joint.type = "fixed"

    for j in urdf_model.robot.joints:
        if j.mimic is not None and j.mimic.joint == gripper_joint.name:
            print(f"  - Found dependent mimic joint: '{j.name}'")
            print(f"    -> Setting type to 'fixed' and removing mimic relationship.")
            j.type = "fixed"
            j.mimic = None  # Crucial: stop it from looking for the driver

    urdf_model._update_actuated_joints()
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