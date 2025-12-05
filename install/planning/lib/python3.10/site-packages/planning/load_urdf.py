import os
import xacro
import yourdfpy
import pyroki as pk
import tempfile
from ament_index_python.packages import get_package_share_directory 
import subprocess

def load_xacro_robot():
    """
    Parses XACRO, loads it into yourdfpy, and creates PyROKI objects.
    
    Returns:
        tuple: (pk.Robot, pk.collision.RobotCollision, yourdfpy.URDF)
    """
    ur_desc_path = get_package_share_directory('ur_description')
    # xacro_path = os.path.join(ur_desc_path, 'urdf', 'ur.urdf.xacro')
    xacro_path = os.path.join('src', 'planning', 'planning', 'ur7e_with_tool.urdf.xacro')

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
        urdf_model = yourdfpy.URDF.load(temp_urdf_path, mesh_dir=ur_desc_path)
    finally:
        # 6. Cleanup
        if os.path.exists(temp_urdf_path):
            os.remove(temp_urdf_path)
            
    return urdf_model
