import os
import xacro
import yourdfpy
import pyroki as pk

def load_xacro_robot(xacro_path: str, mappings: dict = None):
    """
    Parses XACRO, loads it into yourdfpy, and creates PyROKI objects.
    
    Returns:
        tuple: (pk.Robot, pk.collision.RobotCollision, yourdfpy.URDF)
    """
    if not os.path.exists(xacro_path):
        raise FileNotFoundError(f"XACRO file not found: {xacro_path}")

    # 1. Process XACRO -> XML String
    try:
        doc = xacro.process_file(xacro_path, mappings=mappings)
        urdf_string = doc.toxml()
    except Exception as e:
        raise ValueError(f"XACRO processing failed: {e}")

    # 2. Save to a temp file (needed for yourdfpy to resolve mesh paths)
    # We save it in the same folder as the XACRO to keep relative paths valid.
    directory = os.path.dirname(os.path.abspath(xacro_path))
    temp_urdf_path = os.path.join(directory, "temp_processed_robot.urdf")
    
    with open(temp_urdf_path, 'w') as f:
        f.write(urdf_string)

    try:
        urdf_model = yourdfpy.URDF.load(temp_urdf_path)
    finally:
        # 6. Cleanup
        if os.path.exists(temp_urdf_path):
            os.remove(temp_urdf_path)
            
    return urdf_model
