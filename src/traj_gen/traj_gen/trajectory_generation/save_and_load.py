import os
import numpy as np
import jaxls
import jax.numpy as jnp
from jax import Array

def save_trajectory(
            filename: str, 
            start_cfg: np.ndarray,
            target_pos: np.ndarray,
            trajectory: np.ndarray, 
            t_release: float,
            t_target: float,
            timesteps: int, 
            dt: float
    ):
        os.makedirs(os.path.dirname(os.path.abspath(filename)) or ".", exist_ok=True)
        # Save dictionary of arrays
        np.savez_compressed(
            filename,
            start_cfg=start_cfg,
            target_pos=target_pos,
            trajectory=trajectory,
            t_release=t_release,
            t_target=t_target,
            timesteps=np.array(timesteps),
            dt=np.array(dt) # Scalar wrapped in array
        )
        print(f"[IO] Trajectory saved to {filename}")
    
def load_trajectory(filename: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, int, float]:
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Trajectory file not found: {filename}")
        
        data = np.load(filename, allow_pickle=True)

        # Extract data with safety checks
        return data['start_cfg'], data['target_pos'], data['trajectory'], float(data['t_release']), float(data['t_target']), int(data['timesteps']), float(data['dt'])