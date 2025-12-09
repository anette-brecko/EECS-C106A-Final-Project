import os
import numpy as np
import jaxls
import jax.numpy as jnp
from jax import Array
import contextlib
import dataclasses
# --- 1. Define Factory Functions (Top Level) ---
def _default_time():
    return jnp.array([1.0])

def _default_start_cfg():
    return jnp.zeros(6)

def _default_target_pos():
    return jnp.zeros(3)

# --- 2. Define Classes using Named Functions ---
# By using named functions, cloudpickle can save these classes by reference.
class TimeVar(jaxls.Var[Array], default_factory=_default_time): ...
class StartConfigVar(jaxls.Var[Array], default_factory=_default_start_cfg): ...
class TargetPosVar(jaxls.Var[Array], default_factory=_default_target_pos): ...

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

def save_problem(
    problem: jaxls.AnalyzedLeastSquaresProblem,
    filename: str | os.PathLike
):
    import cloudpickle
    # Generate hash for filename
    if not filename:
        raise ValueError("filename must be specified")

    # Make sure filepath exists
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    
    # Save file in filepath
    print(f"[IO] Saving problem to {filename}") 
    
    with open(filename, "wb") as file_handle:
        cloudpickle.dump(problem, file_handle)

def load_problem(filename):
    import cloudpickle
    import jaxls
    import dataclasses
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    
    try:
        with _jaxls_loading_patch(), open(filename, "rb") as f:
            result = cloudpickle.load(f)
        return result
    finally:
        # Restore originals
        jaxls.Var.__init_subclass__ = original_init_subclass
        dataclasses.dataclass = original_dataclass

def save_robot(robot, filename: str):
    """Saves the robot instance using cloudpickle."""
    import cloudpickle
    # Make sure filepath exists
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    
    print(f"[IO] Saving robot to {filename}")
    with open(filename, "wb") as f:
        cloudpickle.dump(robot, f)

def load_problem(filename):
    import os
    import cloudpickle
    import jaxls
    import dataclasses

    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    # --- 1. SETUP: Capture original values safely ---
    raw_attr = jaxls.Var.__init_subclass__
    # This handles both bound methods and plain functions to avoid AttributeError
    original_init_subclass = getattr(raw_attr, "__func__", raw_attr)
    
    original_dataclass = dataclasses.dataclass

    # --- 2. PATCH: Define the temporary replacements ---
    @classmethod  
    def patched_init_subclass(cls, default=None, default_factory=None, **kwargs):
        if default is None and default_factory is None:
            default = 0.0 
        original_init_subclass(cls, default=default, default_factory=default_factory, **kwargs)

    def patched_dataclass(cls=None, **kwargs):
        def wrap(cls):
            if hasattr(cls, '__setattr__') and issubclass(cls, jaxls.Var):
                return cls
            return original_dataclass(cls, **kwargs)
        if cls is None: return wrap
        return wrap(cls)

    # --- 3. APPLY & LOAD ---
    jaxls.Var.__init_subclass__ = patched_init_subclass
    dataclasses.dataclass = patched_dataclass

    try:
        print(f"[IO] Loading problem from {filename}")
        with open(filename, "rb") as f:
            return cloudpickle.load(f)
    finally:
        # --- 4. CLEANUP: Restore originals ---
        jaxls.Var.__init_subclass__ = original_init_subclass
        dataclasses.dataclass = original_dataclass

if __name__ == "__main__":
    import jax
    import jaxls
    import jaxlie

    pose_vars = [jaxls.SE2Var(0), jaxls.SE2Var(1)]

    # Defining cost types.

    @jaxls.Cost.create_factory
    def prior_cost(
        vals: jaxls.VarValues, var: jaxls.SE2Var, target: jaxlie.SE2
    ) -> jax.Array:
        """Prior cost for a pose variable. Penalizes deviations from the target"""
        return (vals[var] @ target.inverse()).log()

    @jaxls.Cost.create_factory
    def between_cost(
        vals: jaxls.VarValues, delta: jaxlie.SE2, var0: jaxls.SE2Var, var1: jaxls.SE2Var
    ) -> jax.Array:
        """'Between' cost for two pose variables. Penalizes deviations from the delta."""
        return ((vals[var0].inverse() @ vals[var1]) @ delta.inverse()).log()

    # Instantiating costs.
    costs = [
        prior_cost(pose_vars[0], jaxlie.SE2.from_xy_theta(0.0, 0.0, 0.0)),
        prior_cost(pose_vars[1], jaxlie.SE2.from_xy_theta(2.0, 0.0, 0.0)),
        between_cost(jaxlie.SE2.from_xy_theta(1.0, 0.0, 0.0), pose_vars[0], pose_vars[1]),
    ]

    problem = jaxls.LeastSquaresProblem(costs, pose_vars).analyze()

    save_problem(problem, '/tmp/test.pkl')
    del problem
    problem2 = load_problem('/tmp/test.pkl')
    solution = problem2.solve()
    print("Pose 0", solution[pose_vars[0]])
    print("Pose 1", solution[pose_vars[1]])
