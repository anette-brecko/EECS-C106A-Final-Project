import os
import numpy as np
import jaxls

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
    import dill
    import dill.settings

    dill.settings['recurse'] = True
    # Generate hash for filename
    if not filename:
        raise ValueError("filename must be specified")

    # Make sure filepath exists
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    
    # Save file in filepath
    print(f"[IO] Saving problem to {filename}") 
    
    with open(filename, "wb") as file_handle:
        dill.dump(problem, file_handle)

def load_problem(
    filename: str | os.PathLike
):
    import dill

    # Make sure filename exists
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    
    with open(filename, "rb") as file_handle:
        return dill.load(file_handle)

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
