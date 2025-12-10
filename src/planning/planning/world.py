import numpy as np
import jax.numpy as jnp

import pyroki as pk
import trimesh
import viser
from viser.extras import ViserUrdf

import time
from typing import Callable

from .trajectory_generation.gen_traj import compute_ee_spatial_jacobian

class World:
    def __init__(self, robot, urdf, target_link_name):
        # Visualize!
        self.server = viser.ViserServer()
        self.urdf_vis = ViserUrdf(self.server, urdf)
        self.robot = robot
        self.target_link_name = target_link_name

        # Table parameters
        self.table_width = 0.6 # Along x-axis
        self.table_length = 2.0 # Along y-axis
        self.table_height = 0.7366
        self.table_offset = np.array([0.0, 0.4, -0.18]) # As measured by center of the edge of the table

        # Floor parameters
        self.min_height = -0.6 # Won't allow robot to go below this height

        # Wall parameters
        self.wall_distance = -.7874 # in y direction (wall is behind robot)

        # Pillar parameters
        self.pillar_length = 0.28
        self.pillar_height = 0.91
        self.visualize_world()
        self._visualize_joints(np.array(robot.joint_var_cls.default_factory()))

    def visualize_all(self, start_cfg, target_pos, traj, t_release, t_target, timesteps, dt):
        ball_traj = self._ball_trajectory(traj, t_release, dt)
        self.visualize_world()
        self.visualize_tf(start_cfg, target_pos)
        self.visualize_ball_trajectory(ball_traj, t_release, t_target, 0.2, 30)
        self.visualize_ee_waypoints(traj)
        return self.animate(ball_traj, traj, t_release, timesteps, dt)

    def visualize_world(self):
        self.server.scene.add_grid("/grid", width=2, height=2, cell_size=0.1)
        
        # Visualize floor
        self.server.scene.add_mesh_trimesh(
            "floor",
            trimesh.creation.box(
                extents=(2.0, 2.0, 0.001),
                transform=trimesh.transformations.translation_matrix(
                    np.array([0, 0.0, self.min_height])
                ),
            )
        )

        # Visualize wall
        self.server.scene.add_mesh_trimesh(
            "wall",
            trimesh.creation.box(
                extents=(2.0, 0.1, 2.0),
                transform=trimesh.transformations.translation_matrix(
                    np.array([0.0, self.wall_distance, 0.0])
                ),
            ),
        )

        # Visualize table
        self.server.scene.add_mesh_trimesh(
            "table",
            trimesh.creation.box(
                extents=(self.table_length, self.table_width, self.table_height),
                transform=trimesh.transformations.translation_matrix(
                    np.array([0, self.table_width / 2.0, -self.table_height / 2]) + self.table_offset
                ),
            ),
        )

        # TODO: Visualize table space

        # Visualize pillar
        self.server.scene.add_mesh_trimesh(
            "pillar",
            trimesh.creation.box(
                extents=(self.pillar_length, self.pillar_length, self.pillar_height),
                transform=trimesh.transformations.translation_matrix(
                    np.array([0, 0.0, -self.pillar_height / 2])
                ),
            ),
        )
                                   
        
    def visualize_tf(self, start_cfg, target_pos):
        start_pos, start_wxyz = self._joints_to_pos_wxyz(start_cfg)
        self.server.scene.add_frame(
            "/start",
            position=start_pos,
            wxyz=start_wxyz,
            axes_length=0.05,
            axes_radius=0.01,
        )
 
        self.server.scene.add_icosphere(
            "/target",
            position=target_pos,
            radius=0.025,
            color=(0, 0, 255)
        )

    def visualize_ball_trajectory(self, ball_traj, t_release, t_target, t_after, num_points):
        #test_ball_trajectory = lambda t: np.array([1.0, 0.0, 5.0]) * (t - t_release) - 0.5 * np.array([0.0, 0.0, 9.81]) * ((t - t_release) ** 2)
        #ball_traj = [test_ball_trajectory(t) for t in times] 
        times = np.linspace(t_release, t_target + t_after, num_points)

        ball_traj = [ball_traj(t) for t in times]
        self.server.scene.add_spline_catmull_rom(
            "/ball_traj",
            points=np.array(ball_traj),
            line_width=2.0,
            segments = 30,
        )

    def visualize_ee_waypoints(self, traj):
        traj_pos = [self._joints_to_pos_wxyz(q)[0] for q in traj]
        self.server.scene.add_point_cloud(
            "/traj_points",
            points=np.array(traj_pos),
            point_size=0.005,
            colors=(0, 0, 255)
        )
        
    def animate(self, ball_traj, robot_traj, t_release, timesteps, dt):
        ee_axis = self.server.scene.add_frame(
            "/ee",
            position=np.zeros(3),
            wxyz=np.zeros(4),
            axes_length=0.05,
            axes_radius=0.01,
        )

        ball = self.server.scene.add_icosphere(
                "/ball",
                position=np.zeros(3),
                radius = 0.02,
                color=(255, 0, 0),
        )

        self.server.gui.reset()
        slider = self.server.gui.add_slider(
            "Timestep", min=0, max=timesteps - 1, step=1, initial_value=0
        )
        playing = self.server.gui.add_checkbox("Playing", initial_value=True)
        execute = self.server.gui.add_button("Execute")
        next_button = self.server.gui.add_button("Next")
        regenerate = self.server.gui.add_button("Regenerate")


        while True:
            if playing.value:
                slider.value = (slider.value + 1) % timesteps

            if regenerate.value:
                return "regenerate"
            elif next_button.value:
                return "next"
            elif execute.value:
                return "execute"
            
            # Update robot
            self.urdf_vis.update_cfg(robot_traj[slider.value])

            # Update ee axis
            ee_axis.position, ee_axis.wxyz = self._joints_to_pos_wxyz(robot_traj[slider.value])
            
            # Update ball
            if slider.value * dt > t_release:
                ball.visible = True
                ball.position = ball_traj(slider.value * dt)
            else: 
                ball.visible = False

            time.sleep(dt)

    def _joints_to_pos_wxyz(self, q):
        """Takes joint positions and return position and orientation (wxyz)"""
        pos = self.robot.forward_kinematics(jnp.array(q))
        twist = np.array(jnp.take(pos, self.robot.links.names.index(self.target_link_name), axis=-2))
        return twist[..., 4:], twist[..., :4]

    def _visualize_joints(self, q):
        self.urdf_vis.update_cfg(q)


    def gen_world_coll(self):
        """Define obstacles in environment."""
        # - Ground
        ground_coll = pk.collision.HalfSpace.from_point_and_normal(
            np.array([0.0, 0.0, self.min_height]), np.array([0.0, 0.0, 1.0])
        )
        # - Wall
        wall_coll = pk.collision.HalfSpace.from_point_and_normal(
            np.array([0, self.wall_distance, 0]), np.array([0.0, 1.0, 0.0])
        )

        # TODO: Pillar
        pillar_coll = pk.collision.Capsule.from_radius_height(
            position=np.array([0.0, 0.0, -self.pillar_height / 2]),
            radius=self.pillar_length / 2,
            height=self.pillar_height,
        )

        # TODO: Table
        table_intervals = np.linspace(start=-self.table_height / 2.0 + self.table_width / 2.0, stop=self.table_height / 2.0 - self.table_width / 2.0, num=2)
        translation = np.concatenate(
            [
                np.full((table_intervals.shape[0], 1), 0.0),
                np.full((table_intervals.shape[0], 1), 0.0),
                table_intervals.reshape(-1, 1),
            ],
            axis=1,
        ) + self.table_offset + np.array([0.0, self.table_width / 2.0, -self.table_height / 2])
        table_coll = pk.collision.Capsule.from_radius_height(
            position=translation,
            wxyz=np.array([.707, .707, 0, 0]), #TODO: Fix if used
            radius=np.full((translation.shape[0], 1), self.table_width / 2),
            height=np.full((translation.shape[0], 1), self.table_length),
        )

        return ground_coll, table_coll, pillar_coll

    
    def _ball_trajectory(self, traj, time_release, dt) -> Callable[[float], np.ndarray]:
        target_link_index = self.robot.links.names.index(self.target_link_name)

        idx_float = time_release / dt
            
        # Get the integer bounds for interpolation
        # Clamp to ensure we don't go out of bounds [0, timesteps-2]
        # We use -2 because we need a "next" neighbor for velocity calculation
        max_idx = traj.shape[0] - 2
        idx_floor = np.clip(np.floor(idx_float).astype(int), 0, max_idx)
        idx_ceil = idx_floor + 1
        
        alpha = idx_float - idx_floor

        q_curr = traj[idx_floor]
        q_next = traj[idx_ceil]
        
        # 4. Interpolate Joint Position and Velocity
        # Linear Interpolation for q
        q_release = (1.0 - alpha) * q_curr + alpha * q_next
        
        # Joint Velocity is the slope between the two points
        q_dot_release = (q_next - q_curr) / dt

        # Get starting position of launch
        q = self.robot.forward_kinematics(q_release)
        x0 = jnp.take(q, target_link_index, axis=-2)[..., 4:]
        
        # Get launch velocity
        jacobian = compute_ee_spatial_jacobian(self.robot, q_release, jnp.array([target_link_index]))
        twist = jacobian @ q_dot_release.squeeze()
        v0 = twist[:3][None]

        gravity_vec = np.array([0.0, 0.0, -9.81]) 
        pred_pos = lambda t: (x0 + v0 * (t - time_release) + 0.5 * gravity_vec * ((t - time_release) ** 2))[0]
        return pred_pos
