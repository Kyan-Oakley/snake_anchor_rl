
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation
from convex_hull import ConvexHullEval
from point_cloud_compression import closest_point_filter

class CreviceEnv(gym.Env):
    def __init__(self, enable_viewer = False, freeze_after_action = False, point_cloud_dim = 1024):
        self.freeze_after_action = freeze_after_action
        self.point_cloud_dim = point_cloud_dim
        self.setup(enable_viewer)

    def setup(self, enable_viewer):
        # Randomly select anchor scene
        scenes = np.array(["parallel_plates_8.0cm",
                           "parallel_plates_9.5cm",
                           "parallel_plates_11.0cm",
                           "parallel_plates_13.0cm",
                           "tapered_converging_3deg_9.5cm",
                           "tapered_converging_5deg_9.5cm",
                           "tapered_diverging_3deg_9.5cm",
                           "tapered_diverging_5deg_9.5cm"])
        chosen_scene = np.random.choice(scenes)
        _root = os.path.dirname(os.path.abspath(__file__))
        point_cloud_path = os.path.join(_root, f"../geo/point_clouds/{chosen_scene}.npy")
        mujoco_path = os.path.join(_root, f"../geo/anchor_scenes/{chosen_scene}.xml")

        # Create point cloud
        self.point_cloud = np.load(point_cloud_path)
        self.point_cloud = closest_point_filter(self.point_cloud, self.point_cloud_dim)
        self.ref_point = np.array([-0.0254, 0, 0.04])

        self.shifted_point_cloud = np.array([self.point_cloud[i] - self.ref_point for i in range(self.point_cloud_dim)])
        
        # Initialize gymnasium
        # base x/y bounds are sized to the crevice point-cloud extents (~+-0.075m across all
        # scenes) plus a small approach margin, not the full workspace, so sampled base poses
        # actually land near the walls instead of missing the crevice entirely.
        low  = np.array([-0.10, -0.15, 0, -np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2], dtype=np.float32)
        high = np.array([0.10, 0.15, 0.2, np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2,  np.pi/2,  np.pi/2,  np.pi/2], dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.point_cloud_dim, 3), dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, shape=(11,), dtype=np.float32)

        # Initialize Mujoco
        self.model = mujoco.MjModel.from_xml_path(mujoco_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.gravity[:] = [0, 0, 0]
        self.enable_viewer = enable_viewer
        if self.enable_viewer:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.counter = 0

        # Find wall ID's for contact points
        self.wall_geom_ids = {
                                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'Wall_1'),
                                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'Wall_2'),
                             }

    def reset(self, seed=None):
        # Hard reset to change the crevice after 10 reps, otherwise reset the snake
        if self.counter == 5:
            self.counter = 0
            self.setup(self.enable_viewer)
        else:
            mujoco.mj_resetData(self.model, self.data)

        # Find and extract point cloud info
        observation = self.shifted_point_cloud
        info = None
        self.counter += 1
        return observation, info

    def step(self, action):
        # Set joint angles in mujoco
        base_xyz = action[0:3]
        base_rpy = action[3:6]
        joint_angles = action[6:]
        self.data.qpos[0:3] = base_xyz
        self.data.qpos[3:7] = Rotation.from_euler("xyz", base_rpy, degrees=False).as_quat()
        self.data.ctrl[:] = joint_angles

        # Check and penalize collisions
        mujoco.mj_forward(self.model, self.data)
        if self.enable_viewer:
            self.viewer.sync()

        for i in range(self.data.ncon):
            if self.data.contact[i].dist < -0.005:  # 5mm penetration threshold
                return self.shifted_point_cloud.astype(np.float32), -10, True, False, {}

        # Settle until qvel converges or timeout
        N_min  = 500    # wait out initial contact transients before checking
        N_max  = 15000
        vel_tol = 0.1

        for i in range(N_max):
            mujoco.mj_step(self.model, self.data)
            if self.enable_viewer and not self.freeze_after_action:
                self.viewer.sync()
            if np.any(np.isnan(self.data.qpos)) or np.any(np.abs(self.data.qpos) > 1e6):
                return self.shifted_point_cloud.astype(np.float32), -10, True, False, {}
            if i >= N_min and np.linalg.norm(self.data.qvel[6:]) < vel_tol:
                break
        else:
            print(f"Warning: did not converge after {N_max} steps — joint qvel={self.data.qvel[-4:]}")

        # Zero velocity and recompute contacts/forces statically — decouples force
        # measurement from transition dynamics regardless of whether simulation converged.
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

        ncon = self.data.ncon
        geom1 = self.data.contact.geom1[:ncon]
        geom2 = self.data.contact.geom2[:ncon]
        seen_pairs = set()
        wall_contact_ids = []
        for i, (g1, g2) in enumerate(zip(geom1.tolist(), geom2.tolist())):
            if (g1 in self.wall_geom_ids) or (g2 in self.wall_geom_ids):
                pair = (g1, g2)
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    wall_contact_ids.append(i)
        
        # Find and package forces
        # mj_contactForce returns [normal, friction_x, friction_y, torque_x, torque_y, torque_z]
        # all in the contact frame. Only the normal component defines the friction cone axis.
        contact_forces = []
        contact_displacements = []
        for idx in wall_contact_ids:
            contact_wrench = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, idx, contact_wrench)
            normal_force_mag = contact_wrench[0]
            world_frame_contact_force = self.data.contact[idx].frame.reshape(3, 3)[0, :] * normal_force_mag
            if np.linalg.norm(world_frame_contact_force) > 1e-4:
                contact_forces.append(-1 * world_frame_contact_force)
                contact_displacements.append(self.data.contact[idx].pos - self.data.geom_xpos[1])

        # Build final elements and return
        reward = self.generate_reward(contact_forces, contact_displacements, base_xyz)
        if len(contact_forces) >= 3: print(f"Action: {action}\n\nReward: {reward}")
        observation = self.shifted_point_cloud.astype(np.float32)
        info = {}

        return observation, reward, True, False, info
    
    def generate_reward(self, contact_forces, contact_displacements, base_xyz):
        """
        Next steps for reward:
        Primary term should be minimum inscribed hypersphere within admissible wrench hull
        Secondary term penalizing bad base joint pose
        Need to add kill term if bodies are physcially overlapping
        Finally as a last metric reward a larger reaching configuration space
        """
        if len(contact_forces) < 3:
            # Dense shaping so the critic sees a gradient before the sparse >=3-contact
            # regime is ever reached: reward more contacts and a closer approach to the
            # crevice instead of a flat -5 for every non-qualifying configuration.
            dist_to_wall = np.min(np.linalg.norm(self.point_cloud - base_xyz, axis=1))
            return -5.0 + len(contact_forces) - dist_to_wall

        vectors_per_cone = 10
        linearized_friction_cones = self.linearize_friction_cones(contact_forces, vectors_per_cone)

        wrench_points = self.generate_wrench_points(linearized_friction_cones, contact_displacements)

        wrench_hull = ConvexHullEval(wrench_points)
        max_radius = wrench_hull.epsilon_metric()
        
        return max_radius

    def linearize_friction_cones(self, contact_forces, n_vectors, friction_coeff=0.5):
        cones = []
        for force in contact_forces:
            cone_points = []

            rand_vec_1 = np.random.rand(3,)
            rand_vec_2 = np.random.rand(3,)
            perp_vec_1 = np.cross(force, rand_vec_1)
            perp_vec_2 = np.cross(force, rand_vec_2)
            while np.linalg.norm(perp_vec_1) < 1e-4 or np.linalg.norm(perp_vec_2) < 1e-4:
                rand_vec_1 = np.random.rand(3,)
                rand_vec_2 = np.random.rand(3,)
                perp_vec_1 = np.cross(force, rand_vec_1)
                perp_vec_2 = np.cross(force, rand_vec_2)

            basis_vector_1 = rand_vec_1 - ((force.T @ rand_vec_1) / (force.T @ force)) * force
            basis_vector_1 = basis_vector_1 * (1 / np.linalg.norm(basis_vector_1))

            basis_vector_2 = rand_vec_2 - ((force.T @ rand_vec_2) / (force.T @ force)) * force - \
                                          ((basis_vector_1.T @ rand_vec_2) / (basis_vector_1 @ basis_vector_1)) * basis_vector_1
            basis_vector_2 = basis_vector_2 * (1 / np.linalg.norm(basis_vector_2))
            
            for i in range(n_vectors):
                inc = 2 * np.pi / n_vectors
                angle = i * inc
                norm_vec = basis_vector_1 * np.cos(angle) + basis_vector_2 * np.sin(angle)
                friction_vec = (friction_coeff * np.linalg.norm(force)) * norm_vec

                friction_cone_element = force + friction_vec
                cone_points.append(friction_cone_element)

            cones.append(cone_points)

        return cones
    
    def generate_wrench_points(self, cones, distances):
        wrench_points = []
        for i, linearized_friction_cone in enumerate(cones):
            displacement = distances[i]
            for force_vector in linearized_friction_cone:
                torque_vector = np.cross(displacement, force_vector)
                wrench_point = np.concatenate((force_vector, torque_vector), axis=None)
                wrench_points.append(wrench_point)
        
        wrench_space = np.array(wrench_points)          
        return wrench_space