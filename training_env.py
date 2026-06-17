import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
from torch import nn
from torch.utils.data import DataLoader
from stable_baselines3 import SAC
from stable_baselines3 import CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from pointnet2_utils import PointNetSetAbstraction
import mujoco
import mujoco.viewer
import time
from point_cloud_compression import closest_point_filter

global POINT_CLOUD_DIM
POINT_CLOUD_DIM = 1024

class CreviceEnv(gym.Env):
    def __init__(self, enable_viewer = False):
        self.setup(enable_viewer)
        
    def setup(self, enable_viewer):
        # Randomly select anchor scene
        scenes = ["parallel_plates_7.0cm",
                  "parallel_plates_8.0cm",
                  "parallel_plates_9.5cm",
                  "parallel_plates_11.0cm",
                  "parallel_plates_13.0cm",
                  "tapered_converging_3deg_9.5cm",
                  "tapered_converging_5deg_9.5cm",
                  "tapered_diverging_3deg_9.5cm",
                  "tapered_diverging_5deg_9.5cm"]
        chosen_scene = np.random.choice(scenes)
        scene_path = f"geo/point_clouds/{chosen_scene}.npy"

        # Create point cloud
        self.point_cloud = np.load(scene_path)
        self.point_cloud = closest_point_filter(self.point_cloud, POINT_CLOUD_DIM)
        self.ref_point = np.array([-0.0254, 0, 0.04])

        self.shifted_point_cloud = np.array([self.point_cloud[i] - self.ref_point for i in range(POINT_CLOUD_DIM)])
        
        # Initialize gymnasium
        low  = np.array([-np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2], dtype=np.float32)
        high = np.array([np.pi/2,  np.pi/2,  np.pi/2,  np.pi/2], dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(POINT_CLOUD_DIM, 3), dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, shape=(4,), dtype=np.float32)

        # Initialize Mujoco
        self.model = mujoco.MjModel.from_xml_path("geo/anchor_scenes/parallel_plates_9.5cm.xml")
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
        if self.counter == 10:
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
        self.data.ctrl[:] = action

        # Settle until qvel converges or timeout
        N_min  = 500    # wait out initial contact transients before checking
        N_max  = 8000
        vel_tol = 0.1

        for i in range(N_max):
            mujoco.mj_step(self.model, self.data)
            if self.enable_viewer:
                self.viewer.sync()
            if np.any(np.isnan(self.data.qpos)) or np.any(np.abs(self.data.qpos) > 1e6):
                return None, -10, True, False, {}
            if i >= N_min and np.linalg.norm(self.data.qvel[6:]) < vel_tol:
                print(f"Converged after {i} steps")
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
        for i, idx in enumerate(wall_contact_ids):
            contact_wrench = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, idx, contact_wrench)
            contact_forces.append(contact_wrench[0])  # normal force only
            g1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, self.data.contact[idx].geom1)
            g2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, self.data.contact[idx].geom2)
            print(contact_wrench[0], g1_name, g2_name)

        # Build final elements and return
        reward = self.generate_reward(contact_forces)
        terminated = True
        observation = truncated = None
        info = {}
        return observation, reward, terminated, truncated, info
    
    def generate_reward(self, contact_forces, friction_coeff=0.5):
        # Sum max friction force per contact (linear in normal force, avoids cubic cone volume scaling)
        total_friction = sum(friction_coeff * abs(f) for f in contact_forces)

        module_mass = 0.255
        center_tether_mass = 0.5
        robot_weight = (module_mass * 18 + center_tether_mass) * 9.80665

        total_reward = (total_friction - robot_weight) / robot_weight

        return total_reward
    
class PointNetExtractor(BaseFeaturesExtractor):
    def __init__(self, D_common = 128):
        super().__init__(self)
        
        # Make instance of pointnet encoder and attention network here
        self.sa1 = PointNetSetAbstraction(
            npoint = POINT_CLOUD_DIM, radius="R1", nsample=32,
            in_channel=3, mlp=[32, 32, 64], group_all=False
        )
        self.sa2 = PointNetSetAbstraction(
            npoint = POINT_CLOUD_DIM//4, radius="R2", nsample=32,
            in_channel=64+3, mlp=[64, 64, 128], group_all=False
        )
        self.sa3 = PointNetSetAbstraction(
            npoint = POINT_CLOUD_DIM//16, radius="R3", nsample=32,
            in_channel=128+3, mlp=[128, 128, 256], group_all=False
        )
        self.attention = JointAttentionReadout(D_common)

    def forward(self, observations):
        # observations shape: (batch, N, 4)
        # run through PointNet++
        xyz1, f1 = self.sa1.forward(observations, None)
        xyz2, f2 = self.sa2.forward(xyz1, f1)
        xyz3, f3 = self.sa3.forward(xyz2, f2)

        # run through attention queries
        features = self.attention.forward(f1, f2, f3)

        # return (batch, features_dim)
        return features
    
class JointAttentionReadout(nn.Module):
    def __init__(self, D_common=128, n_joints=4):
        super().__init__()
        self.scale = D_common ** 0.5

        self.queries = nn.Parameter(torch.randn(n_joints, D_common))

        self.proj1 = nn.Linear(64, D_common)
        self.proj2 = nn.Linear(128, D_common)
        self.proj3 = nn.Linear(256, D_common)

    def forward(self, f1, f2, f3):
        f1 = self.proj1(f1.transpose(1, 2))
        f2 = self.proj2(f2.transpose(1, 2))
        f3 = self.proj3(f3.transpose(1, 2))

        features = torch.cat([f1, f2, f3], dim=1)

        attn = torch.softmax(torch.einsum("bdn,jd->bjn", features, self.queries) / self.scale, dim=1)

        readout = torch.einsum("bjn,bnd->bjd", attn, features)


        return readout.flatten(1)
    


D_common = 128
load_model = False

env = CreviceEnv()

if load_model:
    model = SAC.load("agent/checkpoints/sac_snake_10000_steps", env=env) # Change to the desired checkpoint
else:
    policy_kwargs = dict(
        features_extractor_class = PointNetExtractor,
        features_extractor_kwargs = dict(features_dim = 4 * D_common),
        net_arch = dict(pi = [256, 256], qf = [256, 256])
    )

    model = SAC(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1
    )

    checkpoint_callback = CheckpointCallback(
        save_freq = 10_000,
        save_path = "checkpoints/",
        name_prefix = "jam_net"
    )

# Set timesteps based on how far through training and how much more training is needed
model.learn(total_timesteps = 1_000_000, callback = checkpoint_callback)
model.save("agent/sac_snake_final")