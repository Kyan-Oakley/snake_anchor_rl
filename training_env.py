import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
from torch import nn
from torch.utils.data import DataLoader
from stable_baselines3 import SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from pointnet2_utils import PointNetSetAbstraction
import mujoco
import mujoco.viewer
import time

class CreviceEnv(gym.Env):
    def __init__(self, enable_viewer = False):
        # Initialize gymnasium
        global POINT_CLOUD_DIM
        self.point_cloud = np.load("geo/anchor_scene.npy")
        POINT_CLOUD_DIM = np.size(self.point_cloud, axis = 0)
        self.ref_point = np.array([-0.0254, 0, 0.04])

        self.shifted_point_cloud = np.array([self.point_cloud[i] - self.ref_point for i in range(POINT_CLOUD_DIM)])

        low  = np.array([-np.inf, -np.inf, 0.0,  -np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2], dtype=np.float32)
        high = np.array([ np.inf,  np.inf, np.inf,   np.pi/2,  np.pi/2,  np.pi/2,  np.pi/2], dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(POINT_CLOUD_DIM, 3), dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, shape=(7,), dtype=np.float32)

        # Initialize Mujoco
        self.model = mujoco.MjModel.from_xml_path("geo/xml_anchor_scene.xml")
        self.data = mujoco.MjData(self.model)
        self.model.opt.gravity[:] = [0, 0, 0]
        self.enable_viewer = enable_viewer
        if self.enable_viewer:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        # Find wall ID's for contact points
        self.wall_geom_ids = {
                                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'Wall_1'),
                                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'Wall_2'),
                            }

    def reset(self, seed=None):
        # Reset Mujoco environment
        mujoco.mj_resetData(self.model, self.data)
        mocap_id = self.model.body_mocapid[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'snake_base')]
        self.data.mocap_pos[mocap_id] = self.ref_point
        self.data.mocap_quat[mocap_id] = np.array([1.0, 0.0, 0.0, 0.0])

        # Find and extract point cloud info
        observation = self.shifted_point_cloud
        info = None
        return observation, info

    def step(self, action):
        # Set joint angles and base pos in mujoco
        self.data.ctrl[:] = action[3:]
        self.data.mocap_pos[0] = action[:3]
        self.data.mocap_quat[0] = np.array([1.0, 0.0, 0.0, 0.0])

        # Find contact points
        start_time = time.time()
        curr_time = start_time
        while curr_time - start_time < 0.1:
            curr_time = time.time()
            mujoco.mj_step(self.model, self.data)
            if self.enable_viewer:
                self.viewer.sync()
        ncon = self.data.ncon
        geom1 = self.data.contact.geom1[:ncon]
        geom2 = self.data.contact.geom2[:ncon]
        seen_pairs = set()
        wall_contact_ids = []
        for i, (g1, g2) in enumerate(zip(geom1.tolist(), geom2.tolist())):
            if (g1 in self.wall_geom_ids or g2 in self.wall_geom_ids):
                pair = (g1, g2)
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    wall_contact_ids.append(i)
        
        # Find and package forces
        contact_forces = []
        for idx in wall_contact_ids:
            contact_wrench = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, idx, contact_wrench)
            contact_forces.append(contact_wrench[:3])

        # Build final elements and return
        reward = self.generate_reward(contact_forces)
        terminated = True
        observation = truncated = info = None
        return observation, reward, terminated, truncated, info
    
    def generate_reward(self, contact_forces, friction_coeff=0.5):
        # Naive reward function summing total volume of friction cones then subtracting weight of robot
        total_reward = 0
        for normal_force in contact_forces:
            cone_height = np.linalg.norm(normal_force)
            max_cone_width = friction_coeff * cone_height
            cone_volume = (1/3) * cone_height * np.pi * (max_cone_width ** 2)
            total_reward += cone_volume

        module_mass = 0.255
        center_tether_mass = 0.5
        robot_weight = (module_mass * 18 + center_tether_mass) * 9.80665
        kp = 0.2 # Scaling term to not let friction cone volume dominate
        total_reward -= kp * (robot_weight ** 2) # Square so units agree

        total_reward /= 800 # Normalize for gradients

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
    def __init__(self, D_common, n_joints=4):
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
env = CreviceEnv()
env.reset()
_, reward, _, _, _ = env.step([-0.0254, 0, 0.04, 0, 0, 0, 0])
print(reward)

# policy_kwargs = dict(
#     features_extractor_class = PointNetExtractor,
#     features_extractor_kwargs = dict(features_dim = 4 * D_common),
#     net_arch = dict(pi = [256, 256], qf = [256, 256])
# )

# model = SAC(
#     "MlpPolicy",
#     env,
#     policy_kwargs=policy_kwargs,
#     verbose=1
# )

# model.learn(total_timesteps = 1_000_000)