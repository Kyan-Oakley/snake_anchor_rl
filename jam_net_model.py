import torch
from torch import nn
from torch.utils.data import DataLoader
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from pointnet2_utils import PointNetSetAbstraction

class PointNetExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, D_common=128):
        super().__init__(observation_space, features_dim=5*D_common)
        
        # Make instance of pointnet encoder and attention network here
        self.sa1 = PointNetSetAbstraction(
            npoint = 512, radius=0.04, nsample=32,
            in_channel=3, mlp=[32, 32, 64], group_all=False
        )
        self.sa2 = PointNetSetAbstraction(
            npoint = 128, radius=0.07, nsample=32,
            in_channel=64+3, mlp=[64, 64, 128], group_all=False
        )
        self.sa3 = PointNetSetAbstraction(
            npoint = 32, radius=0.15, nsample=32,
            in_channel=128+3, mlp=[128, 128, 256], group_all=False
        )
        self.attention = JointAttentionReadout(D_common)
        self.out_norm = nn.LayerNorm(5 * D_common)

    def forward(self, observations):
        # observations shape: (batch, N, 3)
        # PointNetSetAbstraction expects [B, C, N]; observations arrive as [B, N, C]
        obs = observations.permute(0, 2, 1)
        xyz1, f1 = self.sa1.forward(obs, None)
        xyz2, f2 = self.sa2.forward(xyz1, f1)
        xyz3, f3 = self.sa3.forward(xyz2, f2)

        # run through attention queries
        features = self.attention.forward(f1, f2, f3)

        # return (batch, features_dim)
        return self.out_norm(features)
    
class JointAttentionReadout(nn.Module):
    def __init__(self, D_common=128, n_joints=5):
        super().__init__()
        self.scale = D_common ** 0.5

        self.queries = nn.Parameter(torch.randn(n_joints, D_common) * 0.1)

        self.proj1 = nn.Sequential(nn.Linear(64, D_common), nn.LayerNorm(D_common))
        self.proj2 = nn.Sequential(nn.Linear(128, D_common), nn.LayerNorm(D_common))
        self.proj3 = nn.Sequential(nn.Linear(256, D_common), nn.LayerNorm(D_common))

    def forward(self, f1, f2, f3):
        f1 = self.proj1(f1.transpose(1, 2))
        f2 = self.proj2(f2.transpose(1, 2))
        f3 = self.proj3(f3.transpose(1, 2))

        features = torch.cat([f1, f2, f3], dim=1)

        attn = torch.softmax(torch.einsum("bnd,jd->bjn", features, self.queries) / self.scale, dim=1)

        readout = torch.einsum("bjn,bnd->bjd", attn, features)


        return readout.flatten(1)