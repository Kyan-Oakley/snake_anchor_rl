import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gym_env import CreviceEnv
from jam_net_model import PointNetExtractor
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
    
def _clip_grad(grad):
    return grad.clamp(-5.0, 5.0) if grad is not None else grad

load_model = False

env = CreviceEnv()

if load_model:
    model = SAC.load("agent/checkpoints/jam_net_9000_steps", # Change to the desired checkpoint
                     env=env,
                     reset_num_timesteps = False,
                     tb_log_name = "SAC" # Must be the folder of the old tensorboard logs
                     )
else:
    policy_kwargs = dict(
        features_extractor_class = PointNetExtractor,
        features_extractor_kwargs = dict(D_common=128),
        net_arch = dict(pi = [256, 256], qf = [256, 256])
    )

    model = SAC(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_starts=1000,
        learning_rate=1e-4,
        verbose=1,
        batch_size=140,
        device="cuda",
        tensorboard_log="training_logs/"
    )

    # Clip gradients on every backward pass to prevent NaN from exploding gradients
    for param in model.policy.parameters():
        if param.requires_grad:
            param.register_hook(_clip_grad)

checkpoint_callback = CheckpointCallback(
    save_freq = 3_000,
    save_path = "agent/checkpoints/",
    name_prefix = "jam_net"
)

# Set timesteps based on how far through training and how much more training is needed
model.learn(total_timesteps = 1_000_000, callback = checkpoint_callback)
model.save("agent/sac_snake_final")
