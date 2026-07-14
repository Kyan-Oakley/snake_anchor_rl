from stable_baselines3 import SAC
from gym_env import CreviceEnv

model = SAC.load("./agent/checkpoints/jam_net_9000_steps")

env = CreviceEnv(enable_viewer=True, freeze_after_action=True)

obs, _ = env.reset()
action, _ = model.predict(obs)
env.step(action)

while env.viewer.is_running():
    pass