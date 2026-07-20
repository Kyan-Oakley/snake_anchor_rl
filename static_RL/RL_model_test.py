from stable_baselines3 import SAC
from gym_env import CreviceEnv
import mujoco
import time

model = SAC.load("./agent/checkpoints/jam_net_12000_steps")

env = CreviceEnv(enable_viewer=True, freeze_after_action=True)

obs, _ = env.reset()
action, _ = model.predict(obs)
env.step(action)

while env.viewer.is_running():
    mujoco.mj_step(env.model, env.data)
    env.viewer.sync()