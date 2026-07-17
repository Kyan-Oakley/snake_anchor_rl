# snake_anchor_rl

Learning-based control for a segmented "snake" robot that wedges itself into a crevice
(gap between walls) to form a mechanically stable anchor, using friction/force-closure
as the reward signal rather than a hand-tuned pose.

Ideally, this will be used as one part of a locomotion method for space traversal of snake robots, where the lack of gravity prevents normal traversal methods. This will be used to form anchors within crevices, while another part will be to reach the rest of the snake robot over to new crevices or to perform other actions.

## How the task is posed

- **Simulation:** [MuJoCo](https://mujoco.readthedocs.io/) scenes (`geo/anchor_scenes/`)
  describing a crevice: parallel plates or tapered/converging-diverging walls at
  various widths/angles. There are also two candidate snake-robot geometries (`geo/J_snake_robot.xml`,
  `geo/ReU_snake_robot.xml`).
- **Observation:** a point cloud of the crevice's inner surface (`geo/gen_anchor_pointcloud.py`
  generates these into `geo/point_clouds/`), downsampled to a fixed size via a KD-tree
  nearest-point filter (`point_cloud_compression.py`).
- **Action:** the robot's base pose as an SE(3) offset in its own rest frame, plus its
  joint angles (`static_RL/gym_env.py:CreviceEnv`) — chosen once per episode, the robot
  is teleported into that configuration, and the motors are commanded a specific position to press into the walls. Then the simulation is run to let contacts settle.
- **Reward:** a force-closure style quality metric, not distance-to-target. Contact
  forces at wall contacts are turned into linearized friction cones, mapped into
  6D wrench space (force + torque), and the convex hull of that wrench cloud is computed
  (`convex_hull.py:ConvexHullEval`). The reward is the *epsilon metric* — the radius of
  the largest hypersphere inscribed in that hull — i.e. how much external disturbance
  wrench the anchor can resist before slipping. Configurations with fewer than 3 wall
  contacts get a dense shaping reward (negative distance to the crevice) instead, so
  there's a gradient before the sparse force-closure regime is reachable.
- **Policy:** Soft Actor-Critic (stable-baselines3) with a custom feature extractor
  (`jam_net_model.py:PointNetExtractor`) — a 3-level PointNet++ set-abstraction stack
  over the point cloud, read out through a small per-joint attention module so each
  joint gets its own query into the multi-scale geometry features.

## Repo layout

```
geo/                    MuJoCo scenes, meshes, and point-cloud generation for crevices
static_RL/              Current RL implementation: single-shot ("static") anchor placement
  gym_env.py              Gymnasium env (CreviceEnv) wrapping the MuJoCo scene + reward
  RL_training_env.py      SAC training entrypoint (stable-baselines3)
  RL_model_test.py        Loads a checkpoint and visualizes one placement in the MuJoCo viewer
  agent/checkpoints/      Saved policy checkpoints (gitignored, generated locally)
  training_logs/          TensorBoard logs (gitignored, generated locally)
convex_hull.py          Wrench-space convex hull + epsilon (force-closure) metric
point_cloud_compression.py  KD-tree based point cloud downsampling
pointnet2_utils.py      PointNet++ set abstraction building blocks
jam_net_model.py        PointNet++ + attention feature extractor used by the SAC policy
```

**Naming note:** `static_RL` is named for the *task*, not the algorithm — each episode is
a single pose decision (place, settle, score), with no multi-step feedback during
placement. That's what the "dynamic" work below is meant to change.

## Status

Training works end-to-end on the 8 static crevice scenes in `geo/anchor_scenes/`
(`static_RL/RL_training_env.py`), checkpointing every 3k steps. The reward function is
still being iterated on (see the shaping note in `gym_env.py:generate_reward`) — the
`main` history has a `reward funneling`/reshaping pass as the most recent tuning attempt.

## Roadmap

- [ ] **Dynamic RL** — A version of this task where placement isn't a single step.
      Rather than the static RL which only acts for one step of an MDP, this acts as an entire MDP that commands specific positions at each step to navigate into a specific jam. The advantage of this is that it will allow us to navigate more complex crevices. The downside is that it is more complicated.
- [ ] **Contextual bandit baseline** — A version of this task designed for one step.
      Rather than a single step SAC algorithm, this is an algorithm that is designed to be single step. Essentially, a contextual bandit is designed to handle the case where an action depends on a state but the next state is not dependent on the action taken. The main change here will be that the training is optimized to handle immediate rewards rather than long term rewards. The advantage of this is that it suits the problem statement better than the current model, but the downside is that contextual bandits typically require a finite action space, so we will need to abstract that away.

## Setup

To set up, just install the packages lsited in `requirements.txt`

## Usage

```bash
# Train (reads/writes static_RL/agent/checkpoints and static_RL/training_logs)
cd static_RL && python RL_training_env.py

# Visualize a trained checkpoint in the MuJoCo viewer
cd static_RL && python RL_model_test.py
```
