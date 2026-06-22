import mujoco
import mujoco.viewer
from pathlib import Path

FILE_PATH = str(Path(__file__).parent / "anchor_scenes" / "tapered_diverging_5deg_9.5cm.xml")

model = mujoco.MjModel.from_xml_path(FILE_PATH)
data = mujoco.MjData(model)
model.opt.gravity[:] = [0, 0, 0]
viewer = mujoco.viewer.launch_passive(model, data)
while viewer.is_running():
    mujoco.mj_step(model, data)
    viewer.sync()
