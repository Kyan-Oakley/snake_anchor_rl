import mujoco
import mujoco.viewer
import numpy as np
import prototype
import time
import matplotlib.pyplot as plt
import pickle
import imageio
import pyvista as pv

# r = 0.65 h = 0.98 c = [1 0.74 0.65] z
# r = 1 h = 0.51 c = [1 1 1.4]
# r = 0.47 h = 0.64 c = [0.43 1.01 2.7]
# 2.45 cm = 1 inch 
# [0.01 2.09 2.01]
# [2.09 0.01 2.01]


NUM_MODULES = 3

# ==============================================================================
# 1. XML Header and Asset Definitions
# ==============================================================================
REU_snake_xml = f"""<mujoco>
    <compiler angle="radian" meshdir="REU_Snake/Mesh/" />
    <asset>
        <mesh name="ReU_mesh" file="ReU_Mesh.STL" scale="0.0254 0.0254 0.0254"/>
        <material name="red" rgba="1 0 0 1"/>
        <material name="blue" rgba="0 0 1 1"/>
        <material name="green_dark" rgba="0 0.6 0 1"/>  
        <material name="green_light" rgba="0 0.4 0 1"/>
        <material name="debug_col" rgba="1 0.5 0.5 0.3"/> 
    </asset>
"""

# ==============================================================================
# 2. Worldbody Initialization & Base Module (mod_0) Definition
# ==============================================================================
REU_snake_xml += f"""
    <worldbody>
        <light pos="0 0 1.5" dir="0 0 -1" directional="true" castshadow="false"/>
        
        <body name="mod_0_in" pos="-0.0253 {-0.74*0.0254*0} 0.2" euler="1.57 0 0">
            <joint name="mod_0_free" type="free"/>
            
            
            <body name="mod_0_body" pos="0 0 0">
                <geom name="mesh0" type="mesh" mesh="ReU_mesh" pos="0 0 0" material="green_dark" contype="2" conaffinity="1" group="0"/>
                <inertial pos="0.02326 0.02258 0.03501" mass="0.17" fullinertia="1.061933e-04 1.055403e-04 6.550863e-05 -1.62875e-06 -1.473256e-05 1.460056e-05"/>
                
                <body name="mod_0_out" pos="0 0 0" euler="0 0 0 ">
                    <geom name="mod_0_out" type="cylinder" size="0.001 0.001" pos="{0.43*0.0254+0.001} {1.01*0.0254} {2.7*0.0254}" euler="0 1.57079632679 0 " material="debug_col" contype="2" conaffinity="1" group="0"/>
                    <inertial pos="{0.43*0.0254+0.001} {1.01*0.0254} {2.7*0.0254}" mass="0.0001" diaginertia="1e-9 1e-9 1e-9"/>
"""

# ==============================================================================
# 3. Procedural Snake Module Generation Loop
# ==============================================================================
for i in range(NUM_MODULES):
    if i < 16:
        REU_snake_xml += f"""
                    <body name="mod_{i+1}_in" pos="{2.09*0.0254} {0.01*0.0254} {2.01*0.0254}" euler="0 0 1.57079632679">
                        <joint name="joint_{i+1}" type="hinge" pos="0.0254 {0.74*0.0254} {0.65*0.0254}" axis="0 1 0" range="-1.57079632679 1.57079632679" limited="true" damping="3.18" frictionloss="0.2"/>
                        <body name="mod_{i+1}_body" pos="0 0 0">
                            <geom name="mesh{i+1}" type="mesh" mesh="ReU_mesh" pos="0 0 0" material="green_dark" contype="2" conaffinity="1" group="0"/>
                            <inertial pos="0.02326 0.02258 0.03501" mass="0.17" fullinertia="1.061933e-04 1.055403e-04 6.550863e-05 -1.62875e-06 -1.473256e-05 1.460056e-05"/>
                            <body name="mod_{i+1}_out" pos="0 0 0" euler="0 0 0 ">
                                <inertial pos="{0.43*0.0254+0.001} {1.01*0.0254} {2.7*0.0254}" mass="0.0001" diaginertia="1e-9 1e-9 1e-9"/>
                                <geom name="mod_{i+1}_out" type="cylinder" size="0.001 0.001" pos="{0.43*0.0254+0.001} {1.01*0.0254} {2.7*0.0254}" euler="0 1.57079632679 0 " material="debug_col" contype="2" conaffinity="1" group="0"/>
"""
    else:
        REU_snake_xml += f"""
                    <body name="mod_{i+1}_in" pos="{2.09*0.0254} {0.01*0.0254} {2.01*0.0254}" euler="0 0 1.57079632679">
                        <joint name="joint_{i+1}" type="hinge" pos="0.0254 {0.74*0.0254} {0.65*0.0254}" axis="0 0 1" range="-1.57079632679 1.57079632679" limited="true" damping="3.18" frictionloss="0.2"/>
                        <body name="mod_{i+1}_body" pos="0 0 0">
                            <geom name="mesh{i+1}" type="mesh" mesh="ReU_mesh" pos="0 0 0" material="green_dark"/>
                            <geom name="mod_{i+1}_link" type="cylinder" pos="0.0254 0.0258 {0.0254*1.4} " size="0.0258 0.006" material="debug_col" /> 
                            <body name="mod_{i+1}_out" pos="0 0 0" euler="0 0 0 ">
                                <geom name="mod_{i+1}_out" type="cylinder" size="0.0125 0.008" pos="{0.43*0.0254+0.001} {1.01*0.0254} {2.7*0.0254}" euler="0 1.57079632679 0 " material="debug_col" contype="2" conaffinity="1" group="0"/>
"""
        
REU_snake_xml += f"""
<body name="mod_{4}_in" pos="{0.43*0.0254+0.001} {1.01*0.0254} {2.7*0.0254}">
<geom name="mod_{4}_in" type="cylinder" size="{0.67*0.00254} {0.00254/2}" pos="0 0 0" material="debug_col" contype="2" conaffinity="1" group="0"/>
<inertial pos="{0.43*0.0254+0.001} {1.01*0.0254} {2.7*0.0254}" mass="0.0001" diaginertia="1e-9 1e-9 1e-9"/>
                </body>
 """      

# ==============================================================================
# 4. Closing All Nested Kinematic Bodies (mod_0 up to mod_N)
# ==============================================================================
# We opened 3 bodies per module plus 3 bodies for mod_0.
for _ in range(NUM_MODULES + 1):
    REU_snake_xml += """
                            </body>
                        </body>
                    </body>
    """

# Close worldbody tag
REU_snake_xml += """
    </worldbody>
"""

# ==============================================================================
# 5. Actuator Elements
# ==============================================================================
REU_snake_xml += """
    <actuator>
"""
for i in range(NUM_MODULES):
    if i == 1:
        REU_snake_xml += f'        <motor name="servo_{i+1}_pos" joint="joint_{i+1}" gear="1" ctrlrange="-10 10" ctrllimited="true"/>\n'
    else:
        REU_snake_xml += f'        <position name="servo_{i+1}_pos" joint="joint_{i+1}" kp="1" ctrlrange="-1.5708 1.5708" ctrllimited="true" forcerange="-10.0 10.0" forcelimited="true"/>\n'

REU_snake_xml += """    </actuator>
"""

# ==============================================================================
# 6. Sensor Subsystem (Includes Base Wrench Sensors and Joint Force Sensors)
# ==============================================================================
REU_snake_xml += """
    <sensor>
"""

for i in range(NUM_MODULES):
    REU_snake_xml += f'        <jointactuatorfrc joint="joint_{i+1}" name="joint_{i+1}_torque_sensor"/>\n'

REU_snake_xml += """    </sensor>
</mujoco>
"""

# Save verified output to target path
with open("REU_snake_robot.xml", "w", encoding="utf-8") as f:
    f.write(REU_snake_xml)




xml_anchor_scene = f"""
<mujoco model="anchoring_workspace_scene">
<visual>
        <scale contactheight="0.05" contactwidth="0.01"/>
    </visual>
  <option cone="elliptic" noslip_iterations="5" noslip_tolerance="1e-6" />
  <option timestep="0.001" gravity="0 0 -9.81"/>

  <asset>
     
    <texture type="skybox" builtin="gradient" rgb1=".3 .5 .7" rgb2="0 0 0" width="32" height="32"/>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>
  </asset>
  
  <include file="REU_snake_robot.xml"/>

  
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="5 5 .05" type="plane" material="grid" group="2"/>
    ``0.0635 wall width''
    <body name="Wall1" pos="{0.0635/2+0.01} 0 0.2">
        <geom name="Wall_1" type="box" size="0.01 0.5 0.5" contype="1" conaffinity="2" group="2" rgba="0.8 0.5 0.2 0.2"/>
    </body>    

    <body name="Wall2" pos="{-0.0635/2-0.01} 0 0.2">
        <geom name="Wall_2" type="box" size="0.01 0.5 0.5" contype="1" conaffinity="2" group="2" rgba="0.8 0.5 0.2 0.2"/>    
    
    </body>

        <body name="Wall3" pos="0 0 0.14">
        <geom name="Wall_3" type="box" size="0.1 0.2 0.05" contype="1" conaffinity="2" group="2" rgba="0.8 0.5 0.2 0.2"/>
    </body>

  </worldbody>
  """
xml_anchor_scene += """
<contact>
"""
for i in range(NUM_MODULES+1):
    xml_anchor_scene += f"""
    
    
    <pair name="f_mod_{i}_w1_link"  geom1="Wall_1" geom2="mesh{i}" condim="6" friction="0.5 0.05 0.005"  margin="0.001" gap="0" solimp="0.99 0.995 0.001 0.5 2" solref="0.004 1.0"/>
    
    
    <pair name="f_mod_{i}_w2_link"  geom1="Wall_2" geom2="mesh{i}" condim="6" friction="0.5 0.05 0.005"  margin="0.001" gap="0" solimp="0.99 0.995 0.001 0.5 2" solref="0.004 1.0"/>

    <pair name="f_mod_{i}_w3_link"  geom1="Wall_3" geom2="mesh{i}" condim="6" friction="0.5 0.05 0.005"  margin="0.001" gap="0" solimp="0.99 0.995 0.001 0.5 2" solref="0.004 1.0"/>
    """

# friction="0.5 0.5 0.5"  solimp="0.8 0.9 0.001" solref="0.004 1"
xml_anchor_scene += f"""
  </contact>
  </mujoco>
  """

#print(xml_anchor_scene)

with open("xml_anchor_scene.xml", "w", encoding="utf-8") as f:
    f.write(xml_anchor_scene)

import pickle
# import prototype # Ensure this is available in your environment

class WrenchExperiment:
    def __init__(self, xml_string, enable_viewer=False, enable_video=False):
        # 1. Initialization
        self.model = mujoco.MjModel.from_xml_path(xml_string)
        self.model.opt.gravity[:] = [0, 0, 0]
        self.data = mujoco.MjData(self.model)
        
        # 2. Viewer and Video Setup
        self.enable_viewer = enable_viewer
        self.enable_video = enable_video
        self.viewer = None
        self.Rapid = True
        self.ArrowRatio = 0.05
        
        # 3. Central Dictionary for Recording
        self.record = {
            'prep_mark_1': [],
            'contacts': [],
            'geom_world_pos_origin': None,
            'experiments': {
                'reference_points': [],
                'wrenches_slip': [],
                'wrenches_non_slip': [],
                'contact_forces': []
            }
        }
        
    def prepare_simulation(self, desired_control_anchor, n_steps=4000):
        """Initializes the simulation to the starting state."""
        self.model.opt.gravity[:] = [0, 0, -9.81]
        mujoco.mj_forward(self.model, self.data)
        
        for j in range(n_steps):
           
            self.data.ctrl[:] = desired_control_anchor * (j + 1) / (n_steps)
 
            mujoco.mj_step(self.model, self.data)

        self.model.opt.gravity[:] = [0, 0, 0]
        for i in range(100):mujoco.mj_forward(self.model, self.data)
        # Record Mark 1 positions
        i = 3
        geom_name = f"mod_{i}_out"
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        self.record['prep_mark_1'].append(self.data.geom_xpos[geom_id].copy())
            
    def extract_contacts(self):
        """Extracts contact information and logs it into the record dictionary."""
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            
            c_force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, i, c_force)
            
            contact_info = {
                'p': contact.pos.copy(),       # Position relative to CoM
                'n': contact.frame[:3].copy(), # Normal vector
                'mu': 0.5,                     # Coefficient of friction
                'f_max': 50,                  # Max normal force capacity
                'initial_force': c_force[0]
            }
            self.record['contacts'].append(contact_info)
            print(f"Contact {i} position: {contact.pos}, Init Force: {c_force[0]}", f"Direction: {contact.frame[:3]}")
            
    def run_single_test(self, wrench, geom_id, body_id, original_position, n=5000, threshold=1e-3, log_interval=200,Rapid=True):
        """Applies a gradually increasing wrench and records slip/non-slip conditions."""
        Rapid = self.Rapid
        for j in range(n):
            current_wrench = wrench * (j + 1) / n

        
            # Apply Wrench
            if j < n-1000:
                self.data.xfrc_applied[body_id, :3] = current_wrench[:3]
                self.data.xfrc_applied[body_id, 3:] = current_wrench[3:]
            else:
                self.data.xfrc_applied[body_id, :3] = wrench[:3]
                self.data.xfrc_applied[body_id, 3:] = wrench[3:]
            mujoco.mj_step(self.model, self.data)
            
            # Viewer and Video hooks
            if self.enable_viewer and self.viewer is not None:
                self._draw_wrench_arrow(current_wrench, body_id, scale=0.05)
                self.viewer.sync()
                #time.sleep(0.001) # Uncomment if visualization is too fast
            if self.enable_video and j%5 == 0:  # Record video every 5 steps to balance quality and performance
                print(f"\r{j}", end="")
                #self._record_video_frame()  # Basic frame capture without arrow
                self._record_video_frame_with_arrow(current_wrench, body_id, scale=0.05)

            # Logging logic based on interval
            if j % log_interval == 0:
                current_geom_pos = self.data.geom_xpos[geom_id].copy()
                self.record['experiments']['reference_points'].append(current_geom_pos)
                
                # Check for slip
                difference = np.linalg.norm(original_position - current_geom_pos, ord=2)
                if difference > threshold:
                    self.record['experiments']['wrenches_slip'].append(current_wrench.copy())
                        
                    if Rapid: 
                        print(j,"slip")
                        break
                else:
                    self.record['experiments']['wrenches_non_slip'].append(current_wrench.copy())
                
                # Record contact forces
                F = np.zeros(self.data.ncon)
                for i in range(self.data.ncon):
                    c_force = np.zeros(6, dtype=np.float64)
                    mujoco.mj_contactForce(self.model, self.data, i, c_force)
                    F[i] = c_force[0]
                self.record['experiments']['contact_forces'].append(F)

    def run_single_test0(self, wrench, geom_id, body_id, original_position, n=5000, log_interval=200, Rapid=True):
        """Applies a gradually increasing wrench and records slip/non-slip conditions based on friction cone limits."""
        Rapid = self.Rapid
        
        # 设定摩擦锥判定阈值 (假设你的物理 mu = 0.5，这里设为 0.495 来提前捕捉临界点)
        mu_threshold = 0.495 
        
        for j in range(n):
            current_wrench = wrench * (j + 1) / n

            # Apply Wrench
            if j < n-1000:
                self.data.xfrc_applied[body_id, :3] = current_wrench[:3]
                self.data.xfrc_applied[body_id, 3:] = current_wrench[3:]
            else:
                self.data.xfrc_applied[body_id, :3] = wrench[:3]
                self.data.xfrc_applied[body_id, 3:] = wrench[3:]
            mujoco.mj_step(self.model, self.data)
            
            # Viewer and Video hooks
            if self.enable_viewer and self.viewer is not None:
                self._draw_wrench_arrow(current_wrench, body_id, scale=0.05)
                self.viewer.sync()
                #time.sleep(0.001) # Uncomment if visualization is too fast
            if self.enable_video and j%5 == 0:  # Record video every 5 steps
                print(f"\r{j}", end="")
                self._record_video_frame_with_arrow(current_wrench, body_id, scale=0.05)

            # Logging logic based on interval
            if j % log_interval == 0:
                current_geom_pos = self.data.geom_xpos[geom_id].copy()
                self.record['experiments']['reference_points'].append(current_geom_pos)
                
                # 初始化摩擦锥破坏标记
                is_physics_broken = False
                
                # Record contact forces & Check dynamic slip
                F = np.zeros(self.data.ncon)
                for i in range(self.data.ncon):
                    c_force = np.zeros(6, dtype=np.float64)
                    mujoco.mj_contactForce(self.model, self.data, i, c_force)
                    
                    normal_force = c_force[0]
                    F[i] = normal_force
                    
                    # 如果该点存在有效接触力，计算其摩擦系数占用比
                    if normal_force > 1e-3:
                        tangential_force = np.linalg.norm(c_force[1:3])
                        current_mu = tangential_force / normal_force
                        
                        # 如果有任何一个接触点的受力比值超过了设定的摩擦系数阈值
                        if current_mu > mu_threshold:
                            is_physics_broken = True

                self.record['experiments']['contact_forces'].append(F)
                
                # Check for slip (使用真实的物理判定代替原本的运动学位移差)
                if is_physics_broken:
                    self.record['experiments']['wrenches_slip'].append(current_wrench.copy())
                        
                    if Rapid: 
                        print(f"\n{j} slip (Friction Cone Broken)")
                        break
                else:
                    self.record['experiments']['wrenches_non_slip'].append(current_wrench.copy())

    def run_all_experiments(self,wrenches= []):
        """Runs the test suite across different moment directions."""
        geom_name = "mod_4_in"
        body_name = "mod_4_in"
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        
        geom_world_pos_origin = self.data.geom_xpos[geom_id].copy()

        self.record['geom_world_pos_origin'] = geom_world_pos_origin
        
        # Save baseline state
        base_data = mujoco.MjData(self.model)
        mujoco.mj_copyData(base_data, self.model, self.data)

        # Optional: Initialize viewer context
        if self.enable_viewer:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        print(len(wrenches))
        #Test wrenches

        if len(wrenches)>0:
            j=0
            for wrench in wrenches:
                j+=1
                print(f"\r{j} trail", end="")
                mujoco.mj_copyData(self.data, self.model, base_data)
                self.run_single_test(wrench, geom_id, body_id, geom_world_pos_origin)
            
        # Test positive moments
        if len(wrenches) ==0 :
            for i in range(3):
                mujoco.mj_copyData(self.data, self.model, base_data)
                wrench = np.zeros(6, dtype=np.float64)
                wrench[3+i] = 5
                self.run_single_test(wrench, geom_id, body_id, geom_world_pos_origin)
                
            # Test negative moments
            for i in range(3):
                mujoco.mj_copyData(self.data, self.model, base_data)
                wrench = np.zeros(6, dtype=np.float64)
                wrench[3+i] = -5
                self.run_single_test(wrench, geom_id, body_id, geom_world_pos_origin)
            
        if self.enable_viewer and self.viewer is not None:
            self.viewer.close()

    def _draw_wrench_arrow(self, current_wrench, body_id, scale=0.05):
        """
        Helper function to render the applied wrench (force part) as an arrow in MuJoCo viewer.
        """
        scale = self.ArrowRatio
        # Safety check: skip if viewer is not enabled or not initialized
        if not self.enable_viewer or self.viewer is None:
            return

        # 1. Clear previous overlay arrows to prevent accumulation
        self.viewer.user_scn.ngeom = 0 
        
        # 2. Extract linear force vector
        force_vector = current_wrench[:3]
        if np.linalg.norm(force_vector) > 1e-5:
            # Use object center of mass as arrow start position
            start_pos = self.data.xpos[body_id].copy()
            # End position = start position + force vector * scale factor
            end_pos = start_pos + force_vector * scale
            
            # 3. Render the arrow
            scn = self.viewer.user_scn
            if scn.ngeom < scn.maxgeom:
                import mujoco
                mujoco.mjv_connector(
                    scn.geoms[scn.ngeom],
                    mujoco.mjtGeom.mjGEOM_ARROW,
                    0.01,         # Arrow thickness
                    start_pos,    # Start position
                    end_pos       # End position
                )
                # Set color to pure red [R, G, B, Alpha]
                scn.geoms[scn.ngeom].rgba = np.array([1.0, 0.0, 0.0, 1.0])
                scn.ngeom += 1

    def _record_video_frame_with_arrow(self, current_wrench, body_id, scale=0.05):
        """
        Captures the current visual state, injects the wrench arrow, 
        and appends the RGB pixel array to the video frame buffer.
        """
        scale = self.ArrowRatio
        # 1. Lazy initialization of the renderer (Copied from your original logic)
        if not hasattr(self, 'renderer') or self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            
        # Initialize the frame container if it doesn't exist
        if not hasattr(self, 'video_frames'):
            self.video_frames = []

        # 2. Sync the renderer with the latest physics state
        self.renderer.update_scene(self.data, camera=-1)
        
        # 3. Inject the arrow into the offscreen renderer's scene
        force_vector = current_wrench[:3]
        if np.linalg.norm(force_vector) > 1e-5:
            start_pos = self.data.xpos[body_id].copy()
            end_pos = start_pos + force_vector * scale
            
            render_scn = self.renderer.scene
            if render_scn.ngeom < render_scn.maxgeom:
                
                mujoco.mjv_connector(
                    render_scn.geoms[render_scn.ngeom], 
                    mujoco.mjtGeom.mjGEOM_ARROW, 
                    0.01,         # Arrow thickness
                    start_pos,    # Start position
                    end_pos       # End position
                )
                # Set color to pure red
                render_scn.geoms[render_scn.ngeom].rgba = np.array([1.0, 0.0, 0.0, 1.0])
                
                # IMPORTANT: Increment geometry counter only; DO NOT reset render_scn.ngeom
                render_scn.ngeom += 1
                
        # 4. Extract pixels and append a deep copy to the buffer
        pixels = self.renderer.render()
        self.video_frames.append(pixels.copy())

    def _record_video_frame(self):
        """
        Captures the current visual state of the MuJoCo simulation 
        and appends the RGB pixel array to the video frame buffer.
        """
        # 1. Lazy initialization of the renderer
        # This avoids allocating GPU/OpenGL resources if video recording is never called.
        if not hasattr(self, 'renderer') or self.renderer is None:
            # You can adjust height and width as needed for your video quality
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            
            # Initialize the frame container if it doesn't exist
            if not hasattr(self, 'video_frames'):
                self.video_frames = []

        # 2. Sync the renderer with the latest physics state
        # 'camera' parameter can be:
        #   -1 : The default free camera
        #    0, 1, ... : Fixed cameras defined in your XML (e.g., <camera name="track" ...>)
        #   "camera_name" : String name of your XML camera
        self.renderer.update_scene(self.data, camera=-1)

        # 3. Render the scene to an RGB pixel array
        # Returns a numpy array of shape (height, width, 3) with dtype=np.uint8
        pixels = self.renderer.render()

        # 4. Append a deep copy of the pixels to the buffer
        # Using .copy() prevents memory references from being overwritten in high-frequency loops
        self.video_frames.append(pixels.copy())

    def save_recorded_video(self, file_path="output_video.mp4", fps=30):
        """Saves the buffered frames into an MP4 video file."""
        if not hasattr(self, 'video_frames') or len(self.video_frames) == 0:
            print("No video frames found to save.")
            return
            
        print(f"Saving {len(self.video_frames)} frames to {file_path}...")
        
        # Utilizing imageio with ffmpeg plugin to compress frames into MP4
        with imageio.get_writer(file_path, fps=fps, codec='libx264', pixelformat='yuv420p') as writer:
            for frame in self.video_frames:
                writer.append_data(frame)
                
        print("Video saved successfully!")
        # Clear the buffer to free up system memory
        self.video_frames.clear()



    def export_data(self, filename="experiment_record.pkl"):
        """Exports the central dictionary to a Pickle file."""
        with open(filename, "wb") as f:
            pickle.dump(self.record, f)
        print(f"Data successfully exported to {filename}")

    def evaluate_and_visualize_prototype(self, generator_class):
        """Generates the wrench hull using the external prototype module."""
        ref_point = self.record['geom_world_pos_origin']
        contacts = self.record['contacts']
        
        generator = generator_class(cone_edges=32)
        f_proto, m_proto = generator.compute_grasp_wrench_hull(contacts, ref_point)
        
        print(f"Force Prototype Vertices: {len(f_proto[0].vertices)}")
        print(f"Force Prototype Volume (Force Capacity): {f_proto[0].volume:.4f}")
        print(f"Moment Prototype Vertices: {len(m_proto[0].vertices)}")
        print(f"Moment Prototype Volume (Balance Capacity): {m_proto[0].volume:.4f}")

        wrenches_slip = np.array(self.record['experiments']['wrenches_slip'])
        wrenches_nonslip = np.array(self.record['experiments']['wrenches_non_slip'])
        
        generator.visualize_contact_setup(contacts, ref_point)
        
        generator.visualize_prototype2(
            f_proto[0], f_proto[1], wrenches_nonslip[:, :3], wrenches_slip[:, :3], 
            "3D Force Subspace of Wrench Hull", color='green'
        )
        generator.visualize_prototype2(
            m_proto[0], m_proto[1], wrenches_nonslip[:, 3:], wrenches_slip[:, 3:], 
            "3D Moment Subspace of Wrench Hull", color='blue'
        )

    def verify_joint_torque_3d(self,joint_name="joint_2", target_torque=0.0):
        """
        Dynamically retrieve the world coordinates of the specified joint and verify 
        whether the current total contact torque equals the target torque based on 
        a full 3D mechanical cross product.
        """
        # 1. Dynamically retrieve the world coordinates and rotation axis of the joint
        model = self.model
        data = self.data
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        j_pos = data.xanchor[joint_id]  # Joint center 3D coordinates
        j_axis = data.xaxis[joint_id]   # Joint rotation axis unit vector
        
        print(f"================ 3D Physical Torque Verification [{joint_name}] ================")
        print(f"Joint World Position (Anchor): [{j_pos[0]:.4f}, {j_pos[1]:.4f}, {j_pos[2]:.4f}]")
        print(f"Joint Rotation Axis (Axis):    [{j_axis[0]:.4f}, {j_axis[1]:.4f}, {j_axis[2]:.4f}]")
        print("-" * 60)

        total_joint_torque = 0.0

        # 2. Iterate through all current contact points
        for i in range(data.ncon):
            contact = data.contact[i]
            
            # Get world coordinates of the contact point
            c_pos = contact.pos  
            
            # Calculate the lever arm vector r pointing from the joint center to the contact point
            r = c_pos - j_pos  
            
            # Extract the 6D wrench in the contact frame 
            # (The first three elements are [normal force, tangential force 1, tangential force 2])
            c_forces = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, c_forces)
            
            # Get the orientation matrix (3x3) of the contact surface in the world frame
            # R[0] is the normal vector, R[1] and R[2] are tangential vectors
            R = contact.frame.reshape(3, 3)
            
            # Transform the contact force to the world frame (combining normal and friction forces)
            f_world = c_forces[0] * R[0] + c_forces[1] * R[1] + c_forces[2] * R[2]
            
            # If this contact point has no force (pure numerical noise), skip it
            if np.linalg.norm(f_world) < 1e-3:
                continue
                
            # 3. Core physical computation
            # Calculate the 3D torque vector generated by this contact force in space (cross product)
            torque_3d = np.cross(r, f_world)
            
            # Project the 3D torque onto the joint's rotation axis (dot product)
            torque_on_joint = np.dot(torque_3d, j_axis)
            
            total_joint_torque += torque_on_joint
            
            print(f"Contact Point {i} | Position: [{c_pos[0]:.3f}, {c_pos[1]:.3f}, {c_pos[2]:.3f}]")
            print(f"               | World Force: {np.linalg.norm(f_world):.2f} N | Contributed Effective Torque: {torque_on_joint:+.4f} Nm")
            print(f" Direction of normal force: {contact.frame[:3]} ")
        

        print("-" * 60)
        print(f" Total resultant torque on the joint inferred from contact forces: {total_joint_torque:+.4f} Nm")
        print(f" Target input torque: {target_torque:.2f} Nm")

        error = abs(abs(total_joint_torque) - target_torque)
        if error < 0.2:
            print(f" Verification passed! The actual resultant torque matches the input {target_torque} Nm perfectly, with an error of only {error:.4f} Nm.")
            return True
        else:
            print(f" Verification warning! A deviation of {error:.4f} Nm exists. Please confirm the joint has not reached its physical limit or generated excessive damping.")
        print("===========================================================================\n")
        return False

    def compute_maximum_inscribed_sphere_0(self,hull, density_ratio=50):
        """
        [Function 1: Independent Calculation of Maximum Inscribed Sphere]
        Input: 
            hull: PyVista convex hull mesh object (obtained via .delaunay_3d().extract_surface())
            density_ratio: Controls voxel grid resolution (higher means more accurate but slower)
        Output: 
            max_inscribed_radius: float, radius of the inscribed sphere
            sphere_center: np.array, [x, y, z] coordinates of the sphere center
        """
        # Ensure the mesh is a closed, watertight manifold
        if hull.n_open_edges > 0:
            print("Warning: Wrench Hull is not closed. Cannot accurately compute the inscribed sphere!")
            return 0.0, np.zeros(3)
            
        # 1. Voxelize the internal volume of the convex hull
        box_length = hull.length
        voxel_spacing = box_length / density_ratio
            
        # Call voxelize using positional spacing argument or safe version fallback
        voxels = hull.voxelize(spacing=voxel_spacing)

        # 2. Compute the Signed Distance Field (SDF) 
        # PyVista convention: inside points are negative, outside points are positive, surface is 0
        voxels.compute_implicit_distance(hull, inplace=True)
        distances = voxels['implicit_distance']
        
        # 3. Locate the point furthest from the surface (the minimum negative value corresponds to the center)
        center_idx = np.argmin(distances)
        max_inscribed_radius = np.abs(distances[center_idx])
        sphere_center = voxels.points[center_idx]
        
        return max_inscribed_radius, sphere_center
    
    def compute_maximum_inscribed_sphere(self,hull):
        """
        [Function 1: Compute the maximum inscribed sphere fixed at the origin [0,0,0]]
        The radius is exactly the shortest distance from the origin to the hull's surface.
        """
        if hull.n_open_edges > 0:
            print("Warning: Wrench Hull is not closed. Calculation may be inaccurate!")
            return 0.0, np.zeros(3)
            
        try:
            # 1. Create a single-point cloud containing only the origin [0, 0, 0]
            origin_point = pv.PolyData(np.array([[0.0, 0.0, 0.0]]))
            
            # 2. Compute the Signed Distance Field (SDF) from the origin to the hull surface
            # PyVista convention: SDF is negative if the point is inside the hull, positive if outside
            origin_point.compute_implicit_distance(hull, inplace=True)
            sdf_value = origin_point['implicit_distance'][0]
            
            # 3. Evaluate the result
            if sdf_value <= 0:
                # The origin is inside (or exactly on) the hull. The absolute value is the safety radius.
                max_radius = abs(sdf_value)
                return max_radius, np.zeros(3)
            else:
                # The origin is outside the hull! (System will slip even with zero external force)
                print("⚠️ Warning: Origin [0,0,0] is outside the Wrench Hull! "
                      "The current configuration cannot maintain a zero-force state. Safety margin is 0.")
                return 0.0, np.zeros(3)
                
        except Exception as e:
            print(f"Warning: Failed to compute origin-centered inscribed sphere: {e}")
            return 0.0, np.zeros(3)
    
    def visualize_wrench_hull_pyvista(self, wrenches_nonslip, wrenches_slip, title, color="orange", 
                                    show_safety_sphere=True, density_ratio_=50,only_sphere=False):
        """
        [Function 2: Handles All 3D Visualization and Rendering]
        Renders raw data points, the convex hull, and automatically references Function 1 
        to calculate and overlay the maximum inscribed safety sphere.
        """
        # Ensure inputs are standard numpy arrays
        wrenches_nonslip = np.array(wrenches_nonslip)
        wrenches_slip = np.array(wrenches_slip)

        pts_nonslip = wrenches_nonslip
        pts_slip = wrenches_slip

        # 1. Initialize PyVista Plotter
        plotter = pv.Plotter()
        plotter.add_title(title, font_size=14)

        # 2. Process and Plot Non-Slip Points & Convex Hull
        if len(pts_nonslip) > 0:
            cloud_nonslip = pv.PolyData(pts_nonslip)
            
            # Plot raw non-slip points as green spheres
            plotter.add_mesh(cloud_nonslip, color='green', point_size=6, 
                            render_points_as_spheres=True, label='Non-Slip Points')
            
            # Compute Convex Hull (Requires at least 4 non-coplanar points in 3D)
            if len(pts_nonslip) >= 4:
                try:
                    hull = cloud_nonslip.delaunay_3d().extract_surface(algorithm='dataset_surface')
                    
                    # Add hull mesh: semi-transparent with clear black edges
                    plotter.add_mesh(hull, color=color, opacity=0.25, 
                                    show_edges=True, edge_color='black', label='Wrench Hull')
                    
                    # ✨ Reference and call the calculation function inside
                    if show_safety_sphere:
                        sphere_radius, sphere_center = self.compute_maximum_inscribed_sphere(hull)

                        # If a valid sphere is found, render it
                        if sphere_radius > 0:
                            inscribed_sphere = pv.Sphere(radius=sphere_radius, center=sphere_center)
                            plotter.add_mesh(inscribed_sphere, color='royalblue', opacity=0.4, 
                                            label=f'Safety Margin (R={sphere_radius:.3f})')
                            
                            # Plot the safety center point
                            plotter.add_points(np.array([sphere_center]), color='black', point_size=12, 
                                            render_points_as_spheres=True, label='Safety Center')
                            
                            print(f"[{title}] Automatically computed inscribed sphere inside visualization:")
                            print(f" -> Radius (R): {sphere_radius:.4f}, Center: {sphere_center}")
                            
                except Exception as e:
                    print(f"Warning: Failed to compute Convex Hull or Inscribed Sphere: {e}")

        # 3. Process and Plot Slip Points (Red)
        if len(pts_slip) > 0:
            cloud_slip = pv.PolyData(pts_slip)
            plotter.add_mesh(cloud_slip, color='red', point_size=6, 
                            render_points_as_spheres=True, label='Slip Points')

        # 4. Finalize Layout and Display
        plotter.show_grid(color='gray', xtitle='Fx / Mx', ytitle='Fy / My', ztitle='Fz / Mz', grid='back')
        plotter.add_legend(loc='lower right', bcolor=None, size=(0.25, 0.25))
        
        # Adjust the camera angle for an ideal isometric viewpoint
        plotter.view_isometric()
        if not only_sphere:
            plotter.show()

    def visualize_wrench_hull_pyvista_0(self,wrenches_nonslip, wrenches_slip,s,ns,title, color="orange",X ="FX(N)",Y="FY(N)",Z="FZ(N)"):
        """
        Visualize the 3D Wrench space point cloud and its Convex Hull using PyVista.
        Automatically handles 6D-to-3D subspace extraction based on the title.
        """

        # Ensure inputs are standard numpy arrays
        wrenches_nonslip = np.array(wrenches_nonslip)
        wrenches_slip = np.array(wrenches_slip)
        wrenches_s = np.array(s)
        wrenches_ns = np.array(ns)
        # 1. Automatically handle 6D Wrench to 3D conversion

        pts_nonslip = wrenches_nonslip
        pts_slip = wrenches_slip

        # 2. Initialize PyVista Plotter
        plotter = pv.Plotter()
        plotter.add_title(title, font_size=11)

        # 3. Process and Plot Non-Slip Points & Convex Hull
        if len(pts_nonslip) > 0:
                
                    # delaunay_3d computes the volumetric mesh, extract_surface gets the outer hull
            cloud_nonslip = pv.PolyData(pts_nonslip)

            
            # Compute Convex Hull (Requires at least 4 non-coplanar points in 3D)
            if len(pts_nonslip) >= 4:
                try:
                   # delaunay_3d computes the volumetric mesh, extract_surface gets the outer hull
                    hull = cloud_nonslip.delaunay_3d().extract_surface(algorithm='dataset_surface')
                    
                    # Add hull mesh: semi-transparent (opacity=0.3) with distinct black edges
                    plotter.add_mesh(hull, color=color, opacity=0.3, 
                                     show_edges=True, edge_color='black', label='Wrench Hull')
                except Exception as e:
                    print(f"Warning: Failed to compute Convex Hull: {e}")

        # 4. Process and Plot Slip Points (Red)
        if len(wrenches_ns) > 0:
            cloud_wrenches_ns = pv.PolyData(wrenches_ns)
            plotter.add_mesh(cloud_wrenches_ns, color='gold', point_size=14,
                             render_points_as_spheres=True, label=' Non-Slip')

        if len(wrenches_s) > 0:
            cloud_wrenches_s = pv.PolyData(wrenches_s)
            plotter.add_mesh(cloud_wrenches_s, color='deeppink', point_size=10,
                             render_points_as_spheres=True, label=' Slip')

        # 5. Finalize Layout and Display

        
        plotter.add_legend(
            loc='lower right',   
            bcolor=None,         
            size=(0.2, 0.2)      
        )
        

        plotter.show_bounds(
            grid='back',          # 网格线显示在背面
            color='black',        # 将网格线和轴线统一设为黑色
            location='outer',     # 标签强制放在包围盒外部，防止穿模
            
            # 刻度数量控制
            n_xlabels=4,          # 建议从 5 降到 4，给长数字留出更多物理空间
            n_ylabels=4,
            n_zlabels=4,
            
            fmt="%.0f",           # 🌟 强烈建议修改：改为 "%.0f" (只显示整数)。你的数据范围在正负200左右，保留2位小数会白白占用空间导致重叠。
            font_size=10,         # 适当调小一点点字体，比如 10
            
            # 自定义轴标题
            xtitle=X,             
            ytitle=Y, 
            ztitle=Z,
            
            # 🌟 强烈建议加上下面这一行：
            padding=0.05          # 在你的 3D 模型和最外层的边界框之间增加 5% 的空白间距，这样坐标数字就不会紧紧贴着模型挤在一起！
        )
        plotter.show()

    def monte_carlo_sphere_sampling(self,num_points, radius=1.0):
        """
        Uniformly sample points on the surface of a 3D sphere using the Monte Carlo method.
        
        Args:
            num_points (int): The number of points to sample.
            radius (float): The radius of the sphere.
            
        Returns:
            np.ndarray: A NumPy array of shape (num_points, 3) containing the 3D coordinates of all points.
        """
        # 1. Generate 3D random points based on the standard normal distribution N(0, 1)
        # shape: (num_points, 3)
        points = np.random.randn(num_points, 3)
        
        # 2. Calculate the distance of each point from the origin (i.e., the L2 norm/magnitude of the vector)
        # keepdims=True ensures the shape of 'norms' is (num_points, 1), facilitating array broadcasting for division
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        
        # 3. Normalize all points to the unit sphere surface, then multiply by the specified radius
        sphere_points = (points / norms) * radius
        
        return sphere_points
    
    def sample_points_on_sphere(self,num_points, radius=1.0):
        """
        Sample points on the surface of a sphere using the Fibonacci lattice method.
        
        Args:
            num_points (int): The number of points to sample.
            radius (float): The radius of the sphere.
            
        Returns:
            np.ndarray: A NumPy array of shape (num_points, 3) containing the 3D coordinates of all points.
        """
        wrenches = []

        r = 200
        n = 8
        for i in range(n):
            alp = 2*np.pi*(i)/n
            for j in range(n):
                phi = 2*np.pi*(j)/n
                wrenches.append(r*np.array([np.cos(alp)*np.cos(phi),np.sin(alp)*np.cos(phi),np.sin(phi),0,0,0]))

        r = 10
        n = 8
        for i in range(n):
            alp = 2*np.pi*(i)/n
            for j in range(n):
                phi = 2*np.pi*(j)/n
                wrenches.append(r*np.array([0,0,0,np.cos(alp)*np.cos(phi),np.sin(alp)*np.cos(phi),np.sin(phi)]))
        return wrenches
 

# Assuming you already have the data, just call it directly:
# visualize_wrench_hull(wrenches_nonslip, wrenches_slip)
# ==========================================
# Execution Example
# ==========================================
if __name__ == "__main__":
    # xml_anchor_scene = "..." # Define your XML string here
    
    # Initialize with toggles for Viewer and Video
    experiment = WrenchExperiment(
        xml_string="./xml_anchor_scene.xml", 
        enable_viewer=False,  # Set to False to run headless
        enable_video=False,   # Set to True once _record_video_frame is implemented
    )
    
    desired_control = np.array([0, 10, 0])
    
    # Run the pipeline
    experiment.prepare_simulation(desired_control)
    experiment.extract_contacts()
    experiment.verify_joint_torque_3d(joint_name="joint_2", target_torque=0.0)

    wrenches = []

    
    points = experiment.monte_carlo_sphere_sampling(num_points=128, radius=200)
    for pt in points:
        wrenches.append(np.concatenate([pt, np.zeros(3)]))

    #experiment.run_all_experiments(wrenches)

    #wrenches = []
    points = experiment.monte_carlo_sphere_sampling(num_points=128, radius=10)
    for pt in points:
        wrenches.append(np.concatenate([ np.zeros(3), pt]))
    

    experiment.run_all_experiments(wrenches)
    if experiment.enable_video:
        experiment.save_recorded_video(file_path="wrench_experiment_video2.mp4", fps=10)

    # Export the central dictionary
    experiment.export_data("my_experiment_results_hull.pkl")
    wrenches_slip = np.array(experiment.record['experiments']['wrenches_slip'])
    wrenches_nonslip = np.array(experiment.record['experiments']['wrenches_non_slip'])
    #experiment.verify_joint_torque_3d(wrenches_nonslip[:,:3], wrenches_slip[:,:3],title = "3D force subspace projection of grasp wrench hull")
    s = []
    ns = []
    s.append(-1 * np.array([3.99e-04, -2.18e-05, 2.00e+01, 1.91e-02, -4.63e+00, -7.37e-06]))
    s.append(-1 * np.array([4.36e-07, -2.48e-06, 2.00e+01, 2.55e-02, -5.98e+00, -8.64e-07]))
    s.append(-1 * np.array([-6.62e-06, 4.07e-06, 2.00e+01, 3.13e+00, -3.01e+00, 2.46e-06]))
    ns.append(-1 * np.array([2.4e-02, -1.66e-04, 2.00e+01, 2.61e-02, -1.53e+00, -3.93e-05]))
    s = np.array(s)
    ns = np.array(ns)

    experiment.visualize_wrench_hull_pyvista_0(wrenches_nonslip[:,:3], wrenches_slip[:,:3],s[:,:3],ns[:,:3],"3D Force subspace projection of admissible wrenches",color = "blue",X="Fx(N)",Y="Fy(N)",Z="Fz(N)")
    experiment.visualize_wrench_hull_pyvista_0(wrenches_nonslip[:,3:], wrenches_slip[:,3:],s[:,3:],ns[:,3:],"3D Torque subspace projection of admissible wrenches",X="\u03c4x(Nm)",Y="\u03c4y(Nm)",Z="\u03c4z(Nm)")
    # Visualize using your external module
    # experiment.evaluate_and_visualize_prototype(prototype.WrenchPrototypeGenerator)