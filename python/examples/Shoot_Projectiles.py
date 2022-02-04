"""
Spawning a projectile with state control
Using a combination of my walkbot code and issacgym projectile.py example to get started
"""

import math
from isaacgym import gymutil, gymtorch, gymapi
import time
import numpy as np

# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(description="Joint control Methods Example")

# create a simulator
sim_params = gymapi.SimParams()
sim_params.substeps = 2
sim_params.dt = 1.0 / 100.0


sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 8
sim_params.physx.num_velocity_iterations = 2

sim_params.physx.num_threads = args.num_threads
sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

if sim is None:
    print("*** Failed to create sim")
    quit()


# add ground plane
plane_params = gymapi.PlaneParams()
# set the normal force to be z dimension
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
gym.add_ground(sim, plane_params)

# set up the env grid
num_envs = 1
spacing = 1.5
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, 0.0, spacing)

# add cartpole urdf asset
asset_root = "../../assets"
asset_file = "urdf/WalkBot/urdf/WalkBot.urdf"
# asset_file = "urdf/WalkBot_3DOF_330/urdf/WalkBot_3DOF.urdf"

# Load asset with default control type of position for all joints
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = False
asset_options.angular_damping = 0.0
asset_options.max_angular_velocity = 10000
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
asset_options.disable_gravity = True
print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
cubebot_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# initial root pose for cartpole actors
initial_pose = gymapi.Transform()
initial_pose.p = gymapi.Vec3(0.0, 0.0, 1.0)
initial_pose.r = gymapi.Quat(0, 0.0, 0.0, 1.0)

# Create environment 0
# Cart held steady using position target mode.
# Pole held at a 45 degree angle using position target mode.
env0 = gym.create_env(sim, env_lower, env_upper, 2)
cubebot0 = gym.create_actor(env0, cubebot_asset, initial_pose, 'CubeBot', 0, 0)
num_projectiles = 10
proj_asset_options = gymapi.AssetOptions()
proj_asset_options.density = 10.
proj_asset = gym.create_box(sim, 0.05, 0.05, 0.05, proj_asset_options)
projectiles = []
for n in range(num_projectiles):
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(n * 0.5, 1.0, 50)
    pose.r = gymapi.Quat(0, 0, 0, 1)

    # create actors which will collide with actors in any environment
    ahandle = gym.create_actor(env0, proj_asset, pose, "projectile" + str(n), -1, 0)

    # set each projectile to a different, random color
    c = 0.5 + 0.5 * np.random.random(3)
    gym.set_rigid_body_color(env0, ahandle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(c[0], c[1], c[2]))

    projectiles.append(ahandle)
# save initial state for reset
initial_state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))
proj_index = 0

# Configure DOF properties
props = gym.get_actor_dof_properties(env0, cubebot0)
props["driveMode"][:] = gymapi.DOF_MODE_POS
props["stiffness"] = 1000.0
props['damping'][:] = 100.0
props['velocity'][:] = 10.89
props['effort'][:] = 0.52
props['friction'][:] = 0.0

gym.set_actor_dof_properties(env0, cubebot0, props)
# Set DOF drive targets
dof_dict = gym.get_actor_dof_dict(env0, cubebot0)
joint_dict = gym.get_actor_joint_dict(env0, cubebot0)
dof_keys = list(dof_dict.keys())
actor_root_state = gym.acquire_actor_root_state_tensor(sim)
root_states = gymtorch.wrap_tensor(actor_root_state)

num_actors = gym.get_actor_count(env0)
walkbot_state = root_states.view(num_envs, num_actors, 13)[0,0,:]

# targets = torch.tensor([1000, 0, 0, 0, 0, 0])
# gym.set_dof_velocity_target_tensor(env0, gymtorch.unwrap_tensor(targets))

# create viewer using the default camera properties
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise ValueError('*** Failed to create viewer')

# subscribe to input events. This allows input to be used to interact
# with the simulation
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE, "space_shoot")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset")


# Look at the first env
cam_pos = gymapi.Vec3(2, 1, 1)
cam_target = initial_pose.p
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# Simulate
joint_idx = 0
control_idx = 0
loop_counter = 1
max_loops = 50


while not gym.query_viewer_has_closed(viewer):
    gym.refresh_actor_root_state_tensor(sim)
    print(walkbot_state)

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    for evt in gym.query_viewer_action_events(viewer):
        if (evt.action == "space_shoot") and evt.value > 0:
            bot_pose = gymapi.Transform()
            bot_pose.p.x = walkbot_state[0]
            bot_pose.p.y = walkbot_state[1]
            bot_pose.p.z = walkbot_state[2]
            bot_pose.r.x = walkbot_state[3]
            bot_pose.r.y = walkbot_state[4]
            bot_pose.r.z = walkbot_state[5]
            bot_pose.r.w = walkbot_state[6]
            
            proj_dir = bot_pose.r.rotate(gymapi.Vec3(0,0,1))
            spawn = bot_pose.p
            speed = 5
            vel = proj_dir * speed

            angvel = 1.57 - 3.14 * np.random.random(3)

            proj_handle = projectiles[proj_index]
            state = gym.get_actor_rigid_body_states(env0, proj_handle, gymapi.STATE_NONE)
            state['pose']['p'].fill((spawn.x, spawn.y, spawn.z))
            state['pose']['r'].fill((0, 0, 0, 1))
            state['vel']['linear'].fill((vel.x, vel.y, vel.z))
            state['vel']['angular'].fill((angvel[0], angvel[1], angvel[2]))
            gym.set_actor_rigid_body_states(env0, proj_handle, state, gymapi.STATE_ALL)

            proj_index = (proj_index + 1) % len(projectiles)
    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    if(loop_counter == 0):
        print('control idx = {}. handle_list[{}] = {}'.format(control_idx, joint_idx, joint_idx))
        if(control_idx == 0):
            gym.set_dof_target_position(env0, joint_idx, 2.09)
        elif(control_idx == 1):
            gym.set_dof_target_position(env0, joint_idx, -2.09)
        else:
            gym.set_dof_target_position(env0, joint_idx, 0)
        control_idx += 1
        if(control_idx>2):
            control_idx = 0
            joint_idx += 1
            if(joint_idx > 5):
                joint_idx = 0

    loop_counter += 1
    if(loop_counter > max_loops):
        loop_counter=0

 
    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

print('Done')

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
