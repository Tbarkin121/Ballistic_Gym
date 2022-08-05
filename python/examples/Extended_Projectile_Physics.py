"""
Spawning a projectile with state control
Using a combination of my walkbot code and issacgym projectile.py example to get started
"""


import math
from matplotlib.pyplot import spring

from numpy.core.getlimits import _fr1
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
import torch
import numpy as np
import time
import yaml
import timeit
import os
from isaacgym.torch_utils import *

class BulletBot():
    def __init__(self, gym, sim, env_handle, collision_group, collision_filter):
        self.num_projectiles = 10
        self.projectile_index = 0
        self.projectile_list = []
        self.gym = gym
        self.sim = sim
        self.env = env_handle
        self.collision_group = collision_group
        self.collision_filter = collision_filter
        asset_root = "../../assets"
        turret_asset_file = "urdf/Turret/urdf/Turret.urdf"

        #Turret Asset Options 
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.angular_damping = 0.0
        asset_options.max_angular_velocity = 10000
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        self.turret_asset = gym.load_asset(sim, asset_root, turret_asset_file, asset_options)

        #Bullet Asset Options
        bullet_asset_options = gymapi.AssetOptions()
        bullet_asset_options.density = 10.
        bullet_asset_options.linear_damping = 0.
        bullet_asset_options.max_linear_velocity = 10000.
        self.projectile_asset = gym.create_sphere(sim, 0.01, bullet_asset_options)
        rigid_shape_properties = self.gym.get_asset_rigid_shape_properties(self.projectile_asset)
        for p in rigid_shape_properties:
            p.restitution = 1.
            p.friction = 0.5
        self.gym.set_asset_rigid_shape_properties(self.projectile_asset, rigid_shape_properties)

        self.TurretHandle = self.create_turret()
        for n in range(self.num_projectiles):
            ProjectileHandle = self.create_projectile(n)
            self.projectile_list.append(ProjectileHandle)
            
    def create_turret(self):
        # Create the turret 
        # initial root pose for actorz``
        initial_pose = gymapi.Transform()
        initial_pose.p = gymapi.Vec3(0.0, 0.0, 1.0)
        initial_pose.r = gymapi.Quat.from_euler_zyx(np.pi/2, 0, np.pi)
        turret_handle = self.gym.create_actor(self.env, self.turret_asset, initial_pose, 'Turret', self.collision_group, self.collision_filter)        
        return turret_handle

    def create_projectile(self, n):
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(n * 0.5, 1.0, 50)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        projectile_handle = self.gym.create_actor(self.env, self.projectile_asset, pose, "projectile" + str(n), -1, 0)
        c = 0.5 + 0.5 * np.random.random(3)
        self.gym.set_rigid_body_color(self.env, projectile_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(c[0], c[1], c[2]))
        
        return projectile_handle

    def fire_projectile(self, pos, ori): #Firing projectiles is kind of like the goal reset code... right... 
        print("Pew")
        pose = gymapi.Transform()
        print(pos)
        print(ori)
        pose.p.x = pos[0]
        pose.p.y = pos[1]
        pose.p.z = pos[2]
        pose.r.x = ori[0]
        pose.r.y = ori[1]
        pose.r.z = ori[2]
        pose.r.w = ori[3]
                    
        proj_dir = pose.r.rotate(gymapi.Vec3(0,0,1))
        spawn = pose.p + proj_dir*0.11
        speed = 5
        vel = proj_dir * speed

        angvel = 5 -  np.random.random(3)*10

        print(self.projectile_list)
        print(self.projectile_index)

        proj_handle = self.projectile_list[self.projectile_index]
        state = self.gym.get_actor_rigid_body_states(self.env, proj_handle, gymapi.STATE_NONE)
        state['pose']['p'].fill((spawn.x, spawn.y, spawn.z))
        state['pose']['r'].fill((0, 0, 0, 1))
        state['vel']['linear'].fill((vel.x, vel.y, vel.z))
        state['vel']['angular'].fill((angvel[0], angvel[1], angvel[2]))
        self.gym.set_actor_rigid_body_states(self.env, proj_handle, state, gymapi.STATE_ALL)
        self.projectile_index = (self.projectile_index + 1) % self.num_projectiles


class Ballistic_testing():
    def __init__(self):
        with open("../../training/cfg/task/BulletBot.yaml", "r") as cfg:
            try:
                self.cfg = yaml.safe_load(cfg)
            except yaml.YAMLError as exc:
                print(exc)
        self.num_envs = 2
        self.create_sim()
        self.create_envs(self.num_envs, 1, 2)
        self.get_state_tensors()
        # Look at the first env
        cam_pos = gymapi.Vec3(1, -2, 1)
        cam_target = gymapi.Vec3(1, 0, 1)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        self.simulation_loop()

    def create_sim(self):
        # initialize gym
        self.gym = gymapi.acquire_gym()
        # parse arguments
        args = gymutil.parse_arguments(description="Playing with the Ballistics Code")

        # create a simulator
        sim_params = gymapi.SimParams()
        sim_params.substeps = 2
        sim_params.dt = 1.0 / 500.0
        self.dt = sim_params.dt
        # sim_params.flex.shape_collision_margin = 0.25
        # sim_params.flex.num_outer_iterations = 4
        # sim_params.flex.num_inner_iterations = 10
        # sim_params.flex.solver_type = 2
        # sim_params.flex.deterministic_mode = 1
        sim_params.physx.solver_type = 0
        sim_params.physx.num_position_iterations = 8
        sim_params.physx.num_velocity_iterations = 2

        sim_params.physx.num_threads = args.num_threads
        sim_params.physx.use_gpu = args.use_gpu

        sim_params.use_gpu_pipeline = False

        sim_params.up_axis = gymapi.UP_AXIS_Z
        # sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)
        if args.use_gpu_pipeline:
            print("WARNING: Forcing CPU pipeline.")

        self.device = 'cuda:0' #cuda isn't quite working yet... need to do some more debugging
        self.sim = self.gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
        # sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, gymapi.SIM_FLEX, sim_params)
        if self.sim is None:
            print("*** Failed to create sim")
            quit()

        # create viewer using the default camera properties
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            raise ValueError('*** Failed to create viewer')

        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_SPACE, "space_shoot")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "reset")

    def create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(0.5 * -spacing, -spacing, 0.0)
        upper = gymapi.Vec3(0.5 * spacing, spacing, spacing)
        # add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.static_friction = 0.0
        plane_params.dynamic_friction = 0.0
        plane_params.restitution = 1.0
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

        collision_group = 0
        collision_filter = 0

        # Create environment
        self.bulletbot_list = []
        self.envs = []        
        # self.create_force_sensors(support_asset)
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            # Creates a tensegrity bot for an environment
            # Returns a list of handles for the support actors
            BulletBot_Obj = BulletBot(self.gym, self.sim, env_ptr, collision_group, collision_filter)
            self.bulletbot_list.append(BulletBot_Obj)
            self.envs.append(env_ptr)
            
        self.num_actors = self.gym.get_actor_count(self.envs[0])
        self.num_bodies = self.gym.get_env_rigid_body_count(self.envs[0])

        print('num_actors:{}, num_bodies:{}'.format(self.num_actors, self.num_bodies))
        self.body_dict = self.gym.get_actor_rigid_body_dict(env_ptr, self.bulletbot_list[0].TurretHandle)
        self.joint_dict = self.gym.get_actor_joint_dict(env_ptr, self.bulletbot_list[0].TurretHandle)

        print('body_dict:')
        print(self.body_dict)
        for b in self.body_dict:
                print(b)
        print('joint_dict:')
        for j in self.joint_dict:
            print(j)    
    
    def get_state_tensors(self):
        # Getting root state tensors 
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.turret_pos = self.root_states.view(self.num_envs, self.num_actors, 13)[:, 0, 0:3] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
        self.turret_ori = self.root_states.view(self.num_envs, self.num_actors, 13)[:, 0, 3:7] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
        self.turret_linvel = self.root_states.view(self.num_envs, self.num_actors, 13)[:, 0, 7:10] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
        self.turret_angvel = self.root_states.view(self.num_envs, self.num_actors, 13)[:, 0, 10:13] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
        self.bullet_pos = self.root_states.view(self.num_envs, self.num_actors, 13)[:, 1:, 0:3] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
        self.bullet_ori = self.root_states.view(self.num_envs, self.num_actors, 13)[:, 1:, 3:7] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
        self.bullet_linvel = self.root_states.view(self.num_envs, self.num_actors, 13)[:, 1:, 7:10] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
        self.bullet_angvel = self.root_states.view(self.num_envs, self.num_actors, 13)[:, 1:, 10:13] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
        
        # self.gym.refresh_actor_root_state_tensor(self.sim)
        # self.turret_init_pos = self.turret_pos.clone()
        # self.turret_init_ori = self.turret_ori.clone()
        # self.inv_start_rot = quat_conjugate(self.turret_init_ori)

        # Get Rigid Body state tensors
        rb_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_state = gymtorch.wrap_tensor(rb_state_tensor)
        print(self.rb_state.shape)
        self.rb_pos = self.rb_state.view(self.num_envs, self.num_bodies, 13)[..., 0:3] #(num_envs, num_rigid_bodies, 13)[pos,ori,Lin-vel,Ang-vel]
        self.rb_ori = self.rb_state.view(self.num_envs, self.num_bodies, 13)[..., 3:7] #(num_envs, num_rigid_bodies, 13)[pos,ori,Lin-vel,Ang-vel]
        self.rb_linvel = self.rb_state.view(self.num_envs, self.num_bodies, 13)[..., 7:10] #(num_envs, num_rigid_bodies, 13)[pos,ori,Lin-vel,Ang-vel]
        self.rb_angvel = self.rb_state.view(self.num_envs, self.num_bodies, 13)[..., 10:13] #(num_envs, num_rigid_bodies, 13)[pos,ori,Lin-vel,Ang-vel]
        self.barrel_pos = self.rb_state.view(self.num_envs, self.num_bodies, 13)[:, 2, 0:3]
        self.barrel_ori = self.rb_state.view(self.num_envs, self.num_bodies, 13)[:, 2, 3:7]

        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.barrel_init_pos = self.barrel_pos.clone()
        self.barrel_init_ori = self.barrel_ori.clone()
        self.inv_start_rot = quat_conjugate(self.barrel_init_ori)
        # # print('rb_pos')
        # # print(rb_pos)

        # # print(body_names)
        # # print(extremity_names)
        # # print(extremity_indices)

    def calculate_projectile_forces(self, actions):
        pass

    def custom_simulate(self, yaw):
        start = timeit.default_timer()
        # print(self.bullet_pos)
        # print(self.bullet_linvel.shape)
        bullet_accel = torch.tensor([0.0, 0.0, -9.81]).unsqueeze(0).unsqueeze(0).repeat(2,10,1)

        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        
        Y = torch.cat( (self.bullet_pos, self.bullet_linvel), dim=-1)
        Y = self.rk4(0,Y, yaw)

        # print(self.bullet_pos.shape)
        # print(Y[:,:,0:3].shape)
        self.bullet_pos[:, :, 0:3] = Y[:,:, 0:3]
        self.bullet_linvel[:, :, 0:3] = Y[:,:, 3:6]
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
        # time.sleep(1)

        # self.bullet_linvel += bullet_accel*self.dt
        # self.bullet_pos += self.bullet_linvel*self.dt

        # self.projectile_index = (self.projectile_index + 1) % self.num_projectiles
        stop = timeit.default_timer()

        # print('Time = {}'.format(stop-start))
        
    def projectile_accel(self, velocity, yaw):        
        # time.sleep(0.1)
        mass = 1
        density = 1
        formfactor = 1
        diameter = 1
        Cd = 1
        yaw_rotation = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
        yaw_rotation = yaw_rotation.unsqueeze(1).repeat(1,10,1)
        # print(yaw_rotation.shape)
        # print(yaw_rotation)
        # print(velocity.shape)
        # print(velocity)
        
        velocity_transformed0 = quat_rotate(yaw_rotation[0,...], velocity[0,...])
        velocity_transformed1 = quat_rotate(yaw_rotation[1,...], velocity[1,...])
        velocity_transformed = torch.cat( (velocity_transformed0.unsqueeze(0),velocity_transformed1.unsqueeze(0)) )
        velocity_transformed[:,:,2] = 0.0
        # print('vel = {}'.format(velocity_transformed[0,0,:]))

        v = torch.norm(velocity_transformed, dim=-1).unsqueeze(-1)

        DF = (-(3.1415 * density * formfactor * diameter**2 )/8) * Cd * (v * velocity_transformed)

        g = torch.tensor([0.0, 0.0, -9.81]).unsqueeze(0).unsqueeze(0).repeat(2,10,1)

        # Earth Angular Velocity
        # Ω = torch.tensor([7.2921159*10**-5]).repeat(2,10)
        Ω = torch.tensor([1]).repeat(2,10)
        lat = torch.tensor([np.pi/2]).repeat(2,10)

        
        Az = yaw.unsqueeze(-1).repeat(1,10)
        


        ωx = Ω*torch.cos(lat) * torch.sin(-Az)
        ωy = Ω*torch.cos(lat) * torch.cos(-Az)
        ωz = Ω*torch.sin(lat)
        
        # ωx = Ω*torch.cos(lat)
        # ωy = Ω*torch.cos(lat)
        # ωz = Ω*torch.sin(lat)
        
        # print('velocity = {}'.format(velocity))    
        # print('v = {}'.format(v))    
        # print('v_normalized = {}'.format(v_normalized))    
        # print('vx = {}'.format(vx))    
        # print('vy = {}'.format(vy))    
        # print('angle = {}'.format(angle))    
        # print('lat = {}'.format(lat))    
        # print('Az = {}'.format(Az))    
        # print('ωx = {}'.format(ωx.shape))
        # print('ωy = {}'.format(ωy.shape))
        # print('ωz = {}'.format(ωz.shape))
        ω = torch.cat( (ωx.unsqueeze(-1),ωy.unsqueeze(-1),ωz.unsqueeze(-1)), dim=2 )
        # os.system('cls||clear')
        # print(ω[0,0,:])
        # print(velocity[0,0,:])

        ac = -2*torch.cross(ω, velocity_transformed)
        ac[:,:,2] = 0.0


        F = DF + mass*(g+ac)
        # F = torch.zeros_like(DF)
        # F = mass*ac

        yaw_rotation = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), -yaw)
        yaw_rotation = yaw_rotation.unsqueeze(1).repeat(1,10,1)
        # print(yaw_rotation.shape)
        # print(yaw_rotation)
        # print(velocity.shape)
        # print(velocity)
        
        force_transformed0 = quat_rotate(yaw_rotation[0,...], F[0,...])
        force_transformed1 = quat_rotate(yaw_rotation[1,...], F[1,...])
        force_transformed = torch.cat( (force_transformed0.unsqueeze(0),force_transformed1.unsqueeze(0)) )

        return force_transformed/mass

    def fun(self,t,y, yaw):
        bullet_accel = self.projectile_accel(self.bullet_linvel, yaw)
        dYdt = torch.cat( (y[:,:,3:6], bullet_accel), dim=-1)
        return dYdt
    
    def rk4(self, t, Y, yaw):
        k1 = self.fun(t, Y, yaw)
        k2 = self.fun(t + self.dt/2, Y + self.dt * k1/2, yaw) 
        k3 = self.fun(t + self.dt/2, Y + self.dt * k2/2, yaw)
        k4 = self.fun(t + self.dt, Y + self.dt * k3, yaw)
        Y += (1/6)*self.dt*(k1 + 2*k2 + 2*k3 + k4)
        return Y

    def simulation_loop(self):
        loop_count = 0
        while not self.gym.query_viewer_has_closed(self.viewer):
            # print(self.bullet_pos)
            loop_count += 1
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_net_contact_force_tensor(self.sim)

            # os.system('cls||clear')
            # print(self.barrel_ori)
            # print(self.inv_start_rot)
            real_ori = quat_mul(self.barrel_ori, self.inv_start_rot)
            # print(real_ori)
            roll, pitch, yaw = get_euler_xyz(real_ori)
            yaw = -yaw
            # print(yaw)
            yaw_rotation = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
            yaw_rotation = yaw_rotation.unsqueeze(1).repeat(1,10,1)
            velocity_transformed0 = quat_rotate(yaw_rotation[0,...], self.bullet_linvel[0,...])
            # print('vel = {}'.format(self.bullet_linvel[0,0,:]))
            # print('vel_t = {}'.format(velocity_transformed0[0,:]))
            # time.sleep(0.01)
            # print('{}{}{}'.format(roll, pitch, yaw))
            

            
            self.custom_simulate(yaw)
            # self.gym.simulate(self.sim)

            self.gym.fetch_results(self.sim, True)

            for evt in self.gym.query_viewer_action_events(self.viewer):
                if (evt.action == "space_shoot") and evt.value > 0:
                    for bot, i in zip(self.bulletbot_list, range(len(self.bulletbot_list))):
                        bot.fire_projectile(self.barrel_pos[i,:], self.barrel_ori[i,:])

            # update the viewer
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            # Wait for dt to elapse in real time.
            # This synchronizes the physics simulation with the rendering rate.
            self.gym.sync_frame_time(self.sim)

Test = Ballistic_testing()