# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_rotate, quat_from_angle_axis, euler_xyz_from_quat

from .tocabi_amp_env_cfg import TocabiAmpEnvCfg
from .motions import TocabiMotionLoader


class TocabiAmpEnv(DirectRLEnv):
    cfg: TocabiAmpEnvCfg

    def __init__(self, cfg: TocabiAmpEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.num_actions = self.cfg.action_space
        self.power_scale = torch.ones(self.num_envs, 12, device=self.device)
        self.torque_limits = torch.tensor(self.cfg.torque_limits, dtype=torch.float32, device=self.device)

        self._vel_commands = torch.empty(self.num_envs, 3, device=self.device)
        self._vel_commands[:, 0].uniform_(self.cfg.cmd_x_range[0], self.cfg.cmd_x_range[1])
        self._vel_commands[:, 1].uniform_(self.cfg.cmd_y_range[0], self.cfg.cmd_y_range[1])
        self._vel_commands[:, 2].uniform_(self.cfg.cmd_yaw_range[0], self.cfg.cmd_yaw_range[1])

        self.init_angle = torch.tensor([ 0.0, 0.0, -0.24, 0.6, -0.36, 0.0,
                0.0, 0.0, -0.24, 0.6, -0.36, 0.0,
                0.0, 0.0, 0.0,
                0.3, 0.3, 1.5, -1.27, -1.0, 0.0, -1.0, 0.0,
                0.0, 0.0,
                -0.3, -0.3, -1.5, 1.27, 1.0, 0.0, 1.0, 0.0], device=self.device, dtype=torch.float32)
        self.p_gains = torch.tensor(self.cfg.p_gain, dtype=torch.float32, device=self.device) / 9.0
        self.d_gains = torch.tensor(self.cfg.d_gain, dtype=torch.float32, device=self.device) / 3.0

        # load motion
        self._motion_loader = TocabiMotionLoader(motion_file=self.cfg.motion_file, device=self.device)

        # DOF and key body indexes
        key_body_names = ["L_AnkleRoll_Link", "R_AnkleRoll_Link"]

        self.ref_body_index = self.robot.data.body_names.index(self.cfg.reference_body)
        self.key_body_indexes = [self.robot.data.body_names.index(name) for name in key_body_names]
        self.desired_order_indexes = [self.robot.data.joint_names.index(name) for name in self.cfg.desired_order]
        self.upper_body_indexes = [self.robot.data.joint_names.index(name) for name in self.cfg.desired_order[12:]]
        self.lower_body_indexes = [self.robot.data.joint_names.index(name) for name in self.cfg.desired_order[:12]]

        self.total_mass = torch.sum(self.robot.root_physx_view.get_masses(),dim=1).unsqueeze(1).to(self.device)

        # contact sensor ids
        self.contact_sensor_lfeet_id, _ = self.contact_sensor.find_bodies("L_AnkleRoll_Link")
        self.contact_sensor_rfeet_id, _ = self.contact_sensor.find_bodies("R_AnkleRoll_Link")

        # modified observation
        self.num_obs_his = self.cfg.num_obs_hist
        self.num_obs_skip = self.cfg.num_obs_skip
        self.obs_history = torch.zeros(self.num_envs, self.num_obs_his * self.num_obs_skip * self.cfg.num_obs_per_step, dtype=torch.float, requires_grad=False, device=self.device)
        self.action_history = torch.zeros(self.num_envs, self.num_obs_his * self.num_obs_skip * self.num_actions, dtype=torch.float, requires_grad=False, device=self.device)
        self._obs_buf = torch.zeros(self.num_envs, self.cfg.observation_space, dtype=torch.float, device=self.device)

        # for action delay
        self.delay_idx_tensor = torch.zeros(self.num_envs,2,device=self.device,dtype=torch.long)
        self.simul_len_tensor = torch.zeros(self.num_envs,2,device=self.device,dtype=torch.long)
        self.delay_idx_tensor[:,1] = 1
        self.simul_len_tensor[:,1] = 0
        for i in range(self.num_envs):
            self.delay_idx_tensor[i,0] = i
            self.simul_len_tensor[i,0] = i
        self.action_log = torch.zeros(self.num_envs, round(0.01/self.physics_dt)+1, 12, device= self.device , dtype=torch.float)

        # reconfigure AMP observation space according to the number of observations and create the buffer
        self.amp_observation_size = self.cfg.num_amp_observations * self.cfg.amp_observation_space
        self.amp_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.amp_observation_size,))
        self.amp_observation_buffer = torch.zeros(
            (self.num_envs, self.cfg.num_amp_observations, self.cfg.amp_observation_space), device=self.device
        )

        self.actions_pre = self.actions.clone()


    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self.contact_sensor
        # add ground plane
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
            ),
        )
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()
        self.action_history = torch.cat((self.action_history[:, self.num_actions:], actions), dim=-1)

    def _apply_action(self):
        lower_body_target = self.actions * self.torque_limits.unsqueeze(0) * self.power_scale
        upper_body_target = self.p_gains[self.num_actions:]*(self.init_angle[self.num_actions:] - self.robot.data.joint_pos[:,self.upper_body_indexes]) + \
                            self.d_gains[self.num_actions:]*(-self.robot.data.joint_vel[:,self.upper_body_indexes])
        
        #action_log -> tensor(num_envs, time(current~past 9), dofs(33))
        self.action_log[:,0:-1,:] = self.action_log[:,1:,:].clone() 
        self.action_log[:,-1,:] = lower_body_target
        self.simul_len_tensor[:,1] +=1
        self.simul_len_tensor[:,1] = self.simul_len_tensor[:,1].clamp(max=round(0.01/self.physics_dt)+1, min=0)
        mask = self.simul_len_tensor[:,1] > self.delay_idx_tensor[:,1] 
        bigmask = torch.zeros(self.num_envs, 12,device=self.device, dtype=torch.bool)
        bigmask[:,:] = mask[:].unsqueeze(-1)
        delayed_lower_torque = torch.where(bigmask, self.action_log[self.delay_idx_tensor[:,0],self.delay_idx_tensor[:,1],:], \
                                    self.action_log[self.simul_len_tensor[:,0],-self.simul_len_tensor[:,1],:])

        target = torch.cat([delayed_lower_torque, upper_body_target], dim=1)
        # target = torch.cat([lower_body_target, upper_body_target], dim=1)
        self.robot.set_joint_effort_target(target, joint_ids=self.desired_order_indexes)

    def _get_observations(self) -> dict:
        # build task observation
        obs = compute_obs(
            self.robot.data.joint_pos[:, self.lower_body_indexes],
            self.robot.data.joint_vel[:, self.lower_body_indexes],
            self.robot.data.body_pos_w[:, self.ref_body_index],
            self.robot.data.body_quat_w[:, self.ref_body_index],
            self.robot.data.body_lin_vel_w[:, self.ref_body_index],
            self.robot.data.body_ang_vel_w[:, self.ref_body_index],
            self.robot.data.body_pos_w[:, self.key_body_indexes],
            self.actions_pre,
            self._vel_commands,
        )

        # update observation history
        self.obs_history = torch.cat((self.obs_history[:, self.cfg.num_obs_per_step:], obs), dim=-1)

        # fill obs buffer with skipped obs history
        for i in range(self.num_obs_his):
            self._obs_buf[:, i * self.cfg.num_obs_per_step : (i + 1) * self.cfg.num_obs_per_step] = \
                self.obs_history[:, self.cfg.num_obs_per_step * (self.cfg.num_obs_skip * (i + 1) - 1) : self.cfg.num_obs_per_step * (self.cfg.num_obs_skip * (i + 1))]
        # fill obs buffer with skipped action history
        action_start_idx = self.num_obs_his * self.cfg.num_obs_per_step
        for i in range(self.num_obs_his):
            self._obs_buf[:, action_start_idx + self.num_actions * i : action_start_idx + self.num_actions * (i + 1)] = \
                self.action_history[:, self.num_actions * (self.cfg.num_obs_skip * (i + 1)) : self.num_actions * (self.cfg.num_obs_skip * (i + 1) + 1)]

        # update AMP observation history
        for i in reversed(range(self.cfg.num_amp_observations - 1)):
            self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
        # build AMP observation
        self.amp_observation_buffer[:, 0] = compute_amp_obs(
            self.robot.data.joint_pos[:, self.lower_body_indexes],
            self.robot.data.joint_vel[:, self.lower_body_indexes],
            self.robot.data.body_pos_w[:, self.ref_body_index],
            self.robot.data.body_quat_w[:, self.ref_body_index],
            self.robot.data.body_lin_vel_w[:, self.ref_body_index],
            self.robot.data.body_ang_vel_w[:, self.ref_body_index],
            self.robot.data.body_pos_w[:, self.key_body_indexes],
        )
        self.extras = {"amp_obs": self.amp_observation_buffer.view(-1, self.amp_observation_size)}

        self.actions_pre = self.actions.clone() 
        # return {"policy": obs}
        return {"policy": self._obs_buf}

    def _get_rewards(self) -> torch.Tensor:
        total_rewards = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        ## task rewards
        lin_vel_error = 1.0 * torch.exp(-0.6 * torch.square(self._vel_commands[:, 0] - self.robot.data.root_lin_vel_b[:, 0]))
        yaw_rate_error = 1.0 * torch.exp(-0.6 * torch.square(self._vel_commands[:, 2] - self.robot.data.root_ang_vel_b[:, 2]))

        ## smooth contact rewards
        left_foot_threshold = self.contact_sensor.data.net_forces_w[:, self.contact_sensor_lfeet_id, 2] > 1.4 * 9.81 * self.total_mass
        right_foot_threshold = self.contact_sensor.data.net_forces_w[:, self.contact_sensor_rfeet_id, 2] > 1.4 * 9.81 * self.total_mass
        thres = left_foot_threshold | right_foot_threshold
        _penalty_contact_force_threshold = -0.2 * torch.where(thres.squeeze(1), torch.ones_like(total_rewards), torch.zeros_like(total_rewards))

        contact_force_penalty_threshold = 0.1 * (1-torch.exp(-0.007*(torch.norm(torch.clamp(self.contact_sensor.data.net_forces_w[:, self.contact_sensor_lfeet_id, 2] - 1.4 * 9.81 * self.total_mass, min=0.0), dim=1) \
                                                         + torch.norm(torch.clamp(self.contact_sensor.data.net_forces_w[:, self.contact_sensor_rfeet_id, 2] - 1.4 * 9.81 * self.total_mass, min=0.0), dim=1))))
        _reward_contact_force = torch.where(thres.squeeze(1), contact_force_penalty_threshold, 0.1 * torch.ones_like(total_rewards))


        total_rewards += lin_vel_error + yaw_rate_error + _penalty_contact_force_threshold + _reward_contact_force

        # return torch.ones((self.num_envs,), dtype=torch.float32, device=self.sim.device)
        return total_rewards


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.early_termination:
            fall_z_died = self.robot.data.body_pos_w[:, self.ref_body_index, 2] < self.cfg.termination_height
            
            base_rot = self.robot.data.body_quat_w[:, self.ref_body_index]
            base_roll, base_pitch, _ = quat2euler(base_rot)
            fall_rot_died = torch.logical_or(torch.abs(base_roll) > 3.14/4.0, torch.abs(base_pitch) > 3.14/4.0) 

            died = torch.logical_or(fall_z_died, fall_rot_died)
        else:
            died = torch.zeros_like(time_out)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if self.cfg.reset_strategy == "default":
            root_state, joint_pos, joint_vel = self._reset_strategy_default(env_ids)
        elif self.cfg.reset_strategy.startswith("random"):
            start = "start" in self.cfg.reset_strategy
            root_state, joint_pos, joint_vel = self._reset_strategy_random(env_ids, start)
            joint_pos = torch.cat([joint_pos, self.init_angle[self.num_actions:].unsqueeze(0).repeat(joint_pos.shape[0], 1)], dim=1).to(self.robot.data.joint_pos.dtype)
            joint_vel = torch.cat([joint_vel, torch.zeros_like(self.init_angle[self.num_actions:]).unsqueeze(0).repeat(joint_vel.shape[0], 1)], dim=1).to(self.robot.data.joint_vel.dtype)
        else:
            raise ValueError(f"Unknown reset strategy: {self.cfg.reset_strategy}")

        self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        if self.cfg.reset_strategy == "default":
            self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        else:
            self.robot.write_joint_state_to_sim(joint_pos, joint_vel, self.desired_order_indexes, env_ids)

        self.actions_pre[env_ids, :] = 0.0

        self.obs_history[env_ids,:] = 0
        self.action_history[env_ids,:] = 0  

        self._vel_commands[env_ids, 0].uniform_(self.cfg.cmd_x_range[0], self.cfg.cmd_x_range[1])
        self._vel_commands[env_ids, 1].uniform_(self.cfg.cmd_y_range[0], self.cfg.cmd_y_range[1])
        self._vel_commands[env_ids, 2].uniform_(self.cfg.cmd_yaw_range[0], self.cfg.cmd_yaw_range[1])

        self.action_log[env_ids] = torch.zeros(1+round(0.01/self.physics_dt),12,device=self.device,dtype=torch.float,requires_grad=False)
        self.delay_idx_tensor[env_ids,1] = torch.randint(low=1+int(0.002/self.physics_dt),high=1+round(0.01/self.physics_dt),size=(len(env_ids),1), device=self.device,requires_grad=False).squeeze(-1)
        #low 5, high 12 for 2000 / 250Hz
        self.simul_len_tensor[env_ids,1] = 0     

    # reset strategies
    def _reset_strategy_default(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        return root_state, joint_pos, joint_vel

    def _reset_strategy_random(
        self, env_ids: torch.Tensor, start: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # sample random motion times (or zeros if start is True)
        num_samples = env_ids.shape[0]
        motion_ids = self._motion_loader.sample_motions(num_samples)
        times = np.zeros(num_samples) if start else self._motion_loader.sample_times(motion_ids)
        # sample random motions
        (
            dof_positions,
            dof_velocities,
            base_positions,
            base_rotations,
            base_linear_velocities,
            base_angular_velocities,
            key_body_positions,
        ) = self._motion_loader.sample(motion_ids=motion_ids, times=times)

        # get root transforms (the Tocabi torso)
        # motion_torso_index = self._motion_loader.get_body_index(["base_link"])[0]
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, 0:3] = base_positions + self.scene.env_origins[env_ids]
        root_state[:, 2] += 0.02  # lift the Tocabi slightly to avoid collisions with the ground
        root_state[:, 3:7] = base_rotations
        root_state[:, 7:10] = base_linear_velocities
        root_state[:, 10:13] = base_angular_velocities
        # get DOFs state
        dof_pos = dof_positions
        dof_vel = dof_velocities

        # update AMP observation
        amp_observations = self.collect_reference_motions(num_samples, motion_ids, times).to(dtype=torch.float32)
        self.amp_observation_buffer[env_ids] = amp_observations.view(num_samples, self.cfg.num_amp_observations, -1)

        return root_state, dof_pos, dof_vel

    # env methods

    def collect_reference_motions(self, num_samples: int, motion_ids = None, current_times: np.ndarray | None = None) -> torch.Tensor:
        # sample random motion times (or use the one specified)
        if current_times is None or motion_ids is None:
            motion_ids = self._motion_loader.sample_motions(num_samples)
            current_times = self._motion_loader.sample_times(motion_ids)
        motion_ids = np.tile(np.expand_dims(motion_ids, axis=-1), [1, self.cfg.num_amp_observations]).flatten()
        times = (
            np.expand_dims(current_times, axis=-1)
            - self.step_dt * np.arange(0, self.cfg.num_amp_observations)
        ).flatten()
        # get motions
        (
            dof_positions,
            dof_velocities,
            base_positions,
            base_rotations,
            base_linear_velocities,
            base_angular_velocities,
            key_body_positions,
        ) = self._motion_loader.sample(motion_ids=motion_ids, times=times)
        # compute AMP observation
        amp_observation = compute_amp_obs(
            dof_positions,
            dof_velocities,
            base_positions,
            base_rotations,
            base_linear_velocities,
            base_angular_velocities,
            key_body_positions,
        )
        return amp_observation.view(-1, self.amp_observation_size)


@torch.jit.script
def quaternion_to_tangent_and_normal(q: torch.Tensor) -> torch.Tensor:
    ref_tangent = torch.zeros_like(q[..., :3])
    ref_normal = torch.zeros_like(q[..., :3])
    ref_tangent[..., 0] = 1
    ref_normal[..., -1] = 1
    tangent = quat_rotate(q, ref_tangent)
    normal = quat_rotate(q, ref_normal)
    return torch.cat([tangent, normal], dim=len(tangent.shape) - 1)

@torch.jit.script
def calc_heading(q):
    ref_dir = torch.zeros_like(q[..., 0:3])
    ref_dir[..., 1] = 1
    rot_dir = quat_rotate(q, ref_dir)
    heading = torch.atan2(rot_dir[..., 1], rot_dir[..., 0])
    return heading

@torch.jit.script
def calc_heading_quat_inv(q):
    heading = calc_heading(q)
    axis = torch.zeros_like(q[..., 0:3])
    axis[..., -1] = 1

    heading_q = quat_from_angle_axis(-heading, axis)
    return heading_q

@torch.jit.script
def quat2mat(q):
    mat = torch.zeros(q.shape[0],4,4,device=q.device)
    w = q[:,0]
    x = q[:,1]
    y = q[:,2]
    z = q[:,3]
    mat[:,0,0] = w*w + x*x - y*y -z*z
    mat[:,0,1] = 2*x*y - 2*w*z
    mat[:,0,2] = 2*x*z + 2*w*y
    mat[:,1,0] = 2*x*y + 2*w*z
    mat[:,1,1] = w*w - x*x + y*y - z*z
    mat[:,1,2] = 2*y*z - 2*w*x
    mat[:,2,0] = 2*x*z - 2*w*y
    mat[:,2,1] = 2*y*z + 2*w*x
    mat[:,2,2] = w*w - x*x - y*y + z*z
    mat[:,3,3] = 1
    return mat

@torch.jit.script
def mat2euler(mat):
    FLOAT_EPS = float(2.220446049250313e-16)
    EPS4 = FLOAT_EPS*4
    cy = torch.sqrt(mat[:,0,0]*mat[:,0,0]+mat[:,1,0]*mat[:,1,0])
    condition = cy > EPS4
    z = torch.where(
        condition, 
        torch.atan2(mat[:,1,0],mat[:,0,0]),
        torch.atan2(-mat[:,0,1],mat[:,1,1])
    )
    y = torch.where(
        condition, 
        torch.atan2(-mat[:,2,0],cy),
        torch.atan2(-mat[:,2,0],cy)
    )
    x = torch.where(
        condition, 
        torch.atan2(mat[:,2,1],mat[:,2,2]),
        torch.zeros_like(y,dtype=torch.float)
    )
    return x, y, z

@torch.jit.script
def quat2euler(q):
    '''
    existing function from isaaclab "euler_xyz_from_quat" is not used in the code
    since, gimbal lock is not handled in that function, it is replaced with the following function
    '''
    return mat2euler(quat2mat(q))

@torch.jit.script
def compute_obs(
    dof_positions: torch.Tensor,
    dof_velocities: torch.Tensor,
    root_positions: torch.Tensor,
    root_rotations: torch.Tensor,
    root_linear_velocities: torch.Tensor,
    root_angular_velocities: torch.Tensor,
    key_body_positions: torch.Tensor,
    previous_actions: torch.Tensor,
    velocity_commands: torch.Tensor,
) -> torch.Tensor:
    root_pos_expand = root_positions.unsqueeze(-2)
    local_key_body_positions = key_body_positions - root_pos_expand

    head_rot_expand = calc_heading_quat_inv(root_rotations)
    head_rot_expand = head_rot_expand.repeat((1, local_key_body_positions.shape[1], 1))
    flat_end_pos = local_key_body_positions.view(local_key_body_positions.shape[0]*local_key_body_positions.shape[1], local_key_body_positions.shape[2])
    flat_heading_rot = head_rot_expand.view(head_rot_expand.shape[0]*head_rot_expand.shape[1], head_rot_expand.shape[2])
    local_end_pos = quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(local_key_body_positions.shape[0], local_key_body_positions.shape[1]*local_key_body_positions.shape[2])
    obs = torch.cat(
        (
            # dof_positions,                                          # joint positions
            # dof_velocities,                                         # joint velocities
            # root_positions[:, 2:3],                                 # root body height
            # torch.stack(quat2euler(root_rotations), dim=-1),        # root body orientation (roll, pitch, yaw)
            # root_linear_velocities,                                 # root body linear velocity  
            # root_angular_velocities,                                # root body angular velocity

            root_positions[:, 2:3],                                 # root body height
            torch.stack(quat2euler(root_rotations), dim=-1),        # root body orientation (roll, pitch, yaw)
            root_linear_velocities,                                 # root body linear velocity  
            root_angular_velocities,                                # root body angular velocity
            velocity_commands,                                      # velocity commands
            dof_positions,                                          # joint positions
            dof_velocities,                                         # joint velocities
            # flat_local_key_pos,
        ),
        dim=-1,
    )
    return obs

@torch.jit.script
def compute_amp_obs(
    dof_positions: torch.Tensor,
    dof_velocities: torch.Tensor,
    root_positions: torch.Tensor,
    root_rotations: torch.Tensor,
    root_linear_velocities: torch.Tensor,
    root_angular_velocities: torch.Tensor,
    key_body_positions: torch.Tensor,
) -> torch.Tensor:
    # calculation of the key body positions based on the root position and orientation
    root_pos_expand = root_positions.unsqueeze(-2)
    local_key_body_positions = key_body_positions - root_pos_expand

    head_rot_expand = calc_heading_quat_inv(root_rotations)
    head_rot_expand = head_rot_expand.repeat((1, local_key_body_positions.shape[1], 1))
    flat_end_pos = local_key_body_positions.view(local_key_body_positions.shape[0]*local_key_body_positions.shape[1], local_key_body_positions.shape[2])
    flat_heading_rot = head_rot_expand.view(head_rot_expand.shape[0]*head_rot_expand.shape[1], head_rot_expand.shape[2])
    local_end_pos = quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(local_key_body_positions.shape[0], local_key_body_positions.shape[1]*local_key_body_positions.shape[2])
    # flat_local_key_pos = (key_body_positions - root_positions.unsqueeze(-2)).view(key_body_positions.shape[0], -1)
    obs = torch.cat(
        (
            root_positions[:, 2:3],                                 # root body height
            quaternion_to_tangent_and_normal(root_rotations),       # root body orientation (tangent, normal)
            # root_linear_velocities,                               # root body linear velocity  
            # root_angular_velocities,                              # root body angular velocity
            dof_positions,                                          # joint positions
            dof_velocities,                                         # joint velocities
            flat_local_key_pos,                                     # key body positions
        ),
        dim=-1,
    )
    return obs
