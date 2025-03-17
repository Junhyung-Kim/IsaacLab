# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from dataclasses import MISSING

from isaaclab_assets import TOCABI_CFG

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
import isaaclab.envs.mdp as mdp
from isaaclab.managers import SceneEntityCfg

MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions")

@configclass
class EventCfg:
    """Configuration for Domain Randomization."""

    mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "damping_distribution_params": (-1.4, 1.5),
            "operation": "add",
            "distribution": "uniform",
        },
    )

    armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "armature_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 16.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class TocabiAmpEnvCfg(DirectRLEnvCfg):
    """Humanoid AMP environment config (base class)."""

    # env
    episode_length_s = 16.0
    decimation = 2

    # spaces
    num_obs_hist = 10
    num_obs_skip = 2
    num_obs_per_step = 1 + 3 + 6 + 3 + 12 + 12 # base h, base rpy, base v & w, vel_cmd, q pos, q vel

    action_space = 12
    observation_space = num_obs_hist * (num_obs_per_step + action_space) - action_space
    # observation_space = 52
    state_space = 0
    num_amp_observations = 2
    amp_observation_space = 37


    early_termination = True
    termination_height = 0.6

    motion_file: str = MISSING
    reference_body = "Pelvis_Link"
    reset_strategy = "random"  # default, random, random-start
    # reset_strategy = "default"  # default, random, random-start
    """Strategy to be followed when resetting each environment (humanoid's pose and joint states).

    * default: pose and joint states are set to the initial state of the asset.
    * random: pose and joint states are set by sampling motions at random, uniform times.
    * random-start: pose and joint states are set by sampling motion at the start (time zero).
    """

    # input commands range
    cmd_x_range = [-0.5, 0.5]
    cmd_y_range = [-0.0, 0.0]
    cmd_yaw_range = [-0.4, 0.4]

    desired_order = [
        'L_HipYaw_Joint', 'L_HipRoll_Joint', 'L_HipPitch_Joint', 'L_Knee_Joint', 'L_AnklePitch_Joint', 'L_AnkleRoll_Joint', 
        'R_HipYaw_Joint', 'R_HipRoll_Joint', 'R_HipPitch_Joint', 'R_Knee_Joint', 'R_AnklePitch_Joint', 'R_AnkleRoll_Joint',
        'Waist1_Joint', 'Waist2_Joint', 'Upperbody_Joint',
        'L_Shoulder1_Joint', 'L_Shoulder2_Joint', 'L_Shoulder3_Joint', 'L_Armlink_Joint', 'L_Elbow_Joint', 'L_Forearm_Joint', 'L_Wrist1_Joint', 'L_Wrist2_Joint',
        'Neck_Joint', 'Head_Joint',
        'R_Shoulder1_Joint', 'R_Shoulder2_Joint', 'R_Shoulder3_Joint', 'R_Armlink_Joint', 'R_Elbow_Joint', 'R_Forearm_Joint', 'R_Wrist1_Joint', 'R_Wrist2_Joint'
    ]   

    torque_limits = [333, 232, 263, 289, 222, 166] * 2

    p_gain = [2000.0, 5000.0, 4000.0, 3700.0, 3200.0, 3200.0] * 2 +\
             [6000.0, 10000.0, 10000.0] +\
             [400.0, 1000.0, 400.0, 400.0, 400.0, 400.0, 100.0, 100.0] +\
             [100.0, 100.0] +\
             [400.0, 1000.0, 400.0, 400.0, 400.0, 400.0, 100.0, 100.0]
    
    d_gain = [15.0, 50.0, 20.0, 25.0, 24.0, 24.0] * 2 +\
             [200.0, 100.0, 100.0] +\
             [10.0, 28.0, 10.0, 10.0, 10.0, 10.0, 3.0, 3.0] +\
             [5.0, 5.0] +\
             [10.0, 28.0, 10.0, 10.0, 10.0, 10.0, 3.0, 3.0]


    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 500,
        render_interval=decimation,
        physx=PhysxCfg(
            max_position_iteration_count=4,
            max_velocity_iteration_count=0,
            bounce_threshold_velocity=0.04,
            gpu_found_lost_pairs_capacity=2**23,
            gpu_total_aggregate_pairs_capacity=2**23,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=True)

    # robot
    robot: ArticulationCfg = TOCABI_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )

    # events
    events: EventCfg = EventCfg()

@configclass
class TocabiAmpWalkEnvCfg(TocabiAmpEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "tocabi_motions.yaml")
