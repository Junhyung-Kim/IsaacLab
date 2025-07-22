# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg, ObservationsCfg

##
# Pre-defined configs
##
from isaaclab_assets import H1_MINIMAL_CFG  # isort: skip


@configclass
class H1Rewards(RewardsCfg):
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    lin_vel_z_l2 = None
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_link"),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_link"),
        },
    )
    # Penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_ankle")}
    )
    # Penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw", ".*_hip_roll"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_.*", ".*_elbow"])},
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.1, params={"asset_cfg": SceneEntityCfg("robot", joint_names="torso")}
    )

@configclass
class WorldModelRewards:
    lin_vel_xy_tracking = RewTerm(func=mdp.track_lin_vel_xy_base_frame_exp, weight=1.0, params={"command_name": "base_velocity", "omega": 5.0})
    ang_vel_z_tracking = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={"command_name": "base_velocity", "std": 7.0})
    orientation_tracking = RewTerm(func=mdp.orientation_tracking, weight=1.0, params={"command_name": "base_velocity", "omega": 5.0})
    base_height_tracking = RewTerm(func=mdp.base_height_tracking, weight=0.5, params={"height": 0.9, "omega": 10.0})
    # ----------------------------------------------------------------------------------------------------------------------------------------------
    periodic_force = RewTerm(func=mdp.periodic_force, weight=1.0, params={"scale": 650, "phase_time": 2.0,
                        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_ankle_link", "right_ankle_link"]),
                        "right_foot": "right_ankle_link", "left_foot": "left_ankle_link"})
    periodic_velocity = RewTerm(func=mdp.periodic_velocity, weight=1.0, params={"scale": 0.5, "phase_time": 2.0,
                        "asset_cfg": SceneEntityCfg("robot", body_names=["left_ankle_link", "right_ankle_link"]),
                        "right_foot": "right_ankle_link", "left_foot": "left_ankle_link"})
    # ----------------------------------------------------------------------------------------------------------------------------------------------
    feet_height_tracking = RewTerm(func=mdp.feet_height_tracking, weight=1.0, params={"omega": 5.0, "foot_height": 0.1, "phase_time": 2.0, "kappa": 4.0, "offset": 0.0695,
                        "asset_cfg": SceneEntityCfg("robot", body_names=["left_ankle_link", "right_ankle_link"]),
                        "right_foot": "right_ankle_link", "left_foot": "left_ankle_link"})
    feet_velocity_tracking = RewTerm(func=mdp.feet_velocity_z_tracking, weight=0.5, params={"omega": 3.0, "foot_height": 0.1, "phase_time": 2.0, "kappa": 4.0,
                        "asset_cfg": SceneEntityCfg("robot", body_names=["left_ankle_link", "right_ankle_link"]),
                        "right_foot": "right_ankle_link", "left_foot": "left_ankle_link"})
    # ----------------------------------------------------------------------------------------------------------------------------------------------
    large_contact = RewTerm(func=mdp.large_contact_force, weight=-0.01, params={"threshold": 100.0,
                        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_ankle_link", "right_ankle_link"]),
                        "asset_cfg": SceneEntityCfg("robot")})
    default_joint = RewTerm(func=mdp.default_joint_pos, weight=0.2, params={"omega": 2.0, "asset_cfg": SceneEntityCfg("robot",
                        joint_names=["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
                                     "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"])})
    # ----------------------------------------------------------------------------------------------------------------------------------------------
    joint_deviation_hip = RewTerm(func=mdp.joint_deviation_l1, weight=-0.05, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw"])})
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_.*", ".*_elbow"])},
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.1, params={"asset_cfg": SceneEntityCfg("robot", joint_names="torso")}
    )
    

@configclass
class WorldModelObs:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        clock_input = ObsTerm(func=mdp.clock_input, params={"step_time": 2.0})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class H1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: H1Rewards = H1Rewards()
    # rewards: WorldModelRewards = WorldModelRewards()
    # observations: WorldModelObs = WorldModelObs()
    

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        self.scene.robot = H1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        if self.scene.height_scanner:
            self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"

        # Randomization
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = [".*torso_link"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.base_com = None

        # Rewards
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.dof_torques_l2.weight = 0.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.25e-7

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # Terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = ".*torso_link"


@configclass
class H1RoughEnvCfg_PLAY(H1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
