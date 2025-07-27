# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, ObservationsCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.rough_env_cfg import H1Rewards
from isaaclab_tasks.manager_based.locomotion.velocity.mdp.actions import TocabiActionCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm

from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

##
# Pre-defined configs
##
from isaaclab_assets import Tocabi_CFG  # isort: skip

@configclass
class TocabiActionsCfg:
    # joint_pos = mdp.JointPositionToLimitsActionCfg(asset_name="robot", joint_names=[".*"], rescale_to_limits=True)
    # joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=False)
    joint_pos = TocabiActionCfg(
        asset_name="robot", 
        lower_joint_names=["L_HipYaw_Joint", "L_HipRoll_Joint", "L_HipPitch_Joint", "L_Knee_Joint", "L_AnklePitch_Joint", "L_AnkleRoll_Joint",
                           "R_HipYaw_Joint", "R_HipRoll_Joint", "R_HipPitch_Joint", "R_Knee_Joint", "R_AnklePitch_Joint", "R_AnkleRoll_Joint"],
        upper_joint_names=["Waist1_Joint", "Waist2_Joint", "Upperbody_Joint",
                           "L_Shoulder1_Joint", "L_Shoulder2_Joint", "L_Shoulder3_Joint", "L_Elbow_Joint", "L_Armlink_Joint", "L_Forearm_Joint", "L_Wrist1_Joint", "L_Wrist2_Joint",
                           "Neck_Joint", "Head_Joint",
                           "R_Shoulder1_Joint", "R_Shoulder2_Joint", "R_Shoulder3_Joint", "R_Elbow_Joint", "R_Armlink_Joint", "R_Forearm_Joint", "R_Wrist1_Joint", "R_Wrist2_Joint"],
        pd_control=True,

        p_gains = [2000.0, 5000.0, 4000.0, 3700.0, 3200.0, 3200.0, 
                   2000.0, 5000.0, 4000.0, 3700.0, 3200.0, 3200.0, 
                   6000.0, 10000.0, 10000.0, 
                   400.0, 1000.0, 400.0, 400.0, 400.0, 400.0, 100.0, 100.0, 
                   100.0, 100.0, 
                   400.0, 1000.0, 400.0, 400.0, 400.0, 400.0, 100.0, 100.0],

        d_gains = [15.0, 50.0, 20.0, 25.0, 24.0, 24.0, 
                   15.0, 50.0, 20.0, 25.0, 24.0, 24.0, 
                   200.0, 100.0, 100.0, 
                   10.0, 28.0, 10.0, 10.0, 10.0, 10.0, 3.0, 3.0, 
                   3.0, 3.0, 
                   10.0, 28.0, 10.0, 10.0, 10.0, 10.0, 3.0, 3.0],

        torque_limits= [333, 232, 263, 289, 222, 166,
                        333, 232, 263, 289, 222, 166,
                        303, 303, 303,
                        64, 64, 64, 64, 23, 23, 10, 10,
                        10, 10,
                        64, 64, 64, 64, 23, 23, 10, 10],

    )
    # joint_pos = mdp.TocabiJointPositionActionCfg(
    #     asset_name="robot",
    #     joint_names=["L_HipYaw_Joint", "L_HipRoll_Joint", "L_HipPitch_Joint", "L_Knee_Joint", "L_AnklePitch_Joint", "L_AnkleRoll_Joint",
    #                 "R_HipYaw_Joint", "R_HipRoll_Joint", "R_HipPitch_Joint", "R_Knee_Joint", "R_AnklePitch_Joint", "R_AnkleRoll_Joint"],
    #     lower_joint_names=["L_HipYaw_Joint", "L_HipRoll_Joint", "L_HipPitch_Joint", "L_Knee_Joint", "L_AnklePitch_Joint", "L_AnkleRoll_Joint",
    #                        "R_HipYaw_Joint", "R_HipRoll_Joint", "R_HipPitch_Joint", "R_Knee_Joint", "R_AnklePitch_Joint", "R_AnkleRoll_Joint"],
    #     upper_joint_names=["Waist1_Joint", "Waist2_Joint", "Upperbody_Joint",
    #                        "L_Shoulder1_Joint", "L_Shoulder2_Joint", "L_Shoulder3_Joint", "L_Elbow_Joint", "L_Armlink_Joint", "L_Forearm_Joint", "L_Wrist1_Joint", "L_Wrist2_Joint",
    #                        "Neck_Joint", "Head_Joint",
    #                        "R_Shoulder1_Joint", "R_Shoulder2_Joint", "R_Shoulder3_Joint", "R_Elbow_Joint", "R_Armlink_Joint", "R_Forearm_Joint", "R_Wrist1_Joint", "R_Wrist2_Joint"],
    # )

@configclass
class TocabiObservations(ObservationsCfg):
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.1, n_max=0.1))
        clock_input = ObsTerm(func=mdp.clock_input, params={"step_time": 2.0})
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_ordered_rel, 
                            noise=Unoise(n_min=-0.1, n_max=0.1),
                            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["L_HipYaw_Joint", "L_HipRoll_Joint", "L_HipPitch_Joint", "L_Knee_Joint", "L_AnklePitch_Joint", "L_AnkleRoll_Joint",
                                                                                     "R_HipYaw_Joint", "R_HipRoll_Joint", "R_HipPitch_Joint", "R_Knee_Joint", "R_AnklePitch_Joint", "R_AnkleRoll_Joint"])})
                                                                                    #  "Waist1_Joint", "Waist2_Joint", "Upperbody_Joint",
                                                                                    #  "L_Shoulder1_Joint", "L_Shoulder2_Joint", "L_Shoulder3_Joint", "L_Elbow_Joint", "L_Armlink_Joint", "L_Forearm_Joint", "L_Wrist1_Joint", "L_Wrist2_Joint",
                                                                                    #  "Neck_Joint", "Head_Joint",
                                                                                    #  "R_Shoulder1_Joint", "R_Shoulder2_Joint", "R_Shoulder3_Joint", "R_Elbow_Joint", "R_Armlink_Joint", "R_Forearm_Joint", "R_Wrist1_Joint", "R_Wrist2_Joint"])})
        joint_vel = ObsTerm(func=mdp.joint_vel_ordered_rel, 
                            noise=Unoise(n_min=-0.1, n_max=0.1),
                            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["L_HipYaw_Joint", "L_HipRoll_Joint", "L_HipPitch_Joint", "L_Knee_Joint", "L_AnklePitch_Joint", "L_AnkleRoll_Joint",
                                                                                     "R_HipYaw_Joint", "R_HipRoll_Joint", "R_HipPitch_Joint", "R_Knee_Joint", "R_AnklePitch_Joint", "R_AnkleRoll_Joint"])})
                                                                                    #  "Waist1_Joint", "Waist2_Joint", "Upperbody_Joint",
                                                                                    #  "L_Shoulder1_Joint", "L_Shoulder2_Joint", "L_Shoulder3_Joint", "L_Elbow_Joint", "L_Armlink_Joint", "L_Forearm_Joint", "L_Wrist1_Joint", "L_Wrist2_Joint",
                                                                                    #  "Neck_Joint", "Head_Joint",
                                                                                    #  "R_Shoulder1_Joint", "R_Shoulder2_Joint", "R_Shoulder3_Joint", "R_Elbow_Joint", "R_Armlink_Joint", "R_Forearm_Joint", "R_Wrist1_Joint", "R_Wrist2_Joint"])})
        # actions = ObsTerm(func=mdp.last_action, params={"action_name": "joint_pos"})
        actions = ObsTerm(func=mdp.last_processed_action, params={"action_name": "joint_pos"})
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5

    policy: PolicyCfg = PolicyCfg()


@configclass
class TocabiRewards:
    """Reward terms for the MDP."""

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": 0.2},
    )

    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=0.5, params={"command_name": "base_velocity", "std": 0.2}
    )

    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*AnkleRoll.*"),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*AnkleRoll.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*AnkleRoll.*"),
        },
    )
    # feet_contact_force = RewTerm(
    #     func=mdp.feet_contact_force,
    #     weight=-0.5,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*AnkleRoll.*"), "threshold": 1.3},
    # )


    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_HipYaw_Joint", ".*_HipRoll_Joint"])},
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*Ankle.*")}
    )
    joint_deviation_knee = RewTerm(
        func=mdp.joint_deviation_neg,
        weight=0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Knee_Joint"])},
    )

@configclass
class Tocabi_H1Rewards(H1Rewards):
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*AnkleRoll.*"),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*AnkleRoll.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*AnkleRoll.*"),
        },
    )
    # Penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*Ankle.*")}
    )
    # Penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_HipYaw_Joint", ".*_HipRoll_Joint"])},
    )
    joint_deviation_arms = None
    joint_deviation_torso = None
    undesired_contacts = None
    flat_orientation_l2 = None
    dof_torques_l2 = None
    action_rate_l2 = None
    dof_acc_l2 = None

@configclass
class Tocabi_WordlModelRewards:
    lin_vel_xy_tracking = RewTerm(func=mdp.track_lin_vel_xy_base_frame_exp, weight=1.0, params={"command_name": "base_velocity", "omega": 5.0})
    ang_vel_z_tracking = RewTerm(func=mdp.track_ang_vel_z_world_exp_tocabi, weight=1.0, params={"command_name": "base_velocity", "omega": 7.0})
    orientation_tracking = RewTerm(func=mdp.orientation_tracking, weight=1.0, params={"command_name": "base_velocity", "omega": 5.0})
    base_height_tracking = RewTerm(func=mdp.base_height_tracking, weight=0.5, params={"height": 0.92, "omega": 10.0})
    # ----------------------------------------------------------------------------------------------------------------------------------------------
    periodic_force = RewTerm(func=mdp.periodic_force, weight=1.0, params={"scale": 1300, "phase_time": 2.0,
                        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["L_AnkleRoll_Link", "R_AnkleRoll_Link"]),
                        "left_foot": "L_AnkleRoll_Link", "right_foot": "R_AnkleRoll_Link"})
    # periodic_velocity = RewTerm(func=mdp.periodic_velocity, weight=1.0, params={"scale": 0.5, "phase_time": 2.0,
    #                     "asset_cfg": SceneEntityCfg("robot", body_names=["L_AnkleRoll_Link", "R_AnkleRoll_Link"]),
    #                     "left_foot": "L_AnkleRoll_Link", "right_foot": "R_AnkleRoll_Link"})
    # ----------------------------------------------------------------------------------------------------------------------------------------------
    feet_height_tracking = RewTerm(func=mdp.feet_height_tracking, weight=1.0, params={"omega": 5.0, "foot_height": 0.1, "phase_time": 2.0, "kappa": 4.0, "offset": 0.1585,
                        "asset_cfg": SceneEntityCfg("robot", body_names=["L_AnkleRoll_Link", "R_AnkleRoll_Link"]),
                        "left_foot": "L_AnkleRoll_Link", "right_foot": "R_AnkleRoll_Link"})
    feet_velocity_z_tracking = RewTerm(func=mdp.feet_velocity_z_tracking, weight=0.5, params={"omega": 3.0, "foot_height": 0.1, "phase_time": 2.0, "kappa": 4.0,
                        "asset_cfg": SceneEntityCfg("robot", body_names=["L_AnkleRoll_Link", "R_AnkleRoll_Link"]),
                        "left_foot": "L_AnkleRoll_Link", "right_foot": "R_AnkleRoll_Link"})
    # feet_velocity_y_tracking = RewTerm(func=mdp.feet_velocity_y_tracking, weight=0.5, params={"omega": 5.0,
    #                     "asset_cfg": SceneEntityCfg("robot", body_names=["L_AnkleRoll_Link", "R_AnkleRoll_Link"]),
    #                     "left_foot": "L_AnkleRoll_Link", "right_foot": "R_AnkleRoll_Link"})
    # ----------------------------------------------------------------------------------------------------------------------------------------------
    large_contact = RewTerm(func=mdp.large_contact_force, weight=-0.005, params={"threshold": 100.0,
                        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["L_AnkleRoll_Link", "R_AnkleRoll_Link"]),
                        "asset_cfg": SceneEntityCfg("robot")})
    default_joint = RewTerm(func=mdp.default_joint_pos, weight=0.2, params={"omega": 2.0, "asset_cfg": SceneEntityCfg("robot",
                        joint_names=["L_HipYaw_Joint", "L_HipRoll_Joint", "L_HipPitch_Joint", "L_Knee_Joint", "L_AnklePitch_Joint", "L_AnkleRoll_Joint",
                                     "R_HipYaw_Joint", "R_HipRoll_Joint", "R_HipPitch_Joint", "R_Knee_Joint", "R_AnklePitch_Joint", "R_AnkleRoll_Joint"])})
    # ----------------------------------------------------------------------------------------------------------------------------------------------
    joint_deviation_hip = RewTerm(func=mdp.joint_deviation_l1, weight=-0.05, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_HipRoll_Joint"])})
    joint_deviation_ankle = RewTerm(func=mdp.joint_deviation_l1, weight=-0.05, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_AnkleRoll_Joint"])})
    feet_flat = RewTerm(func=mdp.flat_orientation_l2, weight=-0.05, params={"asset_cfg": SceneEntityCfg("robot", body_names=["L_AnkleRoll_Link", "R_AnkleRoll_Link"])})

@configclass
class TocabiTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="Pelvis_Link"), "threshold": 1.0},
    )
    root_bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.79, "asset_cfg": SceneEntityCfg("robot")},
    )
    
@configclass
class TocabiEventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )
    
    randomize_rigid_body_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    randomize_actuator_properties = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.0, 0.0),
            "armature_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    randomize_joint_properties = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.0, 0.0),
            "damping_distribution_params": (0.0, 3.0),
            "operation": "add",
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
        },
    )

    push_robot = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(5.0, 6.0),
        params={
            "force_range": (-100.0, 100.0),
            # "force_range": (-0.0, 0.0),
            "torque_range": (-0.0, 0.0),    
            "asset_cfg": SceneEntityCfg("robot", body_names=["Pelvis_Link"]),
        },
    )

    # random_torque_injection = EventTerm(
    #     func=mdp.apply_external_force_torque,
    #     mode="reset",
    #     params={
    #         "torque_range": (-0.0, 0.0),
    #         "force_range": (-0.0, 0.0),
    #         "asset_cfg": SceneEntityCfg("robot", body_names=[".*"]),
    #     },
    # )


@configclass
class TocabiRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    # rewards: TocabiRewards = TocabiRewards()
    # rewards: Tocabi_H1Rewards = Tocabi_H1Rewards()
    rewards: Tocabi_WordlModelRewards = Tocabi_WordlModelRewards()
    observations: TocabiObservations = TocabiObservations()
    actions: TocabiActionsCfg = TocabiActionsCfg()
    terminations: TocabiTerminations = TocabiTerminations()
    events: TocabiEventCfg = TocabiEventCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        # self.scene.contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/base_link/.*", history_length=3, track_air_time=True)
        self.scene.robot = Tocabi_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        if self.scene.height_scanner:
            self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/Pelvis_Link"
        
        # Randomization
        # self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        # self.events.base_external_force_torque.params["asset_cfg"].body_names = ["Pelvis_Link"]
        # self.events.reset_base.params = {
        #     "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
        #     "velocity_range": {
        #         "x": (0.0, 0.0),
        #         "y": (0.0, 0.0),
        #         "z": (0.0, 0.0),
        #         "roll": (0.0, 0.0),
        #         "pitch": (0.0, 0.0),
        #         "yaw": (0.0, 0.0),
        #     },
        # }
        # self.events.base_com = None


        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (-0.0, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)

        self.decimation = 2
        self.episode_length_s = 10.0
        self.sim.dt = 0.005

        self.viewer.eye = (-3.0, 3.0, 1.3)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = -1


@configclass
class TocabiRoughEnvCfg_PLAY(TocabiRoughEnvCfg):
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

        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None

