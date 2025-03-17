# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .rough_env_cfg import JetRoughEnvCfg


@configclass
class JetFlatEnvCfg(JetRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # rewards
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.feet_air_time.weight = 5.0
        self.rewards.joint_deviation_hip.params["asset_cfg"].joint_names = [".*_HipRoll", ".*_HipYaw"]
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)


class JetFlatEnvCfg_PLAY(JetFlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False

        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
