# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
AMP Humanoid locomotion environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# gym.register(
#     id="Isaac-Humanoid-AMP-Dance-Direct-v0",
#     entry_point=f"{__name__}.humanoid_amp_env:HumanoidAmpEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.humanoid_amp_env_cfg:HumanoidAmpDanceEnvCfg",
#         "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_dance_amp_cfg.yaml",
#     },
# )

# gym.register(
#     id="Isaac-Humanoid-AMP-Run-Direct-v0",
#     entry_point=f"{__name__}.humanoid_amp_env:HumanoidAmpEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.humanoid_amp_env_cfg:HumanoidAmpRunEnvCfg",
#         "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_run_amp_cfg.yaml",
#     },
# )

gym.register(
    id="Isaac-Tocabi-AMP-Walk-Direct-v0",
    entry_point=f"{__name__}.tocabi_amp_env:TocabiAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tocabi_amp_env_cfg:TocabiAmpWalkEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_walk_amp_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_walk_amp_cfg.yaml",
    },
)
