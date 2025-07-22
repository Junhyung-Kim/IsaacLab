# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-AMP-Flat-Tocabi-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:TocabiFlatEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_amp_cfg.yaml",
    },
)

gym.register(
    id="Isaac-AMP-Flat-Tocabi-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:TocabiFlatEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_amp_cfg.yaml",
    },
)

# gym.register(
#     id="Isaac-AMP-Rough-Tocabi-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.rough_env_cfg:TocabiRoughEnvCfg",
#         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_amp_rough_cfg.yaml",
#     },
# )

# gym.register(
#     id="Isaac-AMP-Rough-Tocabi-Play-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.rough_env_cfg:TocabiRoughEnvCfg_PLAY",
#         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_amp_rough_cfg.yaml",
#     },
# )
