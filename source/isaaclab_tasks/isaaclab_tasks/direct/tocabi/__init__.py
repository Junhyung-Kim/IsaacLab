# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Humanoid locomotion environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Velocity-Flat-Tocabi-Direct-v0",
    entry_point=f"{__name__}.tocabi_env:TocabiEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tocabi_env_cfg:TocabiFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TocabiFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Tocabi-Direct-v0",
    entry_point=f"{__name__}.tocabi_env:TocabiEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tocabi_env_cfg:TocabiRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TocabiRoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Tocabi-Direct-Play-v0",
    entry_point=f"{__name__}.tocabi_env:TocabiEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tocabi_env_cfg:TocabiFlatPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TocabiFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Tocabi-Direct-Play-v0",
    entry_point=f"{__name__}.tocabi_env:TocabiEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tocabi_env_cfg:TocabiRoughPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TocabiRoughPPORunnerCfg",
    },
)