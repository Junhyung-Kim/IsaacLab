from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .cmd_cfgs import WalkingPhaseCommandCfg, FirstFootStepCommandCfg

class WalkingPhaseCommand(CommandTerm):
    """Command generator that generates a walking phase command."""

    cfg: WalkingPhaseCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: WalkingPhaseCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.phase_time_cmd = torch.zeros(self.num_envs, 1, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.phase_time_cmd
    
    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        # sample phase time command
        r = torch.empty(len(env_ids), device=self.device)
        self.phase_time_cmd[env_ids, 0] = r.uniform_(*self.cfg.ranges.phase_time)

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

class FirstFootStepCommand(CommandTerm):
    """Command generator that generates a first foot step command."""

    cfg: FirstFootStepCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: FirstFootStepCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.first_foot_step = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.first_foot_step
    
    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)
        self.first_foot_step[env_ids, 0] = r.uniform_(0.0, 1.0) <= self.cfg.is_rfoot_first

    def _update_command(self):
        pass
    