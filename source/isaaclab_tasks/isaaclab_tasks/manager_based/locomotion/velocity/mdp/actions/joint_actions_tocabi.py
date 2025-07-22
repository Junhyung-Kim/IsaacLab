# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.envs.mdp.actions.joint_actions import JointAction


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import actions_cfg


class JointPositionToLimitsTocabiLowLevelAction(ActionTerm):
    cfg: actions_cfg.JointPositionToLimitsTocabiLowLevelActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor | float
    """The scaling factor applied to the input action."""
    _clip: torch.Tensor
    """The clip applied to the input action."""

    def __init__(self, cfg: actions_cfg.JointPositionToLimitsTocabiLowLevelActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)

        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.lower_joint_names)
        self._joint_ids = [self._asset.data.joint_names.index(joint_name) for joint_name in self.cfg.lower_joint_names]
        self._upper_joint_ids = [self._asset.data.joint_names.index(joint_name) for joint_name in self.cfg.upper_joint_names]
        self._default_upper_joint_pos = self._asset.data.default_joint_pos[:, self._upper_joint_ids]

        self._num_joints = len(self._joint_ids)
        # log the resolved joint names for debugging
        omni.log.info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )

        # Avoid indexing across all joints for efficiency
        if self._num_joints == self._asset.num_joints:
            self._joint_ids = slice(None)

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)

        # parse scale
        if isinstance(cfg.scale, (float, int)):
            self._scale = float(cfg.scale)
        elif isinstance(cfg.scale, dict):
            self._scale = torch.ones(self.num_envs, self.action_dim, device=self.device)
            # resolve the dictionary config
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.scale, self._joint_names)
            self._scale[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported scale type: {type(cfg.scale)}. Supported types are float and dict.")
        # parse clip
        if self.cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor([[-float("inf"), float("inf")]], device=self.device).repeat(
                    self.num_envs, self.action_dim, 1
                )
                index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.clip, self._joint_names)
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
            else:
                raise ValueError(f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict.")

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        # apply affine transformations
        self._processed_actions = actions * self._scale
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )
        # rescale the position targets if configured
        # this is useful when the input actions are in the range [-1, 1]
        if self.cfg.rescale_to_limits:
            # clip to [-1, 1]
            actions = self._processed_actions.clamp(-1.0, 1.0)
            self._raw_actions[:] = actions
            # rescale within the joint limits
            actions = math_utils.unscale_transform(
                actions,
                self._asset.data.soft_joint_pos_limits[:, self._joint_ids, 0],
                self._asset.data.soft_joint_pos_limits[:, self._joint_ids, 1],
            )
            self._processed_actions[:] = actions[:]

    def apply_actions(self):
        # set position targets
        # self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)
        # self._asset.set_joint_position_target(self._default_upper_joint_pos, joint_ids=self._upper_joint_ids)
        target_pos = torch.cat([self.processed_actions, self._default_upper_joint_pos], dim=1)
        joint_ids_ordered = self._joint_ids + self._upper_joint_ids
        self._asset.set_joint_position_target(target_pos, joint_ids=joint_ids_ordered)
    
        

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0



class JointPositionAction(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    cfg: actions_cfg.TocabiJointPositionActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.TocabiJointPositionActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # use default joint positions as offset
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.lower_joint_names)
        self._joint_ids = [self._asset.data.joint_names.index(joint_name) for joint_name in self.cfg.lower_joint_names]
        self._upper_joint_ids = [self._asset.data.joint_names.index(joint_name) for joint_name in self.cfg.upper_joint_names]
        self._default_upper_joint_pos = self._asset.data.default_joint_pos[:, self._upper_joint_ids]

    def apply_actions(self):
        # set position targets
        # self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)
        target_pos = torch.cat([self.processed_actions, self._default_upper_joint_pos], dim=1)
        joint_ids_ordered = self._joint_ids + self._upper_joint_ids
        self._asset.set_joint_position_target(target_pos, joint_ids=joint_ids_ordered)