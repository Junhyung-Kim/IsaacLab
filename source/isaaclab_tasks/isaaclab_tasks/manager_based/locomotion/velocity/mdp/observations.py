from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera
import math

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

from isaaclab.envs.mdp import *


def joint_pos_ordered(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # need to reorder joint_ids to the order of the joint_names
    joint_ids_reordered = [asset.data.joint_names.index(joint_name) for joint_name in asset_cfg.joint_names]
    return asset.data.joint_pos[:, joint_ids_reordered]

def joint_vel_ordered(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """The joint velocities of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids_reordered = [asset.data.joint_names.index(joint_name) for joint_name in asset_cfg.joint_names]
    return asset.data.joint_vel[:, joint_ids_reordered]

def joint_pos_ordered_rel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # need to reorder joint_ids to the order of the joint_names
    joint_ids_reordered = [asset.data.joint_names.index(joint_name) for joint_name in asset_cfg.joint_names]
    return asset.data.joint_pos[:, joint_ids_reordered] - asset.data.default_joint_pos[:, joint_ids_reordered]

def joint_vel_ordered_rel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """The joint velocities of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids_reordered = [asset.data.joint_names.index(joint_name) for joint_name in asset_cfg.joint_names]
    return asset.data.joint_vel[:, joint_ids_reordered] - asset.data.default_joint_vel[:, joint_ids_reordered]

def clock_input(env: ManagerBasedRLEnv, step_time: float) -> torch.Tensor:
    """The input phase of the environment.
    
    The input phase is a clock input tensor that has cycle.
    """
    clock_input_sin = torch.sin(2 * math.pi * env.episode_length_buf.unsqueeze(1) * env.step_dt / step_time)
    clock_input_cos = torch.cos(2 * math.pi * env.episode_length_buf.unsqueeze(1) * env.step_dt / step_time)
    # return torch.cat([clock_input_sin, clock_input_cos], dim=1)
    return clock_input_sin

def last_processed_action(env: ManagerBasedEnv, action_name: str | None = None) -> torch.Tensor:
    """The last input action to the environment.

    The name of the action term for which the action is required. If None, the
    entire action tensor is returned.
    """
    if action_name is None:
        return env.action_manager.action
    else:
        return env.action_manager.get_term(action_name).processed_actions