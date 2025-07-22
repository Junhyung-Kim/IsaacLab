# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the 33-DOFs Mujoco Humanoid robot TOCABI."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##

JET2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/yong/IsaacGym/IsaacLab/source/isaaclab_assets/data/Robots/Jet2/Jet2_.usd",
        # usd_path=f"/home/yong/IsaacGym/IsaacLab/source/isaaclab_assets/data/Robots/Jet2/Jet2_viz_coll.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            # sleep_threshold=0.005,
            # stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    soft_joint_pos_limit_factor=0.9,
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.89),
        joint_pos={
            "L_HipYaw": 0.0,
            "L_HipRoll": 0.034906585,
            "L_HipPitch": -0.034906585,
            "L_KneePitch": 0.733038285,
            "L_AnklePitch": -0.6981317,
            "L_AnkleRoll": -0.034906585,
            "R_HipYaw": 0.0,
            "R_HipRoll": -0.034906585,
            "R_HipPitch": 0.034906585,
            "R_KneePitch": -0.733038285,
            "R_AnklePitch": 0.6981317,
            "R_AnkleRoll": 0.034906585,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "jet_leg": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_HipYaw",
                ".*_HipRoll",
                ".*_HipPitch",
                ".*_KneePitch",
                ".*_AnklePitch",
                ".*_AnkleRoll"],
            stiffness={
                ".*_HipYaw": 600,
                ".*_HipRoll": 100,
                ".*_HipPitch": 100,
                ".*_KneePitch": 600,
                ".*_AnklePitch": 600,
                ".*_AnkleRoll": 100,
            },
            damping=10.0,
            effort_limit_sim=30,
            velocity_limit_sim=1.0,
            armature=0.01,
        ),
    },
)
