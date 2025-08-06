import math
from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass

from .phase_time_cmd import WalkingPhaseCommand, FirstFootStepCommand

@configclass
class WalkingPhaseCommandCfg(CommandTermCfg):
    """Configuration for the walking phase command generator."""

    class_type: type = WalkingPhaseCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    @configclass
    class Ranges:
        """Uniform distribution ranges for the walking phase commands."""

        phase_time: tuple[float, float] = MISSING
        """Range for the phase time (in s)."""

    ranges: Ranges = MISSING
    """Distribution ranges for the walking phase commands."""

@configclass
class FirstFootStepCommandCfg(CommandTermCfg):
    """Configuration for the first foot step command generator."""

    class_type: type = FirstFootStepCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    is_rfoot_first: float = 0.5
    """Probability of the right foot being the first foot."""