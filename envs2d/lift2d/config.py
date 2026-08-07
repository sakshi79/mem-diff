"""
Tunable parameters for the 2D lift env.
"""

from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Lift2DConfig:
    # display/image
    window_size: int = 512
    render_size: int = 96
    render_action: bool = True

    # arena
    wall_thickness: float = 3.0
    gravity: float = 980.0
    damping: float = 0.995

    # gripper
    finger_length: float = 60.0
    finger_width: float = 14.0
    finger_gap_max: float = 42.0
    finger_gap_min: float = 18.0

    # block
    block_size: float = 36.0
    
    # grasp
    grasp_threshold: float = 58.0
    grip_close_threshold: float = 0.3
    grasp_max_force: float = 6000.0

    # control params
    sim_hz: int = 100
    control_hz: int = 10
    k_p: float = 180.0
    k_v: float = 30.0
    base_margin: float = 35.0

    # # Finger (jaw) force control
    # self.k_grip = 60.0  # jaw PD stiffness
    # self.c_grip = 8.0  # jaw PD damping
    # self.max_grip_force = 800.0 


    
