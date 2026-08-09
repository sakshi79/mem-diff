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
    solver_iterations: int = 25     # stiffer contact solve → less jitter

    # gripper
    finger_length: float = 60.0
    finger_width: float = 14.0
    finger_gap_max: float = 42.0
    finger_gap_min: float = 8.0
    finger_mass: float = 0.5

    # block
    block_size: float = 36.0
    block_mass: float = 1.0

    # contact friction (pymunk multiplies the two shapes' friction at a contact)
    finger_friction: float = 1.8
    block_friction:  float = 1.5

    # grasp — force is COMPUTED from physics (see __post_init__), not hand-tuned
    grip_close_threshold: float = 0.3
    grip_safety_factor:   float = 2.5   # margin over the minimum holding force

    # control params
    sim_hz: int = 100
    control_hz: int = 10
    k_p: float = 180.0
    k_v: float = 30.0
    base_margin: float = 35.0

    # Finger (jaw) force control  (max_grip_force is COMPUTED in __post_init__)
    k_grip:         float = 60.0    # jaw PD stiffness
    c_grip:         float = 8.0     # jaw PD damping

    # base motion
    max_base_speed: float = 200.0   # cap so a held block isn't yanked past the friction limit

    def __post_init__(self):
        # Hold the block by friction on BOTH faces: 2·μ·F ≥ m·g
        #   → F = S · m·g / (2μ),  μ = μ_finger · μ_block (pymunk product rule)
        mu = self.finger_friction * self.block_friction
        self.max_grip_force = self.grip_safety_factor * self.block_mass * self.gravity / (2.0 * mu)

