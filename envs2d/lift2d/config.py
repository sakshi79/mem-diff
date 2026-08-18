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

    # task success: reward = 1 when block is lifted ≥ lift_threshold above floor rest
    # Matches robomimic Lift (0.04 m); our units are ~cm, so 4.0.
    lift_threshold: float = 4.0
    # lift_threshold: float = 50.0

    # control params
    sim_hz: int = 100
    control_hz: int = 10
    k_p: float = 180.0
    k_v: float = 30.0
    base_margin: float = 35.0

    # Our horizon is 200 @ control_hz=10 = 20s
    # Robomimic 3D has horizon = 400 @ robosuite's 20Hz, ie 200 steps (steps = horizon/control_freq) — (DESIGN_NOTES.md §7.6)
    # Our control_hz=10, hence for same steps, horizon = 200. Reducing this since our env is also simpler due to less DOF
    horizon: int = 150  # episode length, in env.step() calls

    # Finger (jaw) motor: velocity-controlled, force-limited.
    # max_grip_force (COMPUTED in __post_init__) = per-jaw normal force N = S·m·g/(2μ).
    jaw_speed: float = 200.0   # commanded open/close speed (px/s); impact knob — keep modest
    k_motor:   float = 10.0    # velocity-servo gain (only needs k_motor·jaw_speed ≫ N)

    # base motion
    max_base_speed: float = 200.0 # a speed cap ensures solver gets enough time to solve constraints
    # prevents issues like tunneling effect

    def __post_init__(self):
        # Hold the block by friction on BOTH faces: 2·μ·F ≥ m·g
        #   → F = S · m·g / (2μ),  μ = μ_finger · μ_block (pymunk product rule)
        mu = self.finger_friction * self.block_friction
        self.max_grip_force = self.grip_safety_factor * self.block_mass * self.gravity / (2.0 * mu)

