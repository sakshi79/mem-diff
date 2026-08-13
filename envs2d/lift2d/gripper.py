"""
Parallel-jaw gripper: kinematic base + kinematic fingers.
- Slide state (left_slide, right_slide) is each finger's offset from the
base along the open/close axis, in [finger_gap_min, finger_gap_max].
- Velocity-motor: the slide moves at jaw_speed until contact with the block.
- Grasp is an EXPLICIT state machine. Both fingers stalled + grip_closed
    → grasped=True and the block is position-synced to the base each
    substep. Grasp breaks on grip release OR friction slip.
- Base sits at the TOP of the fingers (fingers hang below).
"""

from __future__ import annotations
from .config import Lift2DConfig
import numpy as np
import pygame 
import pymunk 
from pymunk.vec2d import Vec2d

class Gripper:
    def __init__(self, space: pymunk.Space, cfg: Lift2DConfig, position):
        self.space = space
        self.cfg = cfg
        self.grip_value: float = -1.0

        # slide state - per finger offset from base along open/close axis.
        self.left_slide = cfg.finger_gap_max
        self.right_slide = cfg.finger_gap_max

        # Grasp state
        self.grasped: bool = False
        self.grasp_offset : Vec2d | None = None
        
        # EMA-smoothed base velocity for slip check (filters specikes from direct position tracking)
        self._smoothed_bvel: Vec2d = Vec2d(0,0)
        self._prev_smoothed_bvel: Vec2d = Vec2d(0,0)
        # Commanded base velocity (derived from position change per drive_base).
        # We keep pymunk's base.velocity at 0 to prevent double-motion during
        # space.step; slip detection uses this field instead.
        self._commanded_bvel: Vec2d = Vec2d(0, 0)
        self._contact_left: bool = False
        self._contact_right: bool = False

        # Kinematic base with a sensor dot marker
        self.base = pymunk.Body(body_type = pymunk.Body.KINEMATIC)
        self.base.position = Vec2d(*position)
        dot = pymunk.Circle(self.base, 7)
        dot.sensor = True
        dot.color = pygame.Color("DimGray")
        space.add(self.base, dot)

        # Kinematic fingers
        self.left, self.left_shape = self._make_finger("left")
        self.right, self.right_shape = self._make_finger("right")
        self._sync_finger_positions()

    def _make_finger(self, side: str):
        cfg = self.cfg
        sign = -1.0 if side=="left" else 1.0
        body = pymunk.Body(body_type = pymunk.Body.KINEMATIC)
        body.position = Vec2d(
            self.base.position.x + sign * (cfg.finger_gap_max + cfg.finger_width/2),
            self.base.position.y + cfg.finger_length/2,  # base at top
        )
        shape = pymunk.Poly.create_box(body, (cfg.finger_width, cfg.finger_length))
        shape.friction = cfg.finger_friction
        shape.elasticity = 0.0
        shape.sensor = False
        shape.color = (pygame.Color("SteelBlue") if side=="left" else pygame.Color("CornflowerBlue"))
        self.space.add(body, shape)
        return body, shape

    def _sync_finger_positions(self):
        cfg = self.cfg
        bx, by = self.base.position
        fy = by + cfg.finger_length / 2
        self.left.position = Vec2d(bx - self.left_slide - cfg.finger_width/2, fy)
        self.right.position = Vec2d(bx + self.right_slide + cfg.finger_width/2, fy)
        self.left.velocity = self.base.velocity
        self.right.velocity = self.base.velocity

    # ----------------------  state ---------------------
    def set_grip(self, value: float):
        self.grip_value = float(value)
    
    @property
    def grip_closed(self) -> bool:
        return self.grip_value > self.cfg.grip_close_threshold

    # --------- geometry helpers ---------------
    def _y_overlap_with_block(self, block) -> bool:
        cfg = self.cfg
        fy_center = self.base.position.y + cfg.finger_length/2
        fy_min = fy_center - cfg.finger_length/2
        fy_max = fy_center + cfg.finger_length/2
        by_min = block.position.y - cfg.block_size/2
        by_max = block.position.y + cfg.block_size/2
        return fy_max > by_min and fy_min < by_max

    def _is_finger_stalled(self, side: str, block) -> bool:
        """ Finger's inner face is at/past block AND y-overlaps"""
        cfg = self.cfg
        if not self._y_overlap_with_block(block):
            return False
        bx = self.base.position.x
        eps = 0.5
        if side == "left":
            block_face = block.position.x - cfg.block_size/2
            return abs((bx-self.left_slide) - block_face) < eps
        else:
            block_face = block.position.x + cfg.block_size/2
            return abs((bx + self.right_slide) - block_face) < eps

    # ------------ per step/substep updates ------------
    def drive_fingers(self, dt: float, block: pymunk.Body):
        """ Velocity-motor with cap: slide moves at jaw speed
        until block contact stops it. """

        cfg = self.cfg
        v_cmd = -cfg.jaw_speed if self.grip_closed else +cfg.jaw_speed
        self.left_slide = self._advance_slide("left", self.left_slide, v_cmd, dt, block)
        self.right_slide = self._advance_slide("right", self.right_slide, v_cmd, dt, block)
        self._sync_finger_positions()

    def _advance_slide(self, side, cur, v_cmd, dt, block):
        cfg = self.cfg
        candidate = float(np.clip(cur + v_cmd * dt,
                                  cfg.finger_gap_min, cfg.finger_gap_max))
        if v_cmd < 0 and block is not None and self._y_overlap_with_block(block):
            bx = self.base.position.x
            if side == "left":
                block_face = block.position.x - cfg.block_size / 2
                if (bx - candidate) > block_face:
                    return bx - block_face
            else:
                block_face = block.position.x + cfg.block_size / 2
                if (bx + candidate) < block_face:
                    return block_face - bx
        return candidate

    def update_grasp(self, block: pymunk.Body, dt: float):
        """Grasp state machine, block transport, slip detection.
        Call each substep AFTER drive_fingers, BEFORE space.step."""
        cfg = self.cfg

        # EMA-smoothed base velocity (filter spikes from direct position
        # tracking so slip check isn't over-triggered on cursor jumps)
        alpha = 0.3
        self._prev_smoothed_bvel = self._smoothed_bvel
        self._smoothed_bvel = alpha * self._commanded_bvel + (1 - alpha) * self._smoothed_bvel

        self._contact_left  = self._is_finger_stalled("left",  block)
        self._contact_right = self._is_finger_stalled("right", block)
        both_stalled = self._contact_left and self._contact_right

        # Grasp state transitions
        if self.grip_closed and both_stalled and not self.grasped:
            self.grasped = True
            self.grasp_offset = Vec2d(*block.position) - self.base.position
        if not self.grip_closed and self.grasped:
            self.grasped = False
            self.grasp_offset = None

        # Slip check + transport
        if self.grasped and self.grasp_offset is not None:
            # NOTE: physical slip check disabled — teleport base motion produces
            # infinite instantaneous acceleration, which trips any real slip
            # threshold. Grasp currently breaks only on grip release. Revisit
            # once we have a proper velocity-based slip formulation.
            block.position = self.base.position + self.grasp_offset
            block.velocity = self.base.velocity
            block.angular_velocity = 0.0



    def drive_base(self, target, dt: float):
        cfg = self.cfg
        ws = cfg.window_size
        wt = cfg.wall_thickness
        # Base at TOP of fingers → finger bottom = base.y + finger_length.
        # Fingers extend base.x ± (finger_gap_max + finger_width/2) at full open.
        reach = cfg.finger_gap_max + cfg.finger_width / 2
        x_min, x_max = wt + reach, ws - wt - reach
        y_min, y_max = wt, ws - wt - cfg.finger_length
        tx = float(np.clip(target[0], x_min, x_max))
        ty = float(np.clip(target[1], y_min, y_max))
        old_pos = self.base.position
        dx = tx - old_pos.x
        dy = ty - old_pos.y
        dist = float(np.hypot(dx, dy))
        max_step = cfg.max_base_speed * dt
        # Bound this substep's motion — small enough for pymunk to catch contacts.
        if dist > max_step:
            scale = max_step / dist
            # new_x = old_pos.x + dx*scale
            # new_y = old_pos.y + dy*scale
        else:
            scale = 1.0
        step_x = dx*scale
        step_y = dy*scale
        self.base.velocity = Vec2d(step_x/dt, step_y/dt)
        self._commanded_bvel = self.base.velocity
        

    def set_pose(self, position):
        cfg = self.cfg
        self.base.position = Vec2d(*position)
        self.base.velocity = Vec2d(0, 0)
        self.left_slide = cfg.finger_gap_max
        self.right_slide = cfg.finger_gap_max
        self.grasped = False
        self.grasp_offset = None
        self._smoothed_bvel = Vec2d(0, 0)
        self._prev_smoothed_bvel = Vec2d(0, 0)
        self._commanded_bvel = Vec2d(0, 0)
        self._contact_left = False
        self._contact_right = False
        self.grip_value = -1.0
        self._sync_finger_positions()
