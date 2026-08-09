"""
Parallel-jaw gripper: kinematic base + two finger bodies, grasping objects
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
        self._contact_left: bool = False
        self._contact_right: bool = False

        # Kinematic base
        ### We need to remove this dot sensor and have pure contact/frcition based grasping
        self.base = pymunk.Body(body_type = pymunk.Body.KINEMATIC)
        self.base.position = Vec2d(*position)
        dot = pymunk.Circle(self.base, 7)
        dot.sensor = True
        dot.color = pygame.Color("DimGray")
        space.add(self.base, dot)

        # Fingers
        self.left, self.left_shape = self._make_finger("left")
        self.right, self.right_shape = self._make_finger("right")

    def _make_finger(self, side: str):
        cfg = self.cfg
        sign = -1.0 if side=="left" else 1.0
        moment = pymunk.moment_for_box(cfg.finger_mass, (cfg.finger_width, cfg.finger_length))
        body = pymunk.Body(cfg.finger_mass, moment)
        open_x = self.base.position.x + sign * (cfg.finger_gap_max + cfg.finger_width/2)
        body.position = Vec2d(
            open_x,
            self.base.position.y,
        )
        shape = pymunk.Poly.create_box(body, (cfg.finger_width, cfg.finger_length))
        shape.friction = cfg.finger_friction
        shape.elasticity = 0.0
        shape.color = (pygame.Color("SteelBlue") if side=="left" else pygame.Color("CornflowerBlue"))
        self.space.add(body, shape)

        # Prismatic slide along x in the base's local frame
        # base is kinematic so the jaw rides with the base and can only open/close.
        reach = cfg.finger_gap_max + cfg.finger_width/2
        if side == "left":
            groove_a, groove_b = Vec2d(-reach, 0), Vec2d(0,0)
        else:
            groove_a, groove_b = Vec2d(0,0), Vec2d(reach, 0)
        groove = pymunk.GrooveJoint(self.base, body, groove_a, groove_b, Vec2d(0,0))
        gear = pymunk.GearJoint(self.base, body, 0.0, 1.0)  # lock jaw rotation
        self.space.add(groove, gear)
        return body, shape

    # ----------------------  state ---------------------
    def set_grip(self, value: float):
        self.grip_value = float(value)
    
    @property
    def grip_closed(self) -> bool:
        return self.grip_value > self.cfg.grip_close_threshold

    # @property
    # def grasped(self) -> bool:
    #     return self.grasp_joints is not None

    # ----------------------- per step/substep ----------------------
    def drive_fingers(self, dt):
        """Force-limited jaw actuation. A closed grip commands a gap narrower than
        the block, so the PD saturates at max_grip_force → steady friction hold."""
        cfg = self.cfg
        t = (self.grip_value + 1.0) / 2.0
        half_gap = cfg.finger_gap_max * (1.0 - t) + cfg.finger_gap_min * t
        bx = self.base.position.x
        targets = {
            self.left:  bx - (half_gap + cfg.finger_width / 2.0),
            self.right: bx + (half_gap + cfg.finger_width / 2.0),
        }
        for body, tgt_x in targets.items():
            fx = cfg.k_grip * (tgt_x - body.position.x) - cfg.c_grip * body.velocity.x
            fx = float(np.clip(fx, -cfg.max_grip_force, cfg.max_grip_force))
            body.apply_force_at_world_point(Vec2d(fx, 0.0), body.position)

    def update_contacts(self, block_shape):
        def touching(shape):
            cps = shape.shapes_collide(block_shape)
            return any(p.distance < 1.0 for p in cps.points)
        self._contact_left  = touching(self.left_shape)
        self._contact_right = touching(self.right_shape)

    @property
    def grasped(self) -> bool:
        return self.grip_closed and self._contact_left and self._contact_right

    def drive_base(self, target, dt: float):
        cfg = self.cfg
        accel = (cfg.k_p * (Vec2d(*target) - self.base.position)
            + cfg.k_v * (Vec2d(0,0) - self.base.velocity))
        v = self.base.velocity + accel * dt
        if v.length > cfg.max_base_speed:
            v = v.normalized() * cfg.max_base_speed
        self.base.velocity = v
        m, ws = cfg.base_margin, cfg.window_size
        self.base.position = Vec2d(
            float(np.clip(self.base.position.x, m, ws-m)),
            float(np.clip(self.base.position.y, m, ws-m))
        )

    def set_pose(self, position):
        cfg = self.cfg
        self.base.position = Vec2d(*position)
        self.base.velocity = Vec2d(0, 0)
        bx, by = self.base.position 
        reach = cfg.finger_gap_max + cfg.finger_width / 2
        self.left.position = Vec2d(bx - reach, by)
        self.right.position = Vec2d(bx + reach, by)
        for b in (self.left, self.right):
            b.velocity = Vec2d(0,0)
            b.angular_velocity = 0.0
            b.angle = 0.0
        self.grip_value = -1.0
        self._contact_left = self._contact_right = False

