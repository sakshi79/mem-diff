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
        self.grasp_joints: list | None = None

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
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = Vec2d(
            self.base.position.x + sign * (cfg.finger_gap_max + cfg.finger_width/2),
            self.base.position.y,
        )
        shape = pymunk.Poly.create_box(body, (cfg.finger_width, cfg.finger_length))
        shape.friction = 1.8
        shape.elasticity = 0.0
        shape.color = (pygame.Color("SteelBlue") if side=="left" else pygame.Color("CornflowerBlue"))
        self.space.add(body, shape)
        return body, shape

    # ----------------------  state ---------------------
    def set_grip(self, value: float):
        self.grip_value = float(value)
    
    @property
    def grip_closed(self) -> bool:
        return self.grip_value > self.cfg.grip_close_threshold

    @property
    def grasped(self) -> bool:
        return self.grasp_joints is not None

    # ----------------------- per step/substep ----------------------
    def update_fingers(self):
        cfg = self.cfg
        t = (self.grip_value + 1.0)/2.0
        half_gap = cfg.finger_gap_max * (1.0 - t) + cfg.finger_gap_min * t
        bx, by = self.base.position
        self.left.position = Vec2d(bx - half_gap - cfg.finger_width/2, by)
        self.right.position = Vec2d(bx + half_gap + cfg.finger_width/2, by)
        self.left.velocity = Vec2d(0,0)
        self.right.velocity = Vec2d(0,0)

    def update_grasp(self, block: pymunk.Body):
        cfg = self.cfg
        if self.grip_closed and self.grasp_joints is None:
            block_pos = Vec2d(*block.position)
            left_tip = self.left.position + Vec2d(0, +cfg.finger_length/2)
            right_tip = self.right.position + Vec2d(0, +cfg.finger_length/2)
            dist = min((block_pos - left_tip).length, (block_pos - right_tip).length)
            if dist < cfg.grasp_threshold:
                block.velocity = Vec2d(0,0)
                block.angular_velocity = 0.0
                # Rigid weld to preserve current grasp
                pivot = pymunk.PivotJoint(self.base, block, Vec2d(*block.position))
                pivot.max_force = cfg.grasp_max_force
                gear = pymunk.GearJoint(self.base, block, block.angle - self.base.angle, 1.0)
                self.grasp_joints = [pivot, gear]
                self.space.add(pivot, gear)
        elif not self.grip_closed and self.grasp_joints is not None:
            self.space.remove(*self.grasp_joints)
            self.grasp_joints = None

    def drive_base(self, target, dt: float):
        cfg = self.cfg
        accel = (cfg.k_p * (Vec2d(*target) - self.base.position)
            + cfg.k_v * (Vec2d(0,0) - self.base.velocity))
        self.base.velocity += accel * dt
        m, ws = cfg.base_margin, cfg.window_size
        self.base.position = Vec2d(
            float(np.clip(self.base.position.x, m, ws-m)),
            float(np.clip(self.base.position.y, m, ws-m))
        )

    def set_pose(self, position):
        self.base.position = Vec2d(*position)
        self.base.velocity = Vec2d(0, 0)
        self.update_fingers()

