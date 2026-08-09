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
        open_reach = cfg.finger_gap_max + cfg.finger_width/2
        close_reach = cfg.finger_gap_min + cfg.finger_width/2
        if side == "left":
            groove_a, groove_b = Vec2d(-open_reach, 0), Vec2d(-close_reach,0)
        else:
            groove_a, groove_b = Vec2d(close_reach,0), Vec2d(open_reach, 0)
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
        """Velocity-controlled jaw motor with a force limit. The servo commands the
        finger's SLIDE velocity along the groove (v_finger - v_base), so it's decoupled
        from the base's own motion — commanding "open" always drives the jaws outward
        relative to the base, no matter how fast the base is being dragged."""
        cfg = self.cfg
        v_dir = cfg.jaw_speed if self.grip_closed else -cfg.jaw_speed
        base_vx = self.base.velocity.x
        for body, sign in ((self.left, +1.0), (self.right, -1.0)):
            v_target_slide = sign * v_dir              # inward when closing
            v_slide = body.velocity.x - base_vx        # relative to base
            fx = cfg.k_motor * (v_target_slide - v_slide)
            fx = float(np.clip(fx, -cfg.max_grip_force, cfg.max_grip_force))
            body.apply_force_at_world_point(Vec2d(fx, 0.0), body.position)


        # DEBUG — remove once diagnosed
        f = getattr(Gripper, "_dbg_f", None)
        if f is None:
            f = open("lift2d_debug.log", "w")
            Gripper._dbg_f = f
        bp, bv = self.base.position, self.base.velocity
        Lp, Rp = self.left.position, self.right.position
        Lv, Rv = self.left.velocity, self.right.velocity
        f.write(
            f"grip={self.grip_value:+.2f} v_dir={v_dir:+.0f} "
            f"base=({bp.x:6.1f},{bp.y:6.1f}) bvel=({bv.x:+6.1f},{bv.y:+6.1f}) "
            f"Llocal=({Lp.x-bp.x:+.1f},{Lp.y-bp.y:+.1f}) Lvel=({Lv.x:+6.1f},{Lv.y:+6.1f}) "
            f"Rlocal=({Rp.x-bp.x:+.1f},{Rp.y-bp.y:+.1f}) Rvel=({Rv.x:+6.1f},{Rv.y:+6.1f})\n"
        )
        f.flush()
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
        # Clamp position AND velocity at limits — otherwise a residual "into the wall"
        # velocity keeps the groove joint under tension, which drags the fingers into
        # the floor and creates a friction jam that pins them.
        m, ws = cfg.base_margin, cfg.window_size
        px = float(np.clip(self.base.position.x, m, ws - m))
        py = float(np.clip(self.base.position.y, m, ws - m))
        vx, vy = self.base.velocity.x, self.base.velocity.y
        if px <= m       and vx < 0: vx = 0.0
        if px >= ws - m  and vx > 0: vx = 0.0
        if py <= m       and vy < 0: vy = 0.0
        if py >= ws - m  and vy > 0: vy = 0.0
        self.base.position = Vec2d(px, py)
        self.base.velocity = Vec2d(vx, vy)
        

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

