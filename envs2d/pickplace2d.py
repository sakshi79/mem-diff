"""
pickplace2d.py
==============
2D Pick-and-Place environment — self-contained (env + teleop demo).

env structure:  obs, reward, done, info = env.step(act)

Agent : Parallel gripper  (kinematic base + left & right finger bodies)
Object: Can               (dynamic, subject to gravity)
Target: Colored box / bin (static container on the floor)
Obs   : dict with keys
            'image':     (3, H, W) float32 RGB, normalised to [0, 1]
            'agent_pos': (2,) float32 base position in world coords
Action: (tx, ty, grip)
            tx, ty:  absolute target position for base this step (world units)
            grip:    float in [-1, +1]
                       > 0.3  → close fingers / maintain grasp
                       ≤ 0.3  → open  fingers / release
Reward: 1.0 while the can is resting inside the target box (and released),
        0.0 otherwise
Done  : False (open-ended; reset manually)

Coordinate system
-----------------
pymunk_override.py has positive_y_is_up = False.
Both pymunk and pygame share the same axes: y=0 is the TOP of the screen,
y increases DOWNWARD.  Gravity therefore points in the +y direction.

Controls (teleop demo)
----------------------
  Left-click drag  – move gripper toward cursor
  ↑ ↓ ← →         – also move gripper (additive with mouse)
  Right click      – toggle gripper open / closed
  Space (toggle)   – also open / close gripper
  R                – reset episode
  Q / Escape       – quit

Usage
-----
  python pickplace2d.py                       # interactive teleop
  python pickplace2d.py -o data/pickplace.zarr  # teleop + record to zarr
"""

from __future__ import annotations

import collections
import sys

import click
import cv2
import gym
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
from gym import spaces
from pymunk.vec2d import Vec2d

from diffusion_policy.env.pusht.pymunk_override import DrawOptions
from diffusion_policy.env.pusht.replay_buffer import ReplayBuffer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wall(static_body: pymunk.Body, a, b, radius: float = 3,
               color: str = "LightGray") -> pymunk.Segment:
    seg = pymunk.Segment(static_body, a, b, radius)
    seg.color = pygame.Color(color)
    seg.elasticity = 0.1
    seg.friction = 0.8
    return seg


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class PickPlace2DEnv(gym.Env):
    """2D parallel-gripper pick-and-place task.

    Pick the can up from its start position and drop it inside the colored
    target box that sits on the floor.
    """

    metadata = {
        "render.modes": ["human", "rgb_array"],
        "video.frames_per_second": 10,
    }
    reward_range = (0.0, 1.0)
    # reward = 1.0 while the can rests inside the target box, 0.0 otherwise

    def __init__(
        self,
        render_size: int = 96,
        window_size: int = 512,
        finger_length: float = 60.0,
        finger_width: float = 14.0,
        finger_gap_max: float = 42.0,
        finger_gap_min: float = 18.0,
        can_width: float = 26.0,
        can_height: float = 40.0,
        grasp_threshold: float = 58.0,
        bin_half_width: float = 55.0,
        bin_wall_height: float = 70.0,
        render_action: bool = True,
        reset_to_state=None,
    ):
        self._seed = None
        self.seed()

        self.window_size = ws = window_size
        self.render_size = render_size

        self.finger_length  = finger_length
        self.finger_width   = finger_width
        self.finger_gap_max = finger_gap_max
        self.finger_gap_min = finger_gap_min
        self.can_width      = can_width
        self.can_height     = can_height
        self.grasp_threshold = grasp_threshold
        self.bin_half_width  = bin_half_width
        self.bin_wall_height = bin_wall_height
        self.render_action  = render_action
        self.reset_to_state = reset_to_state

        # Physics timing
        self.sim_hz     = 100
        self.control_hz = self.metadata["video.frames_per_second"]
        self.k_p        = 180.0
        self.k_v        = 30.0

        # Runtime state
        self._grip_value:  float                          = -1.0
        self._grasp_joint: pymunk.Constraint | None       = None
        self.n_contact_points: int                        = 0
        self.latest_action: np.ndarray | None             = None
        self.target_pos:   Vec2d | None                   = None

        # PyGame handles (created lazily in render_frame)
        self.window = None
        self.clock  = None
        self.screen = None

        # Physics bodies (created in _setup / reset)
        self.space:      pymunk.Space | None = None
        self.base_body:  pymunk.Body  | None = None
        self.left_body:  pymunk.Body  | None = None
        self.right_body: pymunk.Body  | None = None
        self.can:        pymunk.Body  | None = None

        # ------------------------------------------------------------------
        # Gym spaces  (matching multi_push convention: CHW float32 image)
        # ------------------------------------------------------------------
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0, high=1,
                shape=(3, render_size, render_size),
                dtype=np.float32,
            ),
            'agent_pos': spaces.Box(
                low=0, high=ws,
                shape=(2,),
                dtype=np.float32,
            ),
        })

        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            high=np.array([float(ws), float(ws), 1.0], dtype=np.float32),
            shape=(3,),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self):
        self._setup()
        state = self.reset_to_state
        if state is None:
            rs  = self.np_random
            ws  = self.window_size
            # Can starts on the floor on one side
            cx  = float(rs.integers(90, ws // 2 - 40))
            cy  = float(ws - 60)                          # resting on floor
            # Target box sits on the floor on the other side, clear of the can
            tx  = float(rs.integers(ws // 2 + 60, ws - 90))
            # Gripper starts up high
            gx  = float(rs.integers(100, ws - 100))
            gy  = float(rs.integers(ws // 4, ws // 2))
            state = np.array([gx, gy, cx, cy, tx], dtype=np.float32)
        self._set_state(state)
        return self._get_obs()

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action):
        dt      = 1.0 / self.sim_hz
        n_steps = self.sim_hz // self.control_hz
        self.n_contact_points = 0

        if action is not None:
            self.latest_action = np.asarray(action, dtype=np.float32).copy()
            tx   = float(action[0])
            ty   = float(action[1])
            grip = float(np.clip(action[2], -1.0, 1.0))
            self._grip_value = grip

            for _ in range(n_steps):
                self._update_fingers()
                self._update_grasp()

                # PD drive toward absolute target
                target = Vec2d(tx, ty)
                accel = (
                    self.k_p * (target - self.base_body.position)
                    + self.k_v * (Vec2d(0, 0) - self.base_body.velocity)
                )
                self.base_body.velocity += accel * dt

                # Keep base inside arena
                margin = 35
                ws = self.window_size
                self.base_body.position = Vec2d(
                    float(np.clip(self.base_body.position.x, margin, ws - margin)),
                    float(np.clip(self.base_body.position.y, margin, ws - margin)),
                )

                self.space.step(dt)

        placed      = self._is_placed()
        reward      = 1.0 if placed else 0.0
        done        = False          # open-ended; caller decides when to stop
        observation = self._get_obs()
        info        = self._get_info()

        assert observation is not None, "env._get_obs() returned None"
        assert info       is not None, "env._get_info() returned None"
        return observation, reward, done, info

    # ------------------------------------------------------------------
    # Placement check
    # ------------------------------------------------------------------

    def _is_placed(self) -> bool:
        """True when the can rests inside the target box and is released."""
        if self.can is None or self.target_pos is None:
            return False
        cx, cy = self.can.position
        # Horizontally within the bin interior
        in_x = abs(cx - self.target_pos.x) < self.bin_half_width
        # Low enough to be sitting inside the bin (near the floor)
        on_floor = cy > (self.window_size - 30 - self.can_height / 2)
        # Must be released (not currently grasped) and roughly still
        released = self._grasp_joint is None
        slow = self.can.velocity.length < 25.0
        return bool(in_x and on_floor and released and slow)

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        return self.render_frame(mode)

    def render_frame(self, mode: str) -> np.ndarray:
        ws = self.window_size

        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((ws, ws))
            pygame.display.set_caption("2D Pick & Place — Parallel Gripper")
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((ws, ws))
        canvas.fill((24, 24, 30))
        self.screen = canvas

        # Target box floor pad (drawn under the physics debug draw)
        if self.target_pos is not None:
            placed = self._is_placed()
            pad_color = (60, 170, 90) if placed else (70, 90, 150)
            tx = self.target_pos.x
            pad = pygame.Rect(
                round(tx - self.bin_half_width),
                round(ws - 8 - self.bin_wall_height),
                round(2 * self.bin_half_width),
                round(self.bin_wall_height),
            )
            surf = pygame.Surface((pad.width, pad.height), pygame.SRCALPHA)
            surf.fill((*pad_color, 70))
            canvas.blit(surf, (pad.left, pad.top))

        # Physics debug draw (walls, bin, fingers, can)
        draw_options = DrawOptions(canvas)
        self.space.debug_draw(draw_options)

        # Gripper connector lines (no y-flip: coords are already screen coords)
        if self.base_body is not None and self.left_body is not None:
            def _pg(v):
                return (round(v.x), round(v.y))

            lp   = _pg(self.left_body.position)
            rp   = _pg(self.right_body.position)
            bp   = _pg(self.base_body.position)
            mid  = ((lp[0] + rp[0]) // 2, (lp[1] + rp[1]) // 2)
            pygame.draw.line(canvas, (100, 120, 160), lp, rp, 3)
            pygame.draw.line(canvas, (100, 120, 160), bp, mid, 3)

        # HUD
        try:
            if not pygame.font.get_init():
                pygame.font.init()
            font = pygame.font.SysFont("monospace", 12)

            if self._grasp_joint is not None:
                canvas.blit(font.render("● GRASPED", True, (255, 220, 50)), (10, 10))
            if self._is_placed():
                canvas.blit(font.render("✔ PLACED", True, (90, 230, 130)), (10, 26))

            # Grip bar
            bar_x, bar_y, bar_w, bar_h = 10, ws - 22, 140, 12
            pygame.draw.rect(canvas, (50, 50, 60),  (bar_x, bar_y, bar_w, bar_h))
            fill_w  = int((self._grip_value + 1) / 2 * bar_w)
            g_color = (80, 210, 110) if self._grip_value > 0.3 else (210, 80, 80)
            pygame.draw.rect(canvas, g_color, (bar_x, bar_y, fill_w, bar_h))
            pygame.draw.rect(canvas, (120, 120, 130), (bar_x, bar_y, bar_w, bar_h), 1)
            canvas.blit(
                font.render(f"grip {self._grip_value:+.2f}", True, (200, 200, 200)),
                (bar_x + bar_w + 6, bar_y),
            )
        except Exception:
            pass

        if mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.control_hz)

        img = np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )
        img = cv2.resize(img, (self.render_size, self.render_size),
                         interpolation=cv2.INTER_AREA)
        return img

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None

    # ------------------------------------------------------------------
    # Teleop agent
    # ------------------------------------------------------------------

    def teleop_agent(self):
        """Arrow-key + Space-toggle teleop agent.

        Returns a namedtuple with an ``act(obs)`` method, compatible with
        the multi_push demo loop pattern.
        """
        TeleopAgent = collections.namedtuple("TeleopAgent", ["act"])

        state = {"grip_closed": False}

        def act(obs):
            # Consume KEYDOWN events for the toggle
            for event in pygame.event.get(pygame.KEYDOWN):
                if event.key == pygame.K_SPACE:
                    state["grip_closed"] = not state["grip_closed"]
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    self.close()
                    sys.exit(0)

            keys = pygame.key.get_pressed()
            key_step = 5.0
            tx = self.base_body.position.x
            ty = self.base_body.position.y
            if keys[pygame.K_LEFT]:  tx -= key_step
            if keys[pygame.K_RIGHT]: tx += key_step
            if keys[pygame.K_UP]:    ty -= key_step
            if keys[pygame.K_DOWN]:  ty += key_step

            # Override with mouse if left button is held
            mouse_btns = pygame.mouse.get_pressed()
            if mouse_btns[0]:
                tx, ty = pygame.mouse.get_pos()

            grip = 1.0 if state["grip_closed"] else -1.0
            return np.array([tx, ty, grip], dtype=np.float32), state["grip_closed"]

        return TeleopAgent(act)

    # ------------------------------------------------------------------
    # Observation / info
    # ------------------------------------------------------------------

    def _get_obs(self) -> dict:
        img     = self.render_frame("rgb_array")
        img_obs = np.moveaxis(img.astype(np.float32) / 255.0, -1, 0)  # HWC→CHW
        return {
            "image":     img_obs,
            "agent_pos": np.array(self.base_body.position, dtype=np.float32),
        }

    def _get_info(self) -> dict:
        return {
            "pos_agent":  np.array(self.base_body.position, dtype=np.float32),
            "vel_agent":  np.array(self.base_body.velocity, dtype=np.float32),
            "can_pose":   np.array(
                list(self.can.position) + [self.can.angle], dtype=np.float32
            ),
            "target_pos": np.array(self.target_pos, dtype=np.float32),
            "grasped":    self._grasp_joint is not None,
            "placed":     self._is_placed(),
            "grip":       float(self._grip_value),
            "n_contacts": int(np.ceil(self.n_contact_points /
                                      (self.sim_hz // self.control_hz))),
        }

    # ------------------------------------------------------------------
    # Physics setup
    # ------------------------------------------------------------------

    def _setup(self):
        self._grasp_joint = None
        self._grip_value  = -1.0

        self.space = pymunk.Space()
        # positive_y_is_up = False → y=0 TOP, gravity pulls to larger y
        self.space.gravity = (0.0, 900.0)
        self.space.damping = 0.75

        ws = self.window_size

        walls = [
            _make_wall(self.space.static_body, (5,      ws - 5), (ws - 5, ws - 5)),  # floor
            _make_wall(self.space.static_body, (5,      5     ), (5,      ws - 5)),  # left
            _make_wall(self.space.static_body, (ws - 5, 5     ), (ws - 5, ws - 5)),  # right
            _make_wall(self.space.static_body, (5,      5     ), (ws - 5, 5     )),  # ceiling
        ]
        self.space.add(*walls)

        # Kinematic base
        self.base_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.base_body.position = Vec2d(ws / 2, ws // 3)
        base_dot = pymunk.Circle(self.base_body, 7)
        base_dot.sensor = True
        base_dot.color  = pygame.Color("DimGray")
        self.space.add(self.base_body, base_dot)

        # Fingers
        self.left_body,  self.left_shape  = self._make_finger("left")
        self.right_body, self.right_shape = self._make_finger("right")

        # Can (dynamic) — created in _set_state so it lands at the pick position
        mass    = 1.0
        inertia = pymunk.moment_for_box(mass, (self.can_width, self.can_height))
        self.can = pymunk.Body(mass, inertia)
        self.can.position = Vec2d(ws / 2, ws - 80)
        can_shape = pymunk.Poly.create_box(self.can, (self.can_width, self.can_height))
        can_shape.friction   = 1.5
        can_shape.elasticity = 0.05
        can_shape.color      = pygame.Color("Goldenrod")
        self.space.add(self.can, can_shape)

        # Target bin walls are (re)built once the target position is known
        self.target_pos   = None
        self._bin_shapes  = []

        # Collision counter
        self.n_contact_points = 0
        ch = self.space.add_collision_handler(0, 0)
        ch.post_solve = self._handle_collision

    def _build_bin(self, tx: float):
        """Create the colored target box (two side walls) at floor x=tx."""
        ws = self.window_size
        floor_y = ws - 5
        top_y   = floor_y - self.bin_wall_height
        left_x  = tx - self.bin_half_width
        right_x = tx + self.bin_half_width

        left_wall  = _make_wall(self.space.static_body,
                                (left_x, floor_y), (left_x, top_y),
                                radius=4, color="MediumSeaGreen")
        right_wall = _make_wall(self.space.static_body,
                                (right_x, floor_y), (right_x, top_y),
                                radius=4, color="MediumSeaGreen")
        self.space.add(left_wall, right_wall)
        self._bin_shapes = [left_wall, right_wall]
        self.target_pos  = Vec2d(tx, floor_y - self.bin_wall_height / 2)

    def _make_finger(self, side: str):
        sign = -1.0 if side == "left" else 1.0
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = Vec2d(
            self.base_body.position.x + sign * (self.finger_gap_max + self.finger_width / 2),
            self.base_body.position.y,
        )
        shape = pymunk.Poly.create_box(body, (self.finger_width, self.finger_length))
        shape.friction   = 1.8
        shape.elasticity = 0.0
        shape.color = (pygame.Color("SteelBlue")
                       if side == "left" else pygame.Color("CornflowerBlue"))
        self.space.add(body, shape)
        return body, shape

    # ------------------------------------------------------------------
    # Finger kinematics
    # ------------------------------------------------------------------

    def _update_fingers(self):
        t        = (self._grip_value + 1.0) / 2.0
        half_gap = self.finger_gap_max * (1.0 - t) + self.finger_gap_min * t

        bx, by = self.base_body.position
        self.left_body.position  = Vec2d(bx - half_gap - self.finger_width / 2, by)
        self.right_body.position = Vec2d(bx + half_gap + self.finger_width / 2, by)
        self.left_body.velocity  = Vec2d(0, 0)
        self.right_body.velocity = Vec2d(0, 0)

    # ------------------------------------------------------------------
    # Grasping
    # ------------------------------------------------------------------

    def _update_grasp(self):
        grip_closed = self._grip_value > 0.3

        if grip_closed and self._grasp_joint is None:
            can_pos = Vec2d(*self.can.position)
            # Fingertip faces downward (large y) → tip at +finger_length/2
            left_tip  = self.left_body.position  + Vec2d(0, +self.finger_length / 2)
            right_tip = self.right_body.position + Vec2d(0, +self.finger_length / 2)
            dist = min((can_pos - left_tip).length,
                       (can_pos - right_tip).length)

            if dist < self.grasp_threshold:
                anchor_base  = self.base_body.world_to_local(Vec2d(*self.can.position))
                joint = pymunk.PinJoint(self.base_body, self.can,
                                        anchor_base, Vec2d(0, 0))
                joint.max_force   = 6000.0
                self._grasp_joint = joint
                self.space.add(joint)

        elif not grip_closed and self._grasp_joint is not None:
            self.space.remove(self._grasp_joint)
            self._grasp_joint = None

    # ------------------------------------------------------------------
    # State initialisation
    # ------------------------------------------------------------------

    def _set_state(self, state):
        """Set world state from [gripper_x, gripper_y, can_x, can_y, target_x]."""
        gx, gy, cx, cy, tx = (float(v) for v in state[:5])
        self.base_body.position = Vec2d(gx, gy)
        self.base_body.velocity = Vec2d(0, 0)
        self.can.position       = Vec2d(cx, cy)
        self.can.velocity       = Vec2d(0, 0)
        self.can.angle          = 0.0
        self._build_bin(tx)
        self._update_fingers()
        self.space.step(1.0 / self.sim_hz)

    # ------------------------------------------------------------------
    # Collision
    # ------------------------------------------------------------------

    def _handle_collision(self, arbiter, space, data):
        self.n_contact_points += len(arbiter.contact_point_set.points)


# ---------------------------------------------------------------------------
# Teleop demo  (mirrors lift2d.py main())
# ---------------------------------------------------------------------------

@click.command()
@click.option("-o", "--output", default=None, required=False,
              help="Path to zarr replay buffer. Omit to run without recording.")
@click.option("--render-size", default=96,  type=int, show_default=True)
@click.option("--window",      default=512, type=int, show_default=True)
def main(output, render_size, window):
    """
    Interactive teleop demo for PickPlace2DEnv.

    Goal
    ----
      Pick the can (gold) off the floor and drop it inside the green box.

    Movement
    --------
      Left-click drag – gripper follows cursor (proportional)
      Arrow keys      – also move gripper (additive with mouse)

    Grasp
    -----
      Right click     – toggle gripper open / closed
      SPACE           – also toggle gripper open / closed

    Session
    -------
      R               – retry / new episode
      Q / Escape      – quit
    """
    plan_idx = 0

    while True:
        # --- Replay buffer (optional) ---
        if output is not None:
            replay_buffer = ReplayBuffer.create_from_path(output, mode='a')
            seed = replay_buffer.n_episodes
            print(f"Episode count: {seed}")
        else:
            replay_buffer = None
            seed = 0

        # --- Build env ---
        env = PickPlace2DEnv(render_size=render_size, window_size=window,
                             render_action=True)
        env.seed(seed)
        agent = env.teleop_agent()
        clock = pygame.time.Clock()

        obs   = env.reset()
        img   = env.render_frame(mode="human")
        info  = env._get_info()

        episode = []
        retry   = False
        done    = False
        grip_closed = False

        # Track target position for keyboard teleop
        tx = float(env.base_body.position.x)
        ty = float(env.base_body.position.y)

        pygame.display.set_caption(f"PickPlace2D  plan_idx:{plan_idx}")

        print("=" * 58)
        print("  2D Pick & Place Env — Teleop Demo")
        print("  Goal: drop the gold can into the green box")
        print("  Left-click drag   : move gripper toward cursor")
        print("  Right click       : toggle gripper open / closed")
        print("  ↑ ↓ ← → (additive): also move gripper")
        print("  SPACE (toggle)    : also open / close gripper")
        print("  R                 : retry")
        print("  Q / Escape        : quit")
        print("=" * 58)

        while not done:
            # ── Event handling ────────────────────────────────────────────
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close(); sys.exit(0)
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        env.close(); sys.exit(0)
                    if event.key == pygame.K_r:
                        retry = True
                    if event.key == pygame.K_SPACE:
                        grip_closed = not grip_closed
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                    grip_closed = not grip_closed

            if retry:
                break

            # ── Build action ──────────────────────────────────────────────
            # Arrow keys (held) – fixed step per frame updates target
            keys = pygame.key.get_pressed()
            key_step = 8.0
            if keys[pygame.K_RIGHT]: tx += key_step
            if keys[pygame.K_LEFT]:  tx -= key_step
            if keys[pygame.K_DOWN]:  ty += key_step
            if keys[pygame.K_UP]:    ty -= key_step

            # Mouse – override target position if left button is held
            mouse_btns = pygame.mouse.get_pressed()
            if mouse_btns[0]:
                tx, ty = pygame.mouse.get_pos()

            # Clamp to action space bounds
            tx = float(np.clip(tx, 0.0, float(window)))
            ty = float(np.clip(ty, 0.0, float(window)))

            grip = 1.0 if grip_closed else -1.0
            act  = np.array([tx, ty, grip], dtype=np.float32)

            # ── Step ───────────────────────────────────────────────────
            obs, reward, done, info = env.step(act)

            # ── Render ────────────────────────────────────────────────
            img = env.render_frame(mode="human")

            # ── Terminal HUD ─────────────────────────────────────────
            grasped_str = "GRASPED ●" if info["grasped"] else "         "
            placed_str  = "PLACED ✔" if info["placed"] else "        "
            grip_str    = "CLOSED" if grip_closed else "open  "
            state_vec   = np.concatenate([info["pos_agent"], info["can_pose"],
                                          info["target_pos"]])
            print(
                f"\r  rew={reward:.3f}  "
                f"can_x={info['can_pose'][0]:5.1f}  "
                f"grip={grip_str}  {grasped_str}  {placed_str}  ",
                end="", flush=True,
            )

            # Record if output path given and action is non-trivial
            if replay_buffer is not None and act is not None:
                episode.append({
                    "img":    img,
                    "state":  np.float32(state_vec),
                    "action": np.float32(act),
                })

            clock.tick(env.control_hz)

        print()
        # Save episode
        if not retry and replay_buffer is not None and len(episode) > 0:
            data_dict = {k: np.stack([x[k] for x in episode])
                         for k in episode[0]}
            replay_buffer.add_episode(data_dict, compressors="disk")
            plan_idx += 1
            print(f"Saved episode {seed}  ({len(episode)} steps)")
        elif retry:
            print("Retrying episode...")

        env.close()


if __name__ == "__main__":
    main()
