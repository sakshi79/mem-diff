"""2D parallel-gripper lift task."""
from __future__ import annotations
import numpy as np
import gym
from gym import spaces
from pymunk.vec2d import Vec2d

from .config import Lift2DConfig
from .physics import build_space, add_walls, make_block, ContactCounter
from .gripper import Gripper
from .render import Renderer


class Lift2DEnv(gym.Env):
    """2D parallel-gripper lift task."""

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}
    reward_range = (0.0, 1.0)

    def __init__(self, config: Lift2DConfig | None = None, reset_to_state=None, **overrides):
        self.cfg = config if config is not None else Lift2DConfig(**overrides)
        self.reset_to_state = reset_to_state
        self.latest_action = None
        self._seed = None
        self.seed()

        self.renderer = Renderer(self.cfg)

        # Runtime handles (created in reset())
        self.space = None
        self.gripper: Gripper | None = None
        self.block = None
        self.block_shape = None
        self.contacts: ContactCounter | None = None

        ws, rs = self.cfg.window_size, self.cfg.render_size
        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0, high=1, shape=(3, rs, rs), dtype=np.float32),
            'agent_pos': spaces.Box(low=0, high=ws, shape=(2,), dtype=np.float32),
        })
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            high=np.array([float(ws), float(ws), 1.0], dtype=np.float32),
            shape=(3,), dtype=np.float32,
        )

    # ---- seeding -----------------------------------------------------------

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)

    # ---- lifecycle ---------------------------------------------------------

    def _setup(self):
        cfg = self.cfg
        self.space = build_space(cfg)
        add_walls(self.space, size=cfg.window_size, thickness=cfg.wall_thickness)
        self.gripper = Gripper(self.space, cfg, position=(cfg.window_size / 2, cfg.window_size // 3))
        self.block, self.block_shape = make_block(
            self.space, cfg, position=(cfg.window_size / 2, cfg.window_size - 80))
        self.contacts = ContactCounter(self.space)

    def reset(self):
        self._setup()
        state = self.reset_to_state
        if state is None:
            rs, ws = self.np_random, self.cfg.window_size
            bx = float(rs.integers(100, ws - 100))
            by = float(ws - self.cfg.wall_thickness - self.cfg.block_size / 2)  # on the floor
            gx = float(rs.integers(100, ws - 100))
            gy = float(rs.integers(ws // 4, ws // 2))
            state = np.array([gx, gy, bx, by], dtype=np.float32)
        self._set_state(state)
        return self._get_obs()

    def _set_state(self, state):
        gx, gy, bx, by = (float(v) for v in state[:4])
        self.gripper.set_pose((gx, gy))
        self.block.position = Vec2d(bx, by)
        self.block.velocity = Vec2d(0, 0)
        self.block.angle = 0.0
        self.space.step(1.0 / self.cfg.sim_hz)

    def step(self, action):
        cfg = self.cfg
        dt = 1.0 / cfg.sim_hz
        n_steps = cfg.sim_hz // cfg.control_hz
        self.contacts.reset()

        if action is not None:
            self.latest_action = np.asarray(action, dtype=np.float32).copy()
            tx, ty = float(action[0]), float(action[1])
            self.gripper.set_grip(np.clip(action[2], -1.0, 1.0))
            for _ in range(n_steps):
                self.gripper.drive_fingers(dt)
                self.gripper.drive_base((tx, ty), dt)
                self.space.step(dt)
            self.gripper.update_contacts(self.block_shape)

        reward = 1.0 if self.gripper.grasped else 0.0
        return self._get_obs(), reward, False, self._get_info()

    # ---- render / obs / info ----------------------------------------------

    def render(self, mode: str = "rgb_array"):
        return self.render_frame(mode)

    def render_frame(self, mode: str):
        return self.renderer.render(self.space, self.gripper, mode)

    def _get_obs(self) -> dict:
        img = self.render_frame("rgb_array")
        return {
            "image": np.moveaxis(img.astype(np.float32) / 255.0, -1, 0),
            "agent_pos": np.array(self.gripper.base.position, dtype=np.float32),
        }

    def _get_info(self) -> dict:
        cfg = self.cfg
        n_steps = cfg.sim_hz // cfg.control_hz
        return {
            "pos_agent": np.array(self.gripper.base.position, dtype=np.float32),
            "vel_agent": np.array(self.gripper.base.velocity, dtype=np.float32),
            "block_pose": np.array(list(self.block.position) + [self.block.angle], dtype=np.float32),
            "grasped": self.gripper.grasped,
            "grip": float(self.gripper.grip_value),
            "n_contacts": int(np.ceil(self.contacts.n / n_steps)),
        }

    def close(self):
        self.renderer.close()