"""Input handling: pygame events / keys / mouse → action + session commands."""
from __future__ import annotations
import numpy as np
import pygame

from .config import Lift2DConfig


class TeleopController:
    """Translates user input into (command, action).

    command ∈ {"action", "retry", "quit"}.  For "action", the second value is
    a np.array([tx, ty, grip]); otherwise it is None.
    """

    KEY_STEP = 8.0

    def __init__(self, cfg: Lift2DConfig, base_pos):
        self.cfg = cfg
        self.tx = float(base_pos[0])
        self.ty = float(base_pos[1])
        self.grip_closed = False

    def poll(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit", None
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    return "quit", None
                if event.key == pygame.K_r:
                    return "retry", None
                if event.key == pygame.K_SPACE:
                    self.grip_closed = not self.grip_closed
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                self.grip_closed = not self.grip_closed

        keys = pygame.key.get_pressed()
        if keys[pygame.K_RIGHT]: self.tx += self.KEY_STEP
        if keys[pygame.K_LEFT]:  self.tx -= self.KEY_STEP
        if keys[pygame.K_DOWN]:  self.ty += self.KEY_STEP
        if keys[pygame.K_UP]:    self.ty -= self.KEY_STEP
        if pygame.mouse.get_pressed()[0]:
            self.tx, self.ty = pygame.mouse.get_pos()

        ws = self.cfg.window_size
        self.tx = float(np.clip(self.tx, 0.0, ws))
        self.ty = float(np.clip(self.ty, 0.0, ws))
        grip = 1.0 if self.grip_closed else -1.0
        return "action", np.array([self.tx, self.ty, grip], dtype=np.float32)