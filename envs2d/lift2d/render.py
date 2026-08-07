"""
PyGame rendering: debug-draw of the space + connector lines + HUD.
"""

from __future__ import annotations
import cv2
import numpy as np
import pygame

from diffusion_policy.env.pusht.pymunk_override import DrawOptions
from .config import Lift2DConfig

def _pg(v):
    return (round(v.x), round(v.y))

class Renderer:
    def __init__(self, cfg: Lift2DConfig):
        self.cfg = cfg
        pygame.init()
        pygame.display.init()
        self.window = pygame.display.set_mode((cfg.window_size, cfg.window_size))
        pygame.display.set_caption("2D Lift Env")
        self.clock = None
        self.screen = None

    def render(self, space, gripper, mode: str = "rgb_array") -> np.ndarray:
        cfg = self.cfg 
        ws = cfg.window_size 
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()
        
        canvas = pygame.Surface((ws, ws))
        canvas.fill((24, 24, 30))
        self.screen = canvas
        space.debug_draw(DrawOptions(canvas))
        lp, rp, bp = _pg(gripper.left.position), _pg(gripper.right.position), _pg(gripper.base.position)
        mid = ((lp[0]+rp[0])//2, (lp[1]+rp[1])//2)
        pygame.draw.line(canvas, (100, 120, 160), lp, rp, 3)
        pygame.draw.line(canvas, (100, 120, 160), bp, mid, 3)
        self._draw_hud(canvas, gripper)
        if mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(cfg.control_hz)
        img = np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes = (1,0,2))
        return cv2.resize(img, (cfg.render_size, cfg.render_size), interpolation = cv2.INTER_AREA)

    def _draw_hud(self, canvas, gripper):
        try:
            if not pygame.font.get_init():
                pygame.font.init()
            font = pygame.font.SysFont("monospace", 12)
            ws = self.cfg.window_size

            if gripper.grasped:
                canvas.blit(font.render("● GRASPED", True, (255, 220, 50)), (10, 10))
            
            bar_x, bar_y, bar_w, bar_h = 10, ws - 22, 140, 12
            pygame.draw.rect(canvas, (50, 50, 60), (bar_x, bar_y, bar_w, bar_h))
            fill_w = int((gripper.grip_value + 1) / 2 * bar_w)
            g_color = (80, 210, 110) if gripper.grip_closed else (210, 80, 80)
            pygame.draw.rect(canvas, g_color, (bar_x, bar_y, fill_w, bar_h))
            pygame.draw.rect(canvas, (120, 120, 130), (bar_x, bar_y, bar_w, bar_h), 1)
            canvas.blit(
                font.render(f"grip {gripper.grip_value:+.2f}", True, (200, 200, 200)),
                (bar_x + bar_w + 6, bar_y),
            )
        except Exception:
            pass
            
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None


