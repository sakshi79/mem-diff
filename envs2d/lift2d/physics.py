"""
Simulation layer: pymunk space, arena walls, block, contact counting.
"""

from __future__ import annotations
import pygame 
import pymunk
from pymunk.vec2d import Vec2d

from .config import Lift2DConfig 

def build_space(cfg: Lift2DConfig) -> pymunk.Space:
    space = pymunk.Space()
    # positive_y_is_up = False → y=0 TOP, gravity pulls toward positive y
    space.gravity = (0.0, cfg.gravity)
    space.damping = cfg.damping
    return space

def add_walls(space: pymunk.Space, size: float, thickness: float = 3.0):
    corners = [(0, 0), (size, 0), (size, size), (0, size)]
    walls = []
    for a, b in zip(corners, corners[1:] + corners[:1]):
        seg = pymunk.Segment(space.static_body, a, b, thickness)
        seg.color = pygame.Color("LightGray")
        seg.friction = 0.1
        seg.elasticity = 0.8
        walls.append(seg)
    space.add(*walls)
    return walls

def make_block(space: pymunk.Space, cfg: Lift2DConfig, position):
    mass = 1.0
    inertia = pymunk.moment_for_box(mass, (cfg.block_size, cfg.block_size))
    body = pymunk.Body(mass, inertia)
    body.position = Vec2d(*position)
    shape = pymunk.Poly.create_box(body, (cfg.block_size, cfg.block_size))
    shape.friction = 1.5
    shape.elasticity = 0.05
    shape.color = pygame.Color("Tomato")
    space.add(body, shape)
    return body, shape

class ContactCounter:
    """Counts post-solve contact points across the space (collision type 0)."""
    def __init__(self, space: pymunk.Space):
        self.n = 0
        handler = space.add_collision_handler(0,0)
        handler.post_solve = self._post_solve

    def reset(self):
        self.n = 0

    def _post_solve(self, arbiter, space, data):
        self.n += len(arbiter.contact_point_set.points)
