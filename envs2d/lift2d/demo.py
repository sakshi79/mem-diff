"""Interactive teleop demo (optionally recording to a zarr replay buffer)."""
from __future__ import annotations
import click
import numpy as np
import pygame

from diffusion_policy.env.pusht.replay_buffer import ReplayBuffer
from .env import Lift2DEnv
from .teleop import TeleopController


def _print_help():
    print("=" * 58)
    print("  2D Lift Env — Teleop Demo")
    print("  Left-click drag   : move gripper toward cursor")
    print("  Right click       : toggle gripper open / closed")
    print("  ↑ ↓ ← → (additive): also move gripper")
    print("  SPACE (toggle)    : also open / close gripper")
    print("  R                 : reset env")
    print("  Q / Escape        : quit")
    print("=" * 58)


@click.command()
@click.option("-o", "--output", default=None, required=False,
              help="Path to zarr replay buffer. Omit to run without recording.")
@click.option("--render-size", default=96, type=int, show_default=True)
@click.option("--window", default=512, type=int, show_default=True)
def main(output, render_size, window):
    env = Lift2DEnv(render_size=render_size, window_size=window, render_action=True)
    clock = pygame.time.Clock()
    plan_idx = 0
    _print_help()

    try:
        while True:                       # one iteration == one episode
            if output is not None:
                replay_buffer = ReplayBuffer.create_from_path(output, mode='a')
                seed = replay_buffer.n_episodes
                print(f"Episode count: {seed}")
            else:
                replay_buffer, seed = None, 0

            env.seed(seed)
            env.reset()                   # rebuilds the scene, same window
            env.render_frame("human")

            controller = TeleopController(env.cfg, env.gripper.base.position)
            episode, retry, done = [], False, False
            pygame.display.set_caption(f"Lift2D  plan_idx:{plan_idx}")

            while not done:
                cmd, act = controller.poll()
                if cmd == "quit":
                    return                 # finally: env.close()
                if cmd == "retry":
                    retry = True
                    break                  # falls through to reset, no window teardown

                obs, reward, done, info = env.step(act)
                img = env.render_frame("human")

                grasped_str = "GRASPED ●" if info["grasped"] else "         "
                g = env.gripper
                gap = g.right.position.x - g.left.position.x
                state_vec = np.concatenate([info["pos_agent"], info["block_pose"]])
                print(f"\r grip_val={g.grip_value:+.2f} gap={gap:5.1f} "
                      f"Lx={g.left.position.x:5.1f} Rx={g.right.position.x:5.1f} "
                      f"block_x={info['block_pose'][0]:5.1f} {grasped_str}",
                      end="", flush=True)

                if replay_buffer is not None:
                    episode.append({"img": img,
                                    "state": np.float32(state_vec),
                                    "action": np.float32(act)})
                clock.tick(env.cfg.control_hz)

            print()
            if not retry and replay_buffer is not None and len(episode) > 0:
                data_dict = {k: np.stack([x[k] for x in episode]) for k in episode[0]}
                replay_buffer.add_episode(data_dict, compressors="disk")
                plan_idx += 1
                print(f"Saved episode {seed}  ({len(episode)} steps)")
            elif retry:
                print("Retrying episode...")
    finally:
        env.close()