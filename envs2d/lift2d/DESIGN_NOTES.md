# Lift2D — Design Notes

Why the 2D lift environment is built the way it is. Companion to the code in
`envs2d/lift2d/`. Read this before changing physics constants or the grasp model.

---

## 1. Coordinate system & units

- **Y points down.** `pymunk_override.py` sets `positive_y_is_up = False`, so pygame and
  pymunk share axes: `y=0` is the top of the window, `y` grows downward, and gravity is
  therefore `+y`. This avoids any y-flip between physics and rendering.
- **Units are ~centimeters.** The world is `window_size = 512` units wide with a 36-unit
  block and 60-unit fingers — a plausible ~5 m scene with a 36 cm block if 1 unit = 1 cm.
  Gravity is set to **980** (cm/s²) to be consistent with that reading.

## 2. Module layout (why it's split)

Originally one ~700-line file mixed event handling, the env, the gripper, and the sim.
Split so each concern is testable and swappable in isolation:

| module        | responsibility |
|---------------|----------------|
| `config.py`   | `Lift2DConfig` dataclass — the **single source of truth** for every tunable |
| `physics.py`  | pymunk space, arena walls, block, contact counting |
| `gripper.py`  | base + finger bodies, jaw actuation, grasp state |
| `render.py`   | pygame canvas, debug-draw, connector lines, HUD |
| `env.py`      | `Lift2DEnv(gym.Env)` — wires the pieces, obs/reward/step |
| `teleop.py`   | pygame events/keys/mouse → action + session commands |
| `demo.py`     | CLI + interactive loop + zarr recording |

- **`__init__.py`** re-exports `Lift2DEnv`, `Lift2DConfig`, `main` so existing imports
  (`from envs2d.lift2d import main`, used by `demo_lift2d.py`) keep working.
- **`__main__.py`** makes the package runnable: `python -m envs2d.lift2d`.

### Config is the single source of truth
The env constructor originally re-declared every parameter as its own keyword default and
passed them into `Lift2DConfig`. Those duplicated defaults silently **overrode** the config
(e.g. `finger_gap_min` was really 18 even though the config said 8) — a real bug that
killed the grip. The constructor is now `__init__(config=None, reset_to_state=None,
**overrides)`: it uses a passed `Lift2DConfig`, or builds one from `**overrides`. No
duplicated defaults, no drift.

## 3. Scene

- **Arena walls** are four segments spanning the full `window_size` (an earlier version
  sized them to `SIM_SIZE=2.0`, so the "floor" was a 2px box in the corner and the block
  fell straight through). Wall thickness = 3.
- **Block spawns resting on the floor** at `y = window - wall_thickness - block_size/2`,
  not dropped from the air — deterministic start state, no settling transient.

## 4. The gripper

### Kinematic base, PD-driven, speed-capped
- The base is a **kinematic** body: the policy commands an absolute target `(tx, ty)` each
  step and a PD controller drives the base toward it (`k_p=180`, `k_v=30`, slightly
  overdamped for a crisp, non-oscillating approach).
- **Base speed is capped (`max_base_speed=200`).** This is the key to a stable friction
  grasp: without a cap, `k_p=180` yanks the gripper ~67px in a single control step — far
  faster than friction can accelerate the block — so a lifted block slips out. Capping
  *speed* (not lowering `k_p`) keeps positioning crisp while keeping acceleration inside
  the friction budget. Trade-off: the gripper visibly lags a very fast cursor drag.

### Dynamic fingers = a real friction grasp (not a weld)
We deliberately rejected the simpler "weld" grasp (pin the block to the base with a joint
on contact). A weld is not physical — the block can't slip or rotate, and reward doesn't
reflect real contact. Instead the block is held **purely by friction and normal force**,
like a real parallel-jaw gripper:

- Fingers are **dynamic** bodies (mass 0.5) so they can exert and feel contact forces.
- Each finger is constrained to the base by a **`GrooveJoint`** (slides only along the
  base's x-axis — the open/close direction, and rides with the base otherwise) plus a
  **`GearJoint`** locking its rotation to the (non-rotating) base. Together these make each
  jaw a 1-DOF slider that can't sag under gravity or spin.
- **Force-limited actuation** (`drive_fingers`): a closed grip commands a jaw gap
  *narrower than the block* (`finger_gap_min=8 < block_size/2=18`), so the jaw PD
  saturates at `max_grip_force` and presses the block with a steady normal force `N`.
  Friction `μ·N` on both faces then holds the block. `finger_gap_min` **must** stay below
  `block_size/2`, or the jaws stop flush against the block and exert zero squeeze.

### Grasp detection is contact-based
`grasped = grip_closed AND left jaw touches block AND right jaw touches block`, tested with
`Shape.shapes_collide` (distance < 1px). There is no grasp joint; reward = 1.0 iff grasped.

## 5. Grip force is computed from physics, not tuned

The grip force is **derived**, not hand-picked. To hold the block by friction on both
jaw faces, friction must balance weight:

```
2·μ·F ≥ m·g   →   F = S · m·g / (2μ)
```

- `μ = finger_friction · block_friction` — pymunk multiplies the two shapes' friction
  coefficients at a contact (default rule). With `1.8 × 1.5 = 2.7`.
- `m = block_mass`, `g = gravity`.
- `S = grip_safety_factor` — a single dimensionless margin covering stiction, the transient
  acceleration of a lift, and numerical slack.

`Lift2DConfig.__post_init__` computes `max_grip_force` from these, so changing the block's
mass or friction (or gravity) **automatically re-derives the grip force** — and because
`make_block` reads the same `block_mass`/`block_friction` from config, `μ` and `m` stay
consistent between the physics and the force formula. With the defaults
(`S=2.5, m=1, g=980, μ=2.7`) this yields `F ≈ 454`.

We deliberately reduced this to a **single** physical knob. An earlier version added a
separate `grip_lift_accel` term (`F = S·m·(g+a)/(2μ)`); it was removed as redundant — the
safety factor already accounts for lift acceleration, and a bare "hold" condition is the
clean physical statement.

### The two failure modes `S` balances
- **Too little force → slips on lift.** `F` must exceed `m·g/(2μ) ≈ 181` per jaw just to
  hold statically (`S=1`), more to accelerate during a lift. Below `S≈2.2` lifts slip.
- **Too much force → the block extrudes** ("watermelon seed"). Because the jaws grip a
  floor-resting block on its *upper* portion (the base can't reach the block's center near
  the floor), a hard, slightly-asymmetric squeeze shoves the block up/out. Worse at high
  `S`. `S=2.5` (≈454 N) is the sweet spot: reliably holds without extruding. If a heavier
  payload slips, raise `S`; if extrusion returns, lower it.

- **`solver_iterations = 25`** (up from the default 10): stiffer contact solving reduces
  jitter when the dynamic jaws press the block.

## 6. Rendering & the demo loop

- The **pygame window is created once** (in `Renderer`, on env construction) — not on
  import — so importing the env for headless/training use doesn't spawn a window.
- The demo creates the env **once** and calls `env.reset()` per episode. Pressing **R**
  resets the scene *in the same window* instead of tearing down pygame and reopening a new
  window (the old loop re-constructed the env every episode).

## 7. Known limitations / realistic behavior

- **Gross misalignment (≳ half a block off-center) spins the block** as a single jaw
  catches a corner. This is realistic contact behavior, not a bug.
- **Instantaneous large target jumps** can out-accelerate friction and drop the block.
  Continuous motion (real teleop, or a smooth policy) does not trigger this.
- The grasp is **tuning-sensitive** by nature (base speed × grip force × friction ×
  solver iterations). The current values are validated by a headless stress suite
  (grip across x, off-center, lift+hold, smooth carry, 200-step idle, re-grip cycling,
  empty-air close, near-wall grips, adversarial toggling, 25/25 random picks).

## 8. Key constants (see `config.py` for the full list)

| constant | value | why |
|----------|-------|-----|
| `gravity` | 980 | centimeter-scale units |
| `finger_gap_min` | 8 | < block half-width (18) so jaws squeeze, not rest flush |
| `grip_safety_factor` | 2.5 | margin in the computed grip force `F = S·m·g/(2μ)` (≈454 N) |
| `block_mass`, `finger_friction`, `block_friction` | 1.0, 1.8, 1.5 | physical inputs to the grip-force formula (and the shapes) |
| `max_base_speed` | 200 | keeps lift/carry acceleration within the friction budget |
| `solver_iterations` | 25 | reduce contact jitter |
| `k_p`, `k_v` | 180, 30 | crisp, overdamped base positioning |

> `max_grip_force` is **not** a config constant — it is computed in `__post_init__` from
> `grip_safety_factor`, `block_mass`, `gravity`, and the two friction coefficients.
