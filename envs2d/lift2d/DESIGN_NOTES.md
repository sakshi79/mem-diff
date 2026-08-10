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

### Kinematic base, direct position tracking
- The base is a **kinematic** body: the policy commands an absolute target `(tx, ty)`
  each step and `drive_base` **teleports** the base to that target (clamped to
  arena-safe bounds derived from finger geometry). Pymunk's `base.velocity` is set
  to zero so `space.step` does NOT re-integrate the motion; slip detection uses a
  separately-tracked `_commanded_bvel` field.
- **Clamp bounds are geometry-derived**, not a fixed margin. With the base at the top
  of the fingers, `y_max = ws − wall_thickness − finger_length` (finger bottom just
  touches the floor top), and `x_min/x_max = wall_thickness ± (finger_gap_max +
  finger_width/2)` (fingers stay inside side walls at full open).
- The base sits at the **top** of the fingers (base.y = finger top). Fingers hang
  `finger_length` below the base.

### Kinematic fingers with a slide-state motor (Plan B)
The block is held by friction, but the fingers are not dynamic bodies — they're
kinematic, positioned each substep from `base.position + slide_offset`. Each finger's
`slide` (its distance from the base along the open/close axis, in `[finger_gap_min,
finger_gap_max]`) is a scalar state we advance at `jaw_speed` and cap at block
contact (predict-and-cap). The physical force chain — motor commands velocity →
normal force `N` grows to `F_max` at equilibrium → friction `μ·N` holds — is modelled
directly at the equilibrium (see "Force-balance model" below). Grip force `F_max`
is the physics-computed value from §5.

Finger shapes are **sensors** — pymunk will detect their overlap for our contact queries
but will not apply contact response impulses to the block. All finger-block interaction
is handled explicitly in `update_grasp`.

### Grasp is an explicit state machine
`grasped` is a boolean state on `Gripper`. Transitions:
- **Enter:** `grip_closed AND both fingers stalled at block` → set `grasped = True`,
  record `grasp_offset = block.position − base.position`.
- **Exit:** `grip_closed = False` (grip release). *(Slip-based exit is currently
  disabled — see §7.3.)*

While `grasped`, the block is **position-synced** to the base each substep:
`block.position = base.position + grasp_offset`, `block.velocity = base.velocity`,
`block.angular_velocity = 0`. This makes the carry rigid.

### Force-balance model (how the grasp actually holds)

The physical chain that lets a friction grasp work is the same regardless of whether we
implement it with dynamic fingers + contact impulses or kinematic fingers + explicit
state (see §7 for the implementation history):

1. The motor commands the jaw's slide velocity `v = jaw_speed` (inward when closing).
2. Free-moving, the jaw accelerates toward `v`.
3. On contact with the block, a normal force `N` develops at the finger–block face.
4. The motor's output is clamped at `F_max = max_grip_force` (physics-derived, §5), so
   the maximum `N` the motor can sustain is `F_max`.
5. Equilibrium: `N` grows until it balances `F_max` → net force on the jaw = 0 →
   slide velocity → 0. The motor is now *stalled*: still commanding `v` but unable to
   achieve it against the block.
6. That sustained `N = F_max` per jaw is the "grip force." Friction at each contact
   is `μ·N`. Total lateral friction available to resist block motion is `2·μ·N`.
7. Slip: if the block's required lateral acceleration exceeds `2·μ·N / m_block`,
   friction can't hold it — the grasp breaks.

**Halted by normal force, not "by contact" per se.** Contact is just the geometric
moment when `N` starts being non-zero; the *halting* is the force balance in step 5.

**We model the equilibrium directly, not the transient.** Rather than dynamically
simulating steps 3–5 (waiting for `N` to grow to `F_max` and the jaw to decelerate),
we treat the equilibrium as an instant snap: whenever the commanded slide would put
the finger inside the block, we cap it at the contact position. Equivalent to
assuming the motor was saturated (`N = F_max`) the whole time, which matches the
dynamic result to within one substep because the servo is already saturated in normal
use (`k_motor·jaw_speed = 2000 ≫ F_max = 454`). The `N = F_max` at contact is what the
slip check spends as friction budget (`2·μ·F_max`).

### Reward: successful lift, not just grasped
`reward = 1.0` when the block is above the floor by at least `lift_threshold` pixels
(default 50). Formally: `block.y < window − wall_thickness − block_size/2 −
lift_threshold`. Reward stays 0 during the pickup transient (block still on floor while
jaws close) and only turns on once the block has been carried upward — matching the
task's "lift" semantics. There is no grasp joint; grasped/not-grasped is state, and
lifting is what the reward measures.

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

## 7. Design decision log (roadblock → implementation → outcome)

Recent design changes, recorded only after implementation and verification. Each entry:
what was broken, what we did and why, and what actually happened when we tested it.

### 7.1 Velocity-motor jaws (replaced position-spring PD)

**Roadblock.** The initial jaw controller was a position PD: `F = k_grip·(target − pos)`
clipped to `max_grip_force`. To reach the stall force `F ≈ 454 N`, the spring had to be
stretched by `F/k_grip = 454/60 ≈ 7.5 px` — i.e. the jaws were perpetually commanded
**7.5 px INSIDE the block**, and the spring pushed back with `F` via the block's contact.
This coupled *force to penetration*: hard "snap" on close, sustained over-press, and
watermelon-seed extrusion under any asymmetric grip.

**Implementation & why.** Replaced the position spring with a velocity-controlled motor
(matches the mental model of a real force-limited servo motor):
`fx = k_motor·(v_target_slide − v_slide)` clipped to `±max_grip_force`. When the jaw
stalls against the block, the clip *is* the per-jaw normal force `N`; friction `μ·N`
does the holding. Force is now decoupled from any position target — no spring stretch,
no penetration coupling.

**Outcome.** Verified live and headless: extrusion (previously reproducible with the
spring) is gone at the physics-computed `F = S·m·g/(2μ) ≈ 454 N`. Grip on cube gives
gap ≈ 50 (jaws touching cube's outer faces + finger width), lifts hold, releases clean.
Note the servo saturates almost always in normal use (`k_motor·jaw_speed = 2000 ≫ 454`),
so effectively the motor delivers `±max_grip_force` in the commanded direction until
the finger is at the stop or in contact. That saturation matters for the next entry.

### 7.2 Servo relative to base (Fix 1) + base velocity clamp at limits (Fix 2)

**Roadblock.** Two failures appeared in the live demo that the headless stress suite
had missed:

1. **Self-closing under mouse drag.** Moving the gripper around with the mouse caused
   the fingers to drift inward — sometimes to gap ≈ 64 (about the mid-open position),
   *without touching the cube*. The gripper looked "soft" instead of rigid.
2. **Floor jam.** Holding the mouse cursor below the arena (base pinned at its y-limit)
   left the fingers stuck at asymmetric positions (one at closed stop, one at open),
   unresponsive to grip toggling.

Log tracing revealed the root causes:

1. During fast base motion, the velocity servo compared the finger's *world* velocity
   to `v_target`. When the base drags the finger at, say, `+169 px/s`, the finger's
   world velocity matches, but that isn't the physical DOF the servo controls — the
   *slide* along the groove is. The absolute servo could still saturate, but the
   diagnosis was fuzzy.
2. When base position was clamped at the arena limit but the PD kept trying to push
   past, the base's velocity accumulated (`bvel.y = +118` in the log). The `GrooveJoint`
   uses base velocity in its constraint computation, so it kept accelerating the fingers
   downward — into the floor. The debug log showed the groove joint's y-constraint
   *violated by 10 px* (`Llocal.y = +10.2`). That put the finger bottom 10 px inside
   the floor's segment radius → huge normal force → floor-finger friction `μ×N`
   dwarfed the servo's 454 N → fingers pinned laterally, wherever they happened to be.

**Implementation & why.**

- **Fix 1 — servo in the base's frame** (`gripper.py drive_fingers`):
  compute `v_slide = body.velocity.x − base.velocity.x` and use that in the servo
  error, instead of `body.velocity.x` alone. Physically correct: the servo controls
  the *sliding* DOF along the groove, which is the actual motor axis; the base's
  world motion shouldn't matter.
- **Fix 2 — clamp base velocity at position limits** (`gripper.py drive_base`):
  after clipping `base.position` to `[margin, ws − margin]`, zero the component of
  `base.velocity` that would push further into the wall/floor. Removes the persistent
  "into the wall" velocity that was straining the groove joint.

**Outcome.**

- **Fix 2 did the operational work.** With it, the floor jam is gone (headless: base
  parked at y-limit → fingers stay at open stops, gap = 98). Live confirmation:
  self-closing near/on the cube and floor-jam behavior both cleared.
- **Fix 1, honestly, did NOT change behavior.** The servo saturates on any error
  larger than `max_grip_force / k_motor = 454 / 10 = 45 px/s`. In practice, errors are
  always far above 45, so both formulations clip to the same `±454`. We kept Fix 1
  because it is the correct physical model — if `k_motor` is later raised or
  `max_grip_force` lowered enough to un-saturate, only the relative servo tracks
  cleanly. It just doesn't earn its keep right now.
- **Residual to be addressed:** after an *abrupt base direction change* (mouse whip),
  fingers overshoot asymmetrically for ~100–200 ms before returning to symmetric.
  Log shows `Lloc` jumping from −49 (open) to −16 (closed) in one control step when
  the base decelerates from `+184` to `−200` in one step — that's ~4000 px/s² of
  deceleration, and the servo's max jaw acceleration is only `max_grip_force /
  finger_mass = 454 / 0.5 = 908 px/s²`. Candidate fix (not yet applied): cap base
  acceleration to what the servo can compensate.

### 7.3 Plan B: kinematic base + kinematic fingers, teleport tracking, explicit grasp

**Roadblock.** Even after §7.1 and §7.2, the live gripper still felt like a soft body,
not a rigid tool. Two symptoms:

1. **Base–finger "spring."** During any fast base motion, the dynamic fingers lagged
   the base (base's PD accel could reach ~18000 px/s² on cursor jumps; finger servo
   maxed at `F_max/finger_mass = 454/0.5 = 908 px/s²`). Directional changes produced
   ~100–200 ms of asymmetric finger drift — one finger drifted to its closed stop
   while the other overshot outward. Visually the gripper "wobbled."
2. **Base with virtual inertia.** The base was kinematic in pymunk terms but PD-driven,
   which mathematically emulated a spring-damper. The user's mental model of "the
   base is just a control point" didn't match. Why should a coordinate need inertia?

**Implementation & why.** We rewrote both actuation layers as kinematic:

- **Base:** teleport directly to the target each frame. `base.position = target`
  (clamped to geometry-safe bounds); `base.velocity = (0, 0)`; commanded velocity
  tracked in `_commanded_bvel` for downstream use. Pymunk's kinematic body semantics
  — position advances by `velocity·dt` during `space.step` — would otherwise cause
  double motion, so zeroing pymunk's velocity is essential.
- **Fingers:** replaced dynamic bodies + `GrooveJoint` + `GearJoint` + velocity servo
  with kinematic bodies whose positions are set each substep from `base.position +
  slide_offset`. The velocity motor still exists conceptually — `slide_offset`
  advances at `jaw_speed` per substep, capped at block contact (predict-and-cap).
  Finger shapes made **sensors** so pymunk doesn't push the block on overlap.
- **Grasp:** explicit state machine. `grasped=True` when grip is closed AND both
  fingers are stalled at the block. Block is position-synced to base each substep.
- **Geometry:** base moved to the top of fingers (was center). Fingers hang below.
  Arena clamps recomputed from finger geometry so the finger bottom never dips
  below the floor.
- **Reward:** changed from "grasped" to "lifted" (block above `lift_threshold` px
  from floor rest). Grasp is a mid-step state; lift is the task.

**Outcome.**

*What works, verified headless and live:*
- Base tracks the mouse 1:1 — zero lag, no soft feel.
- Fingers rigidly follow the base — no asymmetric drift, ever.
- Grasp forms cleanly on close (both jaws stall at block).
- Lift + carry + release cycle works (block rides with base through arbitrary
  translations; falls to floor on release).
- Finger geometry respects the floor (no digging).
- Reward correctly gates on lift (0 during pickup, 1 once block is airborne past
  `lift_threshold`, 0 once released and fallen back below it).

*What's disabled and why (residual):*
- **Physical slip check is off.** Direct position tracking produces infinite
  instantaneous acceleration — any teleport by even a few pixels trips a real
  friction-budget threshold `2μN/m ≈ 2451 px/s²` in one substep. EMA smoothing
  (attempted at α=0.3) doesn't help because the *derivative* of a smoothed spike
  is still a spike. Solutions require a fundamentally different formulation —
  e.g., a velocity-based slip criterion tuned to teleop timescales, or a
  smoothing/rate-limit layer on the *commanded target* before it reaches
  `drive_base`. Deferred.

*What we let go:*
- **Emergent friction physics.** Grasp is now a state we choose, not something
  pymunk discovers via contact impulses. Block spin during grasp is nullified
  (`angular_velocity = 0`). This is cleaner and more predictable for data
  collection, at the cost of losing block-wall bounces mid-carry and similar
  emergent behavior.
- **Dead code paths.** `k_p`, `k_v`, `base_margin`, `k_motor`, `max_base_speed`,
  `finger_mass` are all commented out in config as Plan-A escape hatches. If we
  ever revert to the dynamic-fingers model, they're one uncomment away.

## 8. Known limitations / realistic behavior

- **Gross misalignment (≳ half a block off-center) spins the block** as a single jaw
  catches a corner. This is realistic contact behavior, not a bug.
- **Instantaneous large target jumps** can out-accelerate friction and drop the block.
  Continuous motion (real teleop, or a smooth policy) does not trigger this.
- The grasp is **tuning-sensitive** by nature (base speed × grip force × friction ×
  solver iterations). The current values are validated by a headless stress suite
  (grip across x, off-center, lift+hold, smooth carry, 200-step idle, re-grip cycling,
  empty-air close, near-wall grips, adversarial toggling, 25/25 random picks).

## 9. Key constants (see `config.py` for the full list)

| constant | value | why |
|----------|-------|-----|
| `gravity` | 980 | centimeter-scale units |
| `finger_gap_min` | 8 | < block half-width (18) so jaws squeeze, not rest flush |
| `finger_length` | 60 | fingers hang this far below base; sets the y-max clamp |
| `grip_safety_factor` | 2.5 | margin in the computed grip force `F = S·m·g/(2μ)` (≈454 N) |
| `block_mass`, `finger_friction`, `block_friction` | 1.0, 1.8, 1.5 | physical inputs to the grip-force formula (and the shapes) |
| `jaw_speed` | 200 | slide-motor commanded speed (px/s) — controls how fast jaws open/close |
| `lift_threshold` | 50 | block must be this far above floor rest for reward = 1 |
| `solver_iterations` | 25 | reduce contact jitter (still applied to space) |

> `max_grip_force` is **not** a config constant — it is computed in `__post_init__` from
> `grip_safety_factor`, `block_mass`, `gravity`, and the two friction coefficients.

> `k_p`, `k_v`, `base_margin`, `k_motor`, `max_base_speed`, `finger_mass` are commented
> out in `config.py` — leftover knobs from the pre-Plan-B controllers. Kept as escape
> hatches for potential future revert (see §7.3).
