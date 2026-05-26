# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Lint (CI uses this exact command)
ruff check src/ tests/

# Run all tests (CARLA tests auto-skip if no server is running)
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_config_loading.py -v

# Run a single test by name
python -m pytest tests/test_config_loading.py::test_ppo_reward_profiles -v

# Launch CARLA (always use -dx12 on RTX 5080 Blackwell)
scripts/launch_carla.bat
# or manually:
CARLA_0.9.16/CarlaUE4.exe -dx12 -quality-level=Low -fps=20 -benchmark -windowed -ResX=800 -ResY=600 -nosound -NoSplash
```

No build step — pure Python. CI installs `requirements-ci.txt` (excludes the `carla` wheel, which is Windows-only).

## Architecture

### Five-workflow pipeline

Each notebook maps 1:1 to a WAT agent in `src/agents/`:

| Notebook | Agent | CARLA needed |
|----------|-------|-------------|
| NB01 data collection | `DataCollectionAgent` | yes — subprocess-per-town |
| NB02 behavior cloning | `BehaviorCloningAgent` | no |
| NB03 PPO fine-tuning | `PPOAgent` | yes — subprocess-per-town |
| NB04 evaluation | `EvaluationAgent` | yes — manual restart between towns |
| NB05 causal analysis | `CausalAnalysisAgent` | no |

The agent classes do all the real work; notebooks are thin shells that instantiate one agent and call `.run()`.

### CARLA lifecycle constraint

**Kill the CARLA process between towns — never reuse it.** In-place `client.load_world()` after a sensor cycle aborts on RTX 5080 Blackwell. All agents that touch multiple towns spawn a fresh `CarlaUE4.exe` subprocess per town via `subprocess.Popen` and kill it with `taskkill` after. The 6-second sleep after kill is load-bearing — remove it and the next `create_connection` races the process exit.

### Reward pipeline (three layers)

Rewards stack in this order:

1. **`CarlaEnv.step()`** — base `+1.0/step`, `-3.0` on lane invasion, `-200` and `terminated=True` on collision. Exposes `crossed_lane_marking_types` in `info`.

2. **`RoadRuleMonitor.step()`** — Gymnasium `Wrapper` around `CarlaEnv`. Injects CA driving rule penalties *before* the rollout buffer sees the reward:
   - *Tier 1* (−200, terminate): red light, wrong-way, off-road, double-solid crossing
   - *Tier 2* (per-step/one-time, no terminate): speeding, tailgating, stop sign, solid-lane crossing, failure to yield

3. **`compute_style_reward()` in `ppo.py`** — style-weight-dependent shaping applied in `PPOAgent._collect_rollout()`: speed bonus, jerk penalty, lane-change penalty, abrupt-steering penalty (Tier 3). Weights come from `configs/ppo.yaml → reward_profiles → {chill|standard|hurry}`.

Do not add driving-rule logic to layer 3 (it's style-agnostic) and do not add style-weight logic to layer 2 (it belongs in layer 3).

### Config pattern

Every hyperparameter lives in `configs/*.yaml`. Agents and notebooks load them via `src.config.load_config(name)`. Call `require_keys(cfg, [...], name)` at the top of any new agent to fail fast if the YAML changes. Never hard-code a hyperparameter in agent or notebook code.

### Sensor suites

Three sensor suites share the same environment and training code but use different observation keys and model classes:

| Suite | Observation key | Model class |
|-------|----------------|-------------|
| `single_cam` | `"camera"` (H×W×3) | `ActorCritic` / `DriveNet` |
| `multi_cam` | `"cameras"` (3×H×W×3) | `MultiCamActorCritic` / `MultiCamDriveNet` |
| `lidar` | `"bev"` (H×W×3 float32) | `ActorCritic` / `LidarDriveNet` |

The `ActorCritic` (PPO) and `DriveNet` (BC) are initialized with the same CNN backbone weights. Only the head differs — BC includes metadata embeddings; PPO replaces the head with actor/critic heads and a learned style embedding.

### CARLA sensor callback safety

Every `sensor.listen()` callback is wrapped in `try/except` with a `weakref.ref`. CARLA dispatches callbacks from a C++ Boost.Asio thread — any Python exception that escapes can corrupt the `io_context` and trigger STATUS_STACK_BUFFER_OVERRUN. Drop frames silently rather than propagating exceptions out of a callback.

### Synchronous mode

The environment always runs in synchronous mode (`world.apply_settings(synchronous_mode=True)`). Call `world.tick()` exactly once per `env.step()`. Never call `world.tick()` outside of `step()` while a rollout is in progress — it will desync the sensor frame counter and cause blocking hangs in `_retrieve_camera_frame`.

### PPO → BC weight transfer

`ActorCritic.__init__` filters BC checkpoint keys to `features.*` before loading. BC head weights are deliberately excluded because the head input dimension differs (BC includes metadata embeddings; PPO does not). If you add new backbone layers in `DriveNet`, update the `startswith("features.")` filter in both `ActorCritic` and `MultiCamActorCritic`.

### State vector (6 dimensions, always)

`[speed / 60.0, sin(heading), cos(heading), speed_limit / 130.0, lane_count / 4.0, is_junction]`

The state vector is observation-only — road rule enforcement happens in `RoadRuleMonitor` via CARLA API calls, not from the state vector.

## Skill routing

When the user's request matches an available skill, invoke it via the Skill tool. When in doubt, invoke the skill.

Key routing rules:
- Product ideas/brainstorming → invoke /office-hours
- Strategy/scope → invoke /plan-ceo-review
- Architecture → invoke /plan-eng-review
- Design system/plan review → invoke /design-consultation or /plan-design-review
- Full review pipeline → invoke /autoplan
- Bugs/errors → invoke /investigate
- QA/testing site behavior → invoke /qa or /qa-only
- Code review/diff check → invoke /review
- Visual polish → invoke /design-review
- Ship/deploy/PR → invoke /ship or /land-and-deploy
- Save progress → invoke /context-save
- Resume context → invoke /context-restore
