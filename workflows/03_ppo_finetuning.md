# Workflow: PPO Fine-Tuning

## Objective
Initialize from a behavior-cloning checkpoint and fine-tune via Proximal Policy
Optimization through live CARLA interaction. Run once per town (hardware
constraint: no runtime map switching on RTX 5080 Blackwell).

## Hardware Constraint
Same as data collection: CARLA must be launched with the target town. The
curriculum switches weather presets only — not towns — during a single run.

## Required Inputs
- BC checkpoint: `models/BC_model_best.pt` (or specify alternative)
- CARLA server running at localhost:2000 with target town loaded
- `town` argument

## Tools
| Tool | Call | Purpose |
|---|---|---|
| `ppo.ActorCritic` | `ActorCritic(bc_state_dict, dropout, state_dim, action_dim)` | Actor-critic initialized from BC |
| `ppo.RolloutBuffer` | `RolloutBuffer(n_steps, obs_img_shape, obs_state_shape, action_dim)` | On-policy experience buffer |
| `ppo.ppo_update` | `ppo_update(model, optimizer, buffer, ...)` | PPO gradient update step |
| `CarlaEnv` | `CarlaEnv(host, port, town, image_width=400, image_height=300)` | Live environment |
| `preprocessing.preprocess_obs` | `preprocess_obs(obs)` | Observation → model tensors |

## Curriculum
- **Phase 1** (0–50,000 steps): ClearNoon only — easy weather for policy warm-up.
- **Phase 2** (50,000–200,000 steps): Random weather from the full 6-preset pool
  sampled per episode reset.

Weather is changed by calling `make_weather(env, preset)` at each episode reset.
The environment is not closed between weather changes.

## Reward Function (from `configs/ppo.yaml`)
Base reward from CarlaEnv:
```
r = speed_kmh / 40.0            # progress reward (always active)
  - 200.0   (on collision)       # collision penalty -> episode terminates
  - 10.0    (on lane invasion)   # lane penalty (non-terminal)
```

Style reward shaping (applied on top of base reward):
```
r += speed_bonus * w_speed
r -= jerk_norm * w_jerk
r -= lane_change_event * w_lane
```
where `w_jerk`, `w_speed`, `w_lane` come from the selected style profile.

## Driving Styles
PPO supports three reward profiles that shape driving behaviour without
changing the underlying PPO update math. Style weights are defined in
`configs/ppo.yaml` under `reward_profiles`.

| Style | Jerk Penalty | Speed Bonus | Lane Change Penalty | Effect |
|-------|-------------|-------------|---------------------|--------|
| Chill | 2.0 | 0.5 | 2.0 | Smooth, conservative |
| Standard | 1.0 | 1.0 | 1.0 | Balanced default |
| Hurry | 0.5 | 2.0 | 0.5 | Aggressive, fast |

Jerk is computed as the second derivative of speed (rate of acceleration change).
Lane change events are detected via CARLA waypoint lane ID transitions.

## Hyperparameters (from `configs/ppo.yaml`)
The canonical source is `configs/ppo.yaml`. Reference copy:
- lr: 3e-5
- clip_eps: 0.2
- entropy_coef: 0.01
- value_loss_coef: 0.5
- n_steps: 512 (rollout length before each update)
- batch_size: 64
- n_epochs_ppo: 4
- gamma: 0.99, gae_lambda: 0.95
- total_timesteps: 200,000
- max_grad_norm: 0.5

## Expected Outputs
```
models/
    ppo_<town>_<style>_best.pt           # checkpoint at best mean episode reward
results/
    ppo_<town>_<style>_training_history.json  # per-update losses
    ppo_config.json                       # config snapshot
```

## Sequencing
1. Load BC checkpoint; build ActorCritic; load weights (strict=False for meta keys).
2. Open CarlaEnv (400×300) for the target town. Apply ClearNoon weather.
3. Reset environment; initialise RolloutBuffer(n_steps=512).
4. Collect rollout:
   a. preprocess_obs → img_t, state_t
   b. model.get_action_and_value → action, log_prob, entropy, value
   c. env.step(action.cpu().numpy()) → next_obs, reward, terminated, _, info
   d. buffer.add(img_t, state_t, action, log_prob, reward, value, terminated)
   e. On episode end (terminated or truncated): reset env, sample new weather
      if in Phase 2.
5. When buffer is full: compute_gae(last_value, last_done); call ppo_update.
6. Log mean reward and losses per update.
7. Save checkpoint whenever mean episode reward improves.
8. After total_timesteps: save final checkpoint and training history.

## Edge Cases
- **CARLA crash mid-training**: if env.world.tick() raises, attempt one full
  env.close() + env reset cycle before aborting.
- **Reward collapse after curriculum switch**: if mean reward drops >50% within
  10,000 steps of switching, revert to Phase 1 weather for 5,000 steps.
- **NaN gradients**: clip_grad_norm handles explosion; if loss is NaN skip the
  update batch and log a warning.
- **Episode timeout at 2000 steps**: If most episodes time out rather than
  reaching a destination or colliding, the reward signal is dominated by the
  timeout boundary. The no-progress termination (150 steps of <0.10m movement)
  mitigates this by terminating stalled episodes early, but the fraction of
  timeout-terminated episodes should be monitored in training logs.
