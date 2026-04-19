# Workflow: Multi-Scenario Evaluation

## Objective
Benchmark all trained models across a standardized set of scenarios and collect
metrics needed for statistical comparison and causal analysis.

## Required Inputs
- Model checkpoints in `models/` (BC_model_best.pt + per-town PPO)
- CARLA server running with one of the evaluation towns loaded
- `town` argument (evaluation must be re-run per town due to hardware constraint)

## Models Under Test
All evaluation parameters are defined in `configs/eval.yaml`.

| Checkpoint | Type | Training |
|---|---|---|
| `BC_model_best.pt` | BC | All data + augmentation + meta |
| `ppo_<Town>_chill_best.pt` | PPO | BC init + PPO chill style |
| `ppo_<Town>_standard_best.pt` | PPO | BC init + PPO standard style |
| `ppo_<Town>_hurry_best.pt` | PPO | BC init + PPO hurry style |

## Evaluation Grid (per town)
- **Weather** (3): ClearNoon, HardRainNoon, ClearNight
- **Episodes per condition** (3): independent resets with different spawn points
- Total per town: 3 weathers x 3 episodes x 4 models = 36 episodes
- Total across 3 evaluation towns (Town01, Town03, Town05): 108 episodes

## Tools
| Tool | Call | Purpose |
|---|---|---|
| `torch.load` | Load .pt checkpoint | Restore model weights |
| `CarlaEnv` | `CarlaEnv(host, port, town, image_width=400, image_height=300)` | Live eval environment |
| `preprocessing.preprocess_obs` | `preprocess_obs(obs)` | Obs -> model input |
| `make_weather` | `carla_utils.make_weather(env, preset)` | Set evaluation weather |

## Metrics (per episode)
- **route_completion**: fraction of planned route arc-distance completed (0-1)
- **collision_count**: number of collisions during episode
- **collision_rate**: collisions per 100m driven
- **lane_keeping_frac**: fraction of steps without lane invasion
- **avg_speed_kmh**: mean speed in km/h over the episode
- **distance_m**: total distance driven in metres
- **survived**: True if episode ended without collision
- **total_steps**: number of steps before termination (max 2000)

## Expected Outputs
```
results/
    eval_results.json    # list of episode records with all metrics + metadata
    eval_summary.csv     # aggregated mean +/- std per model x condition
```

Each episode record:
```json
{
  "model": "bc_combined",
  "model_type": "bc",
  "driving_style": "n/a",
  "town": "Town03",
  "weather": "ClearNoon",
  "episode": 0,
  "route_completion": 0.87,
  "collision_count": 0,
  "collision_rate": 0.0,
  "lane_keeping_frac": 0.94,
  "avg_speed_kmh": 28.4,
  "distance_m": 412.3,
  "survived": true,
  "total_steps": 1500
}
```

## Sequencing
1. Load all available model checkpoints into memory.
2. Open CarlaEnv for the target town.
3. For each model x weather x episode:
   a. Apply weather preset.
   b. Reset environment.
   c. Run episode for max 2000 steps using model's greedy action
      (`action = model(img, state, meta=None).detach().numpy()`).
   d. Record per-step info (speed, collision, lane_invasion).
   e. Compute episode-level metrics; append to results list.
4. After all 9 episodes per model per town: close env, move to next town.
5. Write `eval_results.json` and aggregated `eval_summary.csv`.

## Statistical Tests
After all episodes are collected, run permutation tests (10,000 permutations)
on route_completion between each BC/PPO pair and between model variants.
Append p-values to `eval_summary.csv`.

## Edge Cases
- **Model file missing**: skip that model, log warning, continue with others.
- **Episode collision on step 1**: likely a bad spawn point; retry with a
  different spawn (up to 3 retries before logging as failed episode).
- **CARLA freeze**: if world.tick() blocks >5s, restart the env and rerun the
  current episode.
