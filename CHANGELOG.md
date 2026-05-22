# Changelog

All notable changes to DriveNet are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-03-01

### Added
- PPO fine-tuning agent (`src/agents/ppo_agent.py`) with three driving-style reward
  profiles (chill / standard / hurry) and resume checkpoint logic (`_resume.pt`
  auto-saved every 10 k steps, auto-loaded on restart)
- Autonomous overnight pipeline script (`scripts/run_pipeline.py`): collect → BC →
  PPO × 3 styles → eval → causal across all sensor suites, unattended
- Causal analysis agent (`src/agents/causal_agent.py`) using propensity score
  matching to estimate average treatment effects of rain, night driving, traffic
  density, and policy type; reports bootstrap 95 % CIs and Rosenbaum sensitivity bounds
- Live Tesla/Waymo-style visualization window (`src/visualizer.py`): 5-panel
  1280 × 720 pygame display with front/left/right RGB cameras, semantic BEV
  overlay, and HUD (speed, condition, frames progress)
- Multi-camera + LiDAR sensor suite stubs (`src/sensors_multicam.py`,
  `src/model_multicam.py`) for future retraining on richer sensor configurations
- Notebook 05: causal effect estimation across 5 treatments with forest plots and
  sensitivity analysis
- Notebook 06: scenario failure analysis — ranks the 324 ODD conditions by
  collision rate to generate a prioritised re-test queue

### Changed
- `CarlaEnv` extended to support three pluggable sensor suites: `single_cam`,
  `multi_cam`, and `lidar`
- Evaluation agent benchmarks all three sensor suites in a single run, pooling
  results into `results/eval_results.csv` for downstream PSM analysis
- CI workflow split: `carla` package excluded from `requirements-ci.txt`; CARLA
  connection tests skipped automatically on Linux runners

### Fixed
- Stale synchronous-mode guard added to NB01 collection cell; prevents
  `0xC0000409` STATUS_STACK_BUFFER_OVERRUN crash on RTX 5080 Blackwell when
  CARLA is restarted between towns without a clean teardown
- libcarla Boost.Asio thread leak between multi-town collection runs resolved
  via subprocess-per-town isolation (`scripts/collect_one_town.py`); the OS
  reaps leaked threads on subprocess exit, eliminating the `load_world()` hang

## [0.2.0] - 2026-01-15

### Added
- Behavior cloning training pipeline with condition-aware DriveNet CNN policy
  (`src/drivenet.py`); optional learned metadata embeddings (weather, town,
  road type, time of day, traffic) add 12 dimensions to the MLP head
- GPU augmentation pipeline in `src/dataset.py` (ColorJitter, random horizontal
  flip, Gaussian noise) applied on-device to avoid CPU bottleneck
- Multi-scenario evaluation agent (`src/agents/eval_agent.py`) with per-condition
  episode rollouts, bootstrapped 95 % confidence intervals (1 000 resamples), and
  Mann-Whitney U significance testing with Benjamini-Hochberg FDR correction
- Early-stopping and LR-scheduling in `src/training.py` (patience configurable
  via `configs/bc.yaml`)
- Notebook 02: behaviour cloning training and validation-loss curves
- Notebook 04: multi-scenario evaluation with per-condition breakdown and summary
  statistics

### Changed
- Config-driven design: all hyperparameters externalised to `configs/bc.yaml`
  and `configs/eval.yaml`; agents and notebooks share the same YAML source
- DriveNet architecture updated with optional metadata embedding head; base
  convolutional backbone unchanged for reproducibility

## [0.1.0] - 2025-11-01

### Added
- Expert data collection across 324 ODD conditions (6 weather presets ×
  6 towns × 3 times of day × 3 traffic densities), yielding ≈ 97 200 frames
  at 20 FPS
- CARLA 0.9.16 Gymnasium environment wrapper (`src/carla_env.py`) with
  synchronous mode, 20 Hz tick rate, and try/except guards on all sensor
  callbacks to prevent Boost.Asio thread aborts
- `DataCollectionAgent` (`src/agents/collection_agent.py`) with PID follow-car
  (8 m gap), 54-condition per-town grid, and chunked NPZ storage
- Subprocess-per-town isolation pattern for stable multi-town collection on
  Windows with RTX 5080; each town runs in a fresh Python interpreter
- Config-driven collection parameters (`configs/collection.yaml`)
- Project scaffold: `src/`, `configs/`, `notebooks/`, `tests/`, `workflows/`,
  `scripts/` directories with all agent, training, and utility modules
- Notebook 01: interactive data collection with per-town progress reporting

[Unreleased]: https://github.com/joshlaubach/drivenet-carla-av/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/joshlaubach/drivenet-carla-av/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/joshlaubach/drivenet-carla-av/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/joshlaubach/drivenet-carla-av/releases/tag/v0.1.0
