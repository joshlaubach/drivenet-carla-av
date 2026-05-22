# Operational Design Domain Framework

This document describes DriveNet's Operational Design Domain (ODD), maps each
evaluated condition dimension to its safety relevance, summarises the key
causal-analysis findings from Notebook 05, and explains how the evaluation
pipeline supports regression detection across model versions.

---

## ODD Dimension to Safety Relevance Mapping

| ODD Dimension | CARLA Parameterisation | Safety Relevance |
|---|---|---|
| **Weather — Precipitation** | `precipitation` 0–80, `wetness` 0–100, `precipitation_deposits` 0–90 | Reduces tyre grip (braking distance), obscures lane markings, causes water spray that degrades camera contrast |
| **Weather — Visibility** | `cloudiness` 0–90, `fog_density` (not used in current presets) | Reduces object detection range; overcast conditions affect contrast-based lane detection |
| **Time of Day — Night** | `sun_altitude_angle` −90° | Removes ambient illumination; ego relies entirely on headlights; pedestrian and dark-vehicle detection difficulty increases sharply |
| **Time of Day — Sunset** | `sun_altitude_angle` 15° | Low-angle glare in front-facing camera; glare intensity peaks near 10–20° and is a documented AV perception edge case |
| **Traffic Density — High** | 60 NPC vehicles via CARLA traffic manager | Increases merge frequency, cut-in events, and required following-distance adjustments; stresses reactive braking |
| **Traffic Density — Low** | 10 NPC vehicles | Near-free-flow baseline; isolates pure lane-keeping performance from traffic interaction effects |
| **Town Geometry** | 6 maps spanning suburban grid, dense urban, motorway-style, roundabout-heavy | Road topology variation controls for overfitting to a single map's autopilot trajectory distribution |

---

## Key Causal Analysis Findings

Causal analysis (Notebook 05) uses propensity score matching (PSM) to estimate
the average treatment effect (ATE) of each condition dimension on collision rate
and route completion, controlling for the joint distribution of all other
covariates.

**Primary performance drivers identified:**

1. **Weather (precipitation)** — The largest positive ATE on collision rate across
   all model variants.  HardRainNoon shows the highest collision rate delta vs
   ClearNoon (estimated +0.15–0.25 collisions/episode before PSM balancing).
   This effect persists after controlling for traffic density and time of day,
   confirming a direct perception-path mechanism (wet-road lane-marking loss +
   spray).

2. **Traffic density** — Second-largest ATE.  High-density conditions increase
   collision rate significantly across all policies.  The effect is larger for PPO
   variants than for BC, suggesting PPO's faster driving style amplifies collision
   risk in dense traffic.

3. **Time of day** — Night driving degrades route completion (more episode
   timeouts due to stuck or slow behaviour) but has a smaller direct effect on
   collision rate than precipitation or density, likely because the agent slows
   down in low-visibility conditions rather than colliding.

4. **PPO vs. BC (policy type)** — PPO fine-tuning reduces collision rate
   relative to BC baseline across most conditions, with the standard-style variant
   showing the best overall safety–completion trade-off.  Hurry-style PPO improves
   route completion at the cost of higher collision rate in high-density scenarios.

5. **Town geometry** — No statistically significant ATE on collision rate after
   controlling for other dimensions; the BC policy generalises reasonably across
   map topologies given the diversity of training data.

> Note: ATE estimates are population-level averages across matched condition pairs.
> Per-condition breakdowns and Rosenbaum sensitivity bounds are in
> `results/causal_results.json` and Notebook 05.

---

## Regression Detection

The 324-condition evaluation grid serves as a **regression test suite** for model
versions.  The workflow:

1. **Baseline snapshot** — After the initial evaluation run, `results/eval_results.csv`
   records per-episode metrics for every (model, weather, town, time_of_day,
   traffic_density) tuple.  This snapshot is the regression baseline.

2. **Re-evaluation after model update** — When a new model checkpoint is produced
   (e.g., after fine-tuning on additional data or changing the reward function),
   rerun `EvaluationAgent` on the same 324-condition grid.  The pipeline writes
   results to the same CSV schema, allowing direct comparison.

3. **Delta analysis** — Notebook 06 supports loading two result sets and computing
   the per-condition delta in collision rate and route completion.  Conditions
   where the new model is worse by more than one bootstrapped-CI width are flagged
   as regressions.

4. **Prioritised investigation** — Flagged conditions are ranked by regression
   severity (collision-rate increase × episode count) to focus engineer attention.
   This mirrors the Waymo Scenario Operations workflow: regressions in high-risk
   conditions (HardRainNoon + high traffic) receive higher priority than
   regressions in benign conditions (ClearNoon + low traffic).

5. **Automated gate (future work)** — The `pipeline_summary.json` produced by
   `scripts/run_pipeline.py` already records per-suite pass/fail status.  A CI
   step that parses this file and rejects a model with collision rate > threshold
   in any top-10 priority condition would fully automate the regression gate.

This design means the evaluation infrastructure is not single-use: it produces
actionable safety signals on every run and can track policy improvement or
degradation over time across the full ODD.
