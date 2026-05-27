# Scenario Coverage

DriveNet collects expert driving data and evaluates trained policies across a
structured grid of **324 ODD conditions** derived from four scenario dimensions.
This document describes the coverage design, rationale, known gaps, and the
process for identifying high-priority re-test targets.

---

## ODD Dimensions

| Dimension | Values | Count |
|-----------|--------|------:|
| **Weather** | ClearNoon, CloudyNoon, WetNoon, HardRainNoon, ClearSunset, ClearNight | 6 |
| **Town** | Town01, Town02, Town03, Town04, Town05, Town10HD | 6 |
| **Time of Day** | Day (sun alt $75^\circ$), Sunset (sun alt $15^\circ$), Night (sun alt $-90^\circ$) | 3 |
| **Traffic Density** | Low (10 NPC vehicles), Medium (30 NPC vehicles), High (60 NPC vehicles) | 3 |
| **Total** | $6 \times 6 \times 3 \times 3$ | **324** |

Each condition is sampled for **300 frames at 20 FPS** (15 s of driving),
yielding approximately **97,200 total frames** across the full collection run.
Per-town coverage is 54 conditions ($6$ weather $\times$ $3$ ToD $\times$ $3$ traffic); town is
held orthogonal so BC training sees all towns while PPO and evaluation use held-out
town subsets.

---

## Collection Status: Run 2026-05-26

The three-suite collection run completed on **2026-05-26** after **10 hours 22 minutes**
(13:20:44 to 23:43:12 PDT). **12 of 18 suite/town pairs** reached the $\geq 7$ chunk
target ($\geq 16{,}200$ frames each). All 6 failures are attributable to known CARLA +
RTX 5080 Blackwell hardware/map incompatibilities. No data loss occurred on any
successful pair.

### Per-pair results

| Suite | Town | Chunks | Result | Note |
|-------|------|-------:|--------|------|
| single_cam | Town01 | 7 | complete | |
| single_cam | Town02 | 1 | incomplete | NavMesh race, stalled at condition [2/54] |
| single_cam | Town03 | 0 | incomplete | UE4 spawn/routing failure, all 3 retries |
| single_cam | Town04 | 7 | complete | |
| single_cam | Town05 | 7 | complete | |
| single_cam | Town10HD | 7 | complete | |
| multi_cam | Town01 | 7 | complete | |
| multi_cam | Town02 | 0 | incomplete | STATUS_STACK_BUFFER_OVERRUN (0xC0000409) |
| multi_cam | Town03 | 0 | incomplete | UE4 spawn/routing failure, all 3 retries |
| multi_cam | Town04 | 7 | complete | |
| multi_cam | Town05 | 0 | incomplete | CARLA port 2000 never opened, all 3 retries |
| multi_cam | Town10HD | 7 | complete | |
| lidar | Town01 | 7 | complete | |
| lidar | Town02 | 7 | complete | immune to NavMesh race; BEV sensor has no Boost.Asio pressure |
| lidar | Town03 | 0 | incomplete | UE4 "Low level fatal error" on map load |
| lidar | Town04 | 7 | complete | |
| lidar | Town05 | 7 | complete | |
| lidar | Town10HD | 7 | complete | |

### Failure root causes

**Town02 (single_cam, multi_cam).** CARLA's `Town02.bin` NavMesh cache loads
asynchronously mid-collection at approximately condition [2/54]. For `single_cam`
this stalls sensor frame delivery for 10+ minutes until the subprocess is killed.
For `multi_cam` the three simultaneous camera callbacks create enough Boost.Asio
thread pressure that the NavMesh load triggers a Windows Fast-Fail security
exception (`STATUS_STACK_BUFFER_OVERRUN`, exit code `0xC0000409`). The `lidar`
suite is immune because a single BEV sensor does not produce the same callback
thread load.

**Town03 (all suites).** The map's tunnel geometry, elevated ramps, and
roundabouts cause CARLA's route planner to fail actor spawn and routing setup.
All three suites exit in under 2 minutes with 0 chunks across all retries.
This is a CARLA 0.9.16 map geometry limitation on this hardware configuration,
not a sensor-suite issue.

**multi_cam/Town05.** CARLA failed to open port 2000 on all three retry attempts,
likely a stale socket lock from the prior Town04 teardown. The 45 s inter-retry
wait was insufficient to clear it. `single_cam/Town05` and `lidar/Town05` both
succeeded on the same machine, confirming this was a transient process state
issue rather than a map or sensor-suite incompatibility.

### Usable training data

| Suite | Complete towns | Approx. frames |
|-------|---------------|---------------:|
| single_cam | Town01, Town04, Town05, Town10HD | $\approx 67{,}200$ |
| multi_cam | Town01, Town04, Town10HD | $\approx 50{,}400$ |
| lidar | Town01, Town02, Town04, Town05, Town10HD | $\approx 84{,}000$ |

All successful pairs exceed the 7-chunk, $\geq 16{,}200$ frame threshold. The `lidar`
suite has the broadest town coverage (5/6); `multi_cam` the narrowest (3/6).
Town02 and Town03 are excluded from single_cam and multi_cam training data.

---

## Coverage Rationale

These dimensions were chosen to achieve directed testing across the key axes of
variability that affect perception and control safety in an urban autonomous
driving ODD:

**Weather.** Precipitation and low-visibility presets (WetNoon, HardRainNoon,
ClearNight) stress camera-based perception via lens-obscuring artifacts, surface
reflection, and reduced contrast. Including the full spectrum from clear to hard
rain allows the causal analysis to isolate the marginal effect of precipitation on
collision rate, controlling for other conditions via propensity score matching.

**Town.** CARLA maps vary substantially in road geometry, intersection density,
lane width, and kerb design. Using all six available towns (suburban grid, dense
urban, motorway-style, roundabout-heavy) covers a wider range of road topologies
than a single-map study would allow, and prevents the policy from overfitting to
one map's autopilot trajectories.

**Time of Day.** Sun altitude directly affects headlight requirement and glare
exposure. Night driving (sun alt $-90^\circ$) removes ambient illumination entirely,
making lane markings and obstacles dependent on the vehicle's own headlights.
Sunset ($15^\circ$) introduces low-angle glare, a documented edge case for front-facing
camera perception.

**Traffic Density.** NPC vehicle count controls the frequency of merge events,
cut-ins, and following-distance decisions. High-density runs (60 NPCs) stress
the expert autopilot and, by extension, the cloned policy in scenarios where
reactive braking and lane discipline are required.

---

## Known Coverage Gaps

The following scenario types are **not** covered by the current 324-condition grid.
These represent honest limitations relevant to any safety validation claim:

- **Pedestrian-heavy scenarios.** No pedestrian NPCs are spawned. Scenarios
  involving jaywalkers, crosswalk interactions, or school-zone densities are
  entirely absent.
- **Construction zones.** No temporary lane closures, cones, or reduced-speed
  work zones are present in any CARLA map used here.
- **Adversarial / OOT weather.** Coverage is limited to CARLA's six built-in
  weather presets. Fog, hail, snow, and sandstorm conditions are not represented.
  Out-of-distribution weather that differs structurally from training presets may
  cause silent policy failure.
- **Sensor noise injection.** Camera images are rendered without added Gaussian
  noise, compression artifacts, or occlusion events (e.g., mud on lens). The
  LiDAR suite does not model beam dropout or intensity calibration drift.
- **Dynamic object variety.** NPC vehicles are drawn from CARLA's default
  vehicle blueprint library. Unusual vehicle geometries (bicycles, motorcycles,
  large trucks, emergency vehicles with active lights) are not systematically
  included.
- **Unprotected left turns / roundabouts.** CARLA autopilot has documented
  weaknesses in these manoeuvre types. The BC policy inherits this gap.
- **GPS / localisation noise.** The agent uses ground-truth CARLA waypoints for
  route following. No localisation uncertainty is modelled.
- **Multi-agent interaction.** All NPCs are controlled by CARLA's built-in
  traffic manager. Scenarios requiring prediction of aggressive or irrational
  driver behaviour are not covered.

---

## Scenario Failure Priority

After each evaluation run, the pipeline produces `results/priority_retest_scenarios.csv`
(generated by Notebook 06). The ranking methodology:

1. **Group by condition.** Episodes are aggregated across the three evaluation
   towns (Town01, Town03, Town05) for each (weather, time_of_day, traffic_density)
   combination.

2. **Score by collision rate.** Conditions are ranked in descending order of
   mean collision rate across all evaluated model variants (BC, PPO chill,
   standard, hurry). This identifies the scenario combinations under which *all*
   policies perform poorly, pointing to systematic environment-driven risk rather
   than a single policy failure mode.

3. **Secondary sort by route completion.** Ties in collision rate are broken by
   ascending route completion, surfacing conditions where episodes terminate early
   even without a recorded collision (timeout, stuck vehicle).

4. **Priority re-test queue.** The top-ranked conditions are the first candidates
   for deeper investigation: increasing episode count from 10 to 50, adding
   adversarial weather variants, or validating a retrained model on the same grid
   to detect regressions.

This process mirrors Waymo-style directed testing: the 324-condition baseline grid
is not a pass/fail gate but a diagnostic instrument. Its highest-failure cells
define where simulation resources should be concentrated next.
