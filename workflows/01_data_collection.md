# Workflow: Expert Data Collection

## Objective
Collect expert driving demonstrations across 324 conditions for a single town.
One agent run covers all conditions for one town; the full dataset requires six
runs (one per town, with CARLA restarted between runs).

Target: 300 frames per condition x 54 conditions per town x 6 towns = 97,200 frames.
All collection parameters are defined in `configs/collection.yaml`.

## Hardware Constraint
Map switching at runtime crashes the RTX 5080 (Blackwell/UE4 null-pointer in
TaskGraphThreadHP). Launch CARLA with the target town and collect all 54
conditions before restarting for the next town:

```
CarlaUE4-Win64-Shipping.exe -dx12 /Game/Carla/Maps/<Town>
```

## Condition Grid (per town)
See `configs/collection.yaml` for weather presets, TOD sun angles, and traffic
NPC counts. Summary:
- **Weather** (6): ClearNoon, CloudyNoon, WetNoon, HardRainNoon, ClearSunset, ClearNight
- **Time of day** (3): day, sunset, night (sun angles in config)
- **Traffic density** (3): low, medium, high (NPC counts in config)

Total per town: 6 x 3 x 3 = 54 conditions.

## Required Inputs
- CARLA 0.9.16 server running at localhost:2000, launched with target town
- `town` argument specifying which of the 6 towns to collect
- Output directory: `data/` (default)

## Tools
| Tool | Call | Purpose |
|---|---|---|
| `CarlaEnv` | `CarlaEnv(host, port, town, image_width, image_height)` | Spawn ego vehicle and sensors (resolution per `configs/collection.yaml`) |
| `make_weather` | `carla_utils.make_weather(env, preset_name)` | Apply base weather config |
| `vehicle.set_autopilot` | `env.vehicle.set_autopilot(True, tm_port)` | Enable CARLA autopilot as expert |
| `vehicle.get_control` | `env.vehicle.get_control()` | Read expert action after each tick |
| `world.tick` | `env.world.tick()` | Advance simulation one step |
| `np.savez_compressed` | Save chunk_XXXX.npz | Persist frames to disk |

## Expected Outputs
```
data/<Town>/
    chunk_0000.npz   # chunk size per configs/collection.yaml
    chunk_0001.npz
    ...
    collision_log.npz
```

Each `.npz` file contains:
- `images`: (N, 600, 800, 3) uint8 -- raw camera frames
- `states`: (N, 2) float32 -- [speed_kmh, heading_degrees]
- `actions`: (N, 3) float32 -- [steer, throttle, brake]
- `locations`: (N, 2) float32 -- [x, y] in world coordinates
- `tl_states`: (N,) uint8 -- traffic light state (0=Red,1=Yellow,2=Green,3=Off)
- `speed_limits`: (N,) float32 -- posted speed limit in km/h
- `weather_preset`: (N,) str -- base weather name
- `town`: (N,) str -- town name
- `road_type`: (N,) str -- highway / rural / urban
- `time_of_day`: (N,) str -- day / sunset / night
- `traffic_density`: (N,) str -- low / medium / high

## Sequencing
1. Verify CARLA reachability; raise if not reachable.
2. Open `CarlaEnv` for the target town (800x600 camera).
3. For each of the 54 conditions:
   a. Apply base weather config (cloudiness, precipitation, wetness).
   b. Override sun_altitude_angle for the time-of-day tier.
   c. Spawn NPC vehicles at traffic density count.
   d. Call `env.reset()` to spawn ego vehicle and sensors.
   e. Enable autopilot on ego via traffic manager.
   f. Collect 300 frames: tick -> read control -> read obs -> save.
   g. On collision: log it, reset env, re-enable autopilot, continue.
   h. Destroy NPC vehicles before next condition.
4. Flush remaining frames to the current chunk file.
5. Save `collision_log.npz` with condition labels and collision counts.
6. Call `env.close()`.

## Edge Cases
- **CARLA unreachable**: fail immediately with clear message before starting collection.
- **Collision during collection**: reset ego, re-enable autopilot, continue
  accumulating until 300 clean frames for the condition are saved.
- **Spawn failure**: retry up to 3 times on different spawn points; skip condition
  and log if all attempts fail.
- **Camera timeout**: `CarlaEnv._retrieve_camera_data()` raises `RuntimeError`
  after 2 s; treat as a soft reset (same as collision).
- **NPC spawn failures**: log count but continue; partial NPC traffic is acceptable.
