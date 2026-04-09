# CLAUDE.md — DriveNet Project Instructions

## Hardware
- GPU: RTX 5080 (Blackwell architecture, UE4 4.26)
- OS: Windows 11, Python 3.11, CUDA 12.8
- 32 GB system RAM

## Critical CARLA Constraints

### RHI flag
`scripts/launch_carla.bat` uses `-dx12`. Past debugging established that
`-dx11` can deadlock camera sensors permanently on this GPU. Do not mix RHI
flags across scripts.

### No runtime map switching
`CarlaEnv.__init__()` raises `RuntimeError` if the requested town doesn't match
the currently loaded map. This is intentional — `client.load_world()` triggers a
Vulkan null-pointer crash (address 0x98, TaskGraphThreadHP) on Blackwell.

**Each notebook run and each agent run targets one town. Restart CARLA between
towns:**
```
CarlaUE4-Win64-Shipping.exe -dx12 /Game/Carla/Maps/Town03
```

## Checkpoint Names
| File | Produced by |
|---|---|
| `models/BC_model_best.pt` | Notebook 02 / `BehaviorCloningAgent` |
| `models/PPO_model_best.pt` | Notebook 03 |
| `models/ppo_{Town}_best.pt` | `PPOAgent` (per-town) |

`PPOAgent` and `EvaluationAgent` expect `BC_model_best.pt` as the BC input.
If you use the notebook-produced `PPO_model_best.pt` with `EvaluationAgent`,
add it manually to `MODEL_SPECS` in `src/agents/eval_agent.py`.

## WAT Framework
The project uses a three-layer architecture in addition to the notebooks:

- **Workflows** (`workflows/*.md`): plain-English SOPs. Edit these when the plan
  changes. No code.
- **Agents** (`src/agents/*.py`): coordinators that read workflows and sequence
  tool calls. Handle failures and retries. Do not implement ML math.
- **Tools** (`src/*.py`): deterministic modules (`DriveNet`, `train_model`,
  `ppo_update`, `CarlaEnv`). No knowledge of workflows or agents.

The notebooks and agents are parallel paths — both work against the same tools
and produce the same checkpoint/result file formats. You do not need to choose
one over the other.

## Data Format
Chunks saved as `data/{Town}/chunk_XXXX.npz` with keys:
- `images`: (N, 600, 800, 3) uint8
- `states`: (N, 2) float32 — [speed_kmh, heading_degrees]
- `actions`: (N, 3) float32 — [steer, throttle, brake]
- `locations`, `tl_states`, `speed_limits`: navigation metadata
- `weather_preset`, `town`, `road_type`, `time_of_day`, `traffic_density`: string labels

Preprocessing crops rows [130:530] and resizes to 100×200 for BC.
PPO uses a different crop: rows [65:265] on 300-px-tall camera images.

## Condition Grid
- 6 towns × 6 weathers × 3 times-of-day × 3 traffic densities = 324 conditions
- 300 frames per condition × 324 = 97,200 frames total
- Per town: 54 conditions (collected in one CARLA session)

## Code Conventions
- Do not call `client.load_world()` anywhere — it crashes on this hardware.
- `make_weather()` in `src/carla_utils.py` takes `(env, preset_name)`, not a
  `CarlaWeatherParameters` object directly.
- `DrivingDataset` expects pre-cropped/resized images (uint8 numpy arrays).
  Pass raw images through `crop_and_resize()` before constructing the dataset.
- Weighted MSE loss: weights = [1.0, 1.0, 5.0] — brake is upweighted 5× to
  compensate for class imbalance (most frames have brake=0).
