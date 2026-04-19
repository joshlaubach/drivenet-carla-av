# Workflow: Behavior Cloning

## Objective
Train a single condition-aware DriveNet model (`BC_model`) on all expert data
collected in notebook 01, using GPU augmentation and full metadata embeddings.

## Required Inputs
- `data/` directory populated by the DataCollectionAgent (all 6 towns)
- GPU with CUDA support (tested on RTX 5080)

## Tools
| Tool | Call | Purpose |
|---|---|---|
| `np.load` | Load chunk_XXXX.npz files | Load raw expert data |
| `preprocessing.crop_and_resize` | `crop_and_resize(images)` | Crop sky/hood; resize to 100x200 |
| `preprocessing.encode_metadata` | `encode_metadata(data, *_codes)` | String labels -> integer codes |
| `dataset.DrivingDataset` | `DrivingDataset(images, states, actions, meta, indices)` | PyTorch dataset |
| `dataset.GPUAugmenter` | `GPUAugmenter(device, **aug_kwargs)` | On-GPU augmentation |
| `drivenet.DriveNet` | `DriveNet(dropout, state_dim, action_dim, meta_dims)` | Policy network |
| `training.train_model` | `train_model(model, loaders, ...)` | Full training loop with early stopping |

## Model
| Name | Data | Augmentation | Metadata |
|---|---|---|---|
| `BC_model` | All 6 towns, all conditions | GPU augmentation | Weather, town, road_type, tod, traffic |

## Hyperparameters (from `configs/bc.yaml`)
The canonical source for all hyperparameters is `configs/bc.yaml`. If you
change a value, update the YAML file -- agents and notebooks both load from it.
Key values (reference copy):
- batch_size: 128
- lr: 1e-4, weight_decay: 1e-5
- max_epochs: 200, early_stop_patience: 20
- lr_patience: 3, lr_factor: 0.5
- dropout: 0.3
- loss_weights: [1.0, 1.0, 5.0] (steer, throttle, brake -- brake upweighted)
- meta_dims: [6, 6, 3, 3, 3]

## Expected Outputs
```
models/
    BC_model_best.pt
results/
    bc_training_history.json     # train/val loss curves
    bc_test_metrics.json         # per-target MSE on held-out test set
    bc_split_indices.npz         # train/val/test indices for reproducibility
```

## Data Split
- test: 15% of total frames (held out before any training)
- val: 10% of remaining frames
- train: remaining 75%
- Split by random frame index with seed=42 for reproducibility.

## Sequencing
1. Scan `data/` and load all chunk files. Concatenate across all towns.
2. Apply `crop_and_resize` to images; encode metadata strings to integers.
3. Create train/val/test index split (seed=42). Save indices to `results/bc_split_indices.npz`.
4. Build DrivingDataset with GPUAugmenter for train split; plain datasets for val/test.
5. Create DriveNet(meta_dims=[6,6,3,3,3]); call train_model with weighted MSE criterion.
6. Run evaluate() on the test set; save metrics.
7. Save training history and test metrics.

## Edge Cases
- **GPU OOM during augmentation**: reduce batch_size by half and retry once.
- **Early stopping before epoch 10**: likely a data or learning-rate issue; warn
  but do not abort.
- **Missing chunk files**: log which towns are absent; train on available data
  but note reduced coverage in results.

## Known Limitations
- **No collision recovery data**: The CARLA autopilot avoids collisions, so
  the BC model never sees collision-adjacent states during training. This means
  the learned policy has no recovery behavior. PPO fine-tuning partially
  addresses this through the collision penalty in the reward function.
