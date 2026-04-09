"""Tests for YAML config loading and validation."""

import pytest

from src.config import load_config, require_keys


# -- All configs load successfully --------------------------------------------

CONFIG_NAMES = ["collection", "bc", "ppo", "eval", "causal"]


@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_config_loads(name: str) -> None:
    cfg = load_config(name)
    assert isinstance(cfg, dict), f"{name}.yaml did not return a dict"
    assert len(cfg) > 0, f"{name}.yaml is empty"


# -- Required keys present ---------------------------------------------------

def test_collection_required_keys() -> None:
    cfg = load_config("collection")
    require_keys(
        cfg,
        ["weather_presets", "weather_params", "tod_sun_angles",
         "traffic_vehicle_counts", "frames_per_condition", "chunk_size",
         "image_width", "image_height", "seed"],
        "collection",
    )


def test_bc_required_keys() -> None:
    cfg = load_config("bc")
    require_keys(
        cfg,
        ["seed", "batch_size", "lr", "weight_decay", "max_epochs",
         "early_stop_patience", "lr_patience", "lr_factor", "dropout",
         "loss_weights", "test_fraction", "val_fraction", "meta_dims",
         "model_name", "weather_codes", "town_codes", "road_type_codes",
         "tod_codes", "traffic_codes"],
        "bc",
    )


def test_ppo_required_keys() -> None:
    cfg = load_config("ppo")
    require_keys(
        cfg,
        ["seed", "dropout", "cam_w", "cam_h", "lr", "clip_eps",
         "entropy_coef", "value_loss_coef", "n_steps", "batch_size",
         "n_epochs_ppo", "gamma", "gae_lambda", "total_timesteps",
         "curriculum_switch_step", "max_grad_norm", "crop_top",
         "crop_bottom", "weather_phase1", "weather_phase2",
         "reward_profiles"],
        "ppo",
    )


def test_eval_required_keys() -> None:
    cfg = load_config("eval")
    require_keys(
        cfg,
        ["eval_weathers", "episodes_per_condition",
         "max_steps_per_episode", "grp_sampling", "model_specs",
         "meta_dims", "crop"],
        "eval",
    )


def test_causal_required_keys() -> None:
    cfg = load_config("causal")
    require_keys(
        cfg,
        ["n_bootstrap", "min_treated", "random_seed", "treatments"],
        "causal",
    )


# -- require_keys raises on missing keys --------------------------------------

def test_require_keys_raises_on_missing() -> None:
    cfg = {"a": 1, "b": 2}
    with pytest.raises(KeyError, match="missing required keys"):
        require_keys(cfg, ["a", "b", "c", "d"], "test")


def test_require_keys_passes_when_all_present() -> None:
    cfg = {"a": 1, "b": 2, "c": 3}
    require_keys(cfg, ["a", "b", "c"], "test")  # should not raise


# -- Config file not found ----------------------------------------------------

def test_load_nonexistent_config_raises() -> None:
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_config("nonexistent_config_xyz")


# -- PPO reward profiles structure -------------------------------------------

def test_ppo_reward_profiles() -> None:
    cfg = load_config("ppo")
    profiles = cfg["reward_profiles"]
    for style in ["chill", "standard", "hurry"]:
        assert style in profiles, f"Missing style '{style}' in reward_profiles"
        for key in ["jerk_penalty", "speed_bonus", "lane_change_penalty"]:
            assert key in profiles[style], f"Missing '{key}' in {style} profile"
            assert isinstance(profiles[style][key], (int, float))
