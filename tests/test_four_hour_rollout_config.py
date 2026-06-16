from src.config import load_configs


def test_main_protocol_is_four_hour_recursive_rollout():
    data_cfg, model_cfg, _ = load_configs(".")
    assert data_cfg["window"]["forecast_steps"] == 16
    assert model_cfg["model"]["conditioning_path_mode"] == "mean"
    assert model_cfg["model"]["event_inference"] == "mc_path"
    assert model_cfg["model"]["use_mc_samples_for_intervals"] is True
