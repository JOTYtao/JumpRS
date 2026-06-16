from pathlib import Path
import yaml


class ConfigError(ValueError):
    pass


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_configs(root="."):
    root = Path(root)
    data = load_yaml(root / "config" / "data.yaml")
    model = load_yaml(root / "config" / "model_jumprs.yaml")
    train = load_yaml(root / "config" / "train.yaml")
    if data["dataset"].get("use_synthetic_data") is not False:
        raise ConfigError("dataset.use_synthetic_data must be false. Synthetic PV power is prohibited.")
    return data, model, train


def resolve_project_path(root, path):
    p = Path(path)
    if p.is_absolute():
        return p
    return Path(root) / p
