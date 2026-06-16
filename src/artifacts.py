from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import torch


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower()).strip("_")
    return slug or "model"


def artifact_dir(root: Path | str, model_name: str, run_name: str = "multisite_current") -> Path:
    return Path(root) / "artifacts" / "models" / run_name / slugify(model_name)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def save_model_artifact(
    model: torch.nn.Module,
    root: Path | str,
    model_name: str,
    run_name: str,
    metadata: dict[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
) -> Path:
    """Save a trained torch model and its run metadata in a standard location."""
    out = artifact_dir(root, model_name, run_name)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": model_name,
            "run_name": run_name,
            "state_dict": model.state_dict(),
            "metadata": _jsonable(metadata or {}),
        },
        out / "model.pt",
    )
    (out / "metadata.json").write_text(
        json.dumps(_jsonable({"model_name": model_name, "run_name": run_name, **(metadata or {})}), indent=2)
        + "\n"
    )
    if history:
        pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)
    return out


def save_spec_artifact(
    root: Path | str,
    model_name: str,
    run_name: str,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save metadata for non-parametric baselines such as persistence."""
    out = artifact_dir(root, model_name, run_name)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metadata.json").write_text(
        json.dumps(_jsonable({"model_name": model_name, "run_name": run_name, **(metadata or {})}), indent=2)
        + "\n"
    )
    return out
