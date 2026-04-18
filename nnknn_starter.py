from __future__ import annotations

import copy
import os
import sys
from pathlib import Path


NNKNN_ENV_VAR = "NNKNN_REPO_DIR"


def locate_nnknn_repo() -> Path:
    """Find a local checkout of https://github.com/Heuzi/NN-kNN."""
    here = Path(__file__).resolve().parent
    candidates = []

    env_path = os.environ.get(NNKNN_ENV_VAR)
    if env_path:
        candidates.append(Path(env_path).expanduser())

    candidates.extend(
        [
            here / "NN-kNN",
            here.parent / "NN-kNN",
            here / "nnknn",
            here.parent / "nnknn",
        ]
    )

    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "model" / "nnknn_model.py").exists():
            return candidate

    raise FileNotFoundError(
        "Could not find a local NN-kNN checkout. Clone "
        "https://github.com/Heuzi/NN-kNN next to this project, place it in "
        "'./NN-kNN', or set the NNKNN_REPO_DIR environment variable."
    )


NNKNN_REPO = locate_nnknn_repo()
if str(NNKNN_REPO) not in sys.path:
    sys.path.insert(0, str(NNKNN_REPO))

from model.cls_model import NN_k_NN
from model.nnknn_model import NN_KNN_Model, cross_validate, default_args, train_with_given_split
from model.reg_model import NN_k_NN_regression


def build_config(task_type: str = "classification") -> dict:
    """Start from the upstream defaults and override the task mode."""
    cfg = copy.deepcopy(default_args)
    cfg["task_type"] = task_type
    return cfg


def main() -> None:
    cfg = build_config("classification")
    print(f"Imported NN-kNN from: {NNKNN_REPO}")
    print("Ready symbols:")
    print(f"  NN_KNN_Model: {NN_KNN_Model.__name__}")
    print(f"  NN_k_NN: {NN_k_NN.__name__}")
    print(f"  NN_k_NN_regression: {NN_k_NN_regression.__name__}")
    print(f"  default task_type: {cfg['task_type']}")
    print("Use train_with_given_split(...) or cross_validate(...) once your tensors are ready.")


if __name__ == "__main__":
    main()
