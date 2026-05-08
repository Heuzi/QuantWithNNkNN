from __future__ import annotations

import copy
import gc
import json
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

import numpy as np
import pandas as pd


EPS = 1e-12
BASELINE_MODEL_NAMES = [
    "zero",
    "mean",
    "momentum_heuristic",
    "ridge",
    "elastic_net",
    "lightgbm",
    "xgboost",
    "sklearn_hist_gb",
    "sklearn_mlp",
    "torch_mlp",
]
SEQUENCE_STATIC_MODEL_NAME = "torch_seq_static"
CLASSIFICATION_MODEL_NAMES = [
    "logistic_regression",
    "elastic_net_classifier",
    "lightgbm_classifier",
    "xgboost_classifier",
    "sklearn_mlp_classifier",
    "torch_mlp_classifier",
]
SEQUENCE_STATIC_CLASSIFICATION_MODEL_NAME = "torch_seq_static_classifier"


class Predictor(Protocol):
    def fit(
        self,
        x: Any,
        y: pd.DataFrame,
        *,
        val_x: Any | None = None,
        val_y: pd.DataFrame | None = None,
    ) -> "Predictor":
        ...

    def predict(self, x: Any) -> np.ndarray:
        ...


@dataclass
class Standardizer:
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "Standardizer":
        if x.size == 0:
            width = x.shape[-1] if x.ndim >= 2 else 0
            self.mean_ = np.zeros(width, dtype=np.float64)
            self.scale_ = np.ones(width, dtype=np.float64)
            return self
        valid = np.isfinite(x)
        counts = valid.sum(axis=0)
        sums = np.where(valid, x, 0.0).sum(axis=0)
        self.mean_ = np.divide(sums, counts, out=np.zeros(x.shape[1], dtype=np.float64), where=counts > 0)
        centered = np.where(valid, x - self.mean_, 0.0)
        variances = np.divide(
            (centered**2).sum(axis=0),
            counts,
            out=np.ones(x.shape[1], dtype=np.float64),
            where=counts > 0,
        )
        self.scale_ = np.sqrt(variances)
        self.scale_[self.scale_ < EPS] = 1.0
        return self

    def fit_batches(self, batches: Iterable[np.ndarray], *, width: int) -> "Standardizer":
        counts = np.zeros(width, dtype=np.float64)
        sums = np.zeros(width, dtype=np.float64)
        sumsq = np.zeros(width, dtype=np.float64)
        for batch in batches:
            values = np.asarray(batch, dtype=np.float64)
            if values.size == 0:
                continue
            valid = np.isfinite(values)
            counts += valid.sum(axis=0)
            safe = np.where(valid, values, 0.0)
            sums += safe.sum(axis=0)
            sumsq += (safe * safe).sum(axis=0)
        self.mean_ = np.divide(sums, counts, out=np.zeros(width, dtype=np.float64), where=counts > 0)
        variances = np.divide(sumsq, counts, out=np.ones(width, dtype=np.float64), where=counts > 0) - self.mean_**2
        variances = np.maximum(variances, 0.0)
        self.scale_ = np.sqrt(variances)
        self.scale_[self.scale_ < EPS] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Standardizer is not fit.")
        x = np.nan_to_num(x, nan=self.mean_)
        return (x - self.mean_) / self.scale_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Standardizer is not fit.")
        return x * self.scale_ + self.mean_


def _as_array(frame: pd.DataFrame | np.ndarray) -> np.ndarray:
    if isinstance(frame, pd.DataFrame):
        return frame.astype(float).to_numpy(dtype=np.float64)
    if hasattr(frame, "to_numpy"):
        return frame.to_numpy(dtype=np.float64)
    return np.asarray(frame, dtype=np.float64)


def _as_2d_prediction_array(values: np.ndarray, *, expected_columns: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"Expected 1-D or 2-D prediction array, got shape {arr.shape}.")
    if arr.shape[1] != expected_columns:
        raise ValueError(f"Expected {expected_columns} prediction columns, got {arr.shape[1]}.")
    return arr


def _hstack_static_arrays(static_categorical: dict[str, np.ndarray], columns: Sequence[str]) -> np.ndarray:
    if not columns:
        return np.empty((len(next(iter(static_categorical.values()), [])), 0), dtype=np.int64)
    return np.column_stack([static_categorical[column] for column in columns]).astype(np.int64)


def _torch_cuda_available() -> bool:
    """True only when this Python's PyTorch build can actually execute CUDA kernels."""
    import torch

    return bool(torch.cuda.is_available())


def _default_torch_device() -> str:
    return "cuda" if _torch_cuda_available() else "cpu"


def _gpu_available() -> bool:
    try:
        if _default_torch_device() == "cuda":
            return True
    except Exception:
        pass
    for env_name in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
        visible_devices = os.environ.get(env_name)
        if visible_devices is not None and visible_devices.strip().lower() not in {"", "-1", "none", "void"}:
            return True
    return False


def _xgboost_device(prefer_gpu: bool) -> str:
    requested = os.environ.get("V1_XGBOOST_DEVICE", "auto").strip().lower()
    if requested in {"cpu", "-1", "none", "void"}:
        return "cpu"
    if requested in {"cuda", "gpu"}:
        return "cuda" if _gpu_available() else "cpu"
    return "cuda" if prefer_gpu and _gpu_available() else "cpu"


def _xgboost_nthread(default: int = 1) -> int:
    raw = os.environ.get("V1_XGBOOST_NTHREAD", "").strip()
    if not raw:
        return default
    try:
        return max(int(raw), 1)
    except ValueError:
        return default


def _xgboost_chunk_rows(default: int = 1_048_576) -> int:
    raw = os.environ.get("V1_XGBOOST_CHUNK_ROWS", "").strip()
    if not raw:
        return default
    try:
        return max(int(raw), 1)
    except ValueError:
        return default


def _xgboost_chunk_epochs(default: int = 1) -> int:
    raw = os.environ.get("V1_XGBOOST_CHUNK_EPOCHS", "").strip()
    if not raw:
        return default
    try:
        return max(int(raw), 1)
    except ValueError:
        return default


def _xgboost_training_mode(row_count: int) -> str:
    requested = os.environ.get("V1_XGBOOST_TRAINING_MODE", "auto").strip().lower()
    if requested in {"chunk", "chunks", "chunked"}:
        return "chunked"
    if requested in {"external", "external_memory", "extmem"}:
        return "external"
    try:
        threshold = max(int(os.environ.get("V1_XGBOOST_CHUNK_THRESHOLD_ROWS", "5000000")), 1)
    except ValueError:
        threshold = 5_000_000
    return "chunked" if int(row_count) >= threshold else "external"


def _torch_loader_kwargs(*, device: str, shuffle: bool, batch_size: int) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "pin_memory": device == "cuda",
    }
    workers_raw = os.environ.get("V1_TORCH_NUM_WORKERS", "0").strip()
    try:
        workers = max(int(workers_raw), 0)
    except ValueError:
        workers = 0
    if workers:
        kwargs["num_workers"] = workers
        kwargs["persistent_workers"] = True
        try:
            prefetch_factor = max(int(os.environ.get("V1_TORCH_PREFETCH_FACTOR", "2")), 1)
        except ValueError:
            prefetch_factor = 2
        kwargs["prefetch_factor"] = prefetch_factor
    return kwargs


def _torch_progress_every(total_batches: int) -> int:
    raw = os.environ.get("V1_TORCH_PROGRESS_EVERY_BATCHES", "").strip()
    if raw:
        try:
            return max(int(raw), 1)
        except ValueError:
            pass
    if total_batches <= 100:
        return 1
    return max(min(total_batches // 100, 500), 25)


def _sequence_standardizer_batches(store: Any, cutoff_date: str) -> Iterable[np.ndarray]:
    raw_batch_size = os.environ.get("V1_SEQUENCE_STANDARDIZER_BATCH_ROWS", "524288").strip()
    try:
        batch_size = max(int(raw_batch_size), 1)
    except ValueError:
        batch_size = 524_288
    raw_progress = os.environ.get("V1_SEQUENCE_STANDARDIZER_PROGRESS_BATCHES", "10").strip()
    try:
        progress_every = max(int(raw_progress), 1)
    except ValueError:
        progress_every = 10
    batches = store.iter_rows_through(cutoff_date, batch_size=batch_size)
    rows_seen = 0
    start_time = time.monotonic()
    for batch_index, batch in enumerate(batches, start=1):
        yield batch
        rows_seen += int(len(batch))
        if batch_index == 1 or batch_index % progress_every == 0:
            print(
                json.dumps(
                    {
                        "step": "sequence_standardizer_scan",
                        "feature_set": getattr(store, "feature_set", ""),
                        "cutoff_date": cutoff_date,
                        "batch": batch_index,
                        "batch_rows": int(len(batch)),
                        "rows_seen": rows_seen,
                        "source_rows_total": int(getattr(store, "shape", (0, 0))[0]),
                        "elapsed_seconds": round(time.monotonic() - start_time, 1),
                    }
                ),
                flush=True,
            )


def _tabular_scaler_batches(x: Any, *, batch_size: int = 262_144, phase: str = "fit") -> Iterable[np.ndarray]:
    total_rows = len(x) if hasattr(x, "__len__") else 0
    total_batches = (total_rows + batch_size - 1) // batch_size if total_rows else 0
    progress_every = _torch_progress_every(max(total_batches, 1))
    rows_seen = 0
    start_time = time.monotonic()
    for batch_index, (_, batch) in enumerate(x.iter_numpy_batches(batch_size=batch_size, shuffle=False), start=1):
        yield batch
        rows_seen += int(len(batch))
        if batch_index == 1 or batch_index % progress_every == 0 or batch_index == total_batches:
            print(
                json.dumps(
                    {
                        "step": "torch_mlp_classifier_scaler_scan",
                        "phase": phase,
                        "batch": batch_index,
                        "total_batches": total_batches,
                        "rows_seen": rows_seen,
                        "rows_total": total_rows,
                        "elapsed_seconds": round(time.monotonic() - start_time, 1),
                    }
                ),
                flush=True,
            )


def _sequence_standardizer_cache_path(store: Any, cutoff_date: str) -> Path | None:
    path = getattr(store, "path", None)
    feature_set = str(getattr(store, "feature_set", "") or "sequence")
    if path is None:
        return None
    root = Path(path).parent.parent
    safe_cutoff = str(cutoff_date).replace("/", "-").replace(":", "-")
    return root / "standardizers" / f"{feature_set}__through_{safe_cutoff}.npz"


def _load_sequence_standardizer_cache(scaler: Standardizer, store: Any, cutoff_date: str, *, width: int) -> bool:
    cache_path = _sequence_standardizer_cache_path(store, cutoff_date)
    path = getattr(store, "path", None)
    if cache_path is None or path is None or not cache_path.exists():
        return False
    try:
        payload = np.load(cache_path)
        expected_mtime = int(Path(path).stat().st_mtime_ns)
        if int(payload["width"]) != int(width):
            return False
        if int(payload["data_mtime_ns"]) != expected_mtime:
            return False
        scaler.mean_ = np.asarray(payload["mean"], dtype=np.float64)
        scaler.scale_ = np.asarray(payload["scale"], dtype=np.float64)
    except Exception:
        return False
    print(
        json.dumps(
            {
                "step": "sequence_standardizer_cache_hit",
                "feature_set": getattr(store, "feature_set", ""),
                "cutoff_date": cutoff_date,
                "cache_path": str(cache_path),
            }
        ),
        flush=True,
    )
    return True


def _save_sequence_standardizer_cache(scaler: Standardizer, store: Any, cutoff_date: str, *, width: int) -> None:
    cache_path = _sequence_standardizer_cache_path(store, cutoff_date)
    path = getattr(store, "path", None)
    if cache_path is None or path is None or scaler.mean_ is None or scaler.scale_ is None:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        mean=np.asarray(scaler.mean_, dtype=np.float64),
        scale=np.asarray(scaler.scale_, dtype=np.float64),
        width=np.asarray(int(width), dtype=np.int64),
        data_mtime_ns=np.asarray(int(Path(path).stat().st_mtime_ns), dtype=np.int64),
    )
    print(
        json.dumps(
            {
                "step": "sequence_standardizer_cache_saved",
                "feature_set": getattr(store, "feature_set", ""),
                "cutoff_date": cutoff_date,
                "cache_path": str(cache_path),
            }
        ),
        flush=True,
    )


class _ContiguousGroupBatchSampler:
    def __init__(
        self,
        group_ranges: Sequence[tuple[int, int]],
        *,
        batch_size: int,
        shuffle_groups: bool,
        random_state: int,
    ) -> None:
        self.group_ranges = [(int(start), int(end)) for start, end in group_ranges if int(end) > int(start)]
        self.batch_size = max(int(batch_size), 1)
        self.shuffle_groups = shuffle_groups
        self.random_state = int(random_state)
        self._iteration = 0

    def __iter__(self):
        order = np.arange(len(self.group_ranges), dtype=np.int64)
        if self.shuffle_groups and len(order):
            rng = np.random.default_rng(self.random_state + self._iteration)
            rng.shuffle(order)
        self._iteration += 1
        for group_position in order:
            start, end = self.group_ranges[int(group_position)]
            for batch_start in range(start, end, self.batch_size):
                yield list(range(batch_start, min(batch_start + self.batch_size, end)))

    def __len__(self) -> int:
        total = 0
        for start, end in self.group_ranges:
            total += (end - start + self.batch_size - 1) // self.batch_size
        return total


def _sequence_data_loader(
    dataset: Any,
    *,
    device: str,
    shuffle: bool,
    batch_size: int,
    random_state: int = 0,
):
    import torch

    effective_batch_size = min(max(int(batch_size), 1), max(len(dataset), 1))
    group_ranges = getattr(dataset, "group_ranges", [])
    if shuffle and group_ranges:
        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=_ContiguousGroupBatchSampler(
                group_ranges,
                batch_size=effective_batch_size,
                shuffle_groups=True,
                random_state=random_state,
            ),
            pin_memory=device == "cuda",
        )
    return torch.utils.data.DataLoader(
        dataset,
        **_torch_loader_kwargs(device=device, shuffle=shuffle, batch_size=effective_batch_size),
    )


def _fit_sequence_store_standardizer(
    scaler: Standardizer,
    store: Any,
    cutoff_date: str,
    *,
    width: int,
) -> Standardizer:
    if hasattr(store, "iter_rows_through"):
        if _load_sequence_standardizer_cache(scaler, store, cutoff_date, width=width):
            return scaler
        scaler.fit_batches(_sequence_standardizer_batches(store, cutoff_date), width=width)
        _save_sequence_standardizer_cache(scaler, store, cutoff_date, width=width)
        return scaler
    train_rows = store.fit_rows_through(cutoff_date)
    return scaler.fit(train_rows.astype(np.float64))


def _move_tensor(value, device, *, dtype=None):
    kwargs = {"device": device, "non_blocking": getattr(device, "type", str(device)) == "cuda"}
    if dtype is not None:
        kwargs["dtype"] = dtype
    return value.to(**kwargs)


def _copy_model_to_cpu(model):
    model_copy = copy.deepcopy(model)
    if hasattr(model_copy, "to"):
        model_copy = model_copy.to("cpu")
    return model_copy


class _ConstantProbabilityClassifier:
    def __init__(self, probability: float) -> None:
        self.probability = float(np.clip(probability, 0.0, 1.0))

    def predict_proba(self, x):
        count = len(x)
        positive = np.full(count, self.probability, dtype=np.float64)
        negative = 1.0 - positive
        return np.column_stack([negative, positive])

    def predict(self, x):
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(float)


class ZeroPredictor:
    def fit(
        self,
        x: Any,
        y: pd.DataFrame,
        *,
        val_x: Any | None = None,
        val_y: pd.DataFrame | None = None,
    ) -> "ZeroPredictor":
        self.output_dim_ = y.shape[1]
        return self

    def predict(self, x: Any) -> np.ndarray:
        if isinstance(x, dict):
            count = len(x["metadata"])
        else:
            count = len(x)
        return np.zeros((count, self.output_dim_), dtype=np.float64)


class MeanPredictor:
    def fit(
        self,
        x: Any,
        y: pd.DataFrame,
        *,
        val_x: Any | None = None,
        val_y: pd.DataFrame | None = None,
    ) -> "MeanPredictor":
        self.mean_ = y.astype(float).mean(axis=0).to_numpy(dtype=np.float64)
        return self

    def predict(self, x: Any) -> np.ndarray:
        if isinstance(x, dict):
            count = len(x["metadata"])
        else:
            count = len(x)
        return np.tile(self.mean_, (count, 1))


class MomentumHeuristicPredictor:
    def fit(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame,
        *,
        val_x: Any | None = None,
        val_y: pd.DataFrame | None = None,
    ) -> "MomentumHeuristicPredictor":
        self.target_mean_ = y.astype(float).mean(axis=0).to_numpy(dtype=np.float64)
        self.momentum_cols_ = [
            col
            for col in x.columns
            if col.endswith("momentum_20d__last") or col.endswith("rolling_return_20d__last")
        ]
        if not self.momentum_cols_:
            self.momentum_cols_ = [col for col in x.columns if "momentum_20d" in col or "rolling_return_20d" in col]
        self.scale_ = np.ones(y.shape[1], dtype=np.float64)
        if self.momentum_cols_:
            signal = x[self.momentum_cols_].astype(float).mean(axis=1).fillna(0.0).to_numpy()
            denom = float(np.nanstd(signal)) or 1.0
            y_values = y.astype(float).to_numpy()
            self.scale_ = np.nan_to_num(np.nanstd(y_values, axis=0) / denom, nan=1.0)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if not self.momentum_cols_:
            return np.tile(self.target_mean_, (len(x), 1))
        available = [col for col in self.momentum_cols_ if col in x.columns]
        signal = x[available].astype(float).mean(axis=1).fillna(0.0).to_numpy()[:, None]
        return signal * self.scale_[None, :]


class SklearnRegressor:
    def __init__(self, estimator_name: str) -> None:
        self.estimator_name = estimator_name

    def _make_estimator(self):
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import ElasticNet, Ridge
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        if self.estimator_name == "sklearn_ridge":
            estimator = Ridge(alpha=10.0)
            return make_pipeline(
                SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True),
                StandardScaler(),
                estimator,
            )
        if self.estimator_name == "sklearn_elastic_net":
            estimator = ElasticNet(
                alpha=0.005,
                l1_ratio=0.2,
                max_iter=1000,
                tol=1e-3,
                selection="random",
                random_state=13,
            )
            return make_pipeline(
                SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True),
                StandardScaler(),
                estimator,
            )
        if self.estimator_name == "sklearn_hist_gb":
            estimator = HistGradientBoostingRegressor(
                max_iter=120,
                learning_rate=0.05,
                max_leaf_nodes=31,
                l2_regularization=0.1,
                random_state=19,
            )
            return make_pipeline(
                SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True),
                MultiOutputRegressor(estimator),
            )
        raise ValueError(f"Unknown sklearn estimator: {self.estimator_name}")

    def fit(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame,
        *,
        val_x: Any | None = None,
        val_y: pd.DataFrame | None = None,
    ) -> "SklearnRegressor":
        self.model_ = self._make_estimator()
        self.model_.fit(x, y)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model_.predict(x), dtype=np.float64)


class SklearnMLPPredictor:
    def __init__(
        self,
        *,
        hidden_layer_sizes: tuple[int, ...] = (96, 48),
        learning_rate_init: float = 0.001,
        max_epochs: int = 80,
        patience: int = 8,
        tol: float = 1e-4,
        random_state: int = 17,
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate_init = learning_rate_init
        self.max_epochs = max_epochs
        self.patience = patience
        self.tol = tol
        self.random_state = random_state

    def _build_model(self):
        from sklearn.neural_network import MLPRegressor

        return MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation="relu",
            learning_rate_init=self.learning_rate_init,
            max_iter=1,
            shuffle=True,
            warm_start=True,
            random_state=self.random_state,
        )

    def _prepare(self, x: pd.DataFrame, *, fit: bool) -> np.ndarray:
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        if fit:
            self.imputer_ = SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True)
            self.scaler_ = StandardScaler()
            values = self.imputer_.fit_transform(x)
            return self.scaler_.fit_transform(values)
        values = self.imputer_.transform(x)
        return self.scaler_.transform(values)

    def fit(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame,
        *,
        val_x: pd.DataFrame | None = None,
        val_y: pd.DataFrame | None = None,
    ) -> "SklearnMLPPredictor":
        x_train = self._prepare(x, fit=True)
        y_train = _as_array(y)
        x_val = self._prepare(val_x, fit=False) if val_x is not None else None
        y_val_arr = _as_array(val_y) if val_y is not None else None

        self.model_ = self._build_model()
        best_state: tuple[list[np.ndarray], list[np.ndarray]] | None = None
        best_metric = np.inf
        best_epoch = 0
        epochs_without_improvement = 0
        for epoch in range(self.max_epochs):
            self.model_.partial_fit(x_train, y_train)
            score_x = x_val if x_val is not None else x_train
            score_y = y_val_arr if y_val_arr is not None else y_train
            pred = np.asarray(self.model_.predict(score_x), dtype=np.float64)
            metric = float(np.nanmean((pred - score_y) ** 2))
            if metric + self.tol < best_metric:
                best_metric = metric
                best_epoch = epoch + 1
                best_state = (
                    [coef.copy() for coef in self.model_.coefs_],
                    [intercept.copy() for intercept in self.model_.intercepts_],
                )
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if x_val is not None and epochs_without_improvement >= self.patience:
                break
        if best_state is not None:
            self.model_.coefs_ = [coef.copy() for coef in best_state[0]]
            self.model_.intercepts_ = [intercept.copy() for intercept in best_state[1]]
        self.best_epoch_ = max(best_epoch, 1)
        return self

    def refit_full(self, x: pd.DataFrame, y: pd.DataFrame) -> "SklearnMLPPredictor":
        x_full = self._prepare(x, fit=True)
        y_full = _as_array(y)
        self.model_ = self._build_model()
        for _ in range(max(getattr(self, "best_epoch_", self.max_epochs), 1)):
            self.model_.partial_fit(x_full, y_full)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        x_values = self._prepare(x, fit=False)
        return np.asarray(self.model_.predict(x_values), dtype=np.float64)


class LightGBMRegressor:
    def __init__(
        self,
        *,
        n_estimators: int = 400,
        patience: int = 25,
        random_state: int = 23,
        prefer_gpu: bool = True,
    ) -> None:
        self.n_estimators = n_estimators
        self.patience = patience
        self.random_state = random_state
        self.prefer_gpu = prefer_gpu

    def _preprocess(self, x: pd.DataFrame, *, fit: bool) -> np.ndarray:
        from sklearn.impute import SimpleImputer

        if hasattr(x, "iter_numpy_batches"):
            # Cached tabular matrices are already float32 feature arrays with
            # missing values filled at cache-build time. Loading the requested
            # split from the memmap is still cheaper than rebuilding pandas
            # feature frames for every fold/model.
            if fit:
                self.imputer_ = None
            return np.nan_to_num(x.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if fit:
            self.imputer_ = SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True)
            return self.imputer_.fit_transform(x)
        if getattr(self, "imputer_", None) is None:
            return np.nan_to_num(x.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        return self.imputer_.transform(x)

    def _lazy_matrix(self, x, y_values: np.ndarray | None, *, phase: str):
        import xgboost as xgb

        try:
            batch_size = max(int(os.environ.get("V1_XGBOOST_BATCH_ROWS", "262144")), 1)
        except ValueError:
            batch_size = 262_144
        iterator = _XGBoostTabularDataIter(
            x,
            y_values,
            batch_size=batch_size,
            cache_prefix=_xgboost_cache_prefix(x, phase=phase, random_state=self.random_state),
        ).iterator
        try:
            return xgb.ExtMemQuantileDMatrix(iterator, max_bin=256)
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "step": "xgboost_classifier_extmem_fallback",
                        "phase": phase,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                ),
                flush=True,
            )
            return xgb.QuantileDMatrix(iterator, max_bin=256)

    def _train_lazy_chunks(
        self,
        x,
        y_values: np.ndarray,
        *,
        params: dict[str, object],
        rounds: int,
        val_x=None,
        y_val_arr: np.ndarray | None = None,
        phase: str,
        start_step: str,
        chunk_step: str,
        complete_step: str,
    ):
        import xgboost as xgb

        chunk_rows = _xgboost_chunk_rows()
        epochs = _xgboost_chunk_epochs()
        row_count = len(x)
        chunks_per_epoch = max((row_count + chunk_rows - 1) // chunk_rows, 1)
        total_chunks = chunks_per_epoch * epochs
        rounds_total = max(int(rounds), 1)
        rounds_remaining = rounds_total
        chunks_remaining = total_chunks
        booster = None
        validation = None
        if val_x is not None and y_val_arr is not None:
            x_val = self._preprocess(val_x, fit=False)
            validation = xgb.DMatrix(x_val, label=y_val_arr)
        print(
            json.dumps(
                {
                    "step": start_step,
                    "phase": phase,
                    "training_mode": "chunked",
                    "rows": row_count,
                    "features": x.shape[1],
                    "n_estimators": rounds_total,
                    "device": params.get("device"),
                    "nthread": params.get("nthread"),
                    "chunk_rows": chunk_rows,
                    "chunks_per_epoch": chunks_per_epoch,
                    "epochs": epochs,
                }
            ),
            flush=True,
        )
        rounds_done = 0
        chunks_done = 0
        start = time.monotonic()
        for epoch in range(epochs):
            iterator = x.iter_numpy_batches(
                batch_size=chunk_rows,
                shuffle=True,
                random_state=self.random_state + epoch,
            )
            for local_rows, batch in iterator:
                if rounds_remaining <= 0:
                    break
                chunk_rounds = max(1, int(np.ceil(rounds_remaining / max(chunks_remaining, 1))))
                values = np.nan_to_num(batch.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
                labels = y_values[local_rows]
                dtrain = xgb.DMatrix(values, label=labels)
                evals = [(validation, "validation")] if validation is not None else []
                booster = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=chunk_rounds,
                    xgb_model=booster,
                    evals=evals,
                    verbose_eval=False,
                )
                rounds_done += chunk_rounds
                rounds_remaining -= chunk_rounds
                chunks_done += 1
                chunks_remaining -= 1
                print(
                    json.dumps(
                        {
                            "step": chunk_step,
                            "phase": phase,
                            "epoch": epoch + 1,
                            "epochs": epochs,
                            "chunk": chunks_done,
                            "chunks_total": total_chunks,
                            "rows": int(len(batch)),
                            "rounds_done": rounds_done,
                            "rounds_total": rounds_total,
                            "elapsed_seconds": round(time.monotonic() - start, 1),
                        }
                    ),
                    flush=True,
                )
                del dtrain, values, labels, batch
                gc.collect()
            if rounds_remaining <= 0:
                break
        if booster is None:
            raise RuntimeError("XGBoost chunked training did not receive any training batches.")
        print(
            json.dumps(
                {
                    "step": complete_step,
                    "phase": phase,
                    "training_mode": "chunked",
                    "best_iteration": rounds_done,
                    "device": params.get("device"),
                    "nthread": params.get("nthread"),
                }
            ),
            flush=True,
        )
        return booster, rounds_done

    def _fit_lazy(
        self,
        x,
        y: pd.DataFrame,
        *,
        val_x=None,
        val_y: pd.DataFrame | None = None,
    ) -> "XGBoostClassifier":
        import xgboost as xgb

        y_train = _as_array(y).reshape(-1).astype(np.float32)
        if np.unique(y_train).size < 2:
            self.model_ = _ConstantProbabilityClassifier(float(y_train[0]) if len(y_train) else 0.0)
            self.best_iteration_ = 1
            self.device_ = "cpu"
            self.gpu_fallback_error_ = ""
            return self
        y_val_arr = _as_array(val_y).reshape(-1).astype(np.float32) if val_y is not None else None
        scale_pos_weight = float((len(y_train) - y_train.sum()) / max(y_train.sum(), 1.0))
        self.gpu_fallback_error_ = ""
        self.device_ = _xgboost_device(self.prefer_gpu)
        nthread = _xgboost_nthread()
        params = {
            "learning_rate": 0.03,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary:logistic",
            "tree_method": "hist",
            "device": self.device_,
            "eval_metric": "logloss",
            "scale_pos_weight": max(scale_pos_weight, 1.0),
            "seed": self.random_state,
            "nthread": nthread,
        }
        if _xgboost_training_mode(len(x)) == "chunked":
            self.model_, self.best_iteration_ = self._train_lazy_chunks(
                x,
                y_train,
                params=params,
                rounds=self.n_estimators,
                val_x=val_x,
                y_val_arr=y_val_arr,
                phase="fit",
                start_step="xgboost_classifier_train_start",
                chunk_step="xgboost_classifier_chunk_train",
                complete_step="xgboost_classifier_train_complete",
            )
            return self
        dtrain = self._lazy_matrix(x, y_train, phase="fit")
        evals = []
        if val_x is not None and y_val_arr is not None:
            x_val = self._preprocess(val_x, fit=False)
            evals.append((xgb.DMatrix(x_val, label=y_val_arr), "validation"))
        print(
            json.dumps(
                {
                    "step": "xgboost_classifier_train_start",
                    "rows": len(x),
                    "features": x.shape[1],
                    "n_estimators": self.n_estimators,
                    "device": self.device_,
                    "nthread": nthread,
                }
            ),
            flush=True,
        )
        try:
            self.model_ = xgb.train(
                params,
                dtrain,
                num_boost_round=self.n_estimators,
                evals=evals,
                early_stopping_rounds=self.patience if evals else None,
                verbose_eval=25 if evals else False,
            )
        except Exception as exc:
            if self.device_ != "cuda":
                raise
            self.gpu_fallback_error_ = f"{type(exc).__name__}: {exc}"
            self.device_ = "cpu"
            params["device"] = self.device_
            self.model_ = xgb.train(
                params,
                dtrain,
                num_boost_round=self.n_estimators,
                evals=evals,
                early_stopping_rounds=self.patience if evals else None,
                verbose_eval=25 if evals else False,
            )
        best_iteration = getattr(self.model_, "best_iteration", None)
        self.best_iteration_ = int(best_iteration + 1) if best_iteration is not None else self.n_estimators
        print(
            json.dumps(
                {
                    "step": "xgboost_classifier_train_complete",
                    "best_iteration": self.best_iteration_,
                    "device": self.device_,
                }
            ),
            flush=True,
        )
        return self

    def fit(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame,
        *,
        val_x: pd.DataFrame | None = None,
        val_y: pd.DataFrame | None = None,
    ) -> "LightGBMRegressor":
        from lightgbm import LGBMRegressor, early_stopping

        x_train = self._preprocess(x, fit=True)
        x_val = self._preprocess(val_x, fit=False) if val_x is not None else None
        y_train = _as_array(y)
        y_val_arr = _as_array(val_y) if val_y is not None else None
        self.models_: list[object] = []
        self.best_iterations_: list[int] = []
        self.gpu_fallback_error_ = ""
        self.device_type_ = "gpu" if self.prefer_gpu and _gpu_available() else "cpu"
        for target_idx in range(y_train.shape[1]):
            estimator_kwargs = {
                "n_estimators": self.n_estimators,
                "learning_rate": 0.03,
                "num_leaves": 31,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "objective": "regression",
                "random_state": self.random_state + target_idx,
                "verbosity": -1,
                "device_type": self.device_type_,
            }
            estimator = LGBMRegressor(**estimator_kwargs)
            try:
                if x_val is not None and y_val_arr is not None:
                    estimator.fit(
                        x_train,
                        y_train[:, target_idx],
                        eval_set=[(x_val, y_val_arr[:, target_idx])],
                        eval_metric="rmse",
                        callbacks=[early_stopping(self.patience, verbose=False)],
                    )
                else:
                    estimator.fit(x_train, y_train[:, target_idx])
            except Exception as exc:
                if self.device_type_ != "gpu":
                    raise
                self.gpu_fallback_error_ = f"{type(exc).__name__}: {exc}"
                self.device_type_ = "cpu"
                estimator = LGBMRegressor(**{**estimator_kwargs, "device_type": self.device_type_})
                if x_val is not None and y_val_arr is not None:
                    estimator.fit(
                        x_train,
                        y_train[:, target_idx],
                        eval_set=[(x_val, y_val_arr[:, target_idx])],
                        eval_metric="rmse",
                        callbacks=[early_stopping(self.patience, verbose=False)],
                    )
                else:
                    estimator.fit(x_train, y_train[:, target_idx])
            best_iteration = int(getattr(estimator, "best_iteration_", 0) or self.n_estimators)
            self.models_.append(estimator)
            self.best_iterations_.append(best_iteration)
        return self

    def refit_full(self, x: pd.DataFrame, y: pd.DataFrame) -> "LightGBMRegressor":
        from lightgbm import LGBMRegressor

        x_full = self._preprocess(x, fit=True)
        y_full = _as_array(y)
        self.models_ = []
        for target_idx in range(y_full.shape[1]):
            estimator_kwargs = {
                "n_estimators": int(self.best_iterations_[target_idx]),
                "learning_rate": 0.03,
                "num_leaves": 31,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "objective": "regression",
                "random_state": self.random_state + target_idx,
                "verbosity": -1,
                "device_type": getattr(self, "device_type_", "cpu"),
            }
            estimator = LGBMRegressor(**estimator_kwargs)
            try:
                estimator.fit(x_full, y_full[:, target_idx])
            except Exception as exc:
                if estimator_kwargs["device_type"] != "gpu":
                    raise
                self.gpu_fallback_error_ = f"{type(exc).__name__}: {exc}"
                self.device_type_ = "cpu"
                estimator = LGBMRegressor(**{**estimator_kwargs, "device_type": self.device_type_})
                estimator.fit(x_full, y_full[:, target_idx])
            self.models_.append(estimator)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        x_values = self._preprocess(x, fit=False)
        preds = [np.asarray(model.predict(x_values), dtype=np.float64) for model in self.models_]
        return np.column_stack(preds)


class XGBoostRegressor:
    def __init__(
        self,
        *,
        n_estimators: int = 400,
        patience: int = 25,
        random_state: int = 29,
        prefer_gpu: bool = True,
    ) -> None:
        self.n_estimators = n_estimators
        self.patience = patience
        self.random_state = random_state
        self.prefer_gpu = prefer_gpu

    def _preprocess(self, x: pd.DataFrame, *, fit: bool) -> np.ndarray:
        from sklearn.impute import SimpleImputer

        if hasattr(x, "iter_numpy_batches"):
            if fit:
                self.imputer_ = None
            return np.nan_to_num(x.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if fit:
            self.imputer_ = SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True)
            return self.imputer_.fit_transform(x)
        if getattr(self, "imputer_", None) is None:
            return np.nan_to_num(x.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        return self.imputer_.transform(x)

    def _lazy_matrix(self, x, y_values: np.ndarray | None, *, phase: str):
        import xgboost as xgb

        try:
            batch_size = max(int(os.environ.get("V1_XGBOOST_BATCH_ROWS", "262144")), 1)
        except ValueError:
            batch_size = 262_144
        iterator = _XGBoostTabularDataIter(
            x,
            y_values,
            batch_size=batch_size,
            cache_prefix=_xgboost_cache_prefix(x, phase=phase, random_state=self.random_state),
        ).iterator
        try:
            return xgb.ExtMemQuantileDMatrix(iterator, max_bin=256)
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "step": "xgboost_classifier_extmem_fallback",
                        "phase": phase,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                ),
                flush=True,
            )
            return xgb.QuantileDMatrix(iterator, max_bin=256)

    def _train_lazy_chunks(
        self,
        x,
        y_values: np.ndarray,
        *,
        params: dict[str, object],
        rounds: int,
        val_x=None,
        y_val_arr: np.ndarray | None = None,
        phase: str,
        start_step: str,
        chunk_step: str,
        complete_step: str,
    ):
        import xgboost as xgb

        chunk_rows = _xgboost_chunk_rows()
        epochs = _xgboost_chunk_epochs()
        row_count = len(x)
        chunks_per_epoch = max((row_count + chunk_rows - 1) // chunk_rows, 1)
        total_chunks = chunks_per_epoch * epochs
        rounds_total = max(int(rounds), 1)
        rounds_remaining = rounds_total
        chunks_remaining = total_chunks
        booster = None
        validation = None
        if val_x is not None and y_val_arr is not None:
            x_val = self._preprocess(val_x, fit=False)
            validation = xgb.DMatrix(x_val, label=y_val_arr)
        print(
            json.dumps(
                {
                    "step": start_step,
                    "phase": phase,
                    "training_mode": "chunked",
                    "rows": row_count,
                    "features": x.shape[1],
                    "n_estimators": rounds_total,
                    "device": params.get("device"),
                    "nthread": params.get("nthread"),
                    "chunk_rows": chunk_rows,
                    "chunks_per_epoch": chunks_per_epoch,
                    "epochs": epochs,
                }
            ),
            flush=True,
        )
        rounds_done = 0
        chunks_done = 0
        start = time.monotonic()
        for epoch in range(epochs):
            iterator = x.iter_numpy_batches(
                batch_size=chunk_rows,
                shuffle=True,
                random_state=self.random_state + epoch,
            )
            for local_rows, batch in iterator:
                if rounds_remaining <= 0:
                    break
                chunk_rounds = max(1, int(np.ceil(rounds_remaining / max(chunks_remaining, 1))))
                values = np.nan_to_num(batch.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
                labels = y_values[local_rows]
                dtrain = xgb.DMatrix(values, label=labels)
                evals = [(validation, "validation")] if validation is not None else []
                booster = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=chunk_rounds,
                    xgb_model=booster,
                    evals=evals,
                    verbose_eval=False,
                )
                rounds_done += chunk_rounds
                rounds_remaining -= chunk_rounds
                chunks_done += 1
                chunks_remaining -= 1
                print(
                    json.dumps(
                        {
                            "step": chunk_step,
                            "phase": phase,
                            "epoch": epoch + 1,
                            "epochs": epochs,
                            "chunk": chunks_done,
                            "chunks_total": total_chunks,
                            "rows": int(len(batch)),
                            "rounds_done": rounds_done,
                            "rounds_total": rounds_total,
                            "elapsed_seconds": round(time.monotonic() - start, 1),
                        }
                    ),
                    flush=True,
                )
                del dtrain, values, labels, batch
                gc.collect()
            if rounds_remaining <= 0:
                break
        if booster is None:
            raise RuntimeError("XGBoost chunked training did not receive any training batches.")
        print(
            json.dumps(
                {
                    "step": complete_step,
                    "phase": phase,
                    "training_mode": "chunked",
                    "best_iteration": rounds_done,
                    "device": params.get("device"),
                    "nthread": params.get("nthread"),
                }
            ),
            flush=True,
        )
        return booster, rounds_done

    def _fit_lazy(
        self,
        x,
        y: pd.DataFrame,
        *,
        val_x=None,
        val_y: pd.DataFrame | None = None,
    ) -> "XGBoostClassifier":
        import xgboost as xgb

        y_train = _as_array(y).reshape(-1).astype(np.float32)
        if np.unique(y_train).size < 2:
            self.model_ = _ConstantProbabilityClassifier(float(y_train[0]) if len(y_train) else 0.0)
            self.best_iteration_ = 1
            self.device_ = "cpu"
            self.gpu_fallback_error_ = ""
            return self
        y_val_arr = _as_array(val_y).reshape(-1).astype(np.float32) if val_y is not None else None
        scale_pos_weight = float((len(y_train) - y_train.sum()) / max(y_train.sum(), 1.0))
        self.gpu_fallback_error_ = ""
        self.device_ = "cuda" if self.prefer_gpu and _gpu_available() else "cpu"
        params = {
            "learning_rate": 0.03,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary:logistic",
            "tree_method": "hist",
            "device": self.device_,
            "eval_metric": "logloss",
            "scale_pos_weight": max(scale_pos_weight, 1.0),
            "seed": self.random_state,
            "nthread": 1,
        }
        dtrain = self._lazy_matrix(x, y_train, phase="fit")
        evals = []
        if val_x is not None and y_val_arr is not None:
            x_val = self._preprocess(val_x, fit=False)
            evals.append((xgb.DMatrix(x_val, label=y_val_arr), "validation"))
        print(
            json.dumps(
                {
                    "step": "xgboost_classifier_train_start",
                    "rows": len(x),
                    "features": x.shape[1],
                    "n_estimators": self.n_estimators,
                    "device": self.device_,
                }
            ),
            flush=True,
        )
        try:
            self.model_ = xgb.train(
                params,
                dtrain,
                num_boost_round=self.n_estimators,
                evals=evals,
                early_stopping_rounds=self.patience if evals else None,
                verbose_eval=25 if evals else False,
            )
        except Exception as exc:
            if self.device_ != "cuda":
                raise
            self.gpu_fallback_error_ = f"{type(exc).__name__}: {exc}"
            self.device_ = "cpu"
            params["device"] = self.device_
            self.model_ = xgb.train(
                params,
                dtrain,
                num_boost_round=self.n_estimators,
                evals=evals,
                early_stopping_rounds=self.patience if evals else None,
                verbose_eval=25 if evals else False,
            )
        best_iteration = getattr(self.model_, "best_iteration", None)
        self.best_iteration_ = int(best_iteration + 1) if best_iteration is not None else self.n_estimators
        print(
            json.dumps(
                {
                    "step": "xgboost_classifier_train_complete",
                    "best_iteration": self.best_iteration_,
                    "device": self.device_,
                }
            ),
            flush=True,
        )
        return self

    def fit(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame,
        *,
        val_x: pd.DataFrame | None = None,
        val_y: pd.DataFrame | None = None,
    ) -> "XGBoostRegressor":
        from xgboost import XGBRegressor

        x_train = self._preprocess(x, fit=True)
        x_val = self._preprocess(val_x, fit=False) if val_x is not None else None
        y_train = _as_array(y)
        y_val_arr = _as_array(val_y) if val_y is not None else None
        self.models_: list[object] = []
        self.best_iterations_: list[int] = []
        self.gpu_fallback_error_ = ""
        self.device_ = "cuda" if self.prefer_gpu and _gpu_available() else "cpu"
        for target_idx in range(y_train.shape[1]):
            estimator_kwargs = {
                "n_estimators": self.n_estimators,
                "learning_rate": 0.03,
                "max_depth": 4,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "objective": "reg:squarederror",
                "random_state": self.random_state + target_idx,
                "n_jobs": 1,
                "tree_method": "hist",
                "device": self.device_,
                "early_stopping_rounds": self.patience if x_val is not None and y_val_arr is not None else None,
            }
            estimator = XGBRegressor(**estimator_kwargs)
            try:
                if x_val is not None and y_val_arr is not None:
                    estimator.fit(
                        x_train,
                        y_train[:, target_idx],
                        eval_set=[(x_val, y_val_arr[:, target_idx])],
                        verbose=False,
                    )
                else:
                    estimator.fit(x_train, y_train[:, target_idx], verbose=False)
            except Exception as exc:
                if self.device_ != "cuda":
                    raise
                self.gpu_fallback_error_ = f"{type(exc).__name__}: {exc}"
                self.device_ = "cpu"
                estimator = XGBRegressor(**{**estimator_kwargs, "device": self.device_})
                if x_val is not None and y_val_arr is not None:
                    estimator.fit(
                        x_train,
                        y_train[:, target_idx],
                        eval_set=[(x_val, y_val_arr[:, target_idx])],
                        verbose=False,
                    )
                else:
                    estimator.fit(x_train, y_train[:, target_idx], verbose=False)
            best_iteration = getattr(estimator, "best_iteration", None)
            if best_iteration is None:
                best_iteration = getattr(estimator, "best_ntree_limit", None)
            self.models_.append(estimator)
            self.best_iterations_.append(int(best_iteration or self.n_estimators))
        return self

    def refit_full(self, x: pd.DataFrame, y: pd.DataFrame) -> "XGBoostRegressor":
        from xgboost import XGBRegressor

        x_full = self._preprocess(x, fit=True)
        y_full = _as_array(y)
        self.models_ = []
        for target_idx in range(y_full.shape[1]):
            estimator_kwargs = {
                "n_estimators": int(self.best_iterations_[target_idx]),
                "learning_rate": 0.03,
                "max_depth": 4,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "objective": "reg:squarederror",
                "random_state": self.random_state + target_idx,
                "n_jobs": 1,
                "tree_method": "hist",
                "device": getattr(self, "device_", "cpu"),
            }
            estimator = XGBRegressor(**estimator_kwargs)
            try:
                estimator.fit(x_full, y_full[:, target_idx], verbose=False)
            except Exception as exc:
                if estimator_kwargs["device"] != "cuda":
                    raise
                self.gpu_fallback_error_ = f"{type(exc).__name__}: {exc}"
                self.device_ = "cpu"
                estimator = XGBRegressor(**{**estimator_kwargs, "device": self.device_})
                estimator.fit(x_full, y_full[:, target_idx], verbose=False)
            self.models_.append(estimator)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        x_values = self._preprocess(x, fit=False)
        preds = [np.asarray(model.predict(x_values), dtype=np.float64) for model in self.models_]
        return np.column_stack(preds)


class _XGBoostTabularDataIter:
    def __init__(
        self,
        x: Any,
        y_values: np.ndarray | None = None,
        *,
        batch_size: int = 262_144,
        cache_prefix: str | None = None,
        progress_step: str = "xgboost_classifier_data_scan",
    ) -> None:
        import xgboost as xgb

        class _Iterator(xgb.DataIter):
            def __init__(self, outer: "_XGBoostTabularDataIter") -> None:
                super().__init__(cache_prefix=outer.cache_prefix, release_data=True)
                self.outer = outer
                self._iterator = None
                self._batch_index = 0
                self._rows_seen = 0
                self._start_time = time.monotonic()

            def reset(self) -> None:
                self._iterator = self.outer.x.iter_numpy_batches(batch_size=self.outer.batch_size, shuffle=False)
                self._batch_index = 0
                self._rows_seen = 0
                self._start_time = time.monotonic()

            def next(self, input_data) -> bool:
                if self._iterator is None:
                    self.reset()
                try:
                    local_rows, batch = next(self._iterator)
                except StopIteration:
                    return False
                self._batch_index += 1
                values = np.nan_to_num(batch.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
                kwargs: dict[str, object] = {"data": values}
                if self.outer.y_values is not None:
                    kwargs["label"] = self.outer.y_values[local_rows]
                input_data(**kwargs)
                self._rows_seen += int(len(batch))
                if (
                    self._batch_index == 1
                    or self._batch_index % self.outer.progress_every == 0
                    or self._rows_seen >= self.outer.rows_total
                ):
                    print(
                        json.dumps(
                            {
                                "step": self.outer.progress_step,
                                "batch": self._batch_index,
                                "rows_seen": self._rows_seen,
                                "rows_total": self.outer.rows_total,
                                "elapsed_seconds": round(time.monotonic() - self._start_time, 1),
                            }
                        ),
                        flush=True,
                    )
                return True

        self.x = x
        self.y_values = y_values
        self.batch_size = max(int(batch_size), 1)
        self.cache_prefix = cache_prefix
        self.progress_step = progress_step
        self.rows_total = len(x) if hasattr(x, "__len__") else 0
        total_batches = (self.rows_total + self.batch_size - 1) // self.batch_size if self.rows_total else 1
        self.progress_every = _torch_progress_every(total_batches)
        self.iterator = _Iterator(self)


def _xgboost_cache_prefix(x: Any, *, phase: str, random_state: int) -> str | None:
    store = getattr(x, "store", None)
    path = getattr(store, "path", None)
    feature_set = str(getattr(store, "feature_set", "tabular") or "tabular")
    if path is None:
        return None
    root = Path(path).parent.parent / "xgboost_cache"
    root.mkdir(parents=True, exist_ok=True)
    return str(root / f"{feature_set}_{phase}_{os.getpid()}_{random_state}")


class TorchMLPRegressor:
    def __init__(
        self,
        *,
        hidden_units: int = 128,
        learning_rate: float = 0.001,
        max_epochs: int = 80,
        patience: int = 10,
        batch_size: int = 512,
        random_state: int = 31,
    ) -> None:
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.random_state = random_state
        self.device = _default_torch_device()
        self.x_scaler = Standardizer()
        self.y_scaler = Standardizer()

    def _resolve_device(self):
        import torch

        requested = self.device if self.device != "cuda" or _torch_cuda_available() else "cpu"
        self.device_ = torch.device(requested)
        if self.device_.type == "cuda":
            torch.backends.cudnn.benchmark = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
        return self.device_

    def _build_model(self, input_dim: int, output_dim: int):
        import torch

        return torch.nn.Sequential(
            torch.nn.Linear(input_dim, self.hidden_units),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.05),
            torch.nn.Linear(self.hidden_units, max(self.hidden_units // 2, 16)),
            torch.nn.ReLU(),
            torch.nn.Linear(max(self.hidden_units // 2, 16), output_dim),
        )

    def _fit_fixed_epochs(self, x_values: np.ndarray, y_values: np.ndarray, epochs: int) -> None:
        import torch

        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(x_values.astype(np.float32)),
            torch.from_numpy(y_values.astype(np.float32)),
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            **_torch_loader_kwargs(
                device=self.device_.type,
                shuffle=True,
                batch_size=min(self.batch_size, len(dataset)),
            ),
        )
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        loss_fn = torch.nn.MSELoss()
        self.model_.train()
        for _ in range(max(epochs, 1)):
            for batch_x, batch_y in loader:
                batch_x = _move_tensor(batch_x, self.device_)
                batch_y = _move_tensor(batch_y, self.device_)
                pred = self.model_(batch_x)
                loss = loss_fn(pred, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def fit(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame,
        *,
        val_x: pd.DataFrame | None = None,
        val_y: pd.DataFrame | None = None,
    ) -> "TorchMLPRegressor":
        import torch

        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        self._resolve_device()
        x_train = self.x_scaler.fit(_as_array(x)).transform(_as_array(x)).astype(np.float32)
        y_train = self.y_scaler.fit(_as_array(y)).transform(_as_array(y)).astype(np.float32)
        x_val = self.x_scaler.transform(_as_array(val_x)).astype(np.float32) if val_x is not None else None
        y_val_arr = self.y_scaler.transform(_as_array(val_y)).astype(np.float32) if val_y is not None else None
        self.model_ = self._build_model(x_train.shape[1], y_train.shape[1]).to(self.device_)
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
        loader = torch.utils.data.DataLoader(
            dataset,
            **_torch_loader_kwargs(
                device=self.device_.type,
                shuffle=True,
                batch_size=min(self.batch_size, len(dataset)),
            ),
        )
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        loss_fn = torch.nn.MSELoss()
        best_state = copy.deepcopy(self.model_.state_dict())
        best_metric = np.inf
        best_epoch = 1
        epochs_without_improvement = 0
        for epoch in range(self.max_epochs):
            self.model_.train()
            for batch_x, batch_y in loader:
                batch_x = _move_tensor(batch_x, self.device_)
                batch_y = _move_tensor(batch_y, self.device_)
                pred = self.model_(batch_x)
                loss = loss_fn(pred, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            metric_x = x_val if x_val is not None else x_train
            metric_y = y_val_arr if y_val_arr is not None else y_train
            self.model_.eval()
            with torch.no_grad():
                pred = self.model_(_move_tensor(torch.from_numpy(metric_x), self.device_)).detach().cpu().numpy()
            metric = float(np.nanmean((pred - metric_y) ** 2))
            if metric + 1e-6 < best_metric:
                best_metric = metric
                best_state = copy.deepcopy(self.model_.state_dict())
                best_epoch = epoch + 1
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if x_val is not None and epochs_without_improvement >= self.patience:
                break
        self.model_.load_state_dict(best_state)
        self.best_epoch_ = best_epoch
        return self

    def refit_full(self, x: pd.DataFrame, y: pd.DataFrame) -> "TorchMLPRegressor":
        x_full = self.x_scaler.fit(_as_array(x)).transform(_as_array(x)).astype(np.float32)
        y_full = self.y_scaler.fit(_as_array(y)).transform(_as_array(y)).astype(np.float32)
        self._resolve_device()
        self.model_ = self._build_model(x_full.shape[1], y_full.shape[1]).to(self.device_)
        self._fit_fixed_epochs(x_full, y_full, getattr(self, "best_epoch_", self.max_epochs))
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        import torch

        self._resolve_device()
        self.model_ = self.model_.to(self.device_)
        x_values = self.x_scaler.transform(_as_array(x)).astype(np.float32)
        with torch.no_grad():
            pred = self.model_(_move_tensor(torch.from_numpy(x_values), self.device_)).detach().cpu().numpy()
        return self.y_scaler.inverse_transform(pred)

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        if state.get("model_") is not None:
            state["model_"] = _copy_model_to_cpu(state["model_"])
        state["device_"] = "cpu"
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self._resolve_device()
        if getattr(self, "model_", None) is not None:
            self.model_ = self.model_.to(self.device_)


class _SequenceStaticDataset:
    def __init__(
        self,
        *,
        store: Any,
        metadata: pd.DataFrame,
        static_categorical: dict[str, np.ndarray],
        window_length: int,
        target_frame: pd.DataFrame | None = None,
        static_columns: Sequence[str],
    ) -> None:
        self.store = store
        self.tickers = metadata["ticker"].astype(str).str.upper().tolist()
        self.end_indices = metadata["window_row_count"].astype(int).to_numpy(dtype=np.int64) - 1
        self.window_length = window_length
        self.static_columns = list(static_columns)
        self.static_values = _hstack_static_arrays(static_categorical, self.static_columns)
        self.targets = _as_array(target_frame).astype(np.float32) if target_frame is not None else None
        self.group_ranges = self._contiguous_group_ranges()

    def _contiguous_group_ranges(self) -> list[tuple[int, int]]:
        if not self.tickers:
            return []
        ranges: list[tuple[int, int]] = []
        start = 0
        current = self.tickers[0]
        for position, ticker in enumerate(self.tickers[1:], start=1):
            if ticker != current:
                ranges.append((start, position))
                start = position
                current = ticker
        ranges.append((start, len(self.tickers)))
        return ranges

    def __len__(self) -> int:
        return len(self.tickers)

    def __getitem__(self, index: int):
        sequence = self.store.get_window(self.tickers[index], int(self.end_indices[index]), self.window_length).astype(
            np.float32
        )
        static_ids = self.static_values[index].astype(np.int64)
        if self.targets is None:
            return sequence, static_ids
        return sequence, static_ids, self.targets[index]

    def __getitems__(self, indices):
        index_array = np.asarray(indices, dtype=np.int64)
        if index_array.size == 0:
            return []
        if (
            hasattr(self.store, "open")
            and hasattr(self.store, "ticker_offsets")
            and all(self.tickers[int(index)] == self.tickers[int(index_array[0])] for index in index_array)
        ):
            symbol = self.tickers[int(index_array[0])]
            if symbol not in self.store.ticker_offsets or symbol not in self.store.ticker_lengths:
                return [self.__getitem__(int(index)) for index in index_array]
            end_indices = self.end_indices[index_array].astype(np.int64)
            min_end = int(end_indices.min())
            max_end = int(end_indices.max())
            start = min_end - self.window_length + 1
            if start >= 0 and max_end < self.store.ticker_lengths[symbol]:
                offset = int(self.store.ticker_offsets[symbol])
                block = np.asarray(
                    self.store.open()[offset + start : offset + max_end + 1],
                    dtype=np.float32,
                )
                sequences = np.stack(
                    [
                        block[int(end_index - min_end) : int(end_index - min_end) + self.window_length]
                        for end_index in end_indices
                    ]
                ).astype(np.float32, copy=False)
                static_values = self.static_values[index_array].astype(np.int64, copy=False)
                if self.targets is None:
                    return [(sequences[position], static_values[position]) for position in range(len(index_array))]
                targets = self.targets[index_array].astype(np.float32, copy=False)
                return [
                    (sequences[position], static_values[position], targets[position])
                    for position in range(len(index_array))
                ]
        return [self.__getitem__(int(index)) for index in index_array]


class _SequenceStaticNet:
    def __init__(
        self,
        *,
        sequence_dim: int,
        static_cardinalities: Sequence[int],
        output_dim: int,
        window_length: int,
        hidden_dim: int = 128,
        static_hidden_dim: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        import torch

        super().__init__()
        self.window_length = window_length
        self.token_projection = torch.nn.Linear(sequence_dim, hidden_dim)
        self.position_embedding = torch.nn.Embedding(window_length, hidden_dim)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        embedding_dims = [min(32, max(4, (cardinality + 1) // 4)) for cardinality in static_cardinalities]
        self.static_embeddings = torch.nn.ModuleList(
            [torch.nn.Embedding(cardinality, emb_dim) for cardinality, emb_dim in zip(static_cardinalities, embedding_dims)]
        )
        static_input_dim = int(sum(embedding_dims))
        self.static_mlp = torch.nn.Sequential(
            torch.nn.Linear(static_input_dim, static_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(static_hidden_dim, static_hidden_dim),
            torch.nn.ReLU(),
        )
        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim + static_hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def parameters(self):
        import itertools

        modules = [
            self.token_projection,
            self.position_embedding,
            self.transformer,
            self.static_embeddings,
            self.static_mlp,
            self.fusion,
        ]
        return itertools.chain.from_iterable(module.parameters() for module in modules)

    def state_dict(self):
        return {
            "token_projection": self.token_projection.state_dict(),
            "position_embedding": self.position_embedding.state_dict(),
            "transformer": self.transformer.state_dict(),
            "static_embeddings": [module.state_dict() for module in self.static_embeddings],
            "static_mlp": self.static_mlp.state_dict(),
            "fusion": self.fusion.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        self.token_projection.load_state_dict(state_dict["token_projection"])
        self.position_embedding.load_state_dict(state_dict["position_embedding"])
        self.transformer.load_state_dict(state_dict["transformer"])
        for module, state in zip(self.static_embeddings, state_dict["static_embeddings"]):
            module.load_state_dict(state)
        self.static_mlp.load_state_dict(state_dict["static_mlp"])
        self.fusion.load_state_dict(state_dict["fusion"])

    def train(self) -> None:
        self.token_projection.train()
        self.position_embedding.train()
        self.transformer.train()
        self.static_embeddings.train()
        self.static_mlp.train()
        self.fusion.train()

    def eval(self) -> None:
        self.token_projection.eval()
        self.position_embedding.eval()
        self.transformer.eval()
        self.static_embeddings.eval()
        self.static_mlp.eval()
        self.fusion.eval()

    def to(self, device):
        self.token_projection = self.token_projection.to(device)
        self.position_embedding = self.position_embedding.to(device)
        self.transformer = self.transformer.to(device)
        self.static_embeddings = self.static_embeddings.to(device)
        self.static_mlp = self.static_mlp.to(device)
        self.fusion = self.fusion.to(device)
        return self

    def __call__(self, sequence, static_ids):
        import torch

        positions = torch.arange(sequence.shape[1], device=sequence.device)
        token = self.token_projection(sequence) + self.position_embedding(positions)[None, :, :]
        encoded = self.transformer(token)
        pooled = encoded.mean(dim=1)
        embedded = [module(static_ids[:, idx]) for idx, module in enumerate(self.static_embeddings)]
        static_repr = self.static_mlp(torch.cat(embedded, dim=1))
        return self.fusion(torch.cat([pooled, static_repr], dim=1))


class TorchSequenceStaticRegressor:
    def __init__(
        self,
        *,
        window_length: int = 60,
        hidden_dim: int = 128,
        learning_rate: float = 0.001,
        max_epochs: int = 20,
        patience: int = 4,
        batch_size: int = 512,
        random_state: int = 37,
    ) -> None:
        self.window_length = window_length
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.random_state = random_state
        self.device = _default_torch_device()
        self.x_scaler = Standardizer()
        self.y_scaler = Standardizer()

    def _resolve_device(self):
        import torch

        requested = self.device if self.device != "cuda" or _torch_cuda_available() else "cpu"
        self.device_ = torch.device(requested)
        if self.device_.type == "cuda":
            torch.backends.cudnn.benchmark = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
        return self.device_

    def _prepare_sequence_tensor(self, sequence, mean_t, scale_t):
        import torch

        mean_view = mean_t.view(1, 1, -1)
        scale_view = scale_t.view(1, 1, -1)
        sequence = sequence.to(dtype=torch.float32)
        sequence = torch.where(torch.isfinite(sequence), sequence, mean_view)
        return (sequence - mean_view) / scale_view

    def _train_cutoff_date(self, metadata: pd.DataFrame) -> str:
        dates = metadata["anchor_date"].dropna().astype(str)
        if dates.empty:
            raise ValueError("Sequence/static training metadata is empty.")
        return str(dates.max())

    def _build_model(self, *, sequence_dim: int, static_cardinalities: Sequence[int], output_dim: int):
        return _SequenceStaticNet(
            sequence_dim=sequence_dim,
            static_cardinalities=static_cardinalities,
            output_dim=output_dim,
            window_length=self.window_length,
            hidden_dim=self.hidden_dim,
        )

    def _train_epochs(self, dataset: _SequenceStaticDataset, epochs: int) -> None:
        import torch

        loader = torch.utils.data.DataLoader(
            dataset,
            **_torch_loader_kwargs(
                device=self.device_.type,
                shuffle=True,
                batch_size=min(self.batch_size, len(dataset)),
            ),
        )
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        loss_fn = torch.nn.MSELoss()
        mean_t = torch.from_numpy(self.x_scaler.mean_.astype(np.float32)).to(self.device_)
        scale_t = torch.from_numpy(self.x_scaler.scale_.astype(np.float32)).to(self.device_)
        self.model_.train()
        for _ in range(max(epochs, 1)):
            for sequence, static_ids, target in loader:
                sequence = self._prepare_sequence_tensor(_move_tensor(sequence, self.device_), mean_t, scale_t)
                pred = self.model_(sequence, _move_tensor(static_ids.long(), self.device_))
                loss = loss_fn(pred, _move_tensor(target.float(), self.device_))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def fit(
        self,
        x: dict[str, object],
        y: pd.DataFrame,
        *,
        val_x: dict[str, object] | None = None,
        val_y: pd.DataFrame | None = None,
    ) -> "TorchSequenceStaticRegressor":
        import torch

        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        self._resolve_device()
        self.static_columns_ = list(x["static_categorical"].keys())
        self.static_vocab_sizes_ = [
            int(np.max(x["static_categorical"][column])) + 1 if len(x["static_categorical"][column]) else 1
            for column in self.static_columns_
        ]
        _fit_sequence_store_standardizer(
            self.x_scaler,
            x["store"],
            self._train_cutoff_date(x["metadata"]),
            width=len(x["store"].feature_columns),
        )
        y_train = self.y_scaler.fit(_as_array(y)).transform(_as_array(y)).astype(np.float32)
        y_val_arr = self.y_scaler.transform(_as_array(val_y)).astype(np.float32) if val_y is not None else None
        train_dataset = _SequenceStaticDataset(
            store=x["store"],
            metadata=x["metadata"],
            static_categorical=x["static_categorical"],
            target_frame=pd.DataFrame(y_train, columns=y.columns),
            window_length=self.window_length,
            static_columns=self.static_columns_,
        )
        val_dataset = None
        if val_x is not None and val_y is not None:
            val_dataset = _SequenceStaticDataset(
                store=val_x["store"],
                metadata=val_x["metadata"],
                static_categorical=val_x["static_categorical"],
                target_frame=pd.DataFrame(y_val_arr, columns=val_y.columns),
                window_length=self.window_length,
                static_columns=self.static_columns_,
            )
        self.model_ = self._build_model(
            sequence_dim=len(x["store"].feature_columns),
            static_cardinalities=self.static_vocab_sizes_,
            output_dim=y_train.shape[1],
        ).to(self.device_)
        self.output_dim_ = y_train.shape[1]
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        loss_fn = torch.nn.MSELoss()
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            **_torch_loader_kwargs(
                device=self.device_.type,
                shuffle=True,
                batch_size=min(self.batch_size, len(train_dataset)),
            ),
        )
        mean_t = torch.from_numpy(self.x_scaler.mean_.astype(np.float32)).to(self.device_)
        scale_t = torch.from_numpy(self.x_scaler.scale_.astype(np.float32)).to(self.device_)
        best_state = copy.deepcopy(self.model_.state_dict())
        best_metric = np.inf
        best_epoch = 1
        epochs_without_improvement = 0
        for epoch in range(self.max_epochs):
            self.model_.train()
            for sequence, static_ids, target in train_loader:
                sequence = self._prepare_sequence_tensor(_move_tensor(sequence, self.device_), mean_t, scale_t)
                pred = self.model_(sequence, _move_tensor(static_ids.long(), self.device_))
                loss = loss_fn(pred, _move_tensor(target.float(), self.device_))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            metric = self._dataset_loss(val_dataset if val_dataset is not None else train_dataset)
            if metric + 1e-6 < best_metric:
                best_metric = metric
                best_state = copy.deepcopy(self.model_.state_dict())
                best_epoch = epoch + 1
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if val_dataset is not None and epochs_without_improvement >= self.patience:
                break
        self.model_.load_state_dict(best_state)
        self.best_epoch_ = best_epoch
        return self

    def _dataset_loss(self, dataset: _SequenceStaticDataset) -> float:
        import torch

        loader = torch.utils.data.DataLoader(
            dataset,
            **_torch_loader_kwargs(
                device=self.device_.type,
                shuffle=False,
                batch_size=min(self.batch_size, len(dataset)),
            ),
        )
        loss_fn = torch.nn.MSELoss()
        mean_t = torch.from_numpy(self.x_scaler.mean_.astype(np.float32)).to(self.device_)
        scale_t = torch.from_numpy(self.x_scaler.scale_.astype(np.float32)).to(self.device_)
        self.model_.eval()
        losses: list[float] = []
        with torch.no_grad():
            for sequence, static_ids, target in loader:
                sequence = self._prepare_sequence_tensor(_move_tensor(sequence, self.device_), mean_t, scale_t)
                pred = self.model_(sequence, _move_tensor(static_ids.long(), self.device_))
                losses.append(float(loss_fn(pred, _move_tensor(target.float(), self.device_)).item()))
        return float(np.mean(losses)) if losses else np.inf

    def refit_full(self, x: dict[str, object], y: pd.DataFrame) -> "TorchSequenceStaticRegressor":
        self._resolve_device()
        _fit_sequence_store_standardizer(
            self.x_scaler,
            x["store"],
            self._train_cutoff_date(x["metadata"]),
            width=len(x["store"].feature_columns),
        )
        y_train = self.y_scaler.fit(_as_array(y)).transform(_as_array(y)).astype(np.float32)
        train_dataset = _SequenceStaticDataset(
            store=x["store"],
            metadata=x["metadata"],
            static_categorical=x["static_categorical"],
            target_frame=pd.DataFrame(y_train, columns=y.columns),
            window_length=self.window_length,
            static_columns=self.static_columns_,
        )
        self.model_ = self._build_model(
            sequence_dim=len(x["store"].feature_columns),
            static_cardinalities=self.static_vocab_sizes_,
            output_dim=y_train.shape[1],
        ).to(self.device_)
        self.output_dim_ = y_train.shape[1]
        self._train_epochs(train_dataset, getattr(self, "best_epoch_", self.max_epochs))
        return self

    def predict(self, x: dict[str, object]) -> np.ndarray:
        import torch

        dataset = _SequenceStaticDataset(
            store=x["store"],
            metadata=x["metadata"],
            static_categorical=x["static_categorical"],
            target_frame=None,
            window_length=self.window_length,
            static_columns=self.static_columns_,
        )
        self._resolve_device()
        self.model_ = self.model_.to(self.device_)
        loader = torch.utils.data.DataLoader(
            dataset,
            **_torch_loader_kwargs(
                device=self.device_.type,
                shuffle=False,
                batch_size=min(self.batch_size, len(dataset)),
            ),
        )
        mean_t = torch.from_numpy(self.x_scaler.mean_.astype(np.float32)).to(self.device_)
        scale_t = torch.from_numpy(self.x_scaler.scale_.astype(np.float32)).to(self.device_)
        outputs: list[np.ndarray] = []
        self.model_.eval()
        with torch.no_grad():
            for sequence, static_ids in loader:
                sequence = self._prepare_sequence_tensor(_move_tensor(sequence, self.device_), mean_t, scale_t)
                pred = self.model_(sequence, _move_tensor(static_ids.long(), self.device_)).detach().cpu().numpy()
                outputs.append(pred)
        if not outputs:
            return np.empty((0, getattr(self, "output_dim_", 0)), dtype=np.float64)
        return self.y_scaler.inverse_transform(np.vstack(outputs))

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        if state.get("model_") is not None:
            state["model_"] = _copy_model_to_cpu(state["model_"])
        state["device_"] = "cpu"
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self._resolve_device()
        if getattr(self, "model_", None) is not None:
            self.model_ = self.model_.to(self.device_)


class LogisticClassifier:
    def __init__(
        self,
        *,
        penalty: str = "l2",
        solver: str = "lbfgs",
        l1_ratio: float | None = None,
        c: float = 1.0,
        random_state: int = 41,
    ) -> None:
        self.penalty = penalty
        self.solver = solver
        self.l1_ratio = l1_ratio
        self.c = c
        self.random_state = random_state

    def _prepare(self, x: pd.DataFrame, *, fit: bool) -> np.ndarray:
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        if fit:
            self.imputer_ = SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True)
            self.scaler_ = StandardScaler()
            values = self.imputer_.fit_transform(x)
            return self.scaler_.fit_transform(values)
        values = self.imputer_.transform(x)
        return self.scaler_.transform(values)

    def fit(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame,
        *,
        val_x: pd.DataFrame | None = None,
        val_y: pd.DataFrame | None = None,
    ) -> "LogisticClassifier":
        from sklearn.linear_model import LogisticRegression

        x_train = self._prepare(x, fit=True)
        target = _as_array(y).reshape(-1)
        if np.unique(target).size < 2:
            self.model_ = _ConstantProbabilityClassifier(float(target[0]) if len(target) else 0.0)
            return self
        kwargs: dict[str, object] = {
            "max_iter": 3000,
            "C": self.c,
            "class_weight": "balanced",
            "random_state": self.random_state,
            "solver": self.solver,
        }
        if self.penalty != "l2":
            kwargs["penalty"] = self.penalty
        if self.l1_ratio is not None:
            kwargs["l1_ratio"] = self.l1_ratio
        self.model_ = LogisticRegression(**kwargs)
        self.model_.fit(x_train, target)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        x_values = self._prepare(x, fit=False)
        return self.model_.predict_proba(x_values)[:, 1:2].astype(np.float64)


class ElasticNetLogisticClassifier(LogisticClassifier):
    def _prepare(self, x: pd.DataFrame, *, fit: bool) -> np.ndarray:
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        if fit:
            self.imputer_ = SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True)
            self.scaler_ = StandardScaler()
            values = self.imputer_.fit_transform(x)
            return self.scaler_.fit_transform(values)
        values = self.imputer_.transform(x)
        return self.scaler_.transform(values)

    def fit(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame,
        *,
        val_x: pd.DataFrame | None = None,
        val_y: pd.DataFrame | None = None,
    ) -> "ElasticNetLogisticClassifier":
        from sklearn.linear_model import SGDClassifier

        x_train = self._prepare(x, fit=True)
        target = _as_array(y).reshape(-1)
        if np.unique(target).size < 2:
            self.model_ = _ConstantProbabilityClassifier(float(target[0]) if len(target) else 0.0)
            return self
        self.model_ = SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            alpha=0.0005,
            l1_ratio=0.15,
            max_iter=1000,
            tol=1e-3,
            class_weight="balanced",
            average=True,
            random_state=42,
        )
        self.model_.fit(x_train, target)
        return self


class LightGBMClassifier:
    def __init__(
        self,
        *,
        n_estimators: int = 500,
        patience: int = 25,
        random_state: int = 43,
        prefer_gpu: bool = True,
    ) -> None:
        self.n_estimators = n_estimators
        self.patience = patience
        self.random_state = random_state
        self.prefer_gpu = prefer_gpu

    def _preprocess(self, x: pd.DataFrame, *, fit: bool) -> np.ndarray:
        from sklearn.impute import SimpleImputer

        if fit:
            self.imputer_ = SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True)
            return self.imputer_.fit_transform(x)
        return self.imputer_.transform(x)

    def fit(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame,
        *,
        val_x: pd.DataFrame | None = None,
        val_y: pd.DataFrame | None = None,
    ) -> "LightGBMClassifier":
        from lightgbm import LGBMClassifier, early_stopping

        x_train = self._preprocess(x, fit=True)
        x_val = self._preprocess(val_x, fit=False) if val_x is not None else None
        y_train = _as_array(y).reshape(-1)
        if np.unique(y_train).size < 2:
            self.model_ = _ConstantProbabilityClassifier(float(y_train[0]) if len(y_train) else 0.0)
            self.best_iteration_ = 1
            self.device_type_ = "cpu"
            self.gpu_fallback_error_ = ""
            return self
        y_val_arr = _as_array(val_y).reshape(-1) if val_y is not None else None
        self.gpu_fallback_error_ = ""
        self.device_type_ = "gpu" if self.prefer_gpu and _gpu_available() else "cpu"
        estimator_kwargs = {
            "n_estimators": self.n_estimators,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary",
            "random_state": self.random_state,
            "verbosity": -1,
            "device_type": self.device_type_,
            "class_weight": "balanced",
        }
        estimator = LGBMClassifier(**estimator_kwargs)
        try:
            if x_val is not None and y_val_arr is not None:
                estimator.fit(
                    x_train,
                    y_train,
                    eval_set=[(x_val, y_val_arr)],
                    eval_metric="binary_logloss",
                    callbacks=[early_stopping(self.patience, verbose=False)],
                )
            else:
                estimator.fit(x_train, y_train)
        except Exception as exc:
            if self.device_type_ != "gpu":
                raise
            self.gpu_fallback_error_ = f"{type(exc).__name__}: {exc}"
            self.device_type_ = "cpu"
            estimator = LGBMClassifier(**{**estimator_kwargs, "device_type": self.device_type_})
            if x_val is not None and y_val_arr is not None:
                estimator.fit(
                    x_train,
                    y_train,
                    eval_set=[(x_val, y_val_arr)],
                    eval_metric="binary_logloss",
                    callbacks=[early_stopping(self.patience, verbose=False)],
                )
            else:
                estimator.fit(x_train, y_train)
        self.model_ = estimator
        self.best_iteration_ = int(getattr(estimator, "best_iteration_", 0) or self.n_estimators)
        return self

    def refit_full(self, x: pd.DataFrame, y: pd.DataFrame) -> "LightGBMClassifier":
        from lightgbm import LGBMClassifier

        x_full = self._preprocess(x, fit=True)
        y_full = _as_array(y).reshape(-1)
        if np.unique(y_full).size < 2:
            self.model_ = _ConstantProbabilityClassifier(float(y_full[0]) if len(y_full) else 0.0)
            return self
        estimator_kwargs = {
            "n_estimators": int(getattr(self, "best_iteration_", self.n_estimators)),
            "learning_rate": 0.03,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary",
            "random_state": self.random_state,
            "verbosity": -1,
            "device_type": getattr(self, "device_type_", "cpu"),
            "class_weight": "balanced",
        }
        self.model_ = LGBMClassifier(**estimator_kwargs)
        try:
            self.model_.fit(x_full, y_full)
        except Exception as exc:
            if estimator_kwargs["device_type"] != "gpu":
                raise
            self.gpu_fallback_error_ = f"{type(exc).__name__}: {exc}"
            self.device_type_ = "cpu"
            self.model_ = LGBMClassifier(**{**estimator_kwargs, "device_type": self.device_type_})
            self.model_.fit(x_full, y_full)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        x_values = self._preprocess(x, fit=False)
        return self.model_.predict_proba(x_values)[:, 1:2].astype(np.float64)


class XGBoostClassifier:
    def __init__(
        self,
        *,
        n_estimators: int = 500,
        patience: int = 25,
        random_state: int = 47,
        prefer_gpu: bool = True,
    ) -> None:
        self.n_estimators = n_estimators
        self.patience = patience
        self.random_state = random_state
        self.prefer_gpu = prefer_gpu

    def _preprocess(self, x: pd.DataFrame, *, fit: bool) -> np.ndarray:
        from sklearn.impute import SimpleImputer

        if hasattr(x, "iter_numpy_batches"):
            if fit:
                self.imputer_ = None
            return np.nan_to_num(x.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if fit:
            self.imputer_ = SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True)
            return self.imputer_.fit_transform(x)
        if getattr(self, "imputer_", None) is None:
            return np.nan_to_num(x.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        return self.imputer_.transform(x)

    def _lazy_matrix(self, x, y_values: np.ndarray | None, *, phase: str):
        import xgboost as xgb

        try:
            batch_size = max(int(os.environ.get("V1_XGBOOST_BATCH_ROWS", "262144")), 1)
        except ValueError:
            batch_size = 262_144
        iterator = _XGBoostTabularDataIter(
            x,
            y_values,
            batch_size=batch_size,
            cache_prefix=_xgboost_cache_prefix(x, phase=phase, random_state=self.random_state),
        ).iterator
        try:
            return xgb.ExtMemQuantileDMatrix(iterator, max_bin=256)
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "step": "xgboost_classifier_extmem_fallback",
                        "phase": phase,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                ),
                flush=True,
            )
            return xgb.QuantileDMatrix(iterator, max_bin=256)

    def _train_lazy_chunks(
        self,
        x,
        y_values: np.ndarray,
        *,
        params: dict[str, object],
        rounds: int,
        val_x=None,
        y_val_arr: np.ndarray | None = None,
        phase: str,
        start_step: str,
        chunk_step: str,
        complete_step: str,
    ):
        import xgboost as xgb

        chunk_rows = _xgboost_chunk_rows()
        epochs = _xgboost_chunk_epochs()
        row_count = len(x)
        chunks_per_epoch = max((row_count + chunk_rows - 1) // chunk_rows, 1)
        total_chunks = chunks_per_epoch * epochs
        rounds_total = max(int(rounds), 1)
        rounds_remaining = rounds_total
        chunks_remaining = total_chunks
        booster = None
        validation = None
        if val_x is not None and y_val_arr is not None:
            x_val = self._preprocess(val_x, fit=False)
            validation = xgb.DMatrix(x_val, label=y_val_arr)
        print(
            json.dumps(
                {
                    "step": start_step,
                    "phase": phase,
                    "training_mode": "chunked",
                    "rows": row_count,
                    "features": x.shape[1],
                    "n_estimators": rounds_total,
                    "device": params.get("device"),
                    "nthread": params.get("nthread"),
                    "chunk_rows": chunk_rows,
                    "chunks_per_epoch": chunks_per_epoch,
                    "epochs": epochs,
                }
            ),
            flush=True,
        )
        rounds_done = 0
        chunks_done = 0
        start = time.monotonic()
        for epoch in range(epochs):
            iterator = x.iter_numpy_batches(
                batch_size=chunk_rows,
                shuffle=True,
                random_state=self.random_state + epoch,
            )
            for local_rows, batch in iterator:
                if rounds_remaining <= 0:
                    break
                chunk_rounds = max(1, int(np.ceil(rounds_remaining / max(chunks_remaining, 1))))
                values = np.nan_to_num(batch.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
                labels = y_values[local_rows]
                dtrain = xgb.DMatrix(values, label=labels)
                evals = [(validation, "validation")] if validation is not None else []
                booster = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=chunk_rounds,
                    xgb_model=booster,
                    evals=evals,
                    verbose_eval=False,
                )
                rounds_done += chunk_rounds
                rounds_remaining -= chunk_rounds
                chunks_done += 1
                chunks_remaining -= 1
                print(
                    json.dumps(
                        {
                            "step": chunk_step,
                            "phase": phase,
                            "epoch": epoch + 1,
                            "epochs": epochs,
                            "chunk": chunks_done,
                            "chunks_total": total_chunks,
                            "rows": int(len(batch)),
                            "rounds_done": rounds_done,
                            "rounds_total": rounds_total,
                            "elapsed_seconds": round(time.monotonic() - start, 1),
                        }
                    ),
                    flush=True,
                )
                del dtrain, values, labels, batch
                gc.collect()
            if rounds_remaining <= 0:
                break
        if booster is None:
            raise RuntimeError("XGBoost chunked training did not receive any training batches.")
        print(
            json.dumps(
                {
                    "step": complete_step,
                    "phase": phase,
                    "training_mode": "chunked",
                    "best_iteration": rounds_done,
                    "device": params.get("device"),
                    "nthread": params.get("nthread"),
                }
            ),
            flush=True,
        )
        return booster, rounds_done

    def _fit_lazy(
        self,
        x,
        y: pd.DataFrame,
        *,
        val_x=None,
        val_y: pd.DataFrame | None = None,
    ) -> "XGBoostClassifier":
        import xgboost as xgb

        y_train = _as_array(y).reshape(-1).astype(np.float32)
        if np.unique(y_train).size < 2:
            self.model_ = _ConstantProbabilityClassifier(float(y_train[0]) if len(y_train) else 0.0)
            self.best_iteration_ = 1
            self.device_ = "cpu"
            self.gpu_fallback_error_ = ""
            return self
        y_val_arr = _as_array(val_y).reshape(-1).astype(np.float32) if val_y is not None else None
        scale_pos_weight = float((len(y_train) - y_train.sum()) / max(y_train.sum(), 1.0))
        self.gpu_fallback_error_ = ""
        self.device_ = _xgboost_device(self.prefer_gpu)
        nthread = _xgboost_nthread()
        params = {
            "learning_rate": 0.03,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary:logistic",
            "tree_method": "hist",
            "device": self.device_,
            "eval_metric": "logloss",
            "scale_pos_weight": max(scale_pos_weight, 1.0),
            "seed": self.random_state,
            "nthread": nthread,
        }
        if _xgboost_training_mode(len(x)) == "chunked":
            self.model_, self.best_iteration_ = self._train_lazy_chunks(
                x,
                y_train,
                params=params,
                rounds=self.n_estimators,
                val_x=val_x,
                y_val_arr=y_val_arr,
                phase="fit",
                start_step="xgboost_classifier_train_start",
                chunk_step="xgboost_classifier_chunk_train",
                complete_step="xgboost_classifier_train_complete",
            )
            return self
        dtrain = self._lazy_matrix(x, y_train, phase="fit")
        evals = []
        if val_x is not None and y_val_arr is not None:
            x_val = self._preprocess(val_x, fit=False)
            evals.append((xgb.DMatrix(x_val, label=y_val_arr), "validation"))
        print(
            json.dumps(
                {
                    "step": "xgboost_classifier_train_start",
                    "rows": len(x),
                    "features": x.shape[1],
                    "n_estimators": self.n_estimators,
                    "device": self.device_,
                    "nthread": nthread,
                }
            ),
            flush=True,
        )
        try:
            self.model_ = xgb.train(
                params,
                dtrain,
                num_boost_round=self.n_estimators,
                evals=evals,
                early_stopping_rounds=self.patience if evals else None,
                verbose_eval=25 if evals else False,
            )
        except Exception as exc:
            if self.device_ != "cuda":
                raise
            self.gpu_fallback_error_ = f"{type(exc).__name__}: {exc}"
            self.device_ = "cpu"
            params["device"] = self.device_
            self.model_ = xgb.train(
                params,
                dtrain,
                num_boost_round=self.n_estimators,
                evals=evals,
                early_stopping_rounds=self.patience if evals else None,
                verbose_eval=25 if evals else False,
            )
        best_iteration = getattr(self.model_, "best_iteration", None)
        self.best_iteration_ = int(best_iteration + 1) if best_iteration is not None else self.n_estimators
        print(
            json.dumps(
                {
                    "step": "xgboost_classifier_train_complete",
                    "best_iteration": self.best_iteration_,
                    "device": self.device_,
                }
            ),
            flush=True,
        )
        return self

    def fit(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame,
        *,
        val_x: pd.DataFrame | None = None,
        val_y: pd.DataFrame | None = None,
    ) -> "XGBoostClassifier":
        from xgboost import XGBClassifier

        if hasattr(x, "iter_numpy_batches"):
            return self._fit_lazy(x, y, val_x=val_x, val_y=val_y)
        x_train = self._preprocess(x, fit=True)
        x_val = self._preprocess(val_x, fit=False) if val_x is not None else None
        y_train = _as_array(y).reshape(-1)
        if np.unique(y_train).size < 2:
            self.model_ = _ConstantProbabilityClassifier(float(y_train[0]) if len(y_train) else 0.0)
            self.best_iteration_ = 1
            self.device_ = "cpu"
            self.gpu_fallback_error_ = ""
            return self
        y_val_arr = _as_array(val_y).reshape(-1) if val_y is not None else None
        scale_pos_weight = float((len(y_train) - y_train.sum()) / max(y_train.sum(), 1.0))
        self.gpu_fallback_error_ = ""
        self.device_ = _xgboost_device(self.prefer_gpu)
        nthread = _xgboost_nthread()
        estimator_kwargs = {
            "n_estimators": self.n_estimators,
            "learning_rate": 0.03,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary:logistic",
            "random_state": self.random_state,
            "n_jobs": nthread,
            "tree_method": "hist",
            "device": self.device_,
            "eval_metric": "logloss",
            "scale_pos_weight": max(scale_pos_weight, 1.0),
            "early_stopping_rounds": self.patience if x_val is not None and y_val_arr is not None else None,
        }
        estimator = XGBClassifier(**estimator_kwargs)
        try:
            if x_val is not None and y_val_arr is not None:
                estimator.fit(x_train, y_train, eval_set=[(x_val, y_val_arr)], verbose=False)
            else:
                estimator.fit(x_train, y_train, verbose=False)
        except Exception as exc:
            if self.device_ != "cuda":
                raise
            self.gpu_fallback_error_ = f"{type(exc).__name__}: {exc}"
            self.device_ = "cpu"
            estimator = XGBClassifier(**{**estimator_kwargs, "device": self.device_})
            if x_val is not None and y_val_arr is not None:
                estimator.fit(x_train, y_train, eval_set=[(x_val, y_val_arr)], verbose=False)
            else:
                estimator.fit(x_train, y_train, verbose=False)
        self.model_ = estimator
        best_iteration = getattr(estimator, "best_iteration", None)
        if best_iteration is None:
            best_iteration = getattr(estimator, "best_ntree_limit", None)
        self.best_iteration_ = int(best_iteration or self.n_estimators)
        return self

    def refit_full(self, x: pd.DataFrame, y: pd.DataFrame) -> "XGBoostClassifier":
        from xgboost import XGBClassifier
        import xgboost as xgb

        if hasattr(x, "iter_numpy_batches"):
            y_full = _as_array(y).reshape(-1).astype(np.float32)
            if np.unique(y_full).size < 2:
                self.model_ = _ConstantProbabilityClassifier(float(y_full[0]) if len(y_full) else 0.0)
                return self
            scale_pos_weight = float((len(y_full) - y_full.sum()) / max(y_full.sum(), 1.0))
            nthread = _xgboost_nthread()
            params = {
                "learning_rate": 0.03,
                "max_depth": 4,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "objective": "binary:logistic",
                "tree_method": "hist",
                "device": getattr(self, "device_", "cpu"),
                "eval_metric": "logloss",
                "scale_pos_weight": max(scale_pos_weight, 1.0),
                "seed": self.random_state,
                "nthread": nthread,
            }
            rounds = int(getattr(self, "best_iteration_", self.n_estimators))
            if _xgboost_training_mode(len(x)) == "chunked":
                self.model_, self.best_iteration_ = self._train_lazy_chunks(
                    x,
                    y_full,
                    params=params,
                    rounds=rounds,
                    phase="refit_full",
                    start_step="xgboost_classifier_refit_start",
                    chunk_step="xgboost_classifier_refit_chunk_train",
                    complete_step="xgboost_classifier_refit_complete",
                )
                return self
            dtrain = self._lazy_matrix(x, y_full, phase="refit_full")
            print(
                json.dumps(
                    {
                        "step": "xgboost_classifier_refit_start",
                        "rows": len(x),
                        "features": x.shape[1],
                        "n_estimators": rounds,
                        "device": params["device"],
                        "nthread": nthread,
                    }
                ),
                flush=True,
            )
            self.model_ = xgb.train(params, dtrain, num_boost_round=rounds, verbose_eval=False)
            print(
                json.dumps(
                    {
                        "step": "xgboost_classifier_refit_complete",
                        "device": params["device"],
                    }
                ),
                flush=True,
            )
            return self
        x_full = self._preprocess(x, fit=True)
        y_full = _as_array(y).reshape(-1)
        if np.unique(y_full).size < 2:
            self.model_ = _ConstantProbabilityClassifier(float(y_full[0]) if len(y_full) else 0.0)
            return self
        scale_pos_weight = float((len(y_full) - y_full.sum()) / max(y_full.sum(), 1.0))
        nthread = _xgboost_nthread()
        estimator_kwargs = {
            "n_estimators": int(getattr(self, "best_iteration_", self.n_estimators)),
            "learning_rate": 0.03,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary:logistic",
            "random_state": self.random_state,
            "n_jobs": nthread,
            "tree_method": "hist",
            "device": getattr(self, "device_", "cpu"),
            "eval_metric": "logloss",
            "scale_pos_weight": max(scale_pos_weight, 1.0),
        }
        self.model_ = XGBClassifier(**estimator_kwargs)
        try:
            self.model_.fit(x_full, y_full, verbose=False)
        except Exception as exc:
            if estimator_kwargs["device"] != "cuda":
                raise
            self.gpu_fallback_error_ = f"{type(exc).__name__}: {exc}"
            self.device_ = "cpu"
            self.model_ = XGBClassifier(**{**estimator_kwargs, "device": self.device_})
            self.model_.fit(x_full, y_full, verbose=False)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        import xgboost as xgb

        if hasattr(x, "iter_numpy_batches") and not hasattr(self.model_, "predict_proba"):
            outputs: list[np.ndarray] = []
            try:
                batch_size = max(int(os.environ.get("V1_XGBOOST_PREDICT_BATCH_ROWS", "262144")), 1)
            except ValueError:
                batch_size = 262_144
            for _, batch in x.iter_numpy_batches(batch_size=batch_size, shuffle=False):
                values = np.nan_to_num(batch.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
                outputs.append(np.asarray(self.model_.predict(xgb.DMatrix(values)), dtype=np.float64).reshape(-1, 1))
            if not outputs:
                return np.empty((0, 1), dtype=np.float64)
            return np.vstack(outputs)
        x_values = self._preprocess(x, fit=False)
        return self.model_.predict_proba(x_values)[:, 1:2].astype(np.float64)


class SklearnMLPClassifier:
    def __init__(
        self,
        *,
        hidden_layer_sizes: tuple[int, ...] = (96, 48),
        learning_rate_init: float = 0.001,
        max_epochs: int = 80,
        patience: int = 8,
        tol: float = 1e-4,
        random_state: int = 53,
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate_init = learning_rate_init
        self.max_epochs = max_epochs
        self.patience = patience
        self.tol = tol
        self.random_state = random_state

    def _build_model(self):
        from sklearn.neural_network import MLPClassifier

        return MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation="relu",
            learning_rate_init=self.learning_rate_init,
            max_iter=1,
            shuffle=True,
            warm_start=True,
            random_state=self.random_state,
        )

    def _prepare(self, x: pd.DataFrame, *, fit: bool) -> np.ndarray:
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        if fit:
            self.imputer_ = SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True)
            self.scaler_ = StandardScaler()
            values = self.imputer_.fit_transform(x)
            return self.scaler_.fit_transform(values)
        values = self.imputer_.transform(x)
        return self.scaler_.transform(values)

    def fit(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame,
        *,
        val_x: pd.DataFrame | None = None,
        val_y: pd.DataFrame | None = None,
    ) -> "SklearnMLPClassifier":
        from sklearn.metrics import log_loss

        x_train = self._prepare(x, fit=True)
        y_train = _as_array(y).reshape(-1)
        if np.unique(y_train).size < 2:
            self.model_ = _ConstantProbabilityClassifier(float(y_train[0]) if len(y_train) else 0.0)
            self.best_epoch_ = 1
            return self
        x_val = self._prepare(val_x, fit=False) if val_x is not None else None
        y_val_arr = _as_array(val_y).reshape(-1) if val_y is not None else None
        self.model_ = self._build_model()
        best_state: tuple[list[np.ndarray], list[np.ndarray]] | None = None
        best_metric = np.inf
        best_epoch = 0
        epochs_without_improvement = 0
        for epoch in range(self.max_epochs):
            if epoch == 0:
                self.model_.partial_fit(x_train, y_train, classes=np.array([0.0, 1.0]))
            else:
                self.model_.partial_fit(x_train, y_train)
            score_x = x_val if x_val is not None else x_train
            score_y = y_val_arr if y_val_arr is not None else y_train
            prob = np.clip(self.model_.predict_proba(score_x)[:, 1], EPS, 1.0 - EPS)
            metric = float(log_loss(score_y, prob, labels=[0.0, 1.0]))
            if metric + self.tol < best_metric:
                best_metric = metric
                best_epoch = epoch + 1
                best_state = (
                    [coef.copy() for coef in self.model_.coefs_],
                    [intercept.copy() for intercept in self.model_.intercepts_],
                )
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if x_val is not None and epochs_without_improvement >= self.patience:
                break
        if best_state is not None:
            self.model_.coefs_ = [coef.copy() for coef in best_state[0]]
            self.model_.intercepts_ = [intercept.copy() for intercept in best_state[1]]
        self.best_epoch_ = max(best_epoch, 1)
        return self

    def refit_full(self, x: pd.DataFrame, y: pd.DataFrame) -> "SklearnMLPClassifier":
        x_full = self._prepare(x, fit=True)
        y_full = _as_array(y).reshape(-1)
        if np.unique(y_full).size < 2:
            self.model_ = _ConstantProbabilityClassifier(float(y_full[0]) if len(y_full) else 0.0)
            return self
        self.model_ = self._build_model()
        for epoch in range(max(getattr(self, "best_epoch_", self.max_epochs), 1)):
            if epoch == 0:
                self.model_.partial_fit(x_full, y_full, classes=np.array([0.0, 1.0]))
            else:
                self.model_.partial_fit(x_full, y_full)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        x_values = self._prepare(x, fit=False)
        return self.model_.predict_proba(x_values)[:, 1:2].astype(np.float64)


class TorchMLPClassifier:
    def __init__(
        self,
        *,
        hidden_units: int = 128,
        learning_rate: float = 0.001,
        max_epochs: int = 80,
        patience: int = 10,
        batch_size: int = 512,
        random_state: int = 59,
    ) -> None:
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.random_state = random_state
        self.device = _default_torch_device()
        self.x_scaler = Standardizer()

    def _resolve_device(self):
        import torch

        requested = self.device if self.device != "cuda" or _torch_cuda_available() else "cpu"
        self.device_ = torch.device(requested)
        if self.device_.type == "cuda":
            torch.backends.cudnn.benchmark = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
        return self.device_

    def _build_model(self, input_dim: int):
        import torch

        return torch.nn.Sequential(
            torch.nn.Linear(input_dim, self.hidden_units),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.05),
            torch.nn.Linear(self.hidden_units, max(self.hidden_units // 2, 16)),
            torch.nn.ReLU(),
            torch.nn.Linear(max(self.hidden_units // 2, 16), 1),
        )

    def _fit_fixed_epochs(self, x_values: np.ndarray, y_values: np.ndarray, epochs: int) -> None:
        import torch

        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(x_values.astype(np.float32)),
            torch.from_numpy(y_values.astype(np.float32)),
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            **_torch_loader_kwargs(device=self.device_.type, shuffle=True, batch_size=min(self.batch_size, len(dataset))),
        )
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        self.model_.train()
        total_batches = len(loader)
        progress_every = _torch_progress_every(total_batches)
        start_time = time.monotonic()
        for epoch in range(max(epochs, 1)):
            for batch_index, (batch_x, batch_y) in enumerate(loader, start=1):
                pred = self.model_(_move_tensor(batch_x, self.device_))
                loss = loss_fn(pred, _move_tensor(batch_y, self.device_))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if batch_index == 1 or batch_index % progress_every == 0 or batch_index == total_batches:
                    print(
                        json.dumps(
                            {
                                "step": "torch_mlp_classifier_batch",
                                "phase": "fixed_epochs",
                                "epoch": epoch + 1,
                                "max_epochs": max(epochs, 1),
                                "batch": batch_index,
                                "total_batches": total_batches,
                                "rows_seen": min(batch_index * self.batch_size, len(dataset)),
                                "rows_total": len(dataset),
                                "loss": float(loss.detach().cpu().item()),
                                "elapsed_seconds": round(time.monotonic() - start_time, 1),
                                "device": str(self.device_),
                            }
                        ),
                        flush=True,
                    )

    def _lazy_logloss(self, x, y_values: np.ndarray, mean_t, scale_t) -> float:
        import torch

        loss_fn = torch.nn.BCEWithLogitsLoss()
        losses: list[float] = []
        self.model_.eval()
        with torch.no_grad():
            for local_rows, batch in x.iter_numpy_batches(batch_size=self.batch_size, shuffle=False):
                xb = torch.from_numpy(batch.astype(np.float32)).to(self.device_)
                xb = (torch.nan_to_num(xb, nan=0.0) - mean_t) / scale_t
                yb = torch.from_numpy(y_values[local_rows]).to(self.device_)
                losses.append(float(loss_fn(self.model_(xb), yb).item()))
        return float(np.mean(losses)) if losses else np.inf

    def _fit_lazy(
        self,
        x,
        y: pd.DataFrame,
        *,
        val_x=None,
        val_y: pd.DataFrame | None = None,
    ) -> "TorchMLPClassifier":
        import torch

        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        self._resolve_device()
        y_train = _as_array(y).reshape(-1, 1).astype(np.float32)
        if np.unique(y_train).size < 2:
            self.constant_probability_ = float(y_train[0, 0]) if len(y_train) else 0.0
            self.best_epoch_ = 1
            return self
        self.x_scaler.fit_batches(_tabular_scaler_batches(x, phase="fit"), width=x.shape[1])
        y_val_arr = _as_array(val_y).reshape(-1, 1).astype(np.float32) if val_y is not None else None
        self.model_ = self._build_model(x.shape[1]).to(self.device_)
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        mean_t = torch.from_numpy(self.x_scaler.mean_.astype(np.float32)).to(self.device_)
        scale_t = torch.from_numpy(self.x_scaler.scale_.astype(np.float32)).to(self.device_)
        best_state = copy.deepcopy(self.model_.state_dict())
        best_metric = np.inf
        best_epoch = 1
        epochs_without_improvement = 0
        total_batches = (len(x) + self.batch_size - 1) // self.batch_size if len(x) else 0
        progress_every = _torch_progress_every(max(total_batches, 1))
        start_time = time.monotonic()
        for epoch in range(self.max_epochs):
            self.model_.train()
            for batch_index, (local_rows, batch) in enumerate(
                x.iter_numpy_batches(
                batch_size=self.batch_size,
                shuffle=True,
                random_state=self.random_state + epoch,
                ),
                start=1,
            ):
                xb = torch.from_numpy(batch.astype(np.float32)).to(self.device_)
                xb = (torch.nan_to_num(xb, nan=0.0) - mean_t) / scale_t
                yb = torch.from_numpy(y_train[local_rows]).to(self.device_)
                logits = self.model_(xb)
                loss = loss_fn(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if batch_index == 1 or batch_index % progress_every == 0 or batch_index == total_batches:
                    print(
                        json.dumps(
                            {
                                "step": "torch_mlp_classifier_batch",
                                "phase": "fit",
                                "epoch": epoch + 1,
                                "max_epochs": self.max_epochs,
                                "batch": batch_index,
                                "total_batches": total_batches,
                                "rows_seen": min(batch_index * self.batch_size, len(x)),
                                "rows_total": len(x),
                                "loss": float(loss.detach().cpu().item()),
                                "elapsed_seconds": round(time.monotonic() - start_time, 1),
                                "device": str(self.device_),
                            }
                        ),
                        flush=True,
                    )
            metric = (
                self._lazy_logloss(val_x, y_val_arr, mean_t, scale_t)
                if val_x is not None and y_val_arr is not None
                else self._lazy_logloss(x, y_train, mean_t, scale_t)
            )
            if metric + 1e-6 < best_metric:
                best_metric = metric
                best_state = copy.deepcopy(self.model_.state_dict())
                best_epoch = epoch + 1
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            print(
                json.dumps(
                    {
                        "step": "torch_mlp_classifier_epoch",
                        "epoch": epoch + 1,
                        "max_epochs": self.max_epochs,
                        "best_epoch": best_epoch,
                        "metric": best_metric,
                        "device": str(self.device_),
                    }
                ),
                flush=True,
            )
            if val_x is not None and epochs_without_improvement >= self.patience:
                break
        self.model_.load_state_dict(best_state)
        self.best_epoch_ = best_epoch
        return self

    def fit(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame,
        *,
        val_x: pd.DataFrame | None = None,
        val_y: pd.DataFrame | None = None,
    ) -> "TorchMLPClassifier":
        import torch

        if hasattr(x, "iter_numpy_batches"):
            return self._fit_lazy(x, y, val_x=val_x, val_y=val_y)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        self._resolve_device()
        x_train = self.x_scaler.fit(_as_array(x)).transform(_as_array(x)).astype(np.float32)
        y_train = _as_array(y).reshape(-1, 1).astype(np.float32)
        if np.unique(y_train).size < 2:
            self.constant_probability_ = float(y_train[0, 0]) if len(y_train) else 0.0
            self.best_epoch_ = 1
            return self
        x_val = self.x_scaler.transform(_as_array(val_x)).astype(np.float32) if val_x is not None else None
        y_val_arr = _as_array(val_y).reshape(-1, 1).astype(np.float32) if val_y is not None else None
        self.model_ = self._build_model(x_train.shape[1]).to(self.device_)
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
        loader = torch.utils.data.DataLoader(
            dataset,
            **_torch_loader_kwargs(device=self.device_.type, shuffle=True, batch_size=min(self.batch_size, len(dataset))),
        )
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        best_state = copy.deepcopy(self.model_.state_dict())
        best_metric = np.inf
        best_epoch = 1
        epochs_without_improvement = 0
        for epoch in range(self.max_epochs):
            self.model_.train()
            for batch_x, batch_y in loader:
                pred = self.model_(_move_tensor(batch_x, self.device_))
                loss = loss_fn(pred, _move_tensor(batch_y, self.device_))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            metric_x = x_val if x_val is not None else x_train
            metric_y = y_val_arr if y_val_arr is not None else y_train
            self.model_.eval()
            with torch.no_grad():
                logits = self.model_(_move_tensor(torch.from_numpy(metric_x), self.device_)).detach().cpu().numpy()
            prob = 1.0 / (1.0 + np.exp(-logits))
            metric = float(np.nanmean(-(metric_y * np.log(np.clip(prob, EPS, 1.0 - EPS)) + (1.0 - metric_y) * np.log(np.clip(1.0 - prob, EPS, 1.0 - EPS)))))
            if metric + 1e-6 < best_metric:
                best_metric = metric
                best_state = copy.deepcopy(self.model_.state_dict())
                best_epoch = epoch + 1
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if x_val is not None and epochs_without_improvement >= self.patience:
                break
        self.model_.load_state_dict(best_state)
        self.best_epoch_ = best_epoch
        return self

    def refit_full(self, x: pd.DataFrame, y: pd.DataFrame) -> "TorchMLPClassifier":
        if hasattr(x, "iter_numpy_batches"):
            previous_epochs = int(getattr(self, "best_epoch_", self.max_epochs))
            old_max_epochs = self.max_epochs
            old_patience = self.patience
            self.max_epochs = max(previous_epochs, 1)
            self.patience = max(previous_epochs + 1, 1)
            try:
                return self._fit_lazy(x, y)
            finally:
                self.max_epochs = old_max_epochs
                self.patience = old_patience
        x_full = self.x_scaler.fit(_as_array(x)).transform(_as_array(x)).astype(np.float32)
        y_full = _as_array(y).reshape(-1, 1).astype(np.float32)
        if np.unique(y_full).size < 2:
            self.constant_probability_ = float(y_full[0, 0]) if len(y_full) else 0.0
            return self
        self._resolve_device()
        self.model_ = self._build_model(x_full.shape[1]).to(self.device_)
        self._fit_fixed_epochs(x_full, y_full, getattr(self, "best_epoch_", self.max_epochs))
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        import torch

        if hasattr(self, "constant_probability_"):
            return np.full((len(x), 1), float(self.constant_probability_), dtype=np.float64)
        self._resolve_device()
        self.model_ = self.model_.to(self.device_)
        self.model_.eval()
        if hasattr(x, "iter_numpy_batches"):
            outputs: list[np.ndarray] = []
            mean_t = torch.from_numpy(self.x_scaler.mean_.astype(np.float32)).to(self.device_)
            scale_t = torch.from_numpy(self.x_scaler.scale_.astype(np.float32)).to(self.device_)
            with torch.no_grad():
                for _, batch in x.iter_numpy_batches(batch_size=self.batch_size, shuffle=False):
                    xb = torch.from_numpy(batch.astype(np.float32)).to(self.device_)
                    xb = (torch.nan_to_num(xb, nan=0.0) - mean_t) / scale_t
                    logits = self.model_(xb).detach().cpu().numpy()
                    outputs.append(1.0 / (1.0 + np.exp(-logits)))
            if not outputs:
                return np.empty((0, 1), dtype=np.float64)
            return np.vstack(outputs).astype(np.float64)
        x_values = self.x_scaler.transform(_as_array(x)).astype(np.float32)
        with torch.no_grad():
            logits = self.model_(_move_tensor(torch.from_numpy(x_values), self.device_)).detach().cpu().numpy()
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float64)

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        if state.get("model_") is not None:
            state["model_"] = _copy_model_to_cpu(state["model_"])
        state["device_"] = "cpu"
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self._resolve_device()
        if getattr(self, "model_", None) is not None:
            self.model_ = self.model_.to(self.device_)


class TorchSequenceStaticClassifier:
    def __init__(
        self,
        *,
        window_length: int = 60,
        hidden_dim: int = 128,
        learning_rate: float = 0.001,
        max_epochs: int = 20,
        patience: int = 4,
        batch_size: int = 512,
        random_state: int = 61,
    ) -> None:
        self.window_length = window_length
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.random_state = random_state
        self.device = _default_torch_device()
        self.x_scaler = Standardizer()

    def _resolve_device(self):
        import torch

        requested = self.device if self.device != "cuda" or _torch_cuda_available() else "cpu"
        self.device_ = torch.device(requested)
        if self.device_.type == "cuda":
            torch.backends.cudnn.benchmark = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
        return self.device_

    def _prepare_sequence_tensor(self, sequence, mean_t, scale_t):
        import torch

        mean_view = mean_t.view(1, 1, -1)
        scale_view = scale_t.view(1, 1, -1)
        sequence = sequence.to(dtype=torch.float32)
        sequence = torch.where(torch.isfinite(sequence), sequence, mean_view)
        return (sequence - mean_view) / scale_view

    def _train_cutoff_date(self, metadata: pd.DataFrame) -> str:
        dates = metadata["anchor_date"].dropna().astype(str)
        if dates.empty:
            raise ValueError("Sequence/static training metadata is empty.")
        return str(dates.max())

    def _build_model(self, *, sequence_dim: int, static_cardinalities: Sequence[int]):
        return _SequenceStaticNet(
            sequence_dim=sequence_dim,
            static_cardinalities=static_cardinalities,
            output_dim=1,
            window_length=self.window_length,
            hidden_dim=self.hidden_dim,
        )

    def _dataset_logloss(self, dataset: _SequenceStaticDataset) -> float:
        import torch

        loader = _sequence_data_loader(
            dataset,
            device=self.device_.type,
            shuffle=False,
            batch_size=self.batch_size,
        )
        loss_fn = torch.nn.BCEWithLogitsLoss()
        mean_t = torch.from_numpy(self.x_scaler.mean_.astype(np.float32)).to(self.device_)
        scale_t = torch.from_numpy(self.x_scaler.scale_.astype(np.float32)).to(self.device_)
        self.model_.eval()
        losses: list[float] = []
        with torch.no_grad():
            for sequence, static_ids, target in loader:
                sequence = self._prepare_sequence_tensor(_move_tensor(sequence, self.device_), mean_t, scale_t)
                logits = self.model_(sequence, _move_tensor(static_ids.long(), self.device_))
                losses.append(float(loss_fn(logits, _move_tensor(target.float(), self.device_)).item()))
        return float(np.mean(losses)) if losses else np.inf

    def _train_epochs(self, dataset: _SequenceStaticDataset, epochs: int) -> None:
        import torch

        loader = _sequence_data_loader(
            dataset,
            device=self.device_.type,
            shuffle=True,
            batch_size=self.batch_size,
            random_state=self.random_state,
        )
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        mean_t = torch.from_numpy(self.x_scaler.mean_.astype(np.float32)).to(self.device_)
        scale_t = torch.from_numpy(self.x_scaler.scale_.astype(np.float32)).to(self.device_)
        self.model_.train()
        total_batches = len(loader)
        progress_every = _torch_progress_every(total_batches)
        start_time = time.monotonic()
        for epoch in range(max(epochs, 1)):
            for batch_index, (sequence, static_ids, target) in enumerate(loader, start=1):
                sequence = self._prepare_sequence_tensor(_move_tensor(sequence, self.device_), mean_t, scale_t)
                logits = self.model_(sequence, _move_tensor(static_ids.long(), self.device_))
                loss = loss_fn(logits, _move_tensor(target.float(), self.device_))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if batch_index == 1 or batch_index % progress_every == 0 or batch_index == total_batches:
                    print(
                        json.dumps(
                            {
                                "step": "torch_seq_static_classifier_batch",
                                "phase": "refit_full",
                                "epoch": epoch + 1,
                                "max_epochs": max(epochs, 1),
                                "batch": batch_index,
                                "total_batches": total_batches,
                                "rows_seen": min(batch_index * self.batch_size, len(dataset)),
                                "rows_total": len(dataset),
                                "loss": float(loss.detach().cpu().item()),
                                "elapsed_seconds": round(time.monotonic() - start_time, 1),
                                "device": str(self.device_),
                            }
                        ),
                        flush=True,
                    )

    def fit(
        self,
        x: dict[str, object],
        y: pd.DataFrame,
        *,
        val_x: dict[str, object] | None = None,
        val_y: pd.DataFrame | None = None,
    ) -> "TorchSequenceStaticClassifier":
        import torch

        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        self._resolve_device()
        self.static_columns_ = list(x["static_categorical"].keys())
        self.static_vocab_sizes_ = [
            int(np.max(x["static_categorical"][column])) + 1 if len(x["static_categorical"][column]) else 1
            for column in self.static_columns_
        ]
        _fit_sequence_store_standardizer(
            self.x_scaler,
            x["store"],
            self._train_cutoff_date(x["metadata"]),
            width=len(x["store"].feature_columns),
        )
        y_train = _as_array(y).reshape(-1, 1).astype(np.float32)
        if np.unique(y_train).size < 2:
            self.constant_probability_ = float(y_train[0, 0]) if len(y_train) else 0.0
            self.best_epoch_ = 1
            return self
        y_val_arr = _as_array(val_y).reshape(-1, 1).astype(np.float32) if val_y is not None else None
        train_dataset = _SequenceStaticDataset(
            store=x["store"],
            metadata=x["metadata"],
            static_categorical=x["static_categorical"],
            target_frame=pd.DataFrame(y_train, columns=y.columns),
            window_length=self.window_length,
            static_columns=self.static_columns_,
        )
        val_dataset = None
        if val_x is not None and val_y is not None:
            val_dataset = _SequenceStaticDataset(
                store=val_x["store"],
                metadata=val_x["metadata"],
                static_categorical=val_x["static_categorical"],
                target_frame=pd.DataFrame(y_val_arr, columns=val_y.columns),
                window_length=self.window_length,
                static_columns=self.static_columns_,
            )
        self.model_ = self._build_model(
            sequence_dim=len(x["store"].feature_columns),
            static_cardinalities=self.static_vocab_sizes_,
        ).to(self.device_)
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        train_loader = _sequence_data_loader(
            train_dataset,
            device=self.device_.type,
            shuffle=True,
            batch_size=self.batch_size,
            random_state=self.random_state,
        )
        mean_t = torch.from_numpy(self.x_scaler.mean_.astype(np.float32)).to(self.device_)
        scale_t = torch.from_numpy(self.x_scaler.scale_.astype(np.float32)).to(self.device_)
        best_state = copy.deepcopy(self.model_.state_dict())
        best_metric = np.inf
        best_epoch = 1
        epochs_without_improvement = 0
        total_batches = len(train_loader)
        progress_every = _torch_progress_every(total_batches)
        start_time = time.monotonic()
        for epoch in range(self.max_epochs):
            self.model_.train()
            for batch_index, (sequence, static_ids, target) in enumerate(train_loader, start=1):
                sequence = self._prepare_sequence_tensor(_move_tensor(sequence, self.device_), mean_t, scale_t)
                logits = self.model_(sequence, _move_tensor(static_ids.long(), self.device_))
                loss = loss_fn(logits, _move_tensor(target.float(), self.device_))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if batch_index == 1 or batch_index % progress_every == 0 or batch_index == total_batches:
                    print(
                        json.dumps(
                            {
                                "step": "torch_seq_static_classifier_batch",
                                "phase": "fit",
                                "epoch": epoch + 1,
                                "max_epochs": self.max_epochs,
                                "batch": batch_index,
                                "total_batches": total_batches,
                                "rows_seen": min(batch_index * self.batch_size, len(train_dataset)),
                                "rows_total": len(train_dataset),
                                "loss": float(loss.detach().cpu().item()),
                                "elapsed_seconds": round(time.monotonic() - start_time, 1),
                                "device": str(self.device_),
                            }
                        ),
                        flush=True,
                    )
            metric = self._dataset_logloss(val_dataset if val_dataset is not None else train_dataset)
            if metric + 1e-6 < best_metric:
                best_metric = metric
                best_state = copy.deepcopy(self.model_.state_dict())
                best_epoch = epoch + 1
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            print(
                json.dumps(
                    {
                        "step": "torch_seq_static_classifier_epoch",
                        "epoch": epoch + 1,
                        "max_epochs": self.max_epochs,
                        "best_epoch": best_epoch,
                        "metric": best_metric,
                        "device": str(self.device_),
                    }
                ),
                flush=True,
            )
            if val_dataset is not None and epochs_without_improvement >= self.patience:
                break
        self.model_.load_state_dict(best_state)
        self.best_epoch_ = best_epoch
        return self

    def refit_full(self, x: dict[str, object], y: pd.DataFrame) -> "TorchSequenceStaticClassifier":
        self._resolve_device()
        _fit_sequence_store_standardizer(
            self.x_scaler,
            x["store"],
            self._train_cutoff_date(x["metadata"]),
            width=len(x["store"].feature_columns),
        )
        y_train = _as_array(y).reshape(-1, 1).astype(np.float32)
        if np.unique(y_train).size < 2:
            self.constant_probability_ = float(y_train[0, 0]) if len(y_train) else 0.0
            return self
        train_dataset = _SequenceStaticDataset(
            store=x["store"],
            metadata=x["metadata"],
            static_categorical=x["static_categorical"],
            target_frame=pd.DataFrame(y_train, columns=y.columns),
            window_length=self.window_length,
            static_columns=self.static_columns_,
        )
        self.model_ = self._build_model(
            sequence_dim=len(x["store"].feature_columns),
            static_cardinalities=self.static_vocab_sizes_,
        ).to(self.device_)
        self._train_epochs(train_dataset, getattr(self, "best_epoch_", self.max_epochs))
        return self

    def predict(self, x: dict[str, object]) -> np.ndarray:
        import torch

        dataset = _SequenceStaticDataset(
            store=x["store"],
            metadata=x["metadata"],
            static_categorical=x["static_categorical"],
            target_frame=None,
            window_length=self.window_length,
            static_columns=self.static_columns_,
        )
        if hasattr(self, "constant_probability_"):
            return np.full((len(dataset), 1), float(self.constant_probability_), dtype=np.float64)
        self._resolve_device()
        self.model_ = self.model_.to(self.device_)
        loader = _sequence_data_loader(
            dataset,
            device=self.device_.type,
            shuffle=False,
            batch_size=self.batch_size,
        )
        mean_t = torch.from_numpy(self.x_scaler.mean_.astype(np.float32)).to(self.device_)
        scale_t = torch.from_numpy(self.x_scaler.scale_.astype(np.float32)).to(self.device_)
        outputs: list[np.ndarray] = []
        self.model_.eval()
        with torch.no_grad():
            for sequence, static_ids in loader:
                sequence = self._prepare_sequence_tensor(_move_tensor(sequence, self.device_), mean_t, scale_t)
                logits = self.model_(sequence, _move_tensor(static_ids.long(), self.device_)).detach().cpu().numpy()
                outputs.append(1.0 / (1.0 + np.exp(-logits)))
        if not outputs:
            return np.empty((0, 1), dtype=np.float64)
        return np.vstack(outputs).astype(np.float64)

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        if state.get("model_") is not None:
            state["model_"] = _copy_model_to_cpu(state["model_"])
        state["device_"] = "cpu"
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self._resolve_device()
        if getattr(self, "model_", None) is not None:
            self.model_ = self.model_.to(self.device_)


def is_sequence_static_model(model_name: str) -> bool:
    return model_name in {SEQUENCE_STATIC_MODEL_NAME, SEQUENCE_STATIC_CLASSIFICATION_MODEL_NAME}


def default_model_names(eval_mode: str = "walk_forward", *, task_type: str = "regression") -> list[str]:
    if task_type == "classification":
        names = list(CLASSIFICATION_MODEL_NAMES)
        if eval_mode == "walk_forward":
            names.append(SEQUENCE_STATIC_CLASSIFICATION_MODEL_NAME)
        return names
    names = list(BASELINE_MODEL_NAMES)
    if eval_mode == "walk_forward":
        names.append(SEQUENCE_STATIC_MODEL_NAME)
    return names


def available_model_names() -> list[str]:
    return default_model_names(eval_mode="walk_forward") + default_model_names(
        eval_mode="walk_forward",
        task_type="classification",
    )


def make_model(
    model_name: str,
    *,
    window_length: int = 60,
    task_type: str = "regression",
    model_kwargs: dict[str, object] | None = None,
) -> Predictor:
    kwargs = dict(model_kwargs or {})
    if task_type == "classification":
        if model_name == "logistic_regression":
            return LogisticClassifier()
        if model_name == "elastic_net_classifier":
            return ElasticNetLogisticClassifier()
        if model_name == "lightgbm_classifier":
            return LightGBMClassifier()
        if model_name == "xgboost_classifier":
            return XGBoostClassifier()
        if model_name == "sklearn_mlp_classifier":
            return SklearnMLPClassifier()
        if model_name == "torch_mlp_classifier":
            return TorchMLPClassifier(**kwargs)
        if model_name == SEQUENCE_STATIC_CLASSIFICATION_MODEL_NAME:
            return TorchSequenceStaticClassifier(window_length=window_length, **kwargs)
        raise ValueError(f"Unknown classification model: {model_name}")
    if model_name == "zero":
        return ZeroPredictor()
    if model_name == "mean":
        return MeanPredictor()
    if model_name == "momentum_heuristic":
        return MomentumHeuristicPredictor()
    if model_name == "ridge":
        return SklearnRegressor("sklearn_ridge")
    if model_name == "sklearn_ridge":
        return SklearnRegressor("sklearn_ridge")
    if model_name == "elastic_net":
        return SklearnRegressor("sklearn_elastic_net")
    if model_name == "sklearn_elastic_net":
        return SklearnRegressor("sklearn_elastic_net")
    if model_name == "sklearn_hist_gb":
        return SklearnRegressor("sklearn_hist_gb")
    if model_name == "lightgbm":
        return LightGBMRegressor()
    if model_name == "xgboost":
        return XGBoostRegressor()
    if model_name == "mlp":
        return TorchMLPRegressor()
    if model_name == "sklearn_mlp":
        return SklearnMLPPredictor()
    if model_name == "torch_mlp":
        return TorchMLPRegressor(**kwargs)
    if model_name == SEQUENCE_STATIC_MODEL_NAME:
        return TorchSequenceStaticRegressor(window_length=window_length, **kwargs)
    raise ValueError(f"Unknown model: {model_name}")


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return None
    xr = _rank(x[mask])
    yr = _rank(y[mask])
    if np.std(xr) < EPS or np.std(yr) < EPS:
        return None
    return float(np.corrcoef(xr, yr)[0, 1])


def evaluate_predictions(
    metadata: pd.DataFrame,
    y_true: pd.DataFrame,
    y_pred: np.ndarray,
    *,
    target_columns: Sequence[str],
    model_name: str,
    feature_set: str,
    split_name: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    true_values = y_true[target_columns].astype(float).to_numpy()
    y_pred = _as_2d_prediction_array(y_pred, expected_columns=len(target_columns))
    for idx, target in enumerate(target_columns):
        actual = true_values[:, idx]
        pred = y_pred[:, idx]
        err = pred - actual
        rank_ics: list[float] = []
        spreads: list[float] = []
        for _, group_idx in metadata.groupby("anchor_date").indices.items():
            group_idx = np.asarray(group_idx)
            if len(group_idx) < 10:
                continue
            ic = _spearman(pred[group_idx], actual[group_idx])
            if ic is not None:
                rank_ics.append(ic)
            group_pred = pred[group_idx]
            group_actual = actual[group_idx]
            q = max(1, int(len(group_idx) * 0.2))
            order = np.argsort(group_pred)
            bottom = group_actual[order[:q]]
            top = group_actual[order[-q:]]
            if np.isfinite(top).any() and np.isfinite(bottom).any():
                spreads.append(float(np.nanmean(top) - np.nanmean(bottom)))
        rows.append(
            {
                "model_name": model_name,
                "feature_set": feature_set,
                "split": split_name,
                "target": target,
                "rmse": float(np.sqrt(np.nanmean(err**2))),
                "mae": float(np.nanmean(np.abs(err))),
                "rank_ic": float(np.nanmean(rank_ics)) if rank_ics else None,
                "directional_hit_rate": float(np.nanmean(np.sign(pred) == np.sign(actual))),
                "top_bottom_spread": float(np.nanmean(spreads)) if spreads else None,
                "row_count": int(len(actual)),
                "date_count": int(metadata["anchor_date"].nunique()),
            }
        )
    return rows


def evaluate_classification_predictions(
    metadata: pd.DataFrame,
    y_true: pd.DataFrame,
    y_score: np.ndarray,
    *,
    target_columns: Sequence[str],
    model_name: str,
    feature_set: str,
    split_name: str,
    realized_returns: pd.DataFrame | None = None,
    realized_return_column: str | None = None,
) -> list[dict[str, object]]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    rows: list[dict[str, object]] = []
    true_values = y_true[target_columns].astype(float).to_numpy()
    y_score = _as_2d_prediction_array(y_score, expected_columns=len(target_columns))
    realized = None
    if realized_returns is not None and realized_return_column and realized_return_column in realized_returns.columns:
        realized = realized_returns[realized_return_column].astype(float).to_numpy()
    for idx, target in enumerate(target_columns):
        actual = true_values[:, idx]
        score = np.clip(y_score[:, idx], EPS, 1.0 - EPS)
        pred_label = (score >= 0.5).astype(float)
        try:
            pr_auc = float(average_precision_score(actual, score))
        except Exception:
            pr_auc = None
        if np.unique(actual[np.isfinite(actual)]).size >= 2:
            try:
                roc_auc = float(roc_auc_score(actual, score))
            except Exception:
                roc_auc = None
        else:
            roc_auc = None
        top_precisions: list[float] = []
        spreads: list[float] = []
        event_rate_spreads: list[float] = []
        for _, group_idx in metadata.groupby("anchor_date").indices.items():
            group_idx = np.asarray(group_idx)
            if len(group_idx) < 10:
                continue
            q = max(1, int(np.ceil(len(group_idx) * 0.1)))
            order = np.argsort(score[group_idx])
            top_idx = group_idx[order[-q:]]
            bottom_idx = group_idx[order[:q]]
            top_precisions.append(float(np.nanmean(actual[top_idx])))
            event_rate_spreads.append(float(np.nanmean(actual[top_idx]) - np.nanmean(actual[bottom_idx])))
            if realized is not None:
                spreads.append(float(np.nanmean(realized[top_idx]) - np.nanmean(realized[bottom_idx])))
        rows.append(
            {
                "model_name": model_name,
                "feature_set": feature_set,
                "split": split_name,
                "target": target,
                "pr_auc": pr_auc,
                "roc_auc": roc_auc,
                "top_decile_precision": float(np.nanmean(top_precisions)) if top_precisions else None,
                "top_bottom_spread": float(np.nanmean(spreads)) if spreads else None,
                "top_bottom_event_rate_spread": float(np.nanmean(event_rate_spreads)) if event_rate_spreads else None,
                "accuracy": float(np.nanmean(pred_label == actual)),
                "positive_rate": float(np.nanmean(actual)),
                "row_count": int(len(actual)),
                "date_count": int(metadata["anchor_date"].nunique()),
                "realized_return_column": realized_return_column,
            }
        )
    return rows


def prediction_frame(
    metadata: pd.DataFrame,
    y_pred: np.ndarray,
    *,
    target_columns: Sequence[str],
    model_name: str,
    feature_set: str,
    leaderboard_rank: int | None = None,
    recommended: bool = False,
    y_true: pd.DataFrame | None = None,
    task_type: str = "regression",
) -> pd.DataFrame:
    metadata_columns = ["ticker", "anchor_date"]
    y_pred = _as_2d_prediction_array(y_pred, expected_columns=len(target_columns))
    if "anchor_close" in metadata.columns:
        metadata_columns.append("anchor_close")
    out = metadata[metadata_columns].copy()
    out["model_name"] = model_name
    out["feature_set"] = feature_set
    out["task_type"] = task_type
    out["leaderboard_rank"] = leaderboard_rank
    out["recommended"] = recommended
    for idx, target in enumerate(target_columns):
        if task_type == "classification":
            pred_col = f"pred_prob_{target}"
            flag_col = f"pred_flag_{target}"
        else:
            pred_col = target.replace("market_adjusted_return", "pred_market_adjusted_return")
        out[pred_col] = y_pred[:, idx]
        if task_type == "classification":
            out[flag_col] = (y_pred[:, idx] >= 0.5).astype(bool)
        if y_true is not None:
            if task_type == "classification":
                actual_col = f"actual_{target}"
            else:
                actual_col = target.replace("market_adjusted_return", "actual_market_adjusted_return")
            out[actual_col] = y_true[target].to_numpy()
    return out


def build_metric_summary(metrics: pd.DataFrame, *, split_name: str = "val") -> pd.DataFrame:
    val = metrics[metrics["split"] == split_name].copy()
    preferred_targets = [target for target in val["target"].unique() if not target.endswith("_1d")]
    if preferred_targets:
        val = val[val["target"].isin(preferred_targets)]
    grouped = (
        val.groupby(["model_name", "feature_set"], dropna=False)
        .agg(
            mean_rank_ic=("rank_ic", "mean"),
            mean_top_bottom_spread=("top_bottom_spread", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_mae=("mae", "mean"),
        )
        .reset_index()
    )
    grouped["selection_score"] = grouped["mean_rank_ic"].fillna(-999.0) + grouped[
        "mean_top_bottom_spread"
    ].fillna(0.0)
    grouped = grouped.sort_values(
        ["selection_score", "mean_rank_ic", "mean_top_bottom_spread"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    grouped["leaderboard_rank"] = np.arange(1, len(grouped) + 1)
    grouped["recommended"] = grouped["leaderboard_rank"] <= min(3, len(grouped))
    grouped["source_split"] = split_name
    return grouped


def build_leaderboard(metrics: pd.DataFrame, *, split_name: str = "val") -> pd.DataFrame:
    return build_metric_summary(metrics, split_name=split_name)


def build_classification_metric_summary(metrics: pd.DataFrame, *, split_name: str = "val") -> pd.DataFrame:
    val = metrics[metrics["split"] == split_name].copy()
    for column in (
        "pr_auc",
        "roc_auc",
        "top_decile_precision",
        "top_bottom_spread",
        "accuracy",
        "positive_rate",
    ):
        if column in val.columns:
            val[column] = pd.to_numeric(val[column], errors="coerce")
    grouped = (
        val.groupby(["model_name", "feature_set"], dropna=False)
        .agg(
            mean_pr_auc=("pr_auc", "mean"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_top_decile_precision=("top_decile_precision", "mean"),
            mean_top_bottom_spread=("top_bottom_spread", "mean"),
            mean_accuracy=("accuracy", "mean"),
            mean_positive_rate=("positive_rate", "mean"),
        )
        .reset_index()
    )
    grouped["selection_score"] = (
        grouped["mean_pr_auc"].fillna(-999.0)
        + grouped["mean_top_decile_precision"].fillna(0.0)
        + grouped["mean_top_bottom_spread"].fillna(0.0)
    )
    grouped = grouped.sort_values(
        ["selection_score", "mean_pr_auc", "mean_roc_auc", "mean_top_decile_precision", "mean_top_bottom_spread"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    grouped["leaderboard_rank"] = np.arange(1, len(grouped) + 1)
    grouped["recommended"] = grouped["leaderboard_rank"] <= min(3, len(grouped))
    grouped["source_split"] = split_name
    return grouped


def build_classification_leaderboard(metrics: pd.DataFrame, *, split_name: str = "val") -> pd.DataFrame:
    return build_classification_metric_summary(metrics, split_name=split_name)


def save_model_bundle(path: str | Path, *, model: Predictor, metadata: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump({"model": model, "metadata": metadata}, handle)


def load_model_bundle(path: str | Path) -> dict[str, object]:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def write_json(path: str | Path, payload: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
