from __future__ import annotations

import copy
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

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
    return np.asarray(frame, dtype=np.float64)


def _hstack_static_arrays(static_categorical: dict[str, np.ndarray], columns: Sequence[str]) -> np.ndarray:
    if not columns:
        return np.empty((len(next(iter(static_categorical.values()), [])), 0), dtype=np.int64)
    return np.column_stack([static_categorical[column] for column in columns]).astype(np.int64)


def _default_torch_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _gpu_available() -> bool:
    try:
        return _default_torch_device() == "cuda"
    except Exception:
        return False


def _torch_loader_kwargs(*, device: str, shuffle: bool, batch_size: int) -> dict[str, object]:
    return {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "pin_memory": False,
    }


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
            estimator = ElasticNet(alpha=0.0005, l1_ratio=0.2, max_iter=3000, random_state=13)
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
    ) -> "LightGBMRegressor":
        from lightgbm import LGBMRegressor, early_stopping

        x_train = self._preprocess(x, fit=True)
        x_val = self._preprocess(val_x, fit=False) if val_x is not None else None
        y_train = _as_array(y)
        y_val_arr = _as_array(val_y) if val_y is not None else None
        self.models_: list[object] = []
        self.best_iterations_: list[int] = []
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
            except Exception:
                if self.device_type_ != "gpu":
                    raise
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
            estimator = LGBMRegressor(
                n_estimators=int(self.best_iterations_[target_idx]),
                learning_rate=0.03,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="regression",
                random_state=self.random_state + target_idx,
                verbosity=-1,
                device_type=getattr(self, "device_type_", "cpu"),
            )
            estimator.fit(x_full, y_full[:, target_idx])
            self.models_.append(estimator)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        x_values = self._preprocess(x, fit=False)
        if getattr(self, "device_", "cpu") == "cuda":
            from xgboost import DMatrix

            dmatrix = DMatrix(x_values)
            preds = [np.asarray(model.get_booster().predict(dmatrix), dtype=np.float64) for model in self.models_]
        else:
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
    ) -> "XGBoostRegressor":
        from xgboost import XGBRegressor

        x_train = self._preprocess(x, fit=True)
        x_val = self._preprocess(val_x, fit=False) if val_x is not None else None
        y_train = _as_array(y)
        y_val_arr = _as_array(val_y) if val_y is not None else None
        self.models_: list[object] = []
        self.best_iterations_: list[int] = []
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
            except Exception:
                if self.device_ != "cuda":
                    raise
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
            estimator = XGBRegressor(
                n_estimators=int(self.best_iterations_[target_idx]),
                learning_rate=0.03,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=self.random_state + target_idx,
                n_jobs=1,
                tree_method="hist",
                device=getattr(self, "device_", "cpu"),
            )
            estimator.fit(x_full, y_full[:, target_idx], verbose=False)
            self.models_.append(estimator)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        x_values = self._preprocess(x, fit=False)
        preds = [np.asarray(model.predict(x_values), dtype=np.float64) for model in self.models_]
        return np.column_stack(preds)


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

        requested = self.device if self.device != "cuda" or _gpu_available() else "cpu"
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
        max_epochs: int = 60,
        patience: int = 10,
        batch_size: int = 256,
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

        requested = self.device if self.device != "cuda" or _gpu_available() else "cpu"
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
        train_rows = x["store"].fit_rows_through(self._train_cutoff_date(x["metadata"]))
        self.x_scaler.fit(train_rows.astype(np.float64))
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
        train_rows = x["store"].fit_rows_through(self._train_cutoff_date(x["metadata"]))
        self.x_scaler.fit(train_rows.astype(np.float64))
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


def is_sequence_static_model(model_name: str) -> bool:
    return model_name == SEQUENCE_STATIC_MODEL_NAME


def default_model_names(eval_mode: str = "walk_forward") -> list[str]:
    names = list(BASELINE_MODEL_NAMES)
    if eval_mode == "walk_forward":
        names.append(SEQUENCE_STATIC_MODEL_NAME)
    return names


def available_model_names() -> list[str]:
    return default_model_names(eval_mode="walk_forward")


def make_model(model_name: str, *, window_length: int = 60) -> Predictor:
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
        return TorchMLPRegressor()
    if model_name == SEQUENCE_STATIC_MODEL_NAME:
        return TorchSequenceStaticRegressor(window_length=window_length)
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
) -> pd.DataFrame:
    metadata_columns = ["ticker", "anchor_date"]
    if "anchor_close" in metadata.columns:
        metadata_columns.append("anchor_close")
    out = metadata[metadata_columns].copy()
    out["model_name"] = model_name
    out["feature_set"] = feature_set
    out["leaderboard_rank"] = leaderboard_rank
    out["recommended"] = recommended
    for idx, target in enumerate(target_columns):
        pred_col = target.replace("market_adjusted_return", "pred_market_adjusted_return")
        out[pred_col] = y_pred[:, idx]
        if y_true is not None:
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
