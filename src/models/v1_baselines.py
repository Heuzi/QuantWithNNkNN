from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
import pandas as pd


EPS = 1e-12


class Predictor(Protocol):
    def fit(self, x: pd.DataFrame, y: pd.DataFrame) -> "Predictor":
        ...

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        ...


@dataclass
class Standardizer:
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "Standardizer":
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


def _as_array(frame: pd.DataFrame) -> np.ndarray:
    return frame.astype(float).to_numpy(dtype=np.float64)


class ZeroPredictor:
    def fit(self, x: pd.DataFrame, y: pd.DataFrame) -> "ZeroPredictor":
        self.output_dim_ = y.shape[1]
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.zeros((len(x), self.output_dim_), dtype=np.float64)


class MeanPredictor:
    def fit(self, x: pd.DataFrame, y: pd.DataFrame) -> "MeanPredictor":
        self.mean_ = y.astype(float).mean(axis=0).to_numpy(dtype=np.float64)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.tile(self.mean_, (len(x), 1))


class MomentumHeuristicPredictor:
    def fit(self, x: pd.DataFrame, y: pd.DataFrame) -> "MomentumHeuristicPredictor":
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
        from sklearn.linear_model import ElasticNet, Ridge
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.neural_network import MLPRegressor as SklearnMLPRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.impute import SimpleImputer
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
        if self.estimator_name == "sklearn_mlp":
            estimator = SklearnMLPRegressor(
                hidden_layer_sizes=(96, 48),
                activation="relu",
                learning_rate_init=0.001,
                max_iter=80,
                random_state=17,
                early_stopping=True,
                n_iter_no_change=8,
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

    def fit(self, x: pd.DataFrame, y: pd.DataFrame) -> "SklearnRegressor":
        self.model_ = self._make_estimator()
        self.model_.fit(x, y)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model_.predict(x), dtype=np.float64)


class LightGBMRegressor:
    def fit(self, x: pd.DataFrame, y: pd.DataFrame) -> "LightGBMRegressor":
        from lightgbm import LGBMRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.pipeline import make_pipeline

        estimator = LGBMRegressor(
            n_estimators=160,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="regression",
            random_state=23,
            verbosity=-1,
        )
        self.model_ = make_pipeline(
            SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True),
            MultiOutputRegressor(estimator),
        )
        self.model_.fit(x, y)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model_.predict(x), dtype=np.float64)


class XGBoostRegressor:
    def fit(self, x: pd.DataFrame, y: pd.DataFrame) -> "XGBoostRegressor":
        from sklearn.impute import SimpleImputer
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.pipeline import make_pipeline
        from xgboost import XGBRegressor

        estimator = XGBRegressor(
            n_estimators=160,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=29,
            n_jobs=1,
        )
        self.model_ = make_pipeline(
            SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True),
            MultiOutputRegressor(estimator),
        )
        self.model_.fit(x, y)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model_.predict(x), dtype=np.float64)


class TorchMLPRegressor:
    def __init__(
        self,
        hidden_units: int = 128,
        learning_rate: float = 0.001,
        epochs: int = 40,
        batch_size: int = 512,
        random_state: int = 31,
    ) -> None:
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.x_scaler = Standardizer()
        self.y_scaler = Standardizer()

    def fit(self, x: pd.DataFrame, y: pd.DataFrame) -> "TorchMLPRegressor":
        import torch

        torch.manual_seed(self.random_state)
        x_values = self.x_scaler.fit(_as_array(x)).transform(_as_array(x)).astype(np.float32)
        y_values = self.y_scaler.fit(_as_array(y)).transform(_as_array(y)).astype(np.float32)
        x_tensor = torch.from_numpy(x_values)
        y_tensor = torch.from_numpy(y_values)
        self.model_ = torch.nn.Sequential(
            torch.nn.Linear(x_values.shape[1], self.hidden_units),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.05),
            torch.nn.Linear(self.hidden_units, max(self.hidden_units // 2, 16)),
            torch.nn.ReLU(),
            torch.nn.Linear(max(self.hidden_units // 2, 16), y_values.shape[1]),
        )
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        loss_fn = torch.nn.MSELoss()
        n = len(x_tensor)
        for _ in range(self.epochs):
            order = torch.randperm(n)
            for start in range(0, n, self.batch_size):
                idx = order[start : start + self.batch_size]
                pred = self.model_(x_tensor[idx])
                loss = loss_fn(pred, y_tensor[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        import torch

        x_values = self.x_scaler.transform(_as_array(x)).astype(np.float32)
        with torch.no_grad():
            pred = self.model_(torch.from_numpy(x_values)).numpy()
        return self.y_scaler.inverse_transform(pred)


def make_model(model_name: str) -> Predictor:
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
        return SklearnRegressor("sklearn_mlp")
    if model_name == "torch_mlp":
        return TorchMLPRegressor()
    raise ValueError(f"Unknown model: {model_name}")


def default_model_names() -> list[str]:
    return [
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
    return out


def build_leaderboard(metrics: pd.DataFrame) -> pd.DataFrame:
    val = metrics[metrics["split"] == "val"].copy()
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
    return grouped


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
