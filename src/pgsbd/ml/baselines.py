"""Plain ML baselines for mapping (WF, T) -> log10(I).

Baselines:
1. Ridge regression
2. HistGradientBoosting (MultiOutput)
3. Nearest Neighbor (k=3, distance-weighted)
4. Plain Multi-Output PyTorch MLP (2 -> 128 -> 128 -> 52)
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class BaseBaseline(ABC):
    """Abstract base class for all ML surrogate baselines."""

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        log_y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        log_y_val: np.ndarray | None = None,
    ) -> BaseBaseline:
        """Fit model on training data."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict log10(currents) for input parameters [WF, T].

        Args:
            X: Array of shape [N, 2] with columns [WF, T].

        Returns:
            Array of shape [N, 52] with predicted log10(currents).
        """


class RidgeBaseline(BaseBaseline):
    """Multi-output Ridge regression baseline."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)
        self.scaler = StandardScaler()

    def fit(
        self,
        X_train: np.ndarray,
        log_y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        log_y_val: np.ndarray | None = None,
    ) -> RidgeBaseline:
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, log_y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class HistGradientBoostingBaseline(BaseBaseline):
    """Multi-output HistGradientBoosting regression baseline."""

    def __init__(self, max_iter: int = 100, random_state: int = 42):
        self.max_iter = max_iter
        self.random_state = random_state
        base_estimator = HistGradientBoostingRegressor(
            max_iter=max_iter,
            min_samples_leaf=10,
            random_state=random_state,
        )
        self.model = MultiOutputRegressor(base_estimator, n_jobs=-1)
        self.scaler = StandardScaler()

    def fit(
        self,
        X_train: np.ndarray,
        log_y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        log_y_val: np.ndarray | None = None,
    ) -> HistGradientBoostingBaseline:
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, log_y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class KNNBaseline(BaseBaseline):
    """Distance-weighted Nearest Neighbors regression baseline."""

    def __init__(self, n_neighbors: int = 3):
        self.n_neighbors = n_neighbors
        self.model = KNeighborsRegressor(n_neighbors=n_neighbors, weights="distance")
        self.scaler = StandardScaler()

    def fit(
        self,
        X_train: np.ndarray,
        log_y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        log_y_val: np.ndarray | None = None,
    ) -> KNNBaseline:
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, log_y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class PyTorchMLP(nn.Module):
    """Plain multi-output feedforward neural network."""

    def __init__(
        self,
        in_features: int = 2,
        out_features: int = 52,
        hidden_dims: tuple[int, ...] = (128, 128),
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = in_features
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.SiLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PlainMLPBaseline(BaseBaseline):
    """Plain Multi-Output PyTorch MLP Baseline."""

    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (128, 128),
        lr: float = 2.0e-3,
        weight_decay: float = 1.0e-4,
        epochs: int = 300,
        batch_size: int = 64,
        seed: int = 42,
    ):
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.scaler = StandardScaler()
        self.model: PyTorchMLP | None = None

    def fit(
        self,
        X_train: np.ndarray,
        log_y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        log_y_val: np.ndarray | None = None,
    ) -> PlainMLPBaseline:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model = PyTorchMLP(
            in_features=2,
            out_features=log_y_train.shape[1],
            hidden_dims=self.hidden_dims,
        )

        device = torch.device("cpu")
        self.model.to(device)

        train_ds = TensorDataset(
            torch.tensor(X_train_scaled, dtype=torch.float32),
            torch.tensor(log_y_train, dtype=torch.float32),
        )
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        if X_val is not None and log_y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            val_x_t = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
            val_y_t = torch.tensor(log_y_val, dtype=torch.float32).to(device)
        else:
            val_x_t, val_y_t = None, None

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=1e-5
        )
        criterion = nn.HuberLoss(delta=1.0)

        best_val_loss = float("inf")
        best_state = None
        patience = 40
        patience_counter = 0

        for _ in range(self.epochs):
            self.model.train()
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                pred = self.model(bx)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()

            scheduler.step()

            if val_x_t is not None and val_y_t is not None:
                self.model.eval()
                with torch.no_grad():
                    val_pred = self.model(val_x_t)
                    val_loss = float(criterion(val_pred, val_y_t).item())

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        with torch.no_grad():
            x_t = torch.tensor(X_scaled, dtype=torch.float32)
            preds = self.model(x_t).numpy()
        return preds

    def save_weights(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("Cannot save unfitted model.")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), p)


def save_scaler_artifact(
    scaler: StandardScaler,
    path: str | Path = "artifacts/scalers.json",
) -> dict[str, Any]:
    """Save scaler parameters (mean, scale/std, var) to JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    scaler_dict = {
        "feature_names": ["WF", "temperature"],
        "mean": [float(m) for m in scaler.mean_],
        "scale": [float(s) for s in scaler.scale_],
        "var": [float(v) for v in scaler.var_],
    }
    with open(p, "w", encoding="utf-8") as f:
        json.dump(scaler_dict, f, indent=2)
    return scaler_dict
