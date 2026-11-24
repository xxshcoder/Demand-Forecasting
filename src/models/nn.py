"""PyTorch MLP with learned store/item embeddings and leakage-safe early stopping."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn as torch_nn
from torch.utils.data import DataLoader, TensorDataset

ID_COLUMNS = ["store_id", "item_id"]


class _MLP(torch_nn.Module):
    def __init__(
        self,
        n_numeric: int,
        n_stores: int,
        n_items: int,
        hidden: list[int],
        dropout: float,
        embedding_dim: int,
    ):
        super().__init__()
        self.store_embedding = torch_nn.Embedding(n_stores + 1, embedding_dim)
        self.item_embedding = torch_nn.Embedding(n_items + 1, embedding_dim)
        layers: list[torch_nn.Module] = []
        input_dim = n_numeric + 2 * embedding_dim
        for width in hidden:
            layers.extend(
                [torch_nn.Linear(input_dim, int(width)), torch_nn.ReLU(), torch_nn.Dropout(dropout)]
            )
            input_dim = int(width)
        layers.append(torch_nn.Linear(input_dim, 1))
        self.network = torch_nn.Sequential(*layers)

    def forward(self, numeric, stores, items):
        combined = torch.cat(
            [numeric, self.store_embedding(stores), self.item_embedding(items)], dim=1
        )
        return self.network(combined).squeeze(1)


class TorchForecaster:
    """Standalone estimator implementing fit/predict for the feature dataframe."""

    def __init__(self, params: dict, feature_columns: list[str]):
        cfg = params["models"]["nn"]
        self.cfg = cfg
        self.feature_columns = list(feature_columns)
        self.numeric_columns = [c for c in self.feature_columns if c not in ID_COLUMNS]
        self.device = torch.device(str(cfg["device"]))
        self.store_map: dict[int, int] = {}
        self.item_map: dict[int, int] = {}
        self.medians: np.ndarray | None = None
        self.means: np.ndarray | None = None
        self.stds: np.ndarray | None = None
        self.network: _MLP | None = None

    def _arrays(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        numeric = frame[self.numeric_columns].to_numpy(dtype=np.float32)
        numeric = np.nan_to_num(numeric, nan=0.0, posinf=0.0, neginf=0.0)
        if self.medians is not None:
            numeric = np.where(np.isnan(numeric), self.medians, numeric)
        stores = np.array(
            frame["store_id"].map(self.store_map).fillna(0).to_numpy(dtype=np.int64), copy=True
        )
        items = np.array(
            frame["item_id"].map(self.item_map).fillna(0).to_numpy(dtype=np.int64), copy=True
        )
        return numeric, stores, items

    def _tensors(self, frame: pd.DataFrame, y=None):
        numeric, stores, items = self._arrays(frame)
        if self.means is not None:
            numeric = (numeric - self.means) / self.stds
        tensors = (
            torch.from_numpy(numeric).to(self.device),
            torch.from_numpy(stores).to(self.device),
            torch.from_numpy(items).to(self.device),
        )
        if y is None:
            return tensors
        target = torch.from_numpy(np.array(y, dtype=np.float32, copy=True)).to(self.device)
        return tensors + (target,)

    def fit(self, frame: pd.DataFrame, y) -> TorchForecaster:
        torch.manual_seed(int(self.cfg["seed"]))
        np.random.seed(int(self.cfg["seed"]))
        self.store_map = {
            value: i + 1 for i, value in enumerate(sorted(frame["store_id"].unique()))
        }
        self.item_map = {value: i + 1 for i, value in enumerate(sorted(frame["item_id"].unique()))}

        numeric = frame[self.numeric_columns].to_numpy(dtype=np.float32)
        self.medians = np.nanmedian(numeric, axis=0)
        self.medians = np.nan_to_num(self.medians, nan=0.0)
        numeric = np.where(np.isnan(numeric), self.medians, numeric)
        self.means = numeric.mean(axis=0)
        self.stds = np.where(numeric.std(axis=0) > 0, numeric.std(axis=0), 1.0)

        # Early-stopping tail comes from the *training* period only.
        es_days = int(self.cfg["early_stop_days"])
        if "date" in frame.columns and frame["date"].nunique() > es_days:
            cutoff = pd.to_datetime(frame["date"]).max() - pd.Timedelta(days=es_days - 1)
            mask = pd.to_datetime(frame["date"]) >= cutoff
        else:
            mask = np.zeros(len(frame), dtype=bool)
            mask[-min(len(frame) // 10, es_days) :] = True
        train_frame, train_y = frame.loc[~mask], np.asarray(y)[~mask]
        es_frame, es_y = frame.loc[mask], np.asarray(y)[mask]
        if len(es_frame) == 0:
            raise ValueError("Neural net early-stopping window is empty")

        self.network = _MLP(
            len(self.numeric_columns),
            len(self.store_map),
            len(self.item_map),
            list(self.cfg["hidden"]),
            float(self.cfg["dropout"]),
            int(self.cfg["embedding_dim"]),
        ).to(self.device)
        optimizer = torch.optim.Adam(self.network.parameters(), lr=float(self.cfg["lr"]))
        loss_fn = torch_nn.MSELoss()
        generator = torch.Generator().manual_seed(int(self.cfg["seed"]))
        loader = DataLoader(
            TensorDataset(*self._tensors(train_frame, train_y)),
            batch_size=int(self.cfg["batch_size"]),
            shuffle=True,
            generator=generator,
        )
        es_tensors = self._tensors(es_frame, es_y)
        best_loss, best_state, stale = float("inf"), None, 0
        for _ in range(int(self.cfg["epochs"])):
            self.network.train()
            for batch in loader:
                optimizer.zero_grad()
                loss = loss_fn(self.network(batch[0], batch[1], batch[2]), batch[3])
                loss.backward()
                optimizer.step()
            self.network.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(self.network(*es_tensors[:3]), es_tensors[3]).item())
            if val_loss < best_loss:
                best_loss, best_state, stale = val_loss, copy.deepcopy(self.network.state_dict()), 0
            else:
                stale += 1
                if stale >= int(self.cfg["early_stop_patience"]):
                    break
        if best_state is not None:
            self.network.load_state_dict(best_state)
        self.network.eval()
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.network is None:
            raise RuntimeError("TorchForecaster must be fitted before predict")
        numeric, stores, items = self._tensors(frame)
        chunks = []
        with torch.no_grad():
            for start in range(0, len(numeric), int(self.cfg["batch_size"])):
                stop = start + int(self.cfg["batch_size"])
                chunks.append(
                    self.network(numeric[start:stop], stores[start:stop], items[start:stop])
                    .cpu()
                    .numpy()
                )
        return np.clip(np.concatenate(chunks), 0, None)

    def save(self, path: str | Path) -> None:
        if self.network is None:
            raise RuntimeError("Nothing to save: model is not fitted")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.network.state_dict(),
                "feature_columns": self.feature_columns,
                "numeric_columns": self.numeric_columns,
                "store_map": self.store_map,
                "item_map": self.item_map,
                "medians": self.medians,
                "means": self.means,
                "stds": self.stds,
                "cfg": self.cfg,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, map_location: str = "cpu") -> TorchForecaster:
        payload = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(payload["cfg"], payload["feature_columns"])
        model.numeric_columns = payload["numeric_columns"]
        model.store_map = payload["store_map"]
        model.item_map = payload["item_map"]
        model.medians = payload["medians"]
        model.means = payload["means"]
        model.stds = payload["stds"]
        model.network = _MLP(
            len(model.numeric_columns),
            len(model.store_map),
            len(model.item_map),
            list(model.cfg["hidden"]),
            float(model.cfg["dropout"]),
            int(model.cfg["embedding_dim"]),
        )
        model.network.load_state_dict(payload["state_dict"])
        model.network.to(model.device)
        model.network.eval()
        return model


def make_nn_pyfunc(model: TorchForecaster):
    """Return an MLflow pyfunc wrapper around a fitted TorchForecaster."""
    import mlflow.pyfunc

    class NnPyfuncModel(mlflow.pyfunc.PythonModel):
        def load_context(self, context) -> None:
            self.model = TorchForecaster.load(context.artifacts["nn_model"])

        def predict(self, context, model_input, params=None):
            return self.model.predict(model_input)

    return NnPyfuncModel()
