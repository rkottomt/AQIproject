"""Scikit-learn baseline models for AQI forecasting."""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from config.constants import DATA_PLOTS_DIR, MODELS_SAVED_DIR

logger = logging.getLogger(__name__)

MODEL_SPECS: dict[str, object] = {
    "LinearRegression": LinearRegression(),
    "Lasso": Lasso(alpha=0.1),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "KNeighbors": KNeighborsRegressor(n_neighbors=5),
    "DecisionTree": DecisionTreeRegressor(max_depth=10, random_state=42),
    "MLP": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
}


class BaselineModels:
    """Train, evaluate, and compare six scikit-learn regressors."""

    def __init__(self, city: str) -> None:
        self.city = city
        self.scaler: Optional[StandardScaler] = None
        self.models: dict[str, object] = {}
        self._save_dir = Path(MODELS_SAVED_DIR)
        self._save_dir.mkdir(parents=True, exist_ok=True)

    def train_all(
        self,
        df: pd.DataFrame,
        features: list[str],
        target: str = "AQI",
    ) -> dict[str, float]:
        """Train every baseline model with an 80/20 temporal split.

        Args:
            df: Feature-engineered DataFrame.
            features: List of feature column names.
            target: Target column name.

        Returns:
            Dict mapping model name → training RMSE.
        """
        available = [f for f in features if f in df.columns]
        X = df[available].values
        y = df[target].values

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        rmses: dict[str, float] = {}
        for name, model_template in MODEL_SPECS.items():
            from sklearn.base import clone
            model = clone(model_template)
            try:
                model.fit(X_train_s, y_train)
                preds = model.predict(X_test_s)
                rmse = float(np.sqrt(np.mean((y_test - preds) ** 2)))
                rmses[name] = rmse
                self.models[name] = model
                logger.info("Baseline %s RMSE=%.2f", name, rmse)
            except Exception:
                logger.error("Training failed for %s.", name, exc_info=True)

        # Persist
        artefact = {
            "models": self.models,
            "scaler": self.scaler,
            "features": available,
        }
        with open(self._save_dir / f"baselines_{self.city}.pkl", "wb") as fh:
            pickle.dump(artefact, fh)

        return rmses

    def evaluate_all(self, test_df: pd.DataFrame, features: list[str],
                     target: str = "AQI") -> pd.DataFrame:
        """Evaluate all trained baselines on a test set.

        Args:
            test_df: Test-split DataFrame.
            features: Feature column names.
            target: Target column.

        Returns:
            DataFrame with columns [model, RMSE, MAE, MAPE].
        """
        if not self.models:
            self._load_models()

        available = [f for f in features if f in test_df.columns]
        X = self.scaler.transform(test_df[available].values)
        y = test_df[target].values

        rows: list[dict] = []
        for name, model in self.models.items():
            preds = model.predict(X)
            n = min(len(y), len(preds))
            rmse = float(np.sqrt(np.mean((y[:n] - preds[:n]) ** 2)))
            mae = float(np.mean(np.abs(y[:n] - preds[:n])))
            mape = float(np.mean(np.abs((y[:n] - preds[:n]) / np.clip(y[:n], 1, None))) * 100)
            rows.append({"model": name, "RMSE": rmse, "MAE": mae, "MAPE": mape})
        return pd.DataFrame(rows)

    @staticmethod
    def plot_rmse_comparison(
        results: pd.DataFrame,
        lstm_rmse: float,
        tft_rmse: float,
        save_path: str,
    ) -> None:
        """Horizontal bar chart of all model RMSEs.

        Args:
            results: DataFrame from ``evaluate_all``.
            lstm_rmse: RMSE of the LSTM model.
            tft_rmse: RMSE of the TFT model.
            save_path: File path for the saved plot.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        all_rows = results[["model", "RMSE"]].copy()
        all_rows = pd.concat([
            all_rows,
            pd.DataFrame([
                {"model": "LSTM", "RMSE": lstm_rmse},
                {"model": "TFT", "RMSE": tft_rmse},
            ]),
        ], ignore_index=True).sort_values("RMSE")

        fig, ax = plt.subplots(figsize=(10, 6))
        colours = []
        for m in all_rows["model"]:
            if m == "LSTM":
                colours.append("#e74c3c")
            elif m == "TFT":
                colours.append("#2ecc71")
            else:
                colours.append("#3498db")
        ax.barh(all_rows["model"], all_rows["RMSE"], color=colours)
        ax.set_xlabel("RMSE")
        ax.set_title("Model RMSE Comparison")
        fig.tight_layout()

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        logger.info("RMSE comparison chart saved to %s", save_path)

    def _load_models(self) -> None:
        path = self._save_dir / f"baselines_{self.city}.pkl"
        if path.exists():
            with open(path, "rb") as fh:
                artefact = pickle.load(fh)
            self.models = artefact["models"]
            self.scaler = artefact["scaler"]
            logger.info("Loaded baseline models from %s", path)
