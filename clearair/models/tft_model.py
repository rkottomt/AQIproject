"""Temporal Fusion Transformer forecaster using pytorch-forecasting."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config.constants import (
    DATA_PLOTS_DIR,
    MODELS_SAVED_DIR,
    TFT_EARLY_STOPPING_PATIENCE,
    TFT_MAX_ENCODER_LENGTH,
    TFT_MAX_EPOCHS,
    TFT_MAX_PREDICTION_LENGTH,
)

logger = logging.getLogger(__name__)


class TFTForecaster:
    """Wraps ``pytorch_forecasting.TemporalFusionTransformer`` for AQI prediction."""

    CYCLIC_KNOWN = [
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "month_sin", "month_cos", "holiday_flag",
    ]

    def __init__(
        self,
        city: str,
        max_encoder_length: int = TFT_MAX_ENCODER_LENGTH,
        max_prediction_length: int = TFT_MAX_PREDICTION_LENGTH,
    ) -> None:
        self.city = city
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
        self.model = None
        self._ckpt_path = Path(MODELS_SAVED_DIR) / f"tft_{city}.ckpt"
        self._ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    def prepare_dataset(
        self, df: pd.DataFrame
    ) -> tuple:
        """Build train/val ``TimeSeriesDataSet`` objects.

        Args:
            df: Feature-engineered DataFrame with a datetime index.

        Returns:
            (train_dataset, val_dataset) tuple.
        """
        try:
            from pytorch_forecasting import TimeSeriesDataSet
        except ImportError:
            logger.error("pytorch-forecasting not installed; cannot prepare TFT dataset.")
            raise

        df = df.copy().reset_index()
        df = df.rename(columns={"index": "datetime"} if "datetime" not in df.columns else {})
        df["time_idx"] = np.arange(len(df))
        df["city_cat"] = self.city

        # Determine known/unknown reals from columns
        known_reals = [c for c in self.CYCLIC_KNOWN if c in df.columns]
        all_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude = {"time_idx", "AQI"} | set(known_reals)
        unknown_reals = [c for c in all_numeric if c not in exclude]

        split_idx = int(len(df) * 0.8)

        common_kwargs = dict(
            time_idx="time_idx",
            target="AQI",
            group_ids=["city_cat"],
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            time_varying_known_reals=known_reals,
            time_varying_unknown_reals=unknown_reals[:30],
            static_categoricals=["city_cat"],
            allow_missing_timesteps=True,
        )

        train_ds = TimeSeriesDataSet(
            df.iloc[:split_idx], **common_kwargs
        )
        val_ds = TimeSeriesDataSet.from_dataset(
            train_ds, df.iloc[split_idx:], stop_randomization=True
        )
        return train_ds, val_ds

    def train(
        self,
        train_dataset,
        val_dataset,
        max_epochs: int = TFT_MAX_EPOCHS,
    ) -> None:
        """Train the TFT model with early stopping and checkpointing.

        Args:
            train_dataset: Training ``TimeSeriesDataSet``.
            val_dataset: Validation ``TimeSeriesDataSet``.
            max_epochs: Maximum training epochs.
        """
        try:
            import pytorch_lightning as pl
            from pytorch_forecasting import TemporalFusionTransformer
            from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
        except ImportError:
            logger.error("pytorch-lightning / pytorch-forecasting not installed.")
            raise

        train_dl = train_dataset.to_dataloader(train=True, batch_size=64, num_workers=0)
        val_dl = val_dataset.to_dataloader(train=False, batch_size=64, num_workers=0)

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=TFT_EARLY_STOPPING_PATIENCE,
            mode="min",
        )
        checkpoint = ModelCheckpoint(
            dirpath=str(self._ckpt_path.parent),
            filename=f"tft_{self.city}",
            monitor="val_loss",
            mode="min",
        )

        tft = TemporalFusionTransformer.from_dataset(
            train_dataset,
            learning_rate=0.03,
            hidden_size=16,
            attention_head_size=1,
            dropout=0.1,
            hidden_continuous_size=8,
            output_size=7,
            loss=None,
            log_interval=10,
            reduce_on_plateau_patience=4,
        )

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="auto",
            callbacks=[early_stop, checkpoint],
            enable_progress_bar=True,
            gradient_clip_val=0.1,
        )
        trainer.fit(tft, train_dataloaders=train_dl, val_dataloaders=val_dl)
        self.model = tft

        # Save training curve
        self._plot_training_curve(trainer)
        logger.info("TFT training complete for %s.", self.city)

    def predict(
        self, df: pd.DataFrame, horizon_hours: int = 24
    ) -> dict:
        """Generate quantile forecasts.

        Args:
            df: Recent data (at least ``max_encoder_length`` rows).
            horizon_hours: Prediction horizon (capped at ``max_prediction_length``).

        Returns:
            Dict with keys: datetime, aqi_p10, aqi_p50, aqi_p90.
        """
        if self.model is None:
            self._load_model()

        if self.model is None:
            logger.error("No trained TFT model available for %s.", self.city)
            return {"datetime": [], "aqi_p10": [], "aqi_p50": [], "aqi_p90": []}

        horizon_hours = min(horizon_hours, self.max_prediction_length)

        try:
            predictions = self.model.predict(
                df.tail(self.max_encoder_length + horizon_hours),
                mode="quantiles",
            )
            preds = predictions.numpy() if hasattr(predictions, "numpy") else np.array(predictions)

            last_dt = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else pd.Timestamp.utcnow()
            dt_range = pd.date_range(
                last_dt + pd.Timedelta(hours=1),
                periods=min(horizon_hours, preds.shape[-2] if preds.ndim > 1 else horizon_hours),
                freq="h",
            )

            if preds.ndim == 3:
                return {
                    "datetime": dt_range.tolist(),
                    "aqi_p10": preds[0, :len(dt_range), 0].tolist(),
                    "aqi_p50": preds[0, :len(dt_range), 3].tolist(),
                    "aqi_p90": preds[0, :len(dt_range), 6].tolist(),
                }
            return {
                "datetime": dt_range.tolist(),
                "aqi_p10": preds.flatten()[:len(dt_range)].tolist(),
                "aqi_p50": preds.flatten()[:len(dt_range)].tolist(),
                "aqi_p90": preds.flatten()[:len(dt_range)].tolist(),
            }
        except Exception:
            logger.error("TFT prediction failed for %s.", self.city, exc_info=True)
            return {"datetime": [], "aqi_p10": [], "aqi_p50": [], "aqi_p90": []}

    def get_variable_importance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract TFT attention-based variable importance.

        Args:
            df: Dataset used for interpretation.

        Returns:
            DataFrame with variable names and importance scores.
        """
        if self.model is None:
            self._load_model()
        if self.model is None:
            return pd.DataFrame(columns=["variable", "importance"])

        try:
            interpretation = self.model.interpret_output(
                self.model.predict(df, mode="raw"), reduction="sum"
            )
            importance = interpretation.get("attention", {})
            records = [{"variable": k, "importance": float(v)}
                       for k, v in importance.items()] if isinstance(importance, dict) else []
            result = pd.DataFrame(records)

            if not result.empty:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                plots_dir = Path(DATA_PLOTS_DIR)
                plots_dir.mkdir(parents=True, exist_ok=True)
                fig, ax = plt.subplots(figsize=(10, 6))
                result.sort_values("importance").plot.barh(
                    x="variable", y="importance", ax=ax
                )
                ax.set_title(f"TFT Attention — {self.city}")
                fig.tight_layout()
                fig.savefig(plots_dir / f"tft_attention_{self.city}.png", dpi=150)
                plt.close(fig)

            return result
        except Exception:
            logger.warning("Variable importance extraction failed.", exc_info=True)
            return pd.DataFrame(columns=["variable", "importance"])

    def evaluate(self, test_df: pd.DataFrame) -> dict:
        """Compute RMSE, MAE, MAPE, and interval coverage on test data.

        Args:
            test_df: Test split DataFrame.

        Returns:
            Dict with RMSE, MAE, MAPE, interval_coverage.
        """
        preds = self.predict(test_df, horizon_hours=len(test_df))
        actual = test_df["AQI"].values[-len(preds.get("aqi_p50", [])):]
        predicted = np.array(preds.get("aqi_p50", []))

        if len(actual) == 0 or len(predicted) == 0:
            return {"RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan,
                    "interval_coverage": np.nan}

        n = min(len(actual), len(predicted))
        actual, predicted = actual[:n], predicted[:n]

        rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
        mae = float(np.mean(np.abs(actual - predicted)))
        mape = float(np.mean(np.abs((actual - predicted) / np.clip(actual, 1, None))) * 100)

        p10 = np.array(preds.get("aqi_p10", predicted))[:n]
        p90 = np.array(preds.get("aqi_p90", predicted))[:n]
        coverage = float(np.mean((actual >= p10) & (actual <= p90)) * 100)

        return {"RMSE": rmse, "MAE": mae, "MAPE": mape,
                "interval_coverage": coverage}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        ckpt = self._ckpt_path
        if not ckpt.exists():
            ckpt = self._ckpt_path.with_suffix(".ckpt")
        if ckpt.exists():
            try:
                from pytorch_forecasting import TemporalFusionTransformer
                self.model = TemporalFusionTransformer.load_from_checkpoint(str(ckpt))
                logger.info("Loaded TFT checkpoint from %s", ckpt)
            except Exception:
                logger.error("Failed to load TFT checkpoint.", exc_info=True)

    def _plot_training_curve(self, trainer) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plots = Path(DATA_PLOTS_DIR)
            plots.mkdir(parents=True, exist_ok=True)
            # Logged metrics may be in callback_metrics
            fig, ax = plt.subplots()
            ax.set_title(f"TFT Training — {self.city}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            fig.savefig(plots / f"tft_training_{self.city}.png", dpi=150)
            plt.close(fig)
        except Exception:
            logger.debug("Could not plot TFT training curve.", exc_info=True)
