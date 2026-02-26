"""Modeling helpers for hybrid travel-cost estimation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass
class TravelModelBundle:
    """Container for fitted pipeline and defaults used at prediction time."""

    pipeline: Pipeline
    categorical_cols: list[str]
    numeric_cols: list[str]
    default_lead_time_days: float
    default_num_legs: float
    default_month: int
    default_short_haul: str
    default_domestic: str
    default_traveler_role: str
    seen_origins: set[str]
    seen_dests: set[str]


def _wmape(actual: np.ndarray, pred: np.ndarray) -> float:
    denom = np.sum(np.abs(actual))
    if denom <= 0:
        return 0.0
    return float(np.sum(np.abs(actual - pred)) / denom)


def _safe_numeric(series: pd.Series, fill_value: float) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(fill_value)


def prepare_training_frame(model_flights: pd.DataFrame) -> pd.DataFrame:
    """Build supervised modeling frame from normalized Navan flight rows."""
    df = model_flights.copy()
    if "booking_start" in df.columns:
        df["booking_start"] = pd.to_datetime(df["booking_start"], errors="coerce")
    else:
        df["booking_start"] = pd.NaT

    df["origin"] = df.get("origin", "").astype(str).str.strip()
    df["dest"] = df.get("dest", "").astype(str).str.strip()
    df["destination_state_norm"] = df.get("destination_state_norm", "").astype(str).str.strip()
    df["short_or_long_haul"] = (
        df.get("Short or Long Haul", "").astype(str).str.strip().replace({"": "UNKNOWN"})
    )
    df["domestic_or_international"] = (
        df.get("Domestic or International", "")
        .astype(str)
        .str.strip()
        .replace({"": "UNKNOWN"})
    )
    df["traveler_role_norm"] = (
        df.get("traveler_role_norm", "").astype(str).str.strip().replace({"": "unknown"})
    )

    df["lead_time_days"] = _safe_numeric(df.get("lead_time_days", np.nan), np.nan)
    df["num_legs"] = _safe_numeric(df.get("num_legs", np.nan), 1.0)
    df["flight_miles"] = _safe_numeric(df.get("Flight miles", np.nan), np.nan)
    df["start_month"] = pd.to_numeric(df["booking_start"].dt.month, errors="coerce")
    df["start_dow"] = pd.to_numeric(df["booking_start"].dt.dayofweek, errors="coerce")
    df["is_weekend_start"] = (df["start_dow"] >= 5).astype(float)
    df["usd_total_paid"] = _safe_numeric(df.get("usd_total_paid", np.nan), np.nan)

    frame = df[
        [
            "origin",
            "dest",
            "destination_state_norm",
            "short_or_long_haul",
            "domestic_or_international",
            "traveler_role_norm",
            "lead_time_days",
            "num_legs",
            "flight_miles",
            "start_month",
            "start_dow",
            "is_weekend_start",
            "booking_start",
            "usd_total_paid",
        ]
    ].copy()
    frame = frame[
        frame["origin"].ne("")
        & frame["dest"].ne("")
        & frame["destination_state_norm"].ne("")
        & frame["usd_total_paid"].notna()
    ].copy()
    frame = frame.reset_index(drop=True)
    return frame


def _build_pipeline(cat_cols: list[str], num_cols: list[str]) -> Pipeline:
    """Create sklearn pipeline used by the hybrid engine."""
    pre = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat_cols,
            ),
            (
                "numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                num_cols,
            ),
        ]
    )

    # GBRT is robust on small-medium tabular data and handles non-linear fare patterns.
    model = GradientBoostingRegressor(
        random_state=42,
        n_estimators=250,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
    )
    return Pipeline(steps=[("preprocess", pre), ("model", model)])


def train_travel_model(
    frame: pd.DataFrame,
    evaluation_cutoff_date: Optional[date] = None,
) -> tuple[TravelModelBundle, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit model and return bundle, metrics, feature importance, train frame, valid frame."""
    if len(frame) < 40:
        raise ValueError(
            "Not enough rows to train hybrid travel model (need at least 40 flights after filtering)."
        )

    cat_cols = [
        "origin",
        "dest",
        "destination_state_norm",
        "short_or_long_haul",
        "domestic_or_international",
        "traveler_role_norm",
    ]
    num_cols = ["lead_time_days", "num_legs", "flight_miles", "start_month", "start_dow", "is_weekend_start"]

    use = frame.sort_values("booking_start").reset_index(drop=True)
    if evaluation_cutoff_date is not None:
        cutoff = pd.Timestamp(evaluation_cutoff_date)
        train_df = use[use["booking_start"] <= cutoff].copy()
        valid_df = use[use["booking_start"] > cutoff].copy()
    else:
        split_idx = max(1, int(len(use) * 0.8))
        split_idx = min(split_idx, len(use) - 1)
        train_df = use.iloc[:split_idx].copy()
        valid_df = use.iloc[split_idx:].copy()

    if len(train_df) < 25 or len(valid_df) < 10:
        split_idx = max(1, int(len(use) * 0.8))
        split_idx = min(split_idx, len(use) - 1)
        train_df = use.iloc[:split_idx].copy()
        valid_df = use.iloc[split_idx:].copy()

    if len(train_df) < 25 or len(valid_df) < 10:
        raise ValueError("Insufficient train/validation sizes after split for hybrid model.")

    pipeline = _build_pipeline(cat_cols, num_cols)
    y_train = np.log1p(train_df["usd_total_paid"].values)
    pipeline.fit(train_df[cat_cols + num_cols], y_train)

    pred_train = np.expm1(pipeline.predict(train_df[cat_cols + num_cols]))
    pred_valid = np.expm1(pipeline.predict(valid_df[cat_cols + num_cols]))
    actual_train = train_df["usd_total_paid"].values
    actual_valid = valid_df["usd_total_paid"].values

    metrics = {
        "rows_total": int(len(use)),
        "rows_train": int(len(train_df)),
        "rows_valid": int(len(valid_df)),
        "booking_start_min": str(use["booking_start"].min()),
        "booking_start_max": str(use["booking_start"].max()),
        "train_mae_usd": float(mean_absolute_error(actual_train, pred_train)),
        "valid_mae_usd": float(mean_absolute_error(actual_valid, pred_valid)),
        "train_rmse_usd": float(np.sqrt(mean_squared_error(actual_train, pred_train))),
        "valid_rmse_usd": float(np.sqrt(mean_squared_error(actual_valid, pred_valid))),
        "train_wmape": _wmape(actual_train, pred_train),
        "valid_wmape": _wmape(actual_valid, pred_valid),
    }

    pre = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    try:
        feature_names = pre.get_feature_names_out()
        importance = pd.DataFrame(
            {"feature": feature_names, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)
    except Exception:
        importance = pd.DataFrame(columns=["feature", "importance"])

    bundle = TravelModelBundle(
        pipeline=pipeline,
        categorical_cols=cat_cols,
        numeric_cols=num_cols,
        default_lead_time_days=float(train_df["lead_time_days"].median(skipna=True)),
        default_num_legs=float(train_df["num_legs"].median(skipna=True)),
        default_month=int(train_df["start_month"].median(skipna=True)) if train_df["start_month"].notna().any() else 6,
        default_short_haul=str(train_df["short_or_long_haul"].mode(dropna=True).iloc[0]),
        default_domestic=str(train_df["domestic_or_international"].mode(dropna=True).iloc[0]),
        default_traveler_role=str(train_df["traveler_role_norm"].mode(dropna=True).iloc[0]),
        seen_origins=set(train_df["origin"].astype(str)),
        seen_dests=set(train_df["dest"].astype(str)),
    )

    return bundle, metrics, importance, train_df, valid_df


def predict_route_cost(
    bundle: TravelModelBundle,
    origin: str,
    destination: str,
    destination_state_norm: str,
    lead_time_days: Optional[float] = None,
    num_legs: Optional[float] = None,
    start_month: Optional[int] = None,
    start_dow: Optional[int] = None,
    is_weekend_start: Optional[float] = None,
    short_or_long_haul: Optional[str] = None,
    domestic_or_international: Optional[str] = None,
    traveler_role_norm: Optional[str] = None,
    flight_miles: Optional[float] = None,
) -> float:
    """Predict USD route cost for a synthetic flight row."""
    month = int(start_month) if start_month is not None else int(bundle.default_month)
    dow = int(start_dow) if start_dow is not None else 2
    weekend = float(is_weekend_start) if is_weekend_start is not None else float(dow >= 5)
    row = pd.DataFrame(
        [
            {
                "origin": str(origin),
                "dest": str(destination),
                "destination_state_norm": str(destination_state_norm),
                "short_or_long_haul": short_or_long_haul or bundle.default_short_haul,
                "domestic_or_international": domestic_or_international or bundle.default_domestic,
                "traveler_role_norm": traveler_role_norm or bundle.default_traveler_role,
                "lead_time_days": (
                    float(lead_time_days)
                    if lead_time_days is not None and np.isfinite(lead_time_days)
                    else float(bundle.default_lead_time_days)
                ),
                "num_legs": (
                    float(num_legs)
                    if num_legs is not None and np.isfinite(num_legs)
                    else float(bundle.default_num_legs)
                ),
                "flight_miles": float(flight_miles) if flight_miles is not None else np.nan,
                "start_month": month,
                "start_dow": dow,
                "is_weekend_start": weekend,
            }
        ]
    )
    pred = float(np.expm1(bundle.pipeline.predict(row)[0]))
    return max(0.0, pred)
