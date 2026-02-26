"""Step 7: Build travel cost model from Navan data (hybrid or heuristic engine)."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import config
from optimization_utils import (
    US_ABBR,
    build_airports_df,
    coerce_excel_datetime,
    normalize_name,
    normalize_state,
)

try:
    from travel_cost_modeling import predict_route_cost, prepare_training_frame, train_travel_model
except ImportError:
    # Heuristic engine can still run without sklearn; hybrid path validates these.
    predict_route_cost = None
    prepare_training_frame = None
    train_travel_model = None


DEFAULT_NAVAN_XLSX = config.EXTERNAL_NAVAN_XLSX
DEFAULT_BTS_PRIOR_CSV = os.path.join(
    config.PROJECT_ROOT, "data", "external", "bts", "state_pair_fares.csv"
)


def resolve_path(candidates: list[str], label: str) -> str:
    """Return first existing file path."""
    for path in candidates:
        if not path:
            continue
        if os.path.exists(path):
            return path
    options = "\n".join(f"- {p}" for p in candidates if p)
    raise FileNotFoundError(f"Could not resolve {label}. Tried:\n{options}")


def parse_management_users(navan_path: str) -> set[str]:
    """Read management roster names from Tech Roster tab."""
    raw = pd.read_excel(navan_path, sheet_name="Tech Roster", header=None)
    tbl = raw.iloc[1:].copy()
    tbl.columns = raw.iloc[0].tolist()
    tbl = tbl.dropna(how="all")
    tbl = tbl[tbl["Role Classification"].notna()]
    tbl = tbl[
        ~tbl["Technician (Roster Name)"]
        .astype(str)
        .str.contains("ROLE COLOR LEGEND", case=False, na=False)
    ]
    mgmt = tbl[
        tbl["Role Classification"].astype(str).str.contains("management", case=False, na=False)
    ]
    names = set()
    for col in ["Technician (Roster Name)", "Aliases in System"]:
        if col in mgmt.columns:
            names |= set(mgmt[col].dropna().astype(str).map(normalize_name))
    return {n for n in names if n}


def build_state_airport_weights(demand: pd.DataFrame) -> dict[str, pd.Series]:
    """Build per-state destination-airport demand weights."""
    grouped = (
        demand.dropna(subset=["state_norm", "nearest_hub_airport", "duration_hours"])
        .groupby(["state_norm", "nearest_hub_airport"])["duration_hours"]
        .sum()
        .reset_index()
    )
    by_state: dict[str, pd.Series] = {}
    for state, group in grouped.groupby("state_norm"):
        s = group.set_index("nearest_hub_airport")["duration_hours"]
        by_state[state] = s / s.sum()
    return by_state


def estimate_route_cost(
    origin: str,
    destination: str,
    route_stats: pd.DataFrame,
    origin_stats: pd.DataFrame,
    dest_stats: pd.DataFrame,
    global_cost: float,
    global_legs: float,
) -> tuple[float, float, int, str]:
    """Estimate route cost with hierarchical fallbacks."""
    direct = route_stats[(route_stats["origin"] == origin) & (route_stats["dest"] == destination)]
    if not direct.empty:
        row = direct.iloc[0]
        return (
            float(row["mean_cost"]),
            float(row["mean_legs"]),
            int(row["trip_count"]),
            "direct_route",
        )

    rev = route_stats[(route_stats["origin"] == destination) & (route_stats["dest"] == origin)]
    if not rev.empty:
        row = rev.iloc[0]
        return (
            float(row["mean_cost"]),
            float(row["mean_legs"]),
            int(row["trip_count"]),
            "reverse_route",
        )

    o = origin_stats[origin_stats["origin"] == origin]
    d = dest_stats[dest_stats["dest"] == destination]
    if not o.empty and not d.empty:
        return (
            float((o.iloc[0]["mean_cost"] + d.iloc[0]["mean_cost"]) / 2),
            float((o.iloc[0]["mean_legs"] + d.iloc[0]["mean_legs"]) / 2),
            int(min(o.iloc[0]["trip_count"], d.iloc[0]["trip_count"])),
            "origin_dest_blend",
        )
    if not o.empty:
        return (
            float(o.iloc[0]["mean_cost"]),
            float(o.iloc[0]["mean_legs"]),
            int(o.iloc[0]["trip_count"]),
            "origin_only",
        )
    if not d.empty:
        return (
            float(d.iloc[0]["mean_cost"]),
            float(d.iloc[0]["mean_legs"]),
            int(d.iloc[0]["trip_count"]),
            "destination_only",
        )
    return float(global_cost), float(global_legs), 0, "global_fallback"


def tier_from_score(score: float) -> str:
    """Map confidence score to tier for matrix consumers."""
    if score >= 0.7:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"


def _wmape(actual: np.ndarray, pred: np.ndarray) -> float:
    denom = np.sum(np.abs(actual))
    if denom <= 0:
        return 0.0
    return float(np.sum(np.abs(actual - pred)) / denom)


def regression_metrics(actual: np.ndarray, pred: np.ndarray, prefix: str) -> dict:
    """Build standard error metrics dictionary."""
    return {
        f"{prefix}_mae_usd": float(np.mean(np.abs(actual - pred))),
        f"{prefix}_rmse_usd": float(np.sqrt(np.mean((actual - pred) ** 2))),
        f"{prefix}_wmape": _wmape(actual, pred),
    }


def build_origin_state_lookup(candidates: pd.DataFrame, tech: pd.DataFrame) -> dict[str, str]:
    """Infer origin airport -> state mapping from candidate/tech tables and airport config."""
    lookup: dict[str, str] = {}

    if not candidates.empty:
        for _, row in candidates.iterrows():
            ap = str(row.get("airport_iata", "")).strip()
            st = normalize_state(row.get("state"))
            if ap and st and ap not in lookup:
                lookup[ap] = st

    if not tech.empty:
        for _, row in tech.iterrows():
            ap = str(row.get("base_airport_iata", "")).strip()
            st = normalize_state(row.get("base_state"))
            if ap and st and ap not in lookup:
                lookup[ap] = st

    airports = build_airports_df(config.MAJOR_AIRPORTS)
    for _, row in airports.iterrows():
        ap = str(row.get("airport_code", "")).strip()
        st = normalize_state(row.get("state_abbr"))
        if ap and st and ap not in lookup:
            lookup[ap] = st
    return lookup


def normalize_colname(value: str) -> str:
    """Normalize a free-form column name to snake_case."""
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def first_present(cols: list[str], candidates: list[str]) -> Optional[str]:
    """Return first present candidate column name."""
    for c in candidates:
        if c in cols:
            return c
    return None


def load_bts_prior(path: str) -> tuple[dict[tuple[str, str], float], dict]:
    """Load optional BTS state-pair fare prior and return mapping + metadata."""
    if not path or not os.path.exists(path):
        return {}, {
            "bts_prior_enabled": False,
            "bts_prior_path": path,
            "reason": "missing_file",
            "state_pairs_loaded": 0,
        }

    raw = pd.read_csv(path)
    cols = [normalize_colname(c) for c in raw.columns]
    rename = {old: new for old, new in zip(raw.columns, cols)}
    df = raw.rename(columns=rename).copy()
    all_cols = list(df.columns)

    origin_col = first_present(
        all_cols,
        ["origin_state", "origin", "from_state", "orig_state", "state_from"],
    )
    dest_col = first_present(
        all_cols,
        ["destination_state", "dest_state", "destination", "to_state", "state_to"],
    )
    fare_col = first_present(
        all_cols,
        [
            "avg_fare_usd",
            "average_fare_usd",
            "fare_usd",
            "avg_fare",
            "mean_fare_usd",
            "itinerary_fare",
        ],
    )
    weight_col = first_present(all_cols, ["passengers", "passenger_count", "trips", "weight"])

    if not origin_col or not dest_col or not fare_col:
        return {}, {
            "bts_prior_enabled": False,
            "bts_prior_path": path,
            "reason": "missing_required_columns",
            "columns_found": all_cols,
            "state_pairs_loaded": 0,
        }

    df["origin_state_norm"] = df[origin_col].map(normalize_state)
    df["destination_state_norm"] = df[dest_col].map(normalize_state)
    df["fare_usd_norm"] = pd.to_numeric(df[fare_col], errors="coerce")
    df = df.dropna(subset=["origin_state_norm", "destination_state_norm", "fare_usd_norm"]).copy()
    df = df[
        df["origin_state_norm"].isin(US_ABBR) & df["destination_state_norm"].isin(US_ABBR)
    ].copy()
    if df.empty:
        return {}, {
            "bts_prior_enabled": False,
            "bts_prior_path": path,
            "reason": "no_usable_rows",
            "state_pairs_loaded": 0,
        }

    if weight_col:
        df["weight"] = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    else:
        df["weight"] = 1.0

    grouped = (
        df.groupby(["origin_state_norm", "destination_state_norm"], as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "avg_fare_usd": float(np.average(g["fare_usd_norm"], weights=g["weight"]))
                    if g["weight"].sum() > 0
                    else float(g["fare_usd_norm"].mean()),
                    "rows": int(len(g)),
                    "weight_total": float(g["weight"].sum()),
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )

    mapping = {
        (str(r["origin_state_norm"]), str(r["destination_state_norm"])): float(r["avg_fare_usd"])
        for _, r in grouped.iterrows()
    }
    meta = {
        "bts_prior_enabled": True,
        "bts_prior_path": path,
        "state_pairs_loaded": int(len(mapping)),
        "rows_loaded": int(len(df)),
    }
    return mapping, meta


def build_matrix_heuristic(
    origins: list[str],
    states: list[str],
    state_weights: dict[str, pd.Series],
    default_weights: pd.Series,
    route_stats: pd.DataFrame,
    origin_stats: pd.DataFrame,
    dest_stats: pd.DataFrame,
    global_cost: float,
    global_legs: float,
) -> pd.DataFrame:
    """Build matrix using legacy heuristic fallback logic only."""
    rows = []
    for origin in origins:
        for state in states:
            weights = state_weights.get(state, default_weights)
            weighted_cost = 0.0
            weighted_legs = 0.0
            sample_size = 0
            tier_weights: dict[str, float] = defaultdict(float)
            for dest, wt in weights.items():
                cost, legs, n, tier = estimate_route_cost(
                    origin=origin,
                    destination=str(dest),
                    route_stats=route_stats,
                    origin_stats=origin_stats,
                    dest_stats=dest_stats,
                    global_cost=global_cost,
                    global_legs=global_legs,
                )
                weighted_cost += wt * cost
                weighted_legs += wt * legs
                sample_size += n
                tier_weights[tier] += float(wt)
            dominant_tier = max(tier_weights.items(), key=lambda x: x[1])[0] if tier_weights else "global_fallback"
            rows.append(
                {
                    "origin_airport": origin,
                    "state_norm": state,
                    "expected_cost_usd": float(weighted_cost),
                    "expected_legs": float(weighted_legs),
                    "sample_size": int(sample_size),
                    "confidence_tier": dominant_tier,
                    "confidence_score": 0.4 if dominant_tier in {"direct_route", "reverse_route"} else 0.25,
                    "model_cost_usd": np.nan,
                    "empirical_route_cost_usd": np.nan,
                    "blended_route_cost_usd": float(weighted_cost),
                    "support_trip_count": int(sample_size),
                    "blend_weight_empirical": 0.0,
                    "prior_source": "heuristic",
                }
            )
    return pd.DataFrame(rows)


def build_hybrid_matrix(
    origins: list[str],
    states: list[str],
    state_weights: dict[str, pd.Series],
    default_weights: pd.Series,
    route_stats: pd.DataFrame,
    origin_stats: pd.DataFrame,
    dest_stats: pd.DataFrame,
    global_cost: float,
    global_legs: float,
    min_direct_route_n: int,
    shrinkage_k: float,
    bundle,
    bts_prior: dict[tuple[str, str], float],
    origin_state_lookup: dict[str, str],
) -> pd.DataFrame:
    """Build hybrid matrix: empirical-route + model + optional BTS prior."""
    direct_map: dict[tuple[str, str], tuple[float, int]] = {}
    for _, r in route_stats.iterrows():
        direct_map[(str(r["origin"]), str(r["dest"]))] = (
            float(r["mean_cost"]),
            int(r["trip_count"]),
        )
    origin_trip_map = {
        str(r["origin"]): int(r["trip_count"])
        for _, r in origin_stats.iterrows()
    }
    dest_trip_map = {
        str(r["dest"]): int(r["trip_count"])
        for _, r in dest_stats.iterrows()
    }

    rows = []
    for origin in origins:
        origin_support = int(origin_trip_map.get(origin, 0))
        for state in states:
            weights = state_weights.get(state, default_weights)
            weighted_cost = 0.0
            weighted_legs = 0.0
            weighted_model_cost = 0.0
            weighted_heur_cost = 0.0
            weighted_emp_cost = 0.0
            weighted_blend_emp = 0.0
            weighted_model_weight = 0.0
            weighted_origin_support = 0.0
            weighted_dest_support = 0.0
            guardrail_hit_weight = 0.0
            weighted_conf = 0.0
            model_weight = 0.0
            emp_weight = 0.0
            sample_size = 0
            support_trip_count = 0
            source_weight: dict[str, float] = defaultdict(float)

            for dest, wt in weights.items():
                wt = float(wt)
                heur_cost, heur_legs, n, _ = estimate_route_cost(
                    origin=origin,
                    destination=str(dest),
                    route_stats=route_stats,
                    origin_stats=origin_stats,
                    dest_stats=dest_stats,
                    global_cost=global_cost,
                    global_legs=global_legs,
                )
                sample_size += int(n)

                empirical = direct_map.get((origin, str(dest)))
                if empirical:
                    empirical_cost, direct_n = empirical
                else:
                    empirical_cost, direct_n = np.nan, 0
                support_trip_count += int(direct_n)
                dest_support = int(dest_trip_map.get(str(dest), 0))

                model_cost = predict_route_cost(
                    bundle=bundle,
                    origin=origin,
                    destination=str(dest),
                    destination_state_norm=state,
                )

                blend_w_emp = 0.0
                source = "model"
                conf = 0.3
                route_cost = float(model_cost)
                model_mix_weight = 0.0
                guardrailed = False

                if direct_n >= min_direct_route_n and np.isfinite(empirical_cost):
                    blend_w_emp = float(direct_n) / float(direct_n + shrinkage_k)
                    if np.isfinite(model_cost):
                        route_cost = (
                            blend_w_emp * float(empirical_cost)
                            + (1.0 - blend_w_emp) * float(model_cost)
                        )
                        source = "navan_empirical_blend"
                    else:
                        route_cost = float(empirical_cost)
                        source = "navan_empirical_only"
                    conf = min(0.98, 0.55 + 0.04 * min(direct_n, 10))
                else:
                    if not np.isfinite(model_cost):
                        route_cost = float(heur_cost)
                        source = "heuristic_fallback"
                        conf = 0.2
                    else:
                        in_training_space = (
                            origin in bundle.seen_origins and str(dest) in bundle.seen_dests
                        )
                        model_mix_weight = 0.15
                        if in_training_space:
                            model_mix_weight += 0.35
                        model_mix_weight += min(0.30, 0.03 * min(origin_support, 10))
                        model_mix_weight += min(0.20, 0.02 * min(dest_support, 10))
                        model_mix_weight = float(min(0.80, max(0.15, model_mix_weight)))
                        route_cost = (
                            model_mix_weight * float(model_cost)
                            + (1.0 - model_mix_weight) * float(heur_cost)
                        )
                        source = "model_heuristic_blend"
                        conf = 0.25 + 0.55 * model_mix_weight

                        # Guardrail sparse model predictions so they do not collapse to near-zero.
                        floor_cost = max(35.0, 0.35 * float(heur_cost))
                        ceil_cost = max(floor_cost + 25.0, 3.0 * float(heur_cost))
                        bounded = float(np.clip(route_cost, floor_cost, ceil_cost))
                        guardrailed = abs(bounded - route_cost) > 1e-6
                        if guardrailed:
                            source = "model_heuristic_blend_guardrailed"
                            route_cost = bounded

                if state in US_ABBR and conf < 0.35 and bts_prior:
                    origin_state = origin_state_lookup.get(origin)
                    prior_cost = bts_prior.get((origin_state, state)) if origin_state else None
                    if prior_cost is not None and np.isfinite(prior_cost):
                        route_cost = float(prior_cost)
                        source = "bts_prior"
                        conf = 0.4
                        blend_w_emp = 0.0

                if not np.isfinite(route_cost):
                    route_cost = float(heur_cost)
                    source = "heuristic_fallback"
                    conf = 0.2
                    blend_w_emp = 0.0
                    model_mix_weight = 0.0

                weighted_cost += wt * route_cost
                weighted_legs += wt * float(heur_legs)
                weighted_heur_cost += wt * float(heur_cost)
                weighted_conf += wt * conf
                weighted_blend_emp += wt * blend_w_emp
                weighted_model_weight += wt * model_mix_weight
                weighted_origin_support += wt * float(origin_support)
                weighted_dest_support += wt * float(dest_support)
                if guardrailed:
                    guardrail_hit_weight += wt
                source_weight[source] += wt

                if np.isfinite(model_cost):
                    weighted_model_cost += wt * float(model_cost)
                    model_weight += wt
                if np.isfinite(empirical_cost):
                    weighted_emp_cost += wt * float(empirical_cost)
                    emp_weight += wt

            conf_score = float(weighted_conf)
            dominant_source = (
                max(source_weight.items(), key=lambda x: x[1])[0] if source_weight else "heuristic_fallback"
            )
            rows.append(
                {
                    "origin_airport": origin,
                    "state_norm": state,
                    "expected_cost_usd": float(weighted_cost),
                    "expected_legs": float(weighted_legs),
                    "sample_size": int(sample_size),
                    "confidence_tier": tier_from_score(conf_score),
                    "confidence_score": conf_score,
                    "model_cost_usd": float(weighted_model_cost / model_weight) if model_weight > 0 else np.nan,
                    "heuristic_cost_usd": float(weighted_heur_cost),
                    "empirical_route_cost_usd": float(weighted_emp_cost / emp_weight) if emp_weight > 0 else np.nan,
                    "blended_route_cost_usd": float(weighted_cost),
                    "support_trip_count": int(support_trip_count),
                    "blend_weight_empirical": float(weighted_blend_emp),
                    "blend_weight_model": float(weighted_model_weight),
                    "origin_support_trip_count": float(weighted_origin_support),
                    "destination_support_trip_count": float(weighted_dest_support),
                    "guardrail_hit_weight": float(guardrail_hit_weight),
                    "prior_source": dominant_source,
                }
            )
    return pd.DataFrame(rows)


def build_origin_anomaly_report(matrix: pd.DataFrame, origin_stats: pd.DataFrame) -> dict:
    """Summarize origin-level matrix anomalies against observed Navan origin means."""
    if matrix.empty:
        return {
            "origin_count": 0,
            "flagged_count": 0,
            "flagged_origins": [],
        }

    matrix_origin = (
        matrix.groupby("origin_airport", as_index=False)
        .agg(
            matrix_mean_expected_cost_usd=("expected_cost_usd", "mean"),
            matrix_min_expected_cost_usd=("expected_cost_usd", "min"),
            matrix_max_expected_cost_usd=("expected_cost_usd", "max"),
            mean_confidence_score=("confidence_score", "mean"),
            dominant_prior_source=(
                "prior_source",
                lambda s: s.value_counts().idxmax() if not s.empty else "unknown",
            ),
            mean_guardrail_hit_weight=("guardrail_hit_weight", "mean"),
        )
        .rename(columns={"origin_airport": "origin"})
    )
    observed = origin_stats.rename(
        columns={
            "trip_count": "observed_trip_count",
            "mean_cost": "observed_mean_cost_usd",
        }
    )[["origin", "observed_trip_count", "observed_mean_cost_usd"]]
    merged = matrix_origin.merge(observed, on="origin", how="left")
    merged["observed_trip_count"] = (
        pd.to_numeric(merged["observed_trip_count"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    merged["observed_mean_cost_usd"] = pd.to_numeric(
        merged["observed_mean_cost_usd"], errors="coerce"
    )
    merged["cost_ratio_vs_observed"] = np.where(
        merged["observed_mean_cost_usd"].fillna(0) > 0,
        merged["matrix_mean_expected_cost_usd"] / merged["observed_mean_cost_usd"],
        np.nan,
    )
    merged["abs_delta_vs_observed_usd"] = (
        merged["matrix_mean_expected_cost_usd"] - merged["observed_mean_cost_usd"]
    ).abs()

    flag_mask = (
        (merged["observed_trip_count"] >= 2)
        & (merged["abs_delta_vs_observed_usd"] >= 120.0)
        & (
            (merged["cost_ratio_vs_observed"] <= 0.4)
            | (merged["cost_ratio_vs_observed"] >= 2.2)
        )
    )
    flagged = merged[flag_mask].copy().sort_values(
        "abs_delta_vs_observed_usd", ascending=False
    )
    top_low = merged.sort_values("matrix_mean_expected_cost_usd").head(10)
    top_high = merged.sort_values("matrix_mean_expected_cost_usd", ascending=False).head(10)

    return {
        "origin_count": int(len(merged)),
        "flagged_count": int(len(flagged)),
        "flagged_origins": flagged.to_dict(orient="records"),
        "lowest_mean_expected_cost_origins": top_low.to_dict(orient="records"),
        "highest_mean_expected_cost_origins": top_high.to_dict(orient="records"),
    }


def build_heuristic_valid_predictions(
    valid_df: pd.DataFrame,
    train_route_stats: pd.DataFrame,
    train_origin_stats: pd.DataFrame,
    train_dest_stats: pd.DataFrame,
    train_global_cost: float,
    train_global_legs: float,
) -> np.ndarray:
    """Predict validation rows using legacy heuristic estimator on train-only stats."""
    preds = []
    for _, row in valid_df.iterrows():
        cost, _, _, _ = estimate_route_cost(
            origin=str(row["origin"]),
            destination=str(row["dest"]),
            route_stats=train_route_stats,
            origin_stats=train_origin_stats,
            dest_stats=train_dest_stats,
            global_cost=train_global_cost,
            global_legs=train_global_legs,
        )
        preds.append(float(cost))
    return np.array(preds, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build travel cost model from Navan export.")
    parser.add_argument("--navan-xlsx", default=None, help="Navan report workbook path.")
    parser.add_argument(
        "--output-dir",
        default=config.OPTIMIZATION_DIR,
        help="Optimization output directory.",
    )
    parser.add_argument(
        "--engine",
        choices=["hybrid", "heuristic"],
        default="hybrid",
        help="Travel-cost matrix engine.",
    )
    parser.add_argument(
        "--min-direct-route-n",
        type=int,
        default=5,
        help="Min direct route sample size before empirical/model blending.",
    )
    parser.add_argument(
        "--shrinkage-k",
        type=float,
        default=10.0,
        help="Shrinkage strength for empirical route blending weight.",
    )
    parser.add_argument(
        "--evaluation-cutoff-date",
        default=None,
        help="Optional YYYY-MM-DD cutoff for model train/valid split.",
    )
    parser.add_argument(
        "--bts-prior-csv",
        default=DEFAULT_BTS_PRIOR_CSV,
        help="Optional US BTS state-pair fare prior CSV.",
    )
    parser.add_argument(
        "--disable-bts-prior",
        action="store_true",
        help="Disable BTS prior even if file is available.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    demand_path = out_dir / "demand_appointments.csv"
    candidates_path = out_dir / "candidate_bases.csv"
    tech_path = out_dir / "tech_master.csv"
    if not demand_path.exists() or not candidates_path.exists() or not tech_path.exists():
        raise FileNotFoundError(
            "Missing optimization inputs. Run scripts/06_build_optimization_inputs.py first."
        )

    navan_path = resolve_path(
        [args.navan_xlsx, os.environ.get("ELEVATE_NAVAN_SOURCE"), DEFAULT_NAVAN_XLSX],
        "Navan workbook",
    )

    clean = pd.read_excel(navan_path, sheet_name="Clean Flights")
    report = pd.read_excel(navan_path, sheet_name="Report")

    management_names = parse_management_users(navan_path)
    clean["traveler_name_norm"] = clean["Traveling User"].map(normalize_name)
    clean["traveler_role_norm"] = clean["Traveler Role"].fillna("").astype(str).str.lower()
    clean["is_management"] = clean["traveler_role_norm"].str.contains("management") | clean[
        "traveler_name_norm"
    ].isin(management_names)
    clean["booking_status_norm"] = clean["Booking Status"].fillna("").astype(str).str.upper()

    clean["usd_total_paid"] = pd.to_numeric(clean["USD Total Paid"], errors="coerce")
    clean["lead_time_days"] = pd.to_numeric(clean["Booking Lead Time (days)"], errors="coerce")
    clean["num_legs"] = pd.to_numeric(clean["Number of Legs"], errors="coerce").fillna(1.0)
    clean["booking_start"] = coerce_excel_datetime(clean["Booking Start Date"])
    clean["booking_end"] = coerce_excel_datetime(clean["Booking End Date"])
    clean["origin"] = clean["Origin Airport"].astype(str).str.strip()
    clean["dest"] = clean["Final Destination Airport"].astype(str).str.strip()
    clean["destination_state_norm"] = clean["Destination State"].map(normalize_state)

    model_flights = clean[~clean["is_management"]].copy()
    model_flights = model_flights[model_flights["booking_status_norm"].eq("TICKETED")]
    model_flights = model_flights[model_flights["origin"].ne("") & model_flights["dest"].ne("")]
    model_flights = model_flights.dropna(subset=["usd_total_paid"])
    rows_before_positive_filter = int(len(model_flights))
    non_positive_rows = int((pd.to_numeric(model_flights["usd_total_paid"], errors="coerce") <= 0).sum())
    if non_positive_rows:
        model_flights = model_flights[pd.to_numeric(model_flights["usd_total_paid"], errors="coerce") > 0].copy()

    route_stats = (
        model_flights.groupby(["origin", "dest"])
        .agg(
            trip_count=("usd_total_paid", "count"),
            mean_cost=("usd_total_paid", "mean"),
            median_cost=("usd_total_paid", "median"),
            mean_legs=("num_legs", "mean"),
            mean_lead_time_days=("lead_time_days", "mean"),
        )
        .reset_index()
    )
    origin_stats = (
        model_flights.groupby("origin")
        .agg(
            trip_count=("usd_total_paid", "count"),
            mean_cost=("usd_total_paid", "mean"),
            mean_legs=("num_legs", "mean"),
        )
        .reset_index()
    )
    dest_stats = (
        model_flights.groupby("dest")
        .agg(
            trip_count=("usd_total_paid", "count"),
            mean_cost=("usd_total_paid", "mean"),
            mean_legs=("num_legs", "mean"),
        )
        .reset_index()
    )
    global_cost = float(model_flights["usd_total_paid"].mean())
    global_legs = float(model_flights["num_legs"].mean())

    demand = pd.read_csv(demand_path)
    candidates = pd.read_csv(candidates_path)
    tech = pd.read_csv(tech_path)
    demand["state_norm"] = demand["state_norm"].map(normalize_state)
    demand["duration_hours"] = pd.to_numeric(demand["duration_hours"], errors="coerce").fillna(0.0)

    origins = set(candidates["airport_iata"].dropna().astype(str))
    origins |= set(tech["base_airport_iata"].dropna().astype(str))
    origins = sorted([o for o in origins if o and o != "nan"])
    states = sorted([s for s in demand["state_norm"].dropna().unique().tolist() if s and s != "nan"])

    state_weights = build_state_airport_weights(demand)
    default_weights = (
        demand.dropna(subset=["nearest_hub_airport"])
        .groupby("nearest_hub_airport")["duration_hours"]
        .sum()
    )
    default_weights = default_weights / default_weights.sum()

    metrics: dict = {
        "engine": args.engine,
        "min_direct_route_n": int(args.min_direct_route_n),
        "shrinkage_k": float(args.shrinkage_k),
        "navan_source": navan_path,
        "model_rows_before_positive_filter": rows_before_positive_filter,
        "removed_non_positive_paid_rows": non_positive_rows,
        "model_flight_rows": int(len(model_flights)),
        "unique_origins_in_model": int(model_flights["origin"].nunique()),
        "unique_dests_in_model": int(model_flights["dest"].nunique()),
    }
    feature_importance = pd.DataFrame(columns=["feature", "importance"])

    if args.engine == "heuristic":
        matrix = build_matrix_heuristic(
            origins=origins,
            states=states,
            state_weights=state_weights,
            default_weights=default_weights,
            route_stats=route_stats,
            origin_stats=origin_stats,
            dest_stats=dest_stats,
            global_cost=global_cost,
            global_legs=global_legs,
        )
        bts_meta = {
            "bts_prior_enabled": False,
            "reason": "engine_heuristic",
            "state_pairs_loaded": 0,
        }
    else:
        if (
            prepare_training_frame is None
            or train_travel_model is None
            or predict_route_cost is None
        ):
            raise ImportError(
                "Hybrid engine requires scikit-learn. Install dependencies from requirements.txt."
            )

        training_frame = prepare_training_frame(model_flights)
        cutoff = None
        if args.evaluation_cutoff_date:
            cutoff = date.fromisoformat(args.evaluation_cutoff_date)

        bundle, model_metrics, feature_importance, train_df, valid_df = train_travel_model(
            training_frame,
            evaluation_cutoff_date=cutoff,
        )
        metrics.update(model_metrics)

        # Compare against legacy heuristic on holdout for measurable improvement.
        train_route_stats = (
            train_df.groupby(["origin", "dest"])
            .agg(
                trip_count=("usd_total_paid", "count"),
                mean_cost=("usd_total_paid", "mean"),
                mean_legs=("num_legs", "mean"),
            )
            .reset_index()
        )
        train_origin_stats = (
            train_df.groupby("origin")
            .agg(
                trip_count=("usd_total_paid", "count"),
                mean_cost=("usd_total_paid", "mean"),
                mean_legs=("num_legs", "mean"),
            )
            .reset_index()
        )
        train_dest_stats = (
            train_df.groupby("dest")
            .agg(
                trip_count=("usd_total_paid", "count"),
                mean_cost=("usd_total_paid", "mean"),
                mean_legs=("num_legs", "mean"),
            )
            .reset_index()
        )
        train_global_cost = float(train_df["usd_total_paid"].mean())
        train_global_legs = float(train_df["num_legs"].mean())

        pred_valid_model = np.expm1(
            bundle.pipeline.predict(valid_df[bundle.categorical_cols + bundle.numeric_cols])
        )
        pred_valid_heur = build_heuristic_valid_predictions(
            valid_df=valid_df,
            train_route_stats=train_route_stats,
            train_origin_stats=train_origin_stats,
            train_dest_stats=train_dest_stats,
            train_global_cost=train_global_cost,
            train_global_legs=train_global_legs,
        )
        actual_valid = valid_df["usd_total_paid"].values
        metrics.update(regression_metrics(actual_valid, pred_valid_model, "valid_model"))
        metrics.update(regression_metrics(actual_valid, pred_valid_heur, "valid_heuristic"))
        if metrics["valid_heuristic_mae_usd"] > 0:
            metrics["valid_mae_improvement_pct_vs_heuristic"] = float(
                (metrics["valid_heuristic_mae_usd"] - metrics["valid_model_mae_usd"])
                / metrics["valid_heuristic_mae_usd"]
                * 100.0
            )
        else:
            metrics["valid_mae_improvement_pct_vs_heuristic"] = 0.0
        if metrics["valid_heuristic_wmape"] > 0:
            metrics["valid_wmape_improvement_pct_vs_heuristic"] = float(
                (metrics["valid_heuristic_wmape"] - metrics["valid_model_wmape"])
                / metrics["valid_heuristic_wmape"]
                * 100.0
            )
        else:
            metrics["valid_wmape_improvement_pct_vs_heuristic"] = 0.0

        if args.disable_bts_prior:
            bts_prior = {}
            bts_meta = {
                "bts_prior_enabled": False,
                "reason": "disabled_by_flag",
                "state_pairs_loaded": 0,
            }
        else:
            bts_prior, bts_meta = load_bts_prior(args.bts_prior_csv)

        origin_state_lookup = build_origin_state_lookup(candidates, tech)
        matrix = build_hybrid_matrix(
            origins=origins,
            states=states,
            state_weights=state_weights,
            default_weights=default_weights,
            route_stats=route_stats,
            origin_stats=origin_stats,
            dest_stats=dest_stats,
            global_cost=global_cost,
            global_legs=global_legs,
            min_direct_route_n=max(1, int(args.min_direct_route_n)),
            shrinkage_k=max(1e-6, float(args.shrinkage_k)),
            bundle=bundle,
            bts_prior=bts_prior,
            origin_state_lookup=origin_state_lookup,
        )

    report["Booking Status"] = report["Booking Status"].astype(str)
    report["usd_total_paid"] = pd.to_numeric(report["USD Total Paid"], errors="coerce")
    ticketed_spend = float(
        report.loc[report["Booking Status"].eq("TICKETED"), "usd_total_paid"].sum()
    )
    canceled_voided_spend = float(
        report.loc[report["Booking Status"].isin(["CANCELED", "VOIDED"]), "usd_total_paid"].sum()
    )
    total_spend = float(report["usd_total_paid"].sum())
    model_ticketed_spend = float(model_flights["usd_total_paid"].sum())
    model_trip_count = int(len(model_flights))
    model_out_of_policy = int(
        model_flights["Out-of-Policy"].astype(str).str.lower().eq("true").sum()
    )

    baseline_kpis = {
        "navan_source": navan_path,
        "ticketed_spend_usd_report": ticketed_spend,
        "canceled_voided_spend_usd_report": canceled_voided_spend,
        "total_spend_usd_report": total_spend,
        "ticketed_trip_count_report": int(report["Booking Status"].eq("TICKETED").sum()),
        "canceled_voided_trip_count_report": int(
            report["Booking Status"].isin(["CANCELED", "VOIDED"]).sum()
        ),
        "model_ticketed_spend_usd_excluding_management": model_ticketed_spend,
        "model_trip_count_excluding_management": model_trip_count,
        "model_out_of_policy_count_excluding_management": model_out_of_policy,
        "canceled_voided_handling": "baseline_constant",
    }

    metrics["matrix_rows"] = int(len(matrix))
    metrics["matrix_origins"] = int(matrix["origin_airport"].nunique())
    metrics["matrix_states"] = int(matrix["state_norm"].nunique())

    source_counts = matrix["prior_source"].value_counts(dropna=False).to_dict()
    source_shares = {k: float(v) / max(len(matrix), 1) for k, v in source_counts.items()}
    coverage_report = {
        "engine": args.engine,
        "matrix_rows": int(len(matrix)),
        "source_counts": source_counts,
        "source_shares": source_shares,
        "confidence_tier_counts": matrix["confidence_tier"].value_counts(dropna=False).to_dict(),
        "mean_confidence_score": float(pd.to_numeric(matrix["confidence_score"], errors="coerce").mean()),
    }

    bts_cov_report = dict(bts_meta)
    us_cells = int(matrix["state_norm"].isin(US_ABBR).sum())
    bts_cells = int((matrix["prior_source"] == "bts_prior").sum()) if "prior_source" in matrix.columns else 0
    bts_cov_report["us_matrix_cells"] = us_cells
    bts_cov_report["cells_using_bts_prior"] = bts_cells
    bts_cov_report["share_us_cells_using_bts_prior"] = float(bts_cells / us_cells) if us_cells else 0.0
    anomaly_report = build_origin_anomaly_report(matrix=matrix, origin_stats=origin_stats)

    out_dir.mkdir(parents=True, exist_ok=True)
    matrix_out = out_dir / "travel_cost_matrix.csv"
    route_out = out_dir / "navan_route_stats.csv"
    flight_out = out_dir / "navan_clean_flights_modeled.csv"
    kpi_out = out_dir / "baseline_kpis.json"
    metrics_out = out_dir / "travel_model_metrics.json"
    fi_out = out_dir / "travel_model_feature_importance.csv"
    coverage_out = out_dir / "travel_matrix_coverage_report.json"
    bts_cov_out = out_dir / "bts_prior_coverage_report.json"
    anomaly_out = out_dir / "travel_matrix_origin_anomaly_report.json"

    matrix.to_csv(matrix_out, index=False)
    route_stats.to_csv(route_out, index=False)
    model_flights.to_csv(flight_out, index=False)
    with open(kpi_out, "w") as f:
        json.dump(baseline_kpis, f, indent=2)
    with open(metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)
    feature_importance.to_csv(fi_out, index=False)
    with open(coverage_out, "w") as f:
        json.dump(coverage_report, f, indent=2)
    with open(bts_cov_out, "w") as f:
        json.dump(bts_cov_report, f, indent=2)
    with open(anomaly_out, "w") as f:
        json.dump(anomaly_report, f, indent=2)

    print(f"Saved: {matrix_out}")
    print(f"Saved: {route_out}")
    print(f"Saved: {flight_out}")
    print(f"Saved: {kpi_out}")
    print(f"Saved: {metrics_out}")
    print(f"Saved: {fi_out}")
    print(f"Saved: {coverage_out}")
    print(f"Saved: {bts_cov_out}")
    print(f"Saved: {anomaly_out}")
    print("\nTravel model metrics:")
    print(json.dumps(metrics, indent=2))
    print("\nBaseline KPIs:")
    print(json.dumps(baseline_kpis, indent=2))
    print("Step 7 complete.")


if __name__ == "__main__":
    main()
