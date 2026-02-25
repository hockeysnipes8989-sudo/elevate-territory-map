"""Step 7: Build travel cost model from Navan data (independent input)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import config
from optimization_utils import (
    coerce_excel_datetime,
    normalize_name,
    normalize_state,
)


DEFAULT_NAVAN_XLSX = config.EXTERNAL_NAVAN_XLSX


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
    """Build per-state destination airport demand weights."""
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


def mode_confidence(tiers: list[str]) -> str:
    """Pick best available confidence tier from tier list."""
    order = [
        "direct_route",
        "reverse_route",
        "origin_dest_blend",
        "origin_only",
        "destination_only",
        "global_fallback",
    ]
    for tier in order:
        if tier in tiers:
            return tier
    return "global_fallback"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build travel cost model from Navan export.")
    parser.add_argument("--navan-xlsx", default=None, help="Navan report workbook path.")
    parser.add_argument(
        "--output-dir",
        default=config.OPTIMIZATION_DIR,
        help="Optimization output directory.",
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

    clean["usd_total_paid"] = pd.to_numeric(clean["USD Total Paid"], errors="coerce")
    clean["lead_time_days"] = pd.to_numeric(clean["Booking Lead Time (days)"], errors="coerce")
    clean["num_legs"] = pd.to_numeric(clean["Number of Legs"], errors="coerce").fillna(1.0)
    clean["booking_start"] = coerce_excel_datetime(clean["Booking Start Date"])
    clean["booking_end"] = coerce_excel_datetime(clean["Booking End Date"])
    clean["origin"] = clean["Origin Airport"].astype(str).str.strip()
    clean["dest"] = clean["Final Destination Airport"].astype(str).str.strip()
    clean["destination_state_norm"] = clean["Destination State"].map(normalize_state)

    model_flights = clean[~clean["is_management"]].copy()
    model_flights = model_flights[model_flights["origin"].ne("") & model_flights["dest"].ne("")]
    model_flights = model_flights.dropna(subset=["usd_total_paid"])

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

    matrix_rows = []
    for origin in origins:
        for state in states:
            weights = state_weights.get(state, default_weights)
            weighted_cost = 0.0
            weighted_legs = 0.0
            sample_size = 0
            tiers = []
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
                tiers.append(tier)
            matrix_rows.append(
                {
                    "origin_airport": origin,
                    "state_norm": state,
                    "expected_cost_usd": weighted_cost,
                    "expected_legs": weighted_legs,
                    "sample_size": int(sample_size),
                    "confidence_tier": mode_confidence(tiers),
                }
            )

    matrix = pd.DataFrame(matrix_rows)

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

    out_dir.mkdir(parents=True, exist_ok=True)
    matrix_out = out_dir / "travel_cost_matrix.csv"
    route_out = out_dir / "navan_route_stats.csv"
    flight_out = out_dir / "navan_clean_flights_modeled.csv"
    kpi_out = out_dir / "baseline_kpis.json"

    matrix.to_csv(matrix_out, index=False)
    route_stats.to_csv(route_out, index=False)
    model_flights.to_csv(flight_out, index=False)
    with open(kpi_out, "w") as f:
        json.dump(baseline_kpis, f, indent=2)

    print(f"Saved: {matrix_out}")
    print(f"Saved: {route_out}")
    print(f"Saved: {flight_out}")
    print(f"Saved: {kpi_out}")
    print("\nBaseline KPIs:")
    print(json.dumps(baseline_kpis, indent=2))
    print("Step 7 complete.")


if __name__ == "__main__":
    main()
