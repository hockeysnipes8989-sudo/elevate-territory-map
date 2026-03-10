"""Step 11: Pre-compute per-(tech/candidate, node) full trip cost table.

Produces full_cost_table.csv in the optimization output directory.
Must run after Step 06 (demand_appointments.csv, tech_master.csv,
candidate_bases.csv must exist).

Flight cost: BTS Q2 2025 itinerary fare × 1.6 corporate premium.
Origin-only — cost does not vary by destination.

Three-tier trip classification (distance-based):
  < 100 mi:    drive_day       — mileage only
  100–300 mi:  drive_overnight — mileage or rental + 1 hotel night
  >= 300 mi:   fly             — flight + duration-scaled hotel + rental car

Drive trips switch from personal mileage to rental-car economics when the
one-way median distance for the node reaches PERSONAL_VEHICLE_MAX_ONE_WAY_MI.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import config
from optimization_utils import haversine_km, normalize_state

KM_PER_MILE = 1.60934
MILES_PER_KM = 1.0 / KM_PER_MILE


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    return haversine_km(lat1, lon1, lat2, lon2) * MILES_PER_KM


def get_flight_cost(airport: str) -> float:
    """BTS Q2 2025 fare × corporate premium. Origin-only, no destination dependency."""
    code = airport.strip().upper()
    raw = config.BTS_RAW_ITINERARY_FARES.get(code, config.BTS_NATIONAL_FALLBACK)
    return raw * config.CORPORATE_TRAVEL_PREMIUM


def compute_trip_span_days(avg_days: float | None) -> int:
    """Round modeled calendar-window duration up to whole trip-span days."""
    if avg_days is None or pd.isna(avg_days):
        avg_days = config.HOTEL_AVG_NIGHTS
    return max(1, int(np.ceil(float(avg_days))))


def compute_ground_transport(
    median_dist_mi: float,
    trip_span_days: int,
) -> dict:
    """Return ground transport mode, days, and cost for a one-way median distance."""
    if median_dist_mi < config.PERSONAL_VEHICLE_MAX_ONE_WAY_MI:
        return {
            "ground_transport_mode": "personal_vehicle",
            "rental_days": 0,
            "mileage_cost_usd": config.IRS_MILEAGE_RATE_USD_PER_MI * 2.0 * median_dist_mi,
            "rental_cost_usd": 0.0,
        }
    return {
        "ground_transport_mode": "rental_car",
        "rental_days": trip_span_days,
        "mileage_cost_usd": 0.0,
        "rental_cost_usd": config.RENTAL_CAR_DAILY_RATE_USD * trip_span_days,
    }


def apply_travel_cost_policy(
    employee_style_unit_cost: float,
    travel_cost_policy: str,
    contractor_cost_multiplier: float,
    contractor_cost_cap_usd: float,
    contractor_dispatch_surcharge_usd: float,
) -> float:
    """Transform employee-style cost into the effective unit cost used by Step 08."""
    if travel_cost_policy == config.TRAVEL_COST_POLICY_CONTRACTOR:
        effective = employee_style_unit_cost * contractor_cost_multiplier + contractor_dispatch_surcharge_usd
        return min(effective, contractor_cost_cap_usd)
    return employee_style_unit_cost


def build_airport_lat_lon() -> dict[str, tuple[float, float]]:
    """Build airport code → (lat, lon) from config.MAJOR_AIRPORTS."""
    return {
        ap["code"]: (float(ap["lat"]), float(ap["lon"]))
        for ap in config.MAJOR_AIRPORTS
    }


def prepare_demand(demand_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize demand_appointments for distance computation."""
    df = demand_df.copy()
    df["state_norm"] = df["state_norm"].map(normalize_state)
    df["skill_class"] = df["skill_class"].astype(str)
    df["node_id"] = df["state_norm"] + "__" + df["skill_class"]
    df = df.dropna(subset=["state_norm", "node_id"])
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])
    return df


def compute_entity_node_costs(
    entities: list[dict],
    demand: pd.DataFrame,
    node_avg_days: dict[str, float] | None = None,
) -> list[dict]:
    """Compute full per-trip cost for all (entity, node) pairs.

    Args:
        entities: list of dicts with keys: id, lat, lon, airport
        demand: normalized demand_appointments DataFrame with node_id column
        node_avg_days: node_id → average appointment duration in days (for hotel scaling)

    Returns:
        list of cost row dicts
    """
    # Group appointments by node_id once for all entities
    node_groups: dict[str, pd.DataFrame] = {}
    for node_id, grp in demand.groupby("node_id"):
        node_groups[node_id] = grp

    rows: list[dict] = []
    warned_nodes: set[str] = set()

    for entity in entities:
        entity_id = entity["id"]
        base_lat = float(entity["lat"]) if pd.notna(entity["lat"]) else float("nan")
        base_lon = float(entity["lon"]) if pd.notna(entity["lon"]) else float("nan")
        airport = str(entity["airport"]).strip()
        travel_cost_policy = str(
            entity.get("travel_cost_policy", config.TRAVEL_COST_POLICY_EMPLOYEE)
        ).strip() or config.TRAVEL_COST_POLICY_EMPLOYEE
        contractor_cost_multiplier = float(
            entity.get("contractor_cost_multiplier", 1.0)
            if pd.notna(entity.get("contractor_cost_multiplier", 1.0))
            else 1.0
        )
        contractor_cost_cap_usd = float(
            entity.get("contractor_cost_cap_usd", np.inf)
            if pd.notna(entity.get("contractor_cost_cap_usd", np.inf))
            else np.inf
        )
        contractor_dispatch_surcharge_usd = float(
            entity.get("contractor_dispatch_surcharge_usd", 0.0)
            if pd.notna(entity.get("contractor_dispatch_surcharge_usd", 0.0))
            else 0.0
        )

        if np.isnan(base_lat) or np.isnan(base_lon):
            warnings.warn(
                f"Entity {entity_id} has no valid lat/lon — skipping all node costs.",
                stacklevel=2,
            )
            continue

        for node_id, node_appts in node_groups.items():
            lats = node_appts["lat"].values
            lons = node_appts["lon"].values

            if len(lats) == 0:
                if node_id not in warned_nodes:
                    warned_nodes.add(node_id)
                    print(f"  WARNING: node {node_id} has 0 valid appointments — skipping.")
                continue

            # Compute haversine from entity base to each appointment in node
            dists_mi = np.array([
                haversine_miles(base_lat, base_lon, float(la), float(lo))
                for la, lo in zip(lats, lons)
            ])
            median_dist = float(np.median(dists_mi))
            avg_days = node_avg_days.get(node_id, config.HOTEL_AVG_NIGHTS) if node_avg_days else config.HOTEL_AVG_NIGHTS
            trip_span_days = compute_trip_span_days(avg_days)

            # --- Three-tier trip classification (distance-based) ---
            if median_dist < config.SAME_DAY_DRIVE_THRESHOLD_MI:
                # Same-day drive: mileage only, tech drives home same day
                trip_mode = "drive_day"
                ground = compute_ground_transport(median_dist, trip_span_days)
                mileage_cost = ground["mileage_cost_usd"]
                flight_cost = 0.0
                rental_cost = ground["rental_cost_usd"]
                hotel_nights = 0
                hotel_cost = 0.0
                ground_transport_mode = ground["ground_transport_mode"]
                rental_days = ground["rental_days"]
                employee_style_unit_cost = mileage_cost + rental_cost
            elif median_dist < config.OVERNIGHT_DRIVE_THRESHOLD_MI:
                # Overnight drive: mileage + exactly 1 hotel night
                trip_mode = "drive_overnight"
                ground = compute_ground_transport(median_dist, trip_span_days)
                mileage_cost = ground["mileage_cost_usd"]
                flight_cost = 0.0
                rental_cost = ground["rental_cost_usd"]
                hotel_nights = 1
                hotel_cost = config.HOTEL_NIGHTLY_RATE_USD
                ground_transport_mode = ground["ground_transport_mode"]
                rental_days = ground["rental_days"]
                employee_style_unit_cost = mileage_cost + rental_cost + hotel_cost
            else:
                # Fly trip: flight + duration-scaled hotel + rental car
                trip_mode = "fly"
                hotel_nights = trip_span_days
                flight_cost = get_flight_cost(airport)
                mileage_cost = 0.0
                rental_days = trip_span_days
                rental_cost = config.RENTAL_CAR_DAILY_RATE_USD * rental_days
                hotel_cost = hotel_nights * config.HOTEL_NIGHTLY_RATE_USD
                ground_transport_mode = "rental_car"
                employee_style_unit_cost = flight_cost + rental_cost + hotel_cost

            effective_unit_cost = apply_travel_cost_policy(
                employee_style_unit_cost=employee_style_unit_cost,
                travel_cost_policy=travel_cost_policy,
                contractor_cost_multiplier=contractor_cost_multiplier,
                contractor_cost_cap_usd=contractor_cost_cap_usd,
                contractor_dispatch_surcharge_usd=contractor_dispatch_surcharge_usd,
            )

            rows.append(
                {
                    "tech_or_candidate_id": entity_id,
                    "node_id": node_id,
                    "trip_mode": trip_mode,
                    "median_dist_mi": round(median_dist, 2),
                    "trip_span_days": trip_span_days,
                    "ground_transport_mode": ground_transport_mode,
                    "flight_cost_usd": round(flight_cost, 2),
                    "mileage_cost_usd": round(mileage_cost, 2),
                    "rental_cost_usd": round(rental_cost, 2),
                    "rental_days": rental_days,
                    "hotel_cost_usd": round(hotel_cost, 2),
                    "hotel_nights": hotel_nights,
                    "travel_cost_policy": travel_cost_policy,
                    "contractor_cost_multiplier": contractor_cost_multiplier,
                    "contractor_cost_cap_usd": (
                        round(contractor_cost_cap_usd, 2)
                        if np.isfinite(contractor_cost_cap_usd)
                        else np.nan
                    ),
                    "contractor_dispatch_surcharge_usd": round(contractor_dispatch_surcharge_usd, 2),
                    "employee_style_unit_cost_usd": round(employee_style_unit_cost, 2),
                    "effective_unit_cost_usd": round(effective_unit_cost, 2),
                    "unit_cost_usd": round(effective_unit_cost, 2),
                }
            )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build full per-(tech/candidate, node) trip cost table."
    )
    parser.add_argument(
        "--output-dir",
        default=config.OPTIMIZATION_DIR,
        help="Optimization output directory.",
    )
    args = parser.parse_args()
    out_dir = Path(args.output_dir)

    # Verify required inputs exist
    tech_path = out_dir / "tech_master.csv"
    candidates_path = out_dir / "candidate_bases.csv"
    demand_path = out_dir / "demand_appointments.csv"
    for p in [tech_path, candidates_path, demand_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}\nRun step 06 first.")

    # Load data
    tech_df = pd.read_csv(tech_path)
    candidates_df = pd.read_csv(candidates_path)
    demand_df = pd.read_csv(demand_path)
    airport_latlon = build_airport_lat_lon()

    print(f"  Flight cost model: BTS Q2 2025 × {config.CORPORATE_TRAVEL_PREMIUM:.1f}x corporate premium")

    demand_prepared = prepare_demand(demand_df)
    n_nodes = demand_prepared["node_id"].nunique()
    n_appts = len(demand_prepared)
    print(f"  Demand: {n_appts} appointments across {n_nodes} nodes.")

    # Compute node-level average duration for duration-scaled hotel costs.
    # duration_hours is treated as calendar-window appointment time in this
    # model, which matches the current optimization workload basis.
    demand_prepared["duration_hours"] = pd.to_numeric(
        demand_prepared["duration_hours"], errors="coerce"
    ).fillna(24.0 * config.HOTEL_AVG_NIGHTS)  # fallback: avg Navan stay
    node_avg_duration = demand_prepared.groupby("node_id")["duration_hours"].mean()
    # Convert calendar-window hours to days: 24 hours = 1 calendar day.
    node_avg_days: dict[str, float] = (node_avg_duration / 24.0).to_dict()
    print(f"  Node avg duration range: {min(node_avg_days.values()):.2f} – {max(node_avg_days.values()):.2f} days")

    # Build entity list for existing techs
    tech_entities: list[dict] = []
    for _, row in tech_df.iterrows():
        lat = pd.to_numeric(row.get("base_lat"), errors="coerce")
        lon = pd.to_numeric(row.get("base_lon"), errors="coerce")
        airport = str(row.get("base_airport_iata", "")).strip()
        tech_entities.append(
            {
                "id": str(row["tech_id"]),
                "lat": lat,
                "lon": lon,
                "airport": airport,
                "travel_cost_policy": row.get("travel_cost_policy", config.TRAVEL_COST_POLICY_EMPLOYEE),
                "contractor_cost_multiplier": row.get("contractor_cost_multiplier", 1.0),
                "contractor_cost_cap_usd": row.get("contractor_cost_cap_usd", np.nan),
                "contractor_dispatch_surcharge_usd": row.get("contractor_dispatch_surcharge_usd", 0.0),
            }
        )

    # Build entity list for candidates (new-hire base locations)
    candidate_entities: list[dict] = []
    for _, row in candidates_df.iterrows():
        lat = pd.to_numeric(row.get("lat"), errors="coerce")
        lon = pd.to_numeric(row.get("lon"), errors="coerce")
        airport = str(row.get("airport_iata", "")).strip()

        # Fall back to airport coordinates when candidate lat/lon is missing
        if pd.isna(lat) or pd.isna(lon):
            if airport in airport_latlon:
                lat, lon = airport_latlon[airport]
                print(
                    f"  Candidate {row['candidate_id']}: "
                    f"using airport {airport} lat/lon as fallback."
                )
            else:
                print(
                    f"  WARNING: Candidate {row['candidate_id']} has no lat/lon "
                    f"and unknown airport — skipping."
                )
                continue

        candidate_entities.append(
            {
                "id": str(row["candidate_id"]),
                "lat": lat,
                "lon": lon,
                "airport": airport,
                "travel_cost_policy": config.TRAVEL_COST_POLICY_EMPLOYEE,
                "contractor_cost_multiplier": 1.0,
                "contractor_cost_cap_usd": np.nan,
                "contractor_dispatch_surcharge_usd": 0.0,
            }
        )

    print(f"Processing {len(tech_entities)} techs × {n_nodes} nodes...")
    tech_rows = compute_entity_node_costs(
        tech_entities, demand_prepared, node_avg_days=node_avg_days,
    )

    print(f"Processing {len(candidate_entities)} candidates × {n_nodes} nodes...")
    candidate_rows = compute_entity_node_costs(
        candidate_entities, demand_prepared, node_avg_days=node_avg_days,
    )

    all_rows = tech_rows + candidate_rows
    out_df = pd.DataFrame(all_rows)

    out_path = out_dir / "full_cost_table.csv"
    out_df.to_csv(out_path, index=False)

    # Summary stats
    n_tech_ids = len(set(r["tech_or_candidate_id"] for r in tech_rows))
    n_cand_ids = len(set(r["tech_or_candidate_id"] for r in candidate_rows))
    n_total = len(out_df)

    if not out_df.empty:
        drive_day_mask = out_df["trip_mode"] == "drive_day"
        drive_overnight_mask = out_df["trip_mode"] == "drive_overnight"
        fly_mask = out_df["trip_mode"] == "fly"
        drive_day_pct = drive_day_mask.mean() * 100.0
        drive_overnight_pct = drive_overnight_mask.mean() * 100.0
        fly_pct = fly_mask.mean() * 100.0
        print(f"\nSaved: {out_path}")
        print(f"  Total rows: {n_total:,}  (techs: {n_tech_ids}, candidates: {n_cand_ids})")
        print(f"  Drive (same-day): {drive_day_pct:.1f}%  Drive (overnight): {drive_overnight_pct:.1f}%  Fly: {fly_pct:.1f}%")
        print(f"  Mean unit cost: ${out_df['unit_cost_usd'].mean():,.2f}")
        if drive_day_mask.any():
            print(f"  Drive (same-day) mean:  ${out_df.loc[drive_day_mask, 'unit_cost_usd'].mean():,.2f}")
        if drive_overnight_mask.any():
            print(f"  Drive (overnight) mean: ${out_df.loc[drive_overnight_mask, 'unit_cost_usd'].mean():,.2f}")
        if fly_mask.any():
            print(f"  Fly mean:               ${out_df.loc[fly_mask, 'unit_cost_usd'].mean():,.2f}")
        print(f"  Hotel nightly rate: ${config.HOTEL_NIGHTLY_RATE_USD:.2f}")
        print(f"  Mean hotel nights: {out_df['hotel_nights'].mean():.1f}")
        print(f"  Mean hotel cost: ${out_df['hotel_cost_usd'].mean():,.2f}")
        print(f"  Mean rental days: {out_df['rental_days'].mean():.1f}")
        # Hotel nights distribution
        nights_dist = out_df['hotel_nights'].value_counts().sort_index()
        print("  Hotel nights distribution:")
        for nights_val, cnt in nights_dist.items():
            print(f"    {int(nights_val)} nights: {cnt} ({cnt/len(out_df)*100:.1f}%)")
        # Flight cost sanity check
        if fly_mask.any():
            fly_flights = out_df.loc[fly_mask, "flight_cost_usd"]
            print(f"  Flight cost range: ${fly_flights.min():,.2f} – ${fly_flights.max():,.2f}")
            print(f"  Flight cost mean:  ${fly_flights.mean():,.2f}")
            print(
                f"\n  Cost constants used:\n"
                f"    BTS corporate premium:       {config.CORPORATE_TRAVEL_PREMIUM:.1f}x\n"
                f"    BTS national fallback:       ${config.BTS_NATIONAL_FALLBACK:.2f}\n"
                f"    IRS mileage rate:            ${config.IRS_MILEAGE_RATE_USD_PER_MI:.2f}/mi\n"
                f"    Rental car daily rate:       ${config.RENTAL_CAR_DAILY_RATE_USD:.2f}/day\n"
                f"    Personal vehicle max 1-way:  {config.PERSONAL_VEHICLE_MAX_ONE_WAY_MI:.0f} miles\n"
                f"    Hotel nightly rate:          ${config.HOTEL_NIGHTLY_RATE_USD:.2f}/night\n"
                f"    Same-day drive threshold:    {config.SAME_DAY_DRIVE_THRESHOLD_MI:.0f} miles\n"
                f"    Overnight drive threshold:   {config.OVERNIGHT_DRIVE_THRESHOLD_MI:.0f} miles"
            )
    else:
        print(f"\nSaved: {out_path} (empty — no valid (entity, node) pairs found)")

    print("Step 11 complete.")


if __name__ == "__main__":
    main()
