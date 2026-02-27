"""Step 11: Pre-compute per-(tech/candidate, node) full trip cost table.

Produces full_cost_table.csv in the optimization output directory.
Must run after Steps 7 and 10 (travel cost matrix must exist).
Re-run when demand_appointments.csv, tech_master.csv, or cost matrix changes.

Drive/fly classification:
  - Distance < DRIVE_THRESHOLD_MILES from tech/candidate base → drive
  - Distance >= DRIVE_THRESHOLD_MILES → fly
  - Canadian techs (base_country != USA) always fly regardless of distance

Drive cost:  IRS_MILEAGE_RATE * 2 * median_dist + HOTEL_AVG
Fly cost:    flight_cost(matrix) + RENTAL_CAR_AVG + HOTEL_AVG
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


def load_cost_matrix(output_dir: Path) -> pd.DataFrame:
    """Load the active travel cost matrix (BTS-corrected if configured)."""
    corrected = output_dir / "travel_cost_matrix_bts_corrected.csv"
    if config.BTS_CORRECTED_MATRIX and corrected.exists():
        path = corrected
        print("  [BTS] Using BTS-calibrated travel cost matrix.")
    else:
        path = output_dir / "travel_cost_matrix.csv"
    if not path.exists():
        raise FileNotFoundError(f"Travel cost matrix not found: {path}\nRun steps 7 and 10 first.")
    return pd.read_csv(path)


def build_flight_cost_lookup(
    cost_matrix: pd.DataFrame,
) -> tuple[dict[tuple[str, str], float], dict[str, float], float]:
    """Build (airport, state_norm) → flight cost lookup with fallbacks."""
    matrix = cost_matrix.copy()
    matrix["origin_airport"] = matrix["origin_airport"].astype(str)
    matrix["state_norm"] = matrix["state_norm"].map(normalize_state)
    matrix["expected_cost_usd"] = pd.to_numeric(matrix["expected_cost_usd"], errors="coerce")
    matrix = matrix.dropna(subset=["origin_airport", "state_norm", "expected_cost_usd"])

    exact = {
        (str(r["origin_airport"]), str(r["state_norm"])): float(r["expected_cost_usd"])
        for _, r in matrix.iterrows()
    }
    origin_avg = matrix.groupby("origin_airport")["expected_cost_usd"].mean().to_dict()
    global_avg = float(matrix["expected_cost_usd"].mean())
    return exact, origin_avg, global_avg


def get_flight_cost(
    airport: str,
    state: str,
    exact: dict[tuple[str, str], float],
    origin_avg: dict[str, float],
    global_avg: float,
) -> float:
    """Flight cost lookup with origin-average and global fallbacks."""
    if (airport, state) in exact:
        return exact[(airport, state)]
    if airport in origin_avg:
        return float(origin_avg[airport])
    return global_avg


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
    flight_exact: dict[tuple[str, str], float],
    flight_origin_avg: dict[str, float],
    flight_global_avg: float,
) -> list[dict]:
    """Compute full per-trip cost for all (entity, node) pairs.

    Args:
        entities: list of dicts with keys: id, lat, lon, airport, is_canadian
        demand: normalized demand_appointments DataFrame with node_id column
        flight_exact/origin_avg/global_avg: flight cost lookup maps

    Returns:
        list of cost row dicts
    """
    # Group appointments by node_id once for all entities
    node_groups: dict[str, pd.DataFrame] = {}
    node_states: dict[str, str] = {}
    for node_id, grp in demand.groupby("node_id"):
        node_groups[node_id] = grp
        node_states[node_id] = str(node_id).split("__", 1)[0]

    rows: list[dict] = []
    warned_nodes: set[str] = set()

    for entity in entities:
        entity_id = entity["id"]
        base_lat = float(entity["lat"]) if pd.notna(entity["lat"]) else float("nan")
        base_lon = float(entity["lon"]) if pd.notna(entity["lon"]) else float("nan")
        airport = str(entity["airport"]).strip()
        is_canadian = bool(entity["is_canadian"])

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
            node_state = node_states.get(node_id, "")

            # Classify mode: Canadian techs always fly; others use distance threshold
            if is_canadian:
                trip_mode = "fly"
            elif median_dist < config.DRIVE_THRESHOLD_MILES:
                trip_mode = "drive"
            else:
                trip_mode = "fly"

            # Compute cost components
            if trip_mode == "drive":
                mileage_cost = config.IRS_MILEAGE_RATE_USD_PER_MI * 2.0 * median_dist
                flight_cost = 0.0
                rental_cost = 0.0
                hotel_cost = config.HOTEL_AVG_USD
                unit_cost = mileage_cost + hotel_cost
            else:
                flight_cost = get_flight_cost(
                    airport, node_state, flight_exact, flight_origin_avg, flight_global_avg
                )
                mileage_cost = 0.0
                rental_cost = config.RENTAL_CAR_AVG_USD
                hotel_cost = config.HOTEL_AVG_USD
                unit_cost = flight_cost + rental_cost + hotel_cost

            rows.append(
                {
                    "tech_or_candidate_id": entity_id,
                    "node_id": node_id,
                    "trip_mode": trip_mode,
                    "median_dist_mi": round(median_dist, 2),
                    "flight_cost_usd": round(flight_cost, 2),
                    "mileage_cost_usd": round(mileage_cost, 2),
                    "rental_cost_usd": round(rental_cost, 2),
                    "hotel_cost_usd": round(hotel_cost, 2),
                    "unit_cost_usd": round(unit_cost, 2),
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
            raise FileNotFoundError(f"Missing: {p}\nRun steps 6 and 7 first.")

    # Load data
    tech_df = pd.read_csv(tech_path)
    candidates_df = pd.read_csv(candidates_path)
    demand_df = pd.read_csv(demand_path)
    cost_matrix = load_cost_matrix(out_dir)

    flight_exact, flight_origin_avg, flight_global_avg = build_flight_cost_lookup(cost_matrix)
    airport_latlon = build_airport_lat_lon()

    demand_prepared = prepare_demand(demand_df)
    n_nodes = demand_prepared["node_id"].nunique()
    n_appts = len(demand_prepared)
    print(f"  Demand: {n_appts} appointments across {n_nodes} nodes.")

    # Build entity list for existing techs
    tech_entities: list[dict] = []
    for _, row in tech_df.iterrows():
        lat = pd.to_numeric(row.get("base_lat"), errors="coerce")
        lon = pd.to_numeric(row.get("base_lon"), errors="coerce")
        country = str(row.get("base_country", "USA")).strip().upper()
        is_canadian = country != "USA"
        airport = str(row.get("base_airport_iata", "")).strip()
        tech_entities.append(
            {
                "id": str(row["tech_id"]),
                "lat": lat,
                "lon": lon,
                "airport": airport,
                "is_canadian": is_canadian,
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
                "is_canadian": False,  # New hires assumed US-based
            }
        )

    print(f"Processing {len(tech_entities)} techs × {n_nodes} nodes...")
    tech_rows = compute_entity_node_costs(
        tech_entities, demand_prepared, flight_exact, flight_origin_avg, flight_global_avg
    )

    print(f"Processing {len(candidate_entities)} candidates × {n_nodes} nodes...")
    candidate_rows = compute_entity_node_costs(
        candidate_entities, demand_prepared, flight_exact, flight_origin_avg, flight_global_avg
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
        drive_mask = out_df["trip_mode"] == "drive"
        fly_mask = out_df["trip_mode"] == "fly"
        drive_pct = drive_mask.mean() * 100.0
        print(f"\nSaved: {out_path}")
        print(f"  Total rows: {n_total:,}  (techs: {n_tech_ids}, candidates: {n_cand_ids})")
        print(f"  Drive: {drive_pct:.1f}%  Fly: {100.0 - drive_pct:.1f}%")
        print(f"  Mean unit cost: ${out_df['unit_cost_usd'].mean():,.2f}")
        if drive_mask.any():
            print(f"  Drive mean: ${out_df.loc[drive_mask, 'unit_cost_usd'].mean():,.2f}")
        if fly_mask.any():
            print(f"  Fly mean:   ${out_df.loc[fly_mask, 'unit_cost_usd'].mean():,.2f}")
        print(
            f"\n  Cost constants used:\n"
            f"    IRS mileage rate: ${config.IRS_MILEAGE_RATE_USD_PER_MI:.2f}/mi\n"
            f"    Rental car avg:   ${config.RENTAL_CAR_AVG_USD:.2f}/trip\n"
            f"    Hotel avg:        ${config.HOTEL_AVG_USD:.2f}/trip\n"
            f"    Drive threshold:  {config.DRIVE_THRESHOLD_MILES:.0f} miles"
        )
    else:
        print(f"\nSaved: {out_path} (empty — no valid (entity, node) pairs found)")

    print("Step 11 complete.")


if __name__ == "__main__":
    main()
