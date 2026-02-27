"""Step 8: Run MILP location-allocation optimization scenarios."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, milp

sys.path.insert(0, os.path.dirname(__file__))
import config
from optimization_utils import CANADA_ABBR, normalize_state


CANADA_NODE_STATES = set(CANADA_ABBR) | {"CANADA"}


def load_inputs(output_dir: Path) -> dict:
    """Load all prerequisite optimization inputs."""
    corrected_path = output_dir / "travel_cost_matrix_bts_corrected.csv"
    if config.BTS_CORRECTED_MATRIX and corrected_path.exists():
        cost_matrix_file = corrected_path
        print("  [BTS] Using BTS-calibrated travel cost matrix.")
    else:
        cost_matrix_file = output_dir / "travel_cost_matrix.csv"
    files = {
        "tech": output_dir / "tech_master.csv",
        "demand": output_dir / "demand_appointments.csv",
        "candidates": output_dir / "candidate_bases.csv",
        "cost_matrix": cost_matrix_file,
        "baseline": output_dir / "baseline_kpis.json",
    }
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        joined = "\n".join(f"- {p}" for p in missing)
        raise FileNotFoundError(f"Missing required input files. Run steps 6 and 7 first.\n{joined}")

    with open(files["baseline"], "r") as f:
        baseline = json.load(f)

    result = {
        "tech": pd.read_csv(files["tech"]),
        "demand": pd.read_csv(files["demand"]),
        "candidates": pd.read_csv(files["candidates"]),
        "cost_matrix": pd.read_csv(files["cost_matrix"]),
        "baseline": baseline,
        "full_cost_df": None,
    }

    if config.FULL_COST_MODEL:
        full_cost_path = output_dir / "full_cost_table.csv"
        if full_cost_path.exists():
            result["full_cost_df"] = pd.read_csv(full_cost_path)
            print(
                f"  [FullCost] Loaded full_cost_table.csv "
                f"({len(result['full_cost_df']):,} rows, "
                f"drive/fly model active)."
            )
        else:
            print(
                "  [FullCost] WARNING: FULL_COST_MODEL=True but full_cost_table.csv not found. "
                "Run scripts/11_build_full_cost_table.py first. "
                "Falling back to flight-cost-only model."
            )

    return result


def build_demand_nodes(demand: pd.DataFrame) -> pd.DataFrame:
    """Aggregate appointments into demand nodes by state + skill class."""
    demand = demand.copy()
    demand["state_norm"] = demand["state_norm"].map(normalize_state)
    demand["duration_hours"] = pd.to_numeric(demand["duration_hours"], errors="coerce").fillna(8.0)
    dropped_mask = demand["state_norm"].isna() | (demand["state_norm"] == "")
    dropped_count = int(dropped_mask.sum())
    if dropped_count:
        dropped_ids = demand.loc[dropped_mask, "appointment_id"].tolist()
        print(f"  build_demand_nodes: dropping {dropped_count} row(s) with null/empty state_norm.")
        print(f"    Appointment IDs: {dropped_ids}")
        pct = dropped_count / max(len(demand), 1) * 100.0
        if pct > 5.0:
            print(f"  WARNING: {pct:.1f}% of demand rows dropped — exceeds 5% threshold. Check state_norm values.")
    demand = demand[~dropped_mask].copy()

    grouped = (
        demand.groupby(["state_norm", "skill_class", "required_hps", "required_ls"], as_index=False)
        .agg(
            appointment_count=("appointment_id", "count"),
            demand_hours=("duration_hours", "sum"),
            territory_count=("territory", "nunique"),
        )
        .sort_values(["state_norm", "skill_class"])
        .reset_index(drop=True)
    )
    grouped["avg_hours_per_appointment"] = grouped["demand_hours"] / grouped["appointment_count"]
    grouped["node_id"] = grouped.apply(
        lambda r: f"{r['state_norm']}__{r['skill_class']}",
        axis=1,
    )
    return grouped


def build_cost_lookup(cost_matrix: pd.DataFrame) -> tuple[dict[tuple[str, str], float], dict[str, float], float]:
    """Build exact + fallback cost maps."""
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


def get_cost(
    origin_airport: str,
    state_norm: str,
    exact_cost: dict[tuple[str, str], float],
    origin_avg: dict[str, float],
    global_avg: float,
) -> float:
    """Fetch route cost with fallback."""
    if (origin_airport, state_norm) in exact_cost:
        return exact_cost[(origin_airport, state_norm)]
    if origin_airport in origin_avg:
        return float(origin_avg[origin_airport])
    return float(global_avg)


def build_full_cost_lookup(
    full_cost_df: pd.DataFrame,
) -> dict[tuple[str, str], float]:
    """Build (tech_or_candidate_id, node_id) → unit_cost_usd from full_cost_table.csv."""
    return {
        (str(row.tech_or_candidate_id), str(row.node_id)): float(row.unit_cost_usd)
        for row in full_cost_df.itertuples(index=False)
    }


def is_canada_node_state(state_norm: str) -> bool:
    """Return True for Canadian province nodes plus generic CANADA node label."""
    return str(state_norm).strip().upper() in CANADA_NODE_STATES


def tech_eligible_for_node(tech: pd.Series, node: pd.Series, contractor_scope: str) -> bool:
    """Check skill and special geography constraints."""
    if int(node["required_hps"]) and int(tech["skill_hps"]) == 0:
        return False
    if int(node["required_ls"]) and int(tech["skill_ls"]) == 0:
        return False
    if int(tech["skill_patient"]) == 0:
        return False

    tech_state = str(tech.get("base_state", ""))
    node_state = str(node.get("state_norm", ""))
    is_contractor = str(tech.get("employment_type", "")).lower() == "contractor"
    florida_only = int(tech.get("constraint_florida_only", 0)) == 1
    canada_wide = int(tech.get("constraint_canada_wide", 0)) == 1
    node_is_canada = is_canada_node_state(node_state)

    if florida_only and node_state != "FL":
        return False
    if is_contractor and contractor_scope == "texas_only" and node_state != "TX":
        return False
    # Canada-wide specialists (Hakim) should only cover Canada; Canada nodes should only use them.
    if node_is_canada and not canada_wide:
        return False
    if canada_wide and not node_is_canada:
        return False

    # If no origin airport, don't allow assignment.
    if not str(tech.get("base_airport_iata", "")).strip():
        return False
    return True


def _infeasible_summary(hire_count: int, message: str, baseline_canceled_voided_usd: float) -> dict:
    """Build a summary row for a scenario that could not be solved.

    Uses None instead of float('nan') so the dict serializes safely via
    json.dumps (NaN is not valid JSON).
    """
    return {
        "scenario_hires": int(hire_count),
        "solver_status": -1,
        "solver_proven_optimal": False,
        "solver_message": message,
        "solver_mip_gap": None,
        "solver_mip_node_count": 0,
        "objective_value": None,
        "total_appointments": None,
        "served_appointments": None,
        "unmet_appointments": None,
        "travel_cost_usd": None,
        "out_of_region_penalty_usd": None,
        "hire_cost_usd": None,
        "unmet_penalty_usd": None,
        "modeled_total_cost_usd": None,
        "hours_per_capacity_unit": None,
        "new_hire_capacity_hours": None,
        "mean_existing_utilization": None,
        "max_existing_utilization": None,
        "baseline_canceled_voided_usd": baseline_canceled_voided_usd,
        "economic_total_with_overhead_usd": None,
    }


def solve_scenario(
    hire_count: int,
    tech: pd.DataFrame,
    nodes: pd.DataFrame,
    candidates: pd.DataFrame,
    exact_cost: dict[tuple[str, str], float],
    origin_avg: dict[str, float],
    global_avg: float,
    contractor_scope: str,
    target_utilization: float,
    out_of_region_penalty: float,
    unmet_penalty: float,
    annual_hire_cost_usd: float,
    max_hires_per_base: int,
    time_limit_sec: int,
    full_cost_lookup: dict[tuple[str, str], float] | None = None,
) -> dict:
    """Solve one MILP scenario for a fixed new-hire count."""
    nodes = nodes.reset_index(drop=True).copy()
    tech = tech.reset_index(drop=True).copy()
    candidates = candidates.reset_index(drop=True).copy()

    total_demand_hours = float(nodes["demand_hours"].sum())
    total_availability = float(tech["availability_fte"].sum())
    hours_per_unit = total_demand_hours / max(total_availability * target_utilization, 1e-6)

    tech["capacity_hours"] = tech["availability_fte"] * hours_per_unit
    new_hire_capacity_hours = hours_per_unit

    var_names: list[str] = []
    lb: list[float] = []
    ub: list[float] = []
    integrality: list[int] = []
    obj: list[float] = []
    meta: list[dict] = []

    x_idx: dict[tuple[int, int], int] = {}
    z_idx: dict[tuple[int, int], int] = {}
    y_idx: dict[int, int] = {}
    u_idx: dict[int, int] = {}
    candidate_indices = list(candidates.index) if hire_count > 0 else []
    _full_cost_fallback_count = 0  # track missing full_cost_lookup entries

    # Existing tech assignment vars: appointments assigned to node.
    for ti, trow in tech.iterrows():
        for ni, nrow in nodes.iterrows():
            if not tech_eligible_for_node(trow, nrow, contractor_scope):
                continue
            idx = len(var_names)
            x_idx[(ti, ni)] = idx
            var_names.append(f"x__{trow['tech_id']}__{nrow['node_id']}")
            lb.append(0.0)
            ub.append(float(nrow["appointment_count"]))
            integrality.append(1)

            if full_cost_lookup is not None:
                _fc_key = (str(trow["tech_id"]), str(nrow["node_id"]))
                if _fc_key not in full_cost_lookup:
                    _full_cost_fallback_count += 1
                base_cost = full_cost_lookup.get(
                    _fc_key,
                    global_avg + config.RENTAL_CAR_AVG_USD + config.HOTEL_AVG_USD,
                )
            else:
                base_cost = get_cost(
                    str(trow["base_airport_iata"]),
                    str(nrow["state_norm"]),
                    exact_cost,
                    origin_avg,
                    global_avg,
                )
            is_out_region = int(str(trow.get("base_state", "")) != str(nrow["state_norm"]))
            penalty = out_of_region_penalty if is_out_region else 0.0
            obj.append(base_cost + penalty)
            meta.append(
                {
                    "var_type": "x",
                    "tech_idx": ti,
                    "node_idx": ni,
                    "base_cost": base_cost,
                    "out_region_penalty": penalty,
                }
            )

    # New-hire assignment vars.
    # New hires are NOT eligible for HPS-required nodes (skill_class "hps" or
    # "hps_ls") — they lack HPS certification. They CAN serve "regular" and "ls"
    # nodes. Canada exclusion (Hakim policy) also enforced.
    for ci in candidate_indices:
        crow = candidates.loc[ci]
        for ni, nrow in nodes.iterrows():
            node_is_canada = is_canada_node_state(str(nrow["state_norm"]))
            node_requires_hps = int(nrow.get("required_hps", 0)) == 1
            idx = len(var_names)
            z_idx[(ci, ni)] = idx
            var_names.append(f"z__{crow['candidate_id']}__{nrow['node_id']}")
            lb.append(0.0)
            # Block new hires from Canada demand and HPS-required nodes.
            if node_is_canada or node_requires_hps:
                ub.append(0.0)
            else:
                ub.append(float(nrow["appointment_count"]))
            integrality.append(1)

            if full_cost_lookup is not None:
                _fc_key = (str(crow["candidate_id"]), str(nrow["node_id"]))
                if _fc_key not in full_cost_lookup:
                    _full_cost_fallback_count += 1
                base_cost = full_cost_lookup.get(
                    _fc_key,
                    global_avg + config.RENTAL_CAR_AVG_USD + config.HOTEL_AVG_USD,
                )
            else:
                base_cost = get_cost(
                    str(crow["airport_iata"]),
                    str(nrow["state_norm"]),
                    exact_cost,
                    origin_avg,
                    global_avg,
                )
            is_out_region = int(str(crow.get("state", "")) != str(nrow["state_norm"]))
            penalty = out_of_region_penalty if is_out_region else 0.0
            obj.append(base_cost + penalty)
            meta.append(
                {
                    "var_type": "z",
                    "candidate_idx": ci,
                    "node_idx": ni,
                    "base_cost": base_cost,
                    "out_region_penalty": penalty,
                }
            )

    if full_cost_lookup is not None and _full_cost_fallback_count > 0:
        print(f"  WARNING: {_full_cost_fallback_count} (tech/candidate, node) pairs missing from "
              f"full_cost_table.csv — used global_avg fallback (${global_avg + config.RENTAL_CAR_AVG_USD + config.HOTEL_AVG_USD:,.2f}).")

    # Candidate hire-count integer vars.
    for ci in candidate_indices:
        crow = candidates.loc[ci]
        idx = len(var_names)
        y_idx[ci] = idx
        var_names.append(f"y__{crow['candidate_id']}")
        lb.append(0.0)
        ub.append(float(min(hire_count, max_hires_per_base)))
        integrality.append(1)
        obj.append(float(annual_hire_cost_usd))
        meta.append({"var_type": "y", "candidate_idx": ci})

    # Unmet demand vars.
    for ni, nrow in nodes.iterrows():
        idx = len(var_names)
        u_idx[ni] = idx
        var_names.append(f"u__{nrow['node_id']}")
        lb.append(0.0)
        ub.append(float(nrow["appointment_count"]))
        integrality.append(1)
        obj.append(float(unmet_penalty))
        meta.append({"var_type": "u", "node_idx": ni})

    n_vars = len(var_names)

    rows = []
    cols = []
    data = []
    lower = []
    upper = []
    r = 0

    # Demand balance constraints.
    for ni, nrow in nodes.iterrows():
        for ti in tech.index:
            idx = x_idx.get((ti, ni))
            if idx is not None:
                rows.append(r)
                cols.append(idx)
                data.append(1.0)
        for ci in candidate_indices:
            idx = z_idx[(ci, ni)]
            rows.append(r)
            cols.append(idx)
            data.append(1.0)
        rows.append(r)
        cols.append(u_idx[ni])
        data.append(1.0)

        demand_count = float(nrow["appointment_count"])
        lower.append(demand_count)
        upper.append(demand_count)
        r += 1

    # Existing tech capacity constraints.
    for ti, trow in tech.iterrows():
        for ni, nrow in nodes.iterrows():
            idx = x_idx.get((ti, ni))
            if idx is None:
                continue
            rows.append(r)
            cols.append(idx)
            data.append(float(nrow["avg_hours_per_appointment"]))
        lower.append(-np.inf)
        upper.append(float(trow["capacity_hours"]))
        r += 1

    # Candidate hire capacity constraints.
    for ci in candidate_indices:
        for ni, nrow in nodes.iterrows():
            idx = z_idx[(ci, ni)]
            rows.append(r)
            cols.append(idx)
            data.append(float(nrow["avg_hours_per_appointment"]))
        rows.append(r)
        cols.append(y_idx[ci])
        data.append(-float(new_hire_capacity_hours))
        lower.append(-np.inf)
        upper.append(0.0)
        r += 1

    # Sum of hires equals scenario count.
    if candidate_indices:
        for ci in candidate_indices:
            rows.append(r)
            cols.append(y_idx[ci])
            data.append(1.0)
        lower.append(float(hire_count))
        upper.append(float(hire_count))
        r += 1

    A = sp.coo_matrix((data, (rows, cols)), shape=(r, n_vars))
    constraints = LinearConstraint(A, np.array(lower), np.array(upper))
    bounds = Bounds(np.array(lb), np.array(ub))

    result = milp(
        c=np.array(obj),
        constraints=constraints,
        integrality=np.array(integrality, dtype=int),
        bounds=bounds,
        options={"time_limit": time_limit_sec, "mip_rel_gap": 0.0},
    )
    if result.x is None:
        raise RuntimeError(
            f"MILP failed for N={hire_count}. status={result.status} message={result.message}"
        )
    if int(result.status) not in (0, 1):
        raise RuntimeError(
            f"MILP did not return a usable solution for N={hire_count}. "
            f"status={result.status} message={result.message}"
        )

    solution = np.array(result.x)
    y_values = {ci: solution[idx] for ci, idx in y_idx.items()}

    # Build detailed outputs and cost breakdown.
    existing_rows = []
    new_rows = []
    util_rows = []

    travel_cost = 0.0
    out_region_cost = 0.0
    unmet_appointments = 0.0

    for (ti, ni), idx in x_idx.items():
        val = float(solution[idx])
        if val <= 1e-6:
            continue
        trow = tech.loc[ti]
        nrow = nodes.loc[ni]
        m = meta[idx]
        base = float(m["base_cost"])
        pen = float(m["out_region_penalty"])
        travel_cost += val * base
        out_region_cost += val * pen
        hours = val * float(nrow["avg_hours_per_appointment"])
        existing_rows.append(
            {
                "scenario_hires": hire_count,
                "tech_id": trow["tech_id"],
                "tech_name": trow["tech_name"],
                "employment_type": trow["employment_type"],
                "base_state": trow["base_state"],
                "base_airport_iata": trow["base_airport_iata"],
                "node_id": nrow["node_id"],
                "state_norm": nrow["state_norm"],
                "skill_class": nrow["skill_class"],
                "assigned_appointments": val,
                "assigned_hours": hours,
                "unit_travel_cost_usd": base,
                "unit_out_region_penalty_usd": pen,
                "total_travel_cost_usd": val * base,
                "total_out_region_penalty_usd": val * pen,
            }
        )

    for (ci, ni), idx in z_idx.items():
        val = float(solution[idx])
        if val <= 1e-6:
            continue
        crow = candidates.loc[ci]
        nrow = nodes.loc[ni]
        m = meta[idx]
        base = float(m["base_cost"])
        pen = float(m["out_region_penalty"])
        travel_cost += val * base
        out_region_cost += val * pen
        hours = val * float(nrow["avg_hours_per_appointment"])
        new_rows.append(
            {
                "scenario_hires": hire_count,
                "candidate_id": crow["candidate_id"],
                "candidate_type": crow["candidate_type"],
                "candidate_city": crow["city"],
                "candidate_state": crow["state"],
                "airport_iata": crow["airport_iata"],
                "node_id": nrow["node_id"],
                "state_norm": nrow["state_norm"],
                "skill_class": nrow["skill_class"],
                "assigned_appointments": val,
                "assigned_hours": hours,
                "unit_travel_cost_usd": base,
                "unit_out_region_penalty_usd": pen,
                "total_travel_cost_usd": val * base,
                "total_out_region_penalty_usd": val * pen,
            }
        )

    for ni, idx in u_idx.items():
        val = float(solution[idx])
        unmet_appointments += val

    existing_df = pd.DataFrame(existing_rows)
    new_df = pd.DataFrame(new_rows)

    # Utilization by tech.
    hours_by_tech = (
        existing_df.groupby("tech_id")["assigned_hours"].sum().to_dict() if not existing_df.empty else {}
    )
    for _, trow in tech.iterrows():
        assigned = float(hours_by_tech.get(trow["tech_id"], 0.0))
        cap = float(trow["capacity_hours"])
        util_rows.append(
            {
                "scenario_hires": hire_count,
                "tech_id": trow["tech_id"],
                "tech_name": trow["tech_name"],
                "capacity_hours": cap,
                "assigned_hours": assigned,
                "utilization": assigned / cap if cap > 0 else np.nan,
            }
        )
    util_df = pd.DataFrame(util_rows)

    placement_rows = []
    if not new_df.empty:
        assigned = (
            new_df.groupby("candidate_id")
            .agg(
                assigned_appointments=("assigned_appointments", "sum"),
                assigned_hours=("assigned_hours", "sum"),
            )
            .reset_index()
        )
    else:
        assigned = pd.DataFrame(columns=["candidate_id", "assigned_appointments", "assigned_hours"])
    assigned_map = assigned.set_index("candidate_id").to_dict("index")
    for ci, yv in y_values.items():
        hires = int(round(yv))
        if hires <= 0:
            continue
        crow = candidates.loc[ci]
        metrics = assigned_map.get(crow["candidate_id"], {"assigned_appointments": 0.0, "assigned_hours": 0.0})
        placement_rows.append(
            {
                "scenario_hires": hire_count,
                "candidate_id": crow["candidate_id"],
                "candidate_type": crow["candidate_type"],
                "city": crow["city"],
                "state": crow["state"],
                "airport_iata": crow["airport_iata"],
                "hires_allocated": hires,
                "assigned_appointments": float(metrics["assigned_appointments"]),
                "assigned_hours": float(metrics["assigned_hours"]),
            }
        )
    placements_df = pd.DataFrame(placement_rows)

    hire_cost = float(sum(int(round(v)) for v in y_values.values()) * annual_hire_cost_usd)
    unmet_cost = float(unmet_appointments * unmet_penalty)
    modeled_total = float(travel_cost + out_region_cost + hire_cost + unmet_cost)

    summary = {
        "scenario_hires": int(hire_count),
        "solver_status": int(result.status),
        "solver_proven_optimal": int(result.status) == 0,
        "solver_message": str(result.message),
        "solver_mip_gap": float(getattr(result, "mip_gap", np.nan)),
        "solver_mip_node_count": int(getattr(result, "mip_node_count", 0) or 0),
        "objective_value": float(result.fun),
        "total_appointments": float(nodes["appointment_count"].sum()),
        "served_appointments": float(nodes["appointment_count"].sum() - unmet_appointments),
        "unmet_appointments": float(unmet_appointments),
        "travel_cost_usd": travel_cost,
        "out_of_region_penalty_usd": out_region_cost,
        "hire_cost_usd": hire_cost,
        "unmet_penalty_usd": unmet_cost,
        "modeled_total_cost_usd": modeled_total,
        "hours_per_capacity_unit": float(hours_per_unit),
        "new_hire_capacity_hours": float(new_hire_capacity_hours),
        "mean_existing_utilization": float(np.nanmean(util_df["utilization"].values)),
        "max_existing_utilization": float(np.nanmax(util_df["utilization"].values)),
    }

    return {
        "summary": summary,
        "existing_assignments": existing_df,
        "new_assignments": new_df,
        "placements": placements_df,
        "tech_utilization": util_df,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MILP hiring/location scenarios.")
    parser.add_argument(
        "--output-dir",
        default=config.OPTIMIZATION_DIR,
        help="Optimization output directory.",
    )
    parser.add_argument("--min-new-hires", type=int, default=0)
    parser.add_argument("--max-new-hires", type=int, default=4)
    parser.add_argument("--target-utilization", type=float, default=0.85)
    parser.add_argument(
        "--out-of-region-penalty",
        type=float,
        default=config.DEFAULT_OUT_OF_REGION_PENALTY_USD,
    )
    parser.add_argument("--unmet-penalty", type=float, default=config.DEFAULT_UNMET_PENALTY_USD)
    parser.add_argument(
        "--annual-hire-cost-usd",
        type=float,
        default=config.DEFAULT_ANNUAL_HIRE_COST_USD,
    )
    parser.add_argument("--time-limit-sec", type=int, default=180)
    parser.add_argument(
        "--max-hires-per-base",
        type=int,
        default=1,
        help="Hard cap on hires allocated to any single candidate base.",
    )
    parser.add_argument(
        "--contractor-assignment-scope",
        choices=["texas_only", "anywhere"],
        default=None,
        help="Override contractor assignment geography.",
    )
    args = parser.parse_args()
    if args.annual_hire_cost_usd < 0:
        raise ValueError("--annual-hire-cost-usd must be non-negative.")
    if args.out_of_region_penalty < 0:
        raise ValueError("--out-of-region-penalty must be non-negative.")
    if args.max_hires_per_base < 1:
        raise ValueError("--max-hires-per-base must be at least 1.")
    if not (0 < args.target_utilization <= 1.0):
        raise ValueError("--target-utilization must be in range (0, 1.0].")

    out_dir = Path(args.output_dir)
    inputs = load_inputs(out_dir)
    tech = inputs["tech"].copy()
    demand = inputs["demand"].copy()
    candidates = inputs["candidates"].copy()
    cost_matrix = inputs["cost_matrix"].copy()
    baseline = inputs["baseline"]

    for col in [
        "skill_hps",
        "skill_ls",
        "skill_patient",
        "constraint_florida_only",
        "constraint_canada_wide",
    ]:
        tech[col] = pd.to_numeric(tech[col], errors="coerce").fillna(0).astype(int)
    tech["availability_fte"] = pd.to_numeric(tech["availability_fte"], errors="coerce").fillna(0.0)
    tech["base_state"] = tech["base_state"].map(normalize_state)

    demand_nodes = build_demand_nodes(demand)
    exact_cost, origin_avg, global_avg = build_cost_lookup(cost_matrix)

    full_cost_lookup: dict[tuple[str, str], float] | None = None
    if config.FULL_COST_MODEL and inputs["full_cost_df"] is not None:
        full_cost_lookup = build_full_cost_lookup(inputs["full_cost_df"])
        print(f"  [FullCost] Lookup built: {len(full_cost_lookup):,} (entity, node) pairs.")

    if args.contractor_assignment_scope:
        contractor_scope = args.contractor_assignment_scope
    else:
        contractor_rows = tech[tech["employment_type"].astype(str).str.lower().eq("contractor")]
        if contractor_rows.empty:
            contractor_scope = "anywhere"
        else:
            first_scope = contractor_rows.iloc[0].get("contractor_assignment_scope", "texas_only")
            contractor_scope = str(first_scope) if str(first_scope) else "texas_only"

    scenario_summaries = []
    all_existing = []
    all_new = []
    all_placements = []
    all_util = []

    canceled_voided_usd = float(baseline.get("canceled_voided_spend_usd_report", 0.0))

    for hire_count in range(args.min_new_hires, args.max_new_hires + 1):
        print(f"Solving scenario N={hire_count} new hires...")

        # Pre-check: if we can't possibly satisfy hire_count given candidates and cap, skip.
        max_hires_possible = len(candidates) * args.max_hires_per_base
        if hire_count > max_hires_possible:
            msg = (
                f"Pre-solve infeasibility: N={hire_count} exceeds max possible hires "
                f"({max_hires_possible} = {len(candidates)} candidates × {args.max_hires_per_base} cap). "
                "Skipping."
            )
            print(f"  SKIP: {msg}")
            scenario_summaries.append(_infeasible_summary(hire_count, msg, canceled_voided_usd))
            continue

        try:
            result = solve_scenario(
                hire_count=hire_count,
                tech=tech,
                nodes=demand_nodes,
                candidates=candidates,
                exact_cost=exact_cost,
                origin_avg=origin_avg,
                global_avg=global_avg,
                contractor_scope=contractor_scope,
                target_utilization=args.target_utilization,
                out_of_region_penalty=args.out_of_region_penalty,
                unmet_penalty=args.unmet_penalty,
                annual_hire_cost_usd=args.annual_hire_cost_usd,
                max_hires_per_base=args.max_hires_per_base,
                time_limit_sec=args.time_limit_sec,
                full_cost_lookup=full_cost_lookup,
            )
        except RuntimeError as exc:
            msg = f"Solver error for N={hire_count}: {exc}"
            print(f"  ERROR: {msg}")
            scenario_summaries.append(_infeasible_summary(hire_count, msg, canceled_voided_usd))
            continue

        summary = result["summary"]
        if not summary["solver_proven_optimal"]:
            print(
                f"  Warning: N={hire_count} ended without proven optimality "
                f"(status={summary['solver_status']}, mip_gap={summary['solver_mip_gap']})."
            )
        summary["baseline_canceled_voided_usd"] = canceled_voided_usd
        summary["economic_total_with_overhead_usd"] = float(
            summary["modeled_total_cost_usd"] + summary["baseline_canceled_voided_usd"]
        )
        scenario_summaries.append(summary)

        if not result["existing_assignments"].empty:
            all_existing.append(result["existing_assignments"])
        if not result["new_assignments"].empty:
            all_new.append(result["new_assignments"])
        if not result["placements"].empty:
            all_placements.append(result["placements"])
        if not result["tech_utilization"].empty:
            all_util.append(result["tech_utilization"])

    summary_df = pd.DataFrame(scenario_summaries).sort_values("scenario_hires")
    existing_df = pd.concat(all_existing, ignore_index=True) if all_existing else pd.DataFrame()
    new_df = pd.concat(all_new, ignore_index=True) if all_new else pd.DataFrame()
    placements_df = pd.concat(all_placements, ignore_index=True) if all_placements else pd.DataFrame()
    util_df = pd.concat(all_util, ignore_index=True) if all_util else pd.DataFrame()

    summary_out = out_dir / "scenario_summary.csv"
    existing_out = out_dir / "scenario_assignments_existing.csv"
    new_out = out_dir / "scenario_assignments_newhires.csv"
    placements_out = out_dir / "scenario_placements.csv"
    util_out = out_dir / "scenario_tech_utilization.csv"
    assumptions_out = out_dir / "model_assumptions.json"

    summary_df.to_csv(summary_out, index=False)
    existing_df.to_csv(existing_out, index=False)
    new_df.to_csv(new_out, index=False)
    placements_df.to_csv(placements_out, index=False)
    util_df.to_csv(util_out, index=False)

    assumptions = {
        "min_new_hires": args.min_new_hires,
        "max_new_hires": args.max_new_hires,
        "target_utilization": args.target_utilization,
        "out_of_region_penalty": args.out_of_region_penalty,
        "unmet_penalty": args.unmet_penalty,
        "annual_hire_cost_usd": args.annual_hire_cost_usd,
        "max_hires_per_base": args.max_hires_per_base,
        "hire_cost_scope": "incremental_new_hires_only",
        "hire_cost_input_mode": "direct_fixed_value",
        "contractor_assignment_scope": contractor_scope,
        "full_cost_model": config.FULL_COST_MODEL and full_cost_lookup is not None,
        "full_cost_model_constants": {
            "irs_mileage_rate_usd_per_mi": config.IRS_MILEAGE_RATE_USD_PER_MI,
            "rental_car_avg_usd": config.RENTAL_CAR_AVG_USD,
            "hotel_avg_usd": config.HOTEL_AVG_USD,
            "drive_threshold_miles": config.DRIVE_THRESHOLD_MILES,
        },
        "hps_timeline_assumption": {
            "production_end_estimate": "2027-03-31",
            "service_tail_end_estimate": "2031-03-31",
        },
    }
    with open(assumptions_out, "w") as f:
        json.dump(assumptions, f, indent=2)

    print(f"Saved: {summary_out}")
    print(f"Saved: {existing_out}")
    print(f"Saved: {new_out}")
    print(f"Saved: {placements_out}")
    print(f"Saved: {util_out}")
    print(f"Saved: {assumptions_out}")
    print("\nScenario summary:")
    print(summary_df.to_string(index=False))
    print("Step 8 complete.")


if __name__ == "__main__":
    main()
