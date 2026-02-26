"""Step 8: Run MILP location-allocation optimization scenarios."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, milp

sys.path.insert(0, os.path.dirname(__file__))
import config
from optimization_utils import normalize_state


def load_inputs(output_dir: Path) -> dict:
    """Load all prerequisite optimization inputs."""
    files = {
        "tech": output_dir / "tech_master.csv",
        "demand": output_dir / "demand_appointments.csv",
        "candidates": output_dir / "candidate_bases.csv",
        "cost_matrix": output_dir / "travel_cost_matrix.csv",
        "baseline": output_dir / "baseline_kpis.json",
    }
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        joined = "\n".join(f"- {p}" for p in missing)
        raise FileNotFoundError(f"Missing required input files. Run steps 6 and 7 first.\n{joined}")

    with open(files["baseline"], "r") as f:
        baseline = json.load(f)

    return {
        "tech": pd.read_csv(files["tech"]),
        "demand": pd.read_csv(files["demand"]),
        "candidates": pd.read_csv(files["candidates"]),
        "cost_matrix": pd.read_csv(files["cost_matrix"]),
        "baseline": baseline,
    }


def build_demand_nodes(demand: pd.DataFrame) -> pd.DataFrame:
    """Aggregate appointments into demand nodes by state + skill class."""
    demand = demand.copy()
    demand["state_norm"] = demand["state_norm"].map(normalize_state)
    demand["duration_hours"] = pd.to_numeric(demand["duration_hours"], errors="coerce").fillna(8.0)
    demand = demand[demand["state_norm"].notna()].copy()
    demand = demand[demand["state_norm"] != ""].copy()

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

    if florida_only and node_state != "FL":
        return False
    if is_contractor and contractor_scope == "texas_only" and node_state != "TX":
        return False

    # If no origin airport, don't allow assignment.
    if not str(tech.get("base_airport_iata", "")).strip():
        return False
    return True


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
    time_limit_sec: int,
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
    for ci in candidate_indices:
        crow = candidates.loc[ci]
        for ni, nrow in nodes.iterrows():
            idx = len(var_names)
            z_idx[(ci, ni)] = idx
            var_names.append(f"z__{crow['candidate_id']}__{nrow['node_id']}")
            lb.append(0.0)
            ub.append(float(nrow["appointment_count"]))
            integrality.append(1)

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

    # Candidate hire-count integer vars.
    for ci in candidate_indices:
        crow = candidates.loc[ci]
        idx = len(var_names)
        y_idx[ci] = idx
        var_names.append(f"y__{crow['candidate_id']}")
        lb.append(0.0)
        ub.append(float(hire_count))
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
    parser.add_argument("--unmet-penalty", type=float, default=5000.0)
    parser.add_argument(
        "--annual-hire-cost-usd",
        type=float,
        default=config.DEFAULT_ANNUAL_HIRE_COST_USD,
    )
    parser.add_argument("--time-limit-sec", type=int, default=180)
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

    out_dir = Path(args.output_dir)
    inputs = load_inputs(out_dir)
    tech = inputs["tech"].copy()
    demand = inputs["demand"].copy()
    candidates = inputs["candidates"].copy()
    cost_matrix = inputs["cost_matrix"].copy()
    baseline = inputs["baseline"]

    for col in ["skill_hps", "skill_ls", "skill_patient", "constraint_florida_only"]:
        tech[col] = pd.to_numeric(tech[col], errors="coerce").fillna(0).astype(int)
    tech["availability_fte"] = pd.to_numeric(tech["availability_fte"], errors="coerce").fillna(0.0)
    tech["base_state"] = tech["base_state"].map(normalize_state)

    demand_nodes = build_demand_nodes(demand)
    exact_cost, origin_avg, global_avg = build_cost_lookup(cost_matrix)

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

    for hire_count in range(args.min_new_hires, args.max_new_hires + 1):
        print(f"Solving scenario N={hire_count} new hires...")
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
            time_limit_sec=args.time_limit_sec,
        )
        summary = result["summary"]
        if not summary["solver_proven_optimal"]:
            print(
                f"  Warning: N={hire_count} ended without proven optimality "
                f"(status={summary['solver_status']}, mip_gap={summary['solver_mip_gap']})."
            )
        summary["baseline_canceled_voided_usd"] = float(
            baseline.get("canceled_voided_spend_usd_report", 0.0)
        )
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
        "hire_cost_scope": "incremental_new_hires_only",
        "hire_cost_input_mode": "direct_fixed_value",
        "contractor_assignment_scope": contractor_scope,
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
