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
from optimization_utils import normalize_state


EXISTING_ASSIGNMENT_COLUMNS = [
    "scenario_hires",
    "tech_id",
    "tech_name",
    "employment_type",
    "base_state",
    "base_airport_iata",
    "base_hub_tier",
    "assignment_scope_mode",
    "assignment_scope_states",
    "anchor_site_name",
    "anchor_reserved_fte",
    "external_field_fte",
    "node_id",
    "state_norm",
    "skill_class",
    "assigned_appointments",
    "assigned_hours",
    "unit_travel_cost_usd",
    "unit_out_region_penalty_usd",
    "unit_timezone_penalty_usd",
    "unit_hub_penalty_usd",
    "zone_policy",
    "zone_jump_count",
    "base_operational_zone_label",
    "node_operational_zone_label",
    "travel_cost_policy",
    "trip_mode",
    "ground_transport_mode",
    "median_dist_mi",
    "trip_span_days",
    "rental_days",
    "employee_style_unit_cost_usd",
    "effective_unit_cost_usd",
    "total_travel_cost_usd",
    "total_out_region_penalty_usd",
    "total_timezone_penalty_usd",
    "total_hub_penalty_usd",
]

NEW_HIRE_ASSIGNMENT_COLUMNS = [
    "scenario_hires",
    "candidate_id",
    "candidate_type",
    "candidate_city",
    "candidate_state",
    "airport_iata",
    "hub_tier",
    "node_id",
    "state_norm",
    "skill_class",
    "assigned_appointments",
    "assigned_hours",
    "unit_travel_cost_usd",
    "unit_out_region_penalty_usd",
    "unit_timezone_penalty_usd",
    "unit_hub_penalty_usd",
    "zone_policy",
    "zone_jump_count",
    "base_operational_zone_label",
    "node_operational_zone_label",
    "travel_cost_policy",
    "trip_mode",
    "ground_transport_mode",
    "median_dist_mi",
    "trip_span_days",
    "rental_days",
    "employee_style_unit_cost_usd",
    "effective_unit_cost_usd",
    "total_travel_cost_usd",
    "total_out_region_penalty_usd",
    "total_timezone_penalty_usd",
    "total_hub_penalty_usd",
]

PLACEMENT_COLUMNS = [
    "scenario_hires",
    "candidate_id",
    "candidate_type",
    "city",
    "state",
    "airport_iata",
    "hires_allocated",
    "assigned_appointments",
    "assigned_hours",
]

UTILIZATION_COLUMNS = [
    "scenario_hires",
    "tech_id",
    "tech_name",
    "capacity_hours",
    "assigned_hours",
    "utilization",
]

CONTRACTOR_USAGE_COLUMNS = [
    "scenario_hires",
    "tech_id",
    "tech_name",
    "assigned_appointments",
    "assigned_hours",
    "total_travel_cost_usd",
    "total_hub_penalty_usd",
    "avg_zone_jump",
    "share_two_zone_plus",
    "share_three_zone_plus",
    "states_served",
]


def load_inputs(output_dir: Path) -> dict:
    """Load all prerequisite optimization inputs."""
    files = {
        "tech": output_dir / "tech_master.csv",
        "demand": output_dir / "demand_appointments.csv",
        "candidates": output_dir / "candidate_bases.csv",
    }
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        joined = "\n".join(f"- {p}" for p in missing)
        raise FileNotFoundError(f"Missing required input files. Run step 06 first.\n{joined}")

    result = {
        "tech": pd.read_csv(files["tech"]),
        "demand": pd.read_csv(files["demand"]),
        "candidates": pd.read_csv(files["candidates"]),
        "full_cost_df": None,
    }

    full_cost_path = output_dir / "full_cost_table.csv"
    if full_cost_path.exists():
        result["full_cost_df"] = pd.read_csv(full_cost_path)
        print(
            f"  [FullCost] Loaded full_cost_table.csv "
            f"({len(result['full_cost_df']):,} rows, "
            f"drive/fly model active)."
        )
    else:
        raise FileNotFoundError(
            f"Missing: {full_cost_path}\n"
            "Run scripts/11_build_full_cost_table.py first."
        )

    return result


def blank_slate_output_dir(base_dir: Path) -> Path:
    """Return the dedicated output directory for blank-slate solves."""
    return base_dir / getattr(config, "BLANK_SLATE_SUBDIR", "blank_slate")


def empty_frame(columns: list[str]) -> pd.DataFrame:
    """Return an empty DataFrame that still preserves CSV headers."""
    return pd.DataFrame(columns=columns)


def compute_hours_per_unit(
    total_demand_hours: float,
    total_availability: float,
    target_utilization: float,
    hire_count: int,
    blank_slate: bool,
) -> float:
    """Compute modeled capacity hours per unit for the active scenario."""
    if total_availability > 0:
        return total_demand_hours / (total_availability * target_utilization)
    if blank_slate and hire_count > 0:
        return total_demand_hours / (hire_count * target_utilization)
    raise RuntimeError(
        "No existing technician capacity is available for this solve. "
        "Use --blank-slate with a positive hire count for a pure new-hire rebuild."
    )


def prepare_demand_for_solve(demand: pd.DataFrame, blank_slate: bool) -> tuple[pd.DataFrame, int]:
    """Return scenario-specific demand rows without mutating the source frame."""
    prepared = demand.copy()
    if not blank_slate:
        return prepared, 0

    if "state_norm" in prepared.columns:
        normalized_state = prepared["state_norm"].map(normalize_state)
    else:
        normalized_state = pd.Series("", index=prepared.index, dtype="object")
    florida_mask = normalized_state.fillna("").eq("FL")
    removed = int(florida_mask.sum())
    if removed:
        prepared = prepared.loc[~florida_mask].copy()
    return prepared, removed


def build_demand_nodes(demand: pd.DataFrame) -> pd.DataFrame:
    """Aggregate appointments into demand nodes by state + skill class."""
    demand = demand.copy()
    demand["state_norm"] = demand["state_norm"].map(normalize_state)
    demand["nearest_hub_operational_zone_rank"] = pd.to_numeric(
        demand.get("nearest_hub_operational_zone_rank"), errors="coerce"
    )
    # duration_hours is the model's calendar-window workload measure. It is not
    # intended to represent exact hands-on labor or weekday-only work time.
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

    group_cols = ["state_norm", "skill_class", "required_hps", "required_ls"]

    grouped = (
        demand.groupby(group_cols, as_index=False)
        .agg(
            appointment_count=("appointment_id", "count"),
            demand_hours=("duration_hours", "sum"),
            territory_count=("territory", "nunique"),
        )
        .sort_values(["state_norm", "skill_class"])
        .reset_index(drop=True)
    )

    zone_counts = (
        demand.groupby(
            group_cols
            + [
                "nearest_hub_operational_zone_label",
                "nearest_hub_operational_zone_rank",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="zone_count")
        .sort_values(
            group_cols + ["zone_count", "nearest_hub_operational_zone_rank"],
            ascending=[True, True, True, True, False, True],
        )
    )
    zone_counts["node_operational_zone_share"] = (
        zone_counts["zone_count"] / zone_counts.groupby(group_cols)["zone_count"].transform("sum")
    )
    zone_mode = zone_counts.drop_duplicates(subset=group_cols, keep="first").copy()
    grouped = grouped.merge(
        zone_mode[
            group_cols
            + [
                "nearest_hub_operational_zone_label",
                "nearest_hub_operational_zone_rank",
                "node_operational_zone_share",
            ]
        ].rename(
            columns={
                "nearest_hub_operational_zone_label": "node_operational_zone_label",
                "nearest_hub_operational_zone_rank": "node_operational_zone_rank",
            }
        ),
        on=group_cols,
        how="left",
    )
    grouped["avg_hours_per_appointment"] = grouped["demand_hours"] / grouped["appointment_count"]
    grouped["node_id"] = grouped.apply(
        lambda r: f"{r['state_norm']}__{r['skill_class']}",
        axis=1,
    )
    return grouped


def build_full_cost_lookup(
    full_cost_df: pd.DataFrame,
) -> dict[tuple[str, str], dict]:
    """Build (tech_or_candidate_id, node_id) → full cost metadata row."""
    return {
        (str(row["tech_or_candidate_id"]), str(row["node_id"])): row
        for row in full_cost_df.to_dict("records")
    }


def zone_jump_count(base_rank: object, node_rank: object) -> int | None:
    """Return broad operational zone jump count."""
    if pd.isna(base_rank) or pd.isna(node_rank):
        return None
    return abs(int(base_rank) - int(node_rank))


def evaluate_zone_policy(zone_jump: int | None, zone_policy: str) -> tuple[bool, float]:
    """Return (feasible, penalty_usd_per_appointment) for a zone policy."""
    if zone_jump is None:
        return True, 0.0
    if zone_policy == config.ZONE_POLICY_STANDARD:
        if zone_jump >= 3:
            return False, 0.0
        if zone_jump == 2:
            return True, float(config.EMPLOYEE_TWO_ZONE_JUMP_PENALTY_USD)
        return True, 0.0
    if zone_policy == config.ZONE_POLICY_CONTRACTOR_SOFT:
        if zone_jump >= 3:
            return True, float(config.CONTRACTOR_THREE_PLUS_ZONE_JUMP_PENALTY_USD)
        if zone_jump == 2:
            return True, float(config.CONTRACTOR_TWO_ZONE_JUMP_PENALTY_USD)
        return True, 0.0
    return True, 0.0


def evaluate_hub_penalty(hub_tier: object, trip_mode: object) -> float:
    """Return per-appointment connectivity surcharge for fly trips only."""
    mode = "" if trip_mode is None else str(trip_mode).strip().lower()
    if mode != "fly":
        return 0.0
    tier = "" if hub_tier is None else str(hub_tier).strip()
    if not tier:
        return float(config.HUB_PENALTY_UNKNOWN)
    return float(config.HUB_PENALTY_MAP.get(tier, config.HUB_PENALTY_UNKNOWN))


def parse_assignment_scope_states(value: object) -> set[str]:
    """Parse semicolon-delimited assignment scope states into a normalized set."""
    if value is None or pd.isna(value):
        return set()
    states: set[str] = set()
    for token in str(value).split(";"):
        raw = token.strip()
        if not raw:
            continue
        states.add(normalize_state(raw))
    return {state for state in states if state}


def tech_assignment_scope_fields(
    tech: pd.Series, contractor_scope: str
) -> tuple[str, str, set[str]]:
    """Return assignment scope mode/state/state-set with fallback for legacy data."""
    mode = str(tech.get("assignment_scope_mode", "")).strip()
    state = str(tech.get("assignment_scope_state", "")).strip()
    states = parse_assignment_scope_states(tech.get("assignment_scope_states", ""))
    if state.lower() == "nan":
        state = ""
    is_contractor = str(tech.get("employment_type", "")).strip().lower() == "contractor"
    if mode:
        return mode, state, states
    if is_contractor and contractor_scope == "texas_only":
        return config.ASSIGNMENT_SCOPE_MODE_STATE_LIMITED, "TX", {"TX"}
    return config.ASSIGNMENT_SCOPE_MODE_NATIONAL, "", set()


def tech_eligible_for_node(tech: pd.Series, node: pd.Series, contractor_scope: str) -> bool:
    """Check skill and special geography constraints."""
    if int(node["required_hps"]) and int(tech["skill_hps"]) == 0:
        return False
    if int(node["required_ls"]) and int(tech["skill_ls"]) == 0:
        return False
    if int(tech["skill_patient"]) == 0:
        return False

    node_state = str(node.get("state_norm", ""))
    florida_only = int(tech.get("constraint_florida_only", 0)) == 1
    scope_mode, scope_state, scope_states = tech_assignment_scope_fields(tech, contractor_scope)
    zone_policy = str(tech.get("zone_policy", config.ZONE_POLICY_STANDARD)).strip() or config.ZONE_POLICY_STANDARD
    jump = zone_jump_count(
        tech.get("base_operational_zone_rank"),
        node.get("node_operational_zone_rank"),
    )
    zone_feasible, _ = evaluate_zone_policy(jump, zone_policy)

    if florida_only and node_state != "FL":
        return False
    if scope_mode == config.ASSIGNMENT_SCOPE_MODE_STATE_LIMITED and node_state != scope_state:
        return False
    if scope_mode == config.ASSIGNMENT_SCOPE_MODE_STATE_SET_LIMITED and node_state not in scope_states:
        return False
    if not zone_feasible:
        return False

    # If no origin airport, don't allow assignment.
    airport_raw = tech.get("base_airport_iata", "")
    if pd.isna(airport_raw):
        return False
    airport = str(airport_raw).strip()
    if not airport or airport.lower() == "nan":
        return False
    return True


def synthetic_new_hire_eligible_for_node(node: pd.Series, blank_slate: bool) -> bool:
    """Return whether a synthetic new hire can cover a demand node."""
    if blank_slate:
        return True
    if int(node.get("required_hps", 0)) == 1:
        return False
    if int(node.get("required_ls", 0)) == 1:
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
        "timezone_penalty_usd": None,
        "hub_penalty_usd": None,
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
    full_cost_lookup: dict[tuple[str, str], dict],
    contractor_scope: str,
    target_utilization: float,
    out_of_region_penalty: float,
    unmet_penalty: float,
    annual_hire_cost_usd: float,
    max_hires_per_base: int,
    time_limit_sec: int,
    blank_slate: bool = False,
    blank_slate_max_appointments_per_hire: int | None = None,
    optimized_max_appointments_per_person: int | None = None,
) -> dict:
    """Solve one MILP scenario for a fixed new-hire count."""
    nodes = nodes.reset_index(drop=True).copy()
    tech = tech.reset_index(drop=True).copy()
    candidates = candidates.reset_index(drop=True).copy()
    if blank_slate:
        tech = tech[
            pd.to_numeric(tech["availability_fte"], errors="coerce").fillna(0.0) > 0
        ].reset_index(drop=True)

    total_demand_hours = float(nodes["demand_hours"].sum())
    total_availability = float(tech["availability_fte"].sum())
    # The solver normalizes capacity against the same appointment-duration
    # demand pool used on the workload side. The resulting utilization output is
    # a modeled load ratio, not a literal payroll or timesheet utilization rate.
    hours_per_unit = compute_hours_per_unit(
        total_demand_hours=total_demand_hours,
        total_availability=total_availability,
        target_utilization=target_utilization,
        hire_count=hire_count,
        blank_slate=blank_slate,
    )

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
    _fallback_trip_span_days = max(1, int(np.ceil(config.HOTEL_AVG_NIGHTS)))
    _fallback_cost = (
        config.BTS_NATIONAL_FALLBACK * config.CORPORATE_TRAVEL_PREMIUM
        + (config.RENTAL_CAR_DAILY_RATE_USD * _fallback_trip_span_days)
        + (config.HOTEL_NIGHTLY_RATE_USD * _fallback_trip_span_days)
    )

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

            _fc_key = (str(trow["tech_id"]), str(nrow["node_id"]))
            if _fc_key not in full_cost_lookup:
                _full_cost_fallback_count += 1
            cost_meta = full_cost_lookup.get(_fc_key, {})
            base_cost = float(cost_meta.get("unit_cost_usd", _fallback_cost))
            is_out_region = int(str(trow.get("base_state", "")) != str(nrow["state_norm"]))
            penalty = out_of_region_penalty if is_out_region else 0.0
            trip_mode = cost_meta.get("trip_mode")
            zone_policy = str(
                trow.get("zone_policy", config.ZONE_POLICY_STANDARD)
            ).strip() or config.ZONE_POLICY_STANDARD
            jump = zone_jump_count(
                trow.get("base_operational_zone_rank"),
                nrow.get("node_operational_zone_rank"),
            )
            _, zone_penalty = evaluate_zone_policy(jump, zone_policy)
            hub_penalty = evaluate_hub_penalty(trow.get("base_hub_tier"), trip_mode)
            obj.append(base_cost + penalty + zone_penalty + hub_penalty)
            meta.append(
                {
                    "var_type": "x",
                    "tech_idx": ti,
                    "node_idx": ni,
                    "base_cost": base_cost,
                    "out_region_penalty": penalty,
                    "zone_policy": zone_policy,
                    "zone_jump_count": jump,
                    "timezone_penalty": zone_penalty,
                    "hub_penalty": hub_penalty,
                    "cost_meta": cost_meta,
                }
            )

    # New-hire assignment vars.
    for ci in candidate_indices:
        crow = candidates.loc[ci]
        for ni, nrow in nodes.iterrows():
            hire_skill_eligible = synthetic_new_hire_eligible_for_node(
                nrow,
                blank_slate=blank_slate,
            )
            idx = len(var_names)
            z_idx[(ci, ni)] = idx
            var_names.append(f"z__{crow['candidate_id']}__{nrow['node_id']}")
            lb.append(0.0)
            # In blank-slate mode every hire is hypothetical and treated as fully trained.
            if not hire_skill_eligible:
                ub.append(0.0)
            else:
                ub.append(float(nrow["appointment_count"]))
            integrality.append(1)

            _fc_key = (str(crow["candidate_id"]), str(nrow["node_id"]))
            if _fc_key not in full_cost_lookup:
                _full_cost_fallback_count += 1
            cost_meta = full_cost_lookup.get(_fc_key, {})
            base_cost = float(cost_meta.get("unit_cost_usd", _fallback_cost))
            is_out_region = int(str(crow.get("state", "")) != str(nrow["state_norm"]))
            penalty = out_of_region_penalty if is_out_region else 0.0
            trip_mode = cost_meta.get("trip_mode")
            jump = zone_jump_count(
                crow.get("operational_zone_rank"),
                nrow.get("node_operational_zone_rank"),
            )
            zone_feasible, zone_penalty = evaluate_zone_policy(jump, config.ZONE_POLICY_STANDARD)
            hub_penalty = evaluate_hub_penalty(crow.get("hub_tier"), trip_mode)
            if not hire_skill_eligible or not zone_feasible:
                ub[-1] = 0.0
            obj.append(base_cost + penalty + zone_penalty + hub_penalty)
            meta.append(
                {
                    "var_type": "z",
                    "candidate_idx": ci,
                    "node_idx": ni,
                    "base_cost": base_cost,
                    "out_region_penalty": penalty,
                    "zone_policy": config.ZONE_POLICY_STANDARD,
                    "zone_jump_count": jump,
                    "timezone_penalty": zone_penalty,
                    "hub_penalty": hub_penalty,
                    "cost_meta": cost_meta,
                }
            )

    if _full_cost_fallback_count > 0:
        print(f"  WARNING: {_full_cost_fallback_count} (tech/candidate, node) pairs missing from "
              f"full_cost_table.csv — used BTS fallback (${_fallback_cost:,.2f}).")

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

    # Optimized-scenario appointment cap across all existing assignable people,
    # including contractors. This is appointment count, not hours.
    if (not blank_slate) and optimized_max_appointments_per_person is not None:
        for ti in tech.index:
            for ni in nodes.index:
                idx = x_idx.get((ti, ni))
                if idx is None:
                    continue
                rows.append(r)
                cols.append(idx)
                data.append(1.0)
            lower.append(-np.inf)
            upper.append(float(optimized_max_appointments_per_person))
            r += 1

    # Per-hire appointment cap for synthetic hires. In optimized mode this cap
    # scales with hires allocated at a base; in blank-slate mode max_hires_per_base
    # is already fixed at 1.
    synthetic_hire_appointment_cap = (
        blank_slate_max_appointments_per_hire
        if blank_slate
        else optimized_max_appointments_per_person
    )
    if synthetic_hire_appointment_cap is not None:
        for ci in candidate_indices:
            for ni in nodes.index:
                idx = z_idx[(ci, ni)]
                rows.append(r)
                cols.append(idx)
                data.append(1.0)
            rows.append(r)
            cols.append(y_idx[ci])
            data.append(-float(synthetic_hire_appointment_cap))
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
    timezone_penalty_cost = 0.0
    hub_penalty_cost = 0.0
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
        zone_penalty = float(m.get("timezone_penalty", 0.0))
        hub_penalty = float(m.get("hub_penalty", 0.0))
        cost_meta = m.get("cost_meta", {})
        travel_cost += val * base
        out_region_cost += val * pen
        timezone_penalty_cost += val * zone_penalty
        hub_penalty_cost += val * hub_penalty
        hours = val * float(nrow["avg_hours_per_appointment"])
        existing_rows.append(
            {
                "scenario_hires": hire_count,
                "tech_id": trow["tech_id"],
                "tech_name": trow["tech_name"],
                "employment_type": trow["employment_type"],
                "base_state": trow["base_state"],
                "base_airport_iata": trow["base_airport_iata"],
                "base_hub_tier": trow.get("base_hub_tier"),
                "assignment_scope_mode": trow.get("assignment_scope_mode"),
                "assignment_scope_states": trow.get("assignment_scope_states"),
                "anchor_site_name": trow.get("anchor_site_name"),
                "anchor_reserved_fte": trow.get("anchor_reserved_fte"),
                "external_field_fte": trow.get("external_field_fte"),
                "node_id": nrow["node_id"],
                "state_norm": nrow["state_norm"],
                "skill_class": nrow["skill_class"],
                "assigned_appointments": val,
                "assigned_hours": hours,
                "unit_travel_cost_usd": base,
                "unit_out_region_penalty_usd": pen,
                "unit_timezone_penalty_usd": zone_penalty,
                "unit_hub_penalty_usd": hub_penalty,
                "zone_policy": m.get("zone_policy"),
                "zone_jump_count": m.get("zone_jump_count"),
                "base_operational_zone_label": trow.get("base_operational_zone_label"),
                "node_operational_zone_label": nrow.get("node_operational_zone_label"),
                "travel_cost_policy": cost_meta.get("travel_cost_policy", trow.get("travel_cost_policy")),
                "trip_mode": cost_meta.get("trip_mode"),
                "ground_transport_mode": cost_meta.get("ground_transport_mode"),
                "median_dist_mi": cost_meta.get("median_dist_mi"),
                "trip_span_days": cost_meta.get("trip_span_days"),
                "rental_days": cost_meta.get("rental_days"),
                "employee_style_unit_cost_usd": cost_meta.get("employee_style_unit_cost_usd", base),
                "effective_unit_cost_usd": cost_meta.get("effective_unit_cost_usd", base),
                "total_travel_cost_usd": val * base,
                "total_out_region_penalty_usd": val * pen,
                "total_timezone_penalty_usd": val * zone_penalty,
                "total_hub_penalty_usd": val * hub_penalty,
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
        zone_penalty = float(m.get("timezone_penalty", 0.0))
        hub_penalty = float(m.get("hub_penalty", 0.0))
        cost_meta = m.get("cost_meta", {})
        travel_cost += val * base
        out_region_cost += val * pen
        timezone_penalty_cost += val * zone_penalty
        hub_penalty_cost += val * hub_penalty
        hours = val * float(nrow["avg_hours_per_appointment"])
        new_rows.append(
            {
                "scenario_hires": hire_count,
                "candidate_id": crow["candidate_id"],
                "candidate_type": crow["candidate_type"],
                "candidate_city": crow["city"],
                "candidate_state": crow["state"],
                "airport_iata": crow["airport_iata"],
                "hub_tier": crow.get("hub_tier"),
                "node_id": nrow["node_id"],
                "state_norm": nrow["state_norm"],
                "skill_class": nrow["skill_class"],
                "assigned_appointments": val,
                "assigned_hours": hours,
                "unit_travel_cost_usd": base,
                "unit_out_region_penalty_usd": pen,
                "unit_timezone_penalty_usd": zone_penalty,
                "unit_hub_penalty_usd": hub_penalty,
                "zone_policy": m.get("zone_policy"),
                "zone_jump_count": m.get("zone_jump_count"),
                "base_operational_zone_label": crow.get("operational_zone_label"),
                "node_operational_zone_label": nrow.get("node_operational_zone_label"),
                "travel_cost_policy": cost_meta.get("travel_cost_policy", config.TRAVEL_COST_POLICY_EMPLOYEE),
                "trip_mode": cost_meta.get("trip_mode"),
                "ground_transport_mode": cost_meta.get("ground_transport_mode"),
                "median_dist_mi": cost_meta.get("median_dist_mi"),
                "trip_span_days": cost_meta.get("trip_span_days"),
                "rental_days": cost_meta.get("rental_days"),
                "employee_style_unit_cost_usd": cost_meta.get("employee_style_unit_cost_usd", base),
                "effective_unit_cost_usd": cost_meta.get("effective_unit_cost_usd", base),
                "total_travel_cost_usd": val * base,
                "total_out_region_penalty_usd": val * pen,
                "total_timezone_penalty_usd": val * zone_penalty,
                "total_hub_penalty_usd": val * hub_penalty,
            }
        )

    for ni, idx in u_idx.items():
        val = float(solution[idx])
        unmet_appointments += val

    existing_df = pd.DataFrame(existing_rows, columns=EXISTING_ASSIGNMENT_COLUMNS)
    new_df = pd.DataFrame(new_rows, columns=NEW_HIRE_ASSIGNMENT_COLUMNS)

    # Legacy "utilization" output by tech. This is assigned modeled workload
    # divided by modeled capacity under the same calendar-window framework.
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
    util_df = pd.DataFrame(util_rows, columns=UTILIZATION_COLUMNS)

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
    placements_df = pd.DataFrame(placement_rows, columns=PLACEMENT_COLUMNS)

    hire_cost = float(sum(int(round(v)) for v in y_values.values()) * annual_hire_cost_usd)
    unmet_cost = float(unmet_appointments * unmet_penalty)
    modeled_total = float(
        travel_cost
        + out_region_cost
        + timezone_penalty_cost
        + hub_penalty_cost
        + hire_cost
        + unmet_cost
    )

    contractor_usage_df = empty_frame(CONTRACTOR_USAGE_COLUMNS)
    if not existing_df.empty:
        contractor_df = existing_df[
            existing_df["employment_type"].astype(str).str.lower().eq("contractor")
        ].copy()
        if not contractor_df.empty:
            contractor_df["zone_jump_count"] = pd.to_numeric(
                contractor_df["zone_jump_count"], errors="coerce"
            ).fillna(0.0)
            contractor_df["weighted_zone_jump"] = (
                contractor_df["assigned_appointments"] * contractor_df["zone_jump_count"]
            )
            contractor_df["two_zone_plus_appointments"] = np.where(
                contractor_df["zone_jump_count"] >= 2,
                contractor_df["assigned_appointments"],
                0.0,
            )
            contractor_df["three_zone_plus_appointments"] = np.where(
                contractor_df["zone_jump_count"] >= 3,
                contractor_df["assigned_appointments"],
                0.0,
            )
            contractor_usage_df = (
                contractor_df.groupby(["scenario_hires", "tech_id", "tech_name"], as_index=False)
                .agg(
                    assigned_appointments=("assigned_appointments", "sum"),
                    assigned_hours=("assigned_hours", "sum"),
                    total_travel_cost_usd=("total_travel_cost_usd", "sum"),
                    total_hub_penalty_usd=("total_hub_penalty_usd", "sum"),
                    weighted_zone_jump=("weighted_zone_jump", "sum"),
                    two_zone_plus_appointments=("two_zone_plus_appointments", "sum"),
                    three_zone_plus_appointments=("three_zone_plus_appointments", "sum"),
                    states_served=("state_norm", "nunique"),
                )
            )
            contractor_usage_df["avg_zone_jump"] = np.where(
                contractor_usage_df["assigned_appointments"] > 0,
                contractor_usage_df["weighted_zone_jump"] / contractor_usage_df["assigned_appointments"],
                0.0,
            )
            contractor_usage_df["share_two_zone_plus"] = np.where(
                contractor_usage_df["assigned_appointments"] > 0,
                contractor_usage_df["two_zone_plus_appointments"] / contractor_usage_df["assigned_appointments"],
                0.0,
            )
            contractor_usage_df["share_three_zone_plus"] = np.where(
                contractor_usage_df["assigned_appointments"] > 0,
                contractor_usage_df["three_zone_plus_appointments"] / contractor_usage_df["assigned_appointments"],
                0.0,
            )
            contractor_usage_df = contractor_usage_df[
                CONTRACTOR_USAGE_COLUMNS
            ]

    mean_existing_utilization = (
        float(np.nanmean(util_df["utilization"].values)) if not util_df.empty else 0.0
    )
    max_existing_utilization = (
        float(np.nanmax(util_df["utilization"].values)) if not util_df.empty else 0.0
    )

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
        "timezone_penalty_usd": timezone_penalty_cost,
        "hub_penalty_usd": hub_penalty_cost,
        "hire_cost_usd": hire_cost,
        "unmet_penalty_usd": unmet_cost,
        "modeled_total_cost_usd": modeled_total,
        "hours_per_capacity_unit": float(hours_per_unit),
        "new_hire_capacity_hours": float(new_hire_capacity_hours),
        "mean_existing_utilization": mean_existing_utilization,
        "max_existing_utilization": max_existing_utilization,
    }

    return {
        "summary": summary,
        "existing_assignments": existing_df,
        "new_assignments": new_df,
        "placements": placements_df,
        "tech_utilization": util_df,
        "contractor_usage": contractor_usage_df,
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
    parser.add_argument(
        "--target-utilization",
        type=float,
        default=0.85,
        help=(
            "Modeled load target used to normalize technician capacity against "
            "the appointment-duration demand pool. This is not a literal "
            "weekday payroll-utilization target."
        ),
    )
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
        help="Legacy override for contractor assignment scope.",
    )
    parser.add_argument(
        "--blank-slate",
        action="store_true",
        help=(
            "Solve a pure rebuild scenario: zero current-tech capacity, allow "
            "new hires on HPS nodes, and write outputs to the blank_slate subdirectory."
        ),
    )
    parser.add_argument(
        "--blank-slate-max-appointments-per-hire",
        type=int,
        default=config.BLANK_SLATE_MAX_APPOINTMENTS_PER_HIRE,
        help=(
            "Hard cap on assigned appointments per synthetic hire in blank-slate "
            "mode. Applies to appointment count, not duration hours."
        ),
    )
    parser.add_argument(
        "--optimized-max-appointments-per-person",
        type=int,
        default=config.OPTIMIZED_MAX_APPOINTMENTS_PER_PERSON,
        help=(
            "Hard cap on assigned appointments per assignable person in normal "
            "optimized scenarios. Applies to current techs, contractors, and "
            "synthetic hires. Uses appointment count, not duration hours."
        ),
    )
    args = parser.parse_args()
    if args.annual_hire_cost_usd < 0:
        raise ValueError("--annual-hire-cost-usd must be non-negative.")
    if args.out_of_region_penalty < 0:
        raise ValueError("--out-of-region-penalty must be non-negative.")
    if args.max_hires_per_base < 1:
        raise ValueError("--max-hires-per-base must be at least 1.")
    if args.blank_slate_max_appointments_per_hire < 1:
        raise ValueError("--blank-slate-max-appointments-per-hire must be at least 1.")
    if args.optimized_max_appointments_per_person < 1:
        raise ValueError("--optimized-max-appointments-per-person must be at least 1.")
    if not (0 < args.target_utilization <= 1.0):
        raise ValueError("--target-utilization must be in range (0, 1.0].")
    if args.blank_slate:
        if args.min_new_hires <= 0:
            raise ValueError("--blank-slate requires --min-new-hires to be positive.")
        if args.max_new_hires <= 0:
            raise ValueError("--blank-slate requires a positive hire count.")
        if args.min_new_hires > args.max_new_hires:
            raise ValueError("--blank-slate requires --min-new-hires to be less than or equal to --max-new-hires.")
        if args.max_hires_per_base != 1:
            raise ValueError("--blank-slate requires --max-hires-per-base 1.")

    input_dir = Path(args.output_dir)
    out_dir = blank_slate_output_dir(input_dir) if args.blank_slate else input_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs(input_dir)

    input_summary_path = input_dir / "optimization_input_summary.json"
    if input_summary_path.exists():
        with open(input_summary_path) as f:
            input_summary = json.load(f)
        data_span_years = float(input_summary.get("data_span_years", 1.0))
    else:
        data_span_years = 1.0
    print(f"  Data span: {data_span_years:.2f} years")

    tech = inputs["tech"].copy()
    demand = inputs["demand"].copy()
    candidates = inputs["candidates"].copy()

    for col in [
        "skill_hps",
        "skill_ls",
        "skill_patient",
        "constraint_florida_only",
    ]:
        tech[col] = pd.to_numeric(tech[col], errors="coerce").fillna(0).astype(int)
    tech["availability_fte"] = pd.to_numeric(tech["availability_fte"], errors="coerce").fillna(0.0)
    tech["base_state"] = tech["base_state"].map(normalize_state)
    tech["base_operational_zone_rank"] = pd.to_numeric(
        tech.get("base_operational_zone_rank"), errors="coerce"
    )
    candidates["operational_zone_rank"] = pd.to_numeric(
        candidates.get("operational_zone_rank"), errors="coerce"
    )
    if "assignment_scope_mode" not in tech.columns:
        is_contractor = tech["employment_type"].astype(str).str.lower().eq("contractor")
        tech["assignment_scope_mode"] = np.where(
            is_contractor & (args.contractor_assignment_scope == "texas_only"),
            config.ASSIGNMENT_SCOPE_MODE_STATE_LIMITED,
            config.ASSIGNMENT_SCOPE_MODE_NATIONAL,
        )
    if "assignment_scope_state" not in tech.columns:
        is_contractor = tech["employment_type"].astype(str).str.lower().eq("contractor")
        tech["assignment_scope_state"] = np.where(
            is_contractor & (args.contractor_assignment_scope == "texas_only"),
            "TX",
            "",
        )
    if "travel_cost_policy" not in tech.columns:
        tech["travel_cost_policy"] = np.where(
            tech["employment_type"].astype(str).str.lower().eq("contractor"),
            config.TRAVEL_COST_POLICY_CONTRACTOR,
            config.TRAVEL_COST_POLICY_EMPLOYEE,
        )
    if "contractor_cost_multiplier" not in tech.columns:
        tech["contractor_cost_multiplier"] = np.where(
            tech["employment_type"].astype(str).str.lower().eq("contractor"),
            config.CONTRACTOR_COST_MULTIPLIER,
            1.0,
        )
    if "contractor_cost_cap_usd" not in tech.columns:
        tech["contractor_cost_cap_usd"] = np.where(
            tech["employment_type"].astype(str).str.lower().eq("contractor"),
            config.CONTRACTOR_COST_CAP_USD,
            np.nan,
        )
    if "contractor_dispatch_surcharge_usd" not in tech.columns:
        tech["contractor_dispatch_surcharge_usd"] = np.where(
            tech["employment_type"].astype(str).str.lower().eq("contractor"),
            config.CONTRACTOR_DISPATCH_SURCHARGE_USD,
            0.0,
        )
    if "zone_policy" not in tech.columns:
        tech["zone_policy"] = np.where(
            tech["employment_type"].astype(str).str.lower().eq("contractor"),
            config.ZONE_POLICY_CONTRACTOR_SOFT,
            config.ZONE_POLICY_STANDARD,
        )
    if args.contractor_assignment_scope:
        contractor_mask = tech["employment_type"].astype(str).str.lower().eq("contractor")
        if args.contractor_assignment_scope == "texas_only":
            tech.loc[contractor_mask, "assignment_scope_mode"] = config.ASSIGNMENT_SCOPE_MODE_STATE_LIMITED
            tech.loc[contractor_mask, "assignment_scope_state"] = "TX"
        else:
            tech.loc[contractor_mask, "assignment_scope_mode"] = config.ASSIGNMENT_SCOPE_MODE_NATIONAL
            tech.loc[contractor_mask, "assignment_scope_state"] = ""

    if args.blank_slate:
        print("  Blank slate mode: zeroing existing technician availability.")
        tech["availability_fte"] = 0.0
    demand, florida_rows_removed = prepare_demand_for_solve(demand, args.blank_slate)
    if args.blank_slate:
        print(
            f"  Blank slate mode: removed {florida_rows_removed} Florida appointment row(s); "
            f"{len(demand):,} demand row(s) remain."
        )

    demand_nodes = build_demand_nodes(demand)

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
    all_contractors = []

    canceled_voided_usd = config.BASELINE_CANCELED_VOIDED_USD

    # Scale hire cost to match the data period.
    # Travel costs cover the full data span (2.08 years of appointments).
    # Hire cost must be scaled to match for like-for-like comparison in the MILP.
    hire_cost_for_period = args.annual_hire_cost_usd * data_span_years
    print(f"  Annual hire cost: ${args.annual_hire_cost_usd:,.0f} × {data_span_years:.2f} years = ${hire_cost_for_period:,.0f} for optimization period")
    if not args.blank_slate:
        print(
            "  Optimized synthetic new hires are modeled as patient-sim-only: "
            "they cannot cover Learning Space / AVS, and their HPS restriction remains active."
        )

    scenario_counts = list(range(args.min_new_hires, args.max_new_hires + 1))

    for hire_count in scenario_counts:
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
                full_cost_lookup=full_cost_lookup,
                contractor_scope=contractor_scope,
                target_utilization=args.target_utilization,
                out_of_region_penalty=args.out_of_region_penalty,
                unmet_penalty=args.unmet_penalty,
                annual_hire_cost_usd=hire_cost_for_period,
                max_hires_per_base=args.max_hires_per_base,
                time_limit_sec=args.time_limit_sec,
                blank_slate=args.blank_slate,
                blank_slate_max_appointments_per_hire=(
                    args.blank_slate_max_appointments_per_hire if args.blank_slate else None
                ),
                optimized_max_appointments_per_person=(
                    None if args.blank_slate else args.optimized_max_appointments_per_person
                ),
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
        if not result["contractor_usage"].empty:
            all_contractors.append(result["contractor_usage"])

    summary_df = pd.DataFrame(scenario_summaries).sort_values("scenario_hires")
    existing_df = (
        pd.concat(all_existing, ignore_index=True)
        if all_existing
        else empty_frame(EXISTING_ASSIGNMENT_COLUMNS)
    )
    new_df = (
        pd.concat(all_new, ignore_index=True)
        if all_new
        else empty_frame(NEW_HIRE_ASSIGNMENT_COLUMNS)
    )
    placements_df = (
        pd.concat(all_placements, ignore_index=True)
        if all_placements
        else empty_frame(PLACEMENT_COLUMNS)
    )
    util_df = (
        pd.concat(all_util, ignore_index=True)
        if all_util
        else empty_frame(UTILIZATION_COLUMNS)
    )
    contractor_usage_df = (
        pd.concat(all_contractors, ignore_index=True)
        if all_contractors
        else empty_frame(CONTRACTOR_USAGE_COLUMNS)
    )

    summary_out = out_dir / "scenario_summary.csv"
    existing_out = out_dir / "scenario_assignments_existing.csv"
    new_out = out_dir / "scenario_assignments_newhires.csv"
    placements_out = out_dir / "scenario_placements.csv"
    util_out = out_dir / "scenario_tech_utilization.csv"
    contractor_out = out_dir / "scenario_contractor_usage.csv"
    assumptions_out = out_dir / "model_assumptions.json"

    summary_df.to_csv(summary_out, index=False)
    existing_df.to_csv(existing_out, index=False)
    new_df.to_csv(new_out, index=False)
    placements_df.to_csv(placements_out, index=False)
    util_df.to_csv(util_out, index=False)
    contractor_usage_df.to_csv(contractor_out, index=False)

    anchored_mask = (
        tech["anchor_site_name"].fillna("").astype(str).str.strip().ne("")
        if "anchor_site_name" in tech.columns
        else pd.Series(False, index=tech.index)
    )
    special_tech_constraints = []
    for _, row in tech[anchored_mask].iterrows():
        reserved = pd.to_numeric(row.get("anchor_reserved_fte"), errors="coerce")
        external = pd.to_numeric(row.get("external_field_fte"), errors="coerce")
        special_tech_constraints.append(
            {
                "tech_name": str(row["tech_name"]),
                "anchor_site_name": str(row.get("anchor_site_name", "")).strip(),
                "anchor_reserved_fte": None if pd.isna(reserved) else float(reserved),
                "external_field_fte": None if pd.isna(external) else float(external),
                "assignment_scope_mode": str(row.get("assignment_scope_mode", "")).strip(),
                "assignment_scope_states": str(row.get("assignment_scope_states", "")).strip(),
                "anchor_notes": str(row.get("anchor_notes", "")).strip(),
            }
        )

    assumptions = {
        "min_new_hires": args.min_new_hires,
        "max_new_hires": args.max_new_hires,
        "blank_slate": bool(args.blank_slate),
        "blank_slate_hire_count": int(args.max_new_hires) if args.blank_slate else None,
        "blank_slate_unique_bases": bool(args.blank_slate),
        "blank_slate_max_appointments_per_hire": (
            int(args.blank_slate_max_appointments_per_hire) if args.blank_slate else None
        ),
        "blank_slate_appointment_cap_note": (
            f"Blank-slate synthetic hires are capped at "
            f"{int(args.blank_slate_max_appointments_per_hire)} assigned appointments each. "
            "This is an appointment-count ceiling, not a duration-hours or utilization cap."
            if args.blank_slate
            else None
        ),
        "optimized_max_appointments_per_person": (
            None if args.blank_slate else int(args.optimized_max_appointments_per_person)
        ),
        "optimized_appointment_cap_note": (
            f"Optimized scenarios cap every assignable person at "
            f"{int(args.optimized_max_appointments_per_person)} assigned appointments over "
            "the full data span. This applies to current techs, contractors, and synthetic "
            "hires, and it is an appointment-count ceiling, not a duration-hours or "
            "utilization cap."
            if not args.blank_slate
            else None
        ),
        "target_utilization": args.target_utilization,
        "out_of_region_penalty": args.out_of_region_penalty,
        "unmet_penalty": args.unmet_penalty,
        "annual_hire_cost_usd": args.annual_hire_cost_usd,
        "max_hires_per_base": args.max_hires_per_base,
        "data_span_years": data_span_years,
        "hire_cost_for_optimization_period": hire_cost_for_period,
        "hire_cost_scope": "incremental_new_hires_only",
        "hire_cost_input_mode": "direct_fixed_value",
        "workload_time_basis": (
            "Appointment duration_hours are treated as calendar-window workload, "
            "not exact hands-on labor time."
        ),
        "capacity_basis": (
            "Technician capacity is normalized against total modeled demand "
            "hours using availability_fte and target_utilization."
        ),
        "utilization_metric_note": (
            "Legacy utilization fields are modeled load ratios under the "
            "calendar-window demand framework, not timesheet or Monday-through-"
            "Friday labor utilization percentages."
        ),
        "target_utilization_note": (
            "target_utilization is the normalization target for the modeled "
            "load ratio, not a literal payroll-utilization benchmark."
        ),
        "hub_connectivity_penalty": {
            "applies_to_trip_mode": "fly_only",
            "classification_source": {
                "us_airports": "faa_cy2024_commercial_service_enplanements",
                "source_year": getattr(config, "FAA_HUB_SOURCE_YEAR", None),
                "source_url": getattr(config, "FAA_HUB_SOURCE_URL", ""),
                "canadian_airports": "manual_canada_override",
            },
            "tier_penalties_usd_per_appointment": {
                "large_hub": config.HUB_PENALTY_LARGE,
                "medium_hub": config.HUB_PENALTY_MEDIUM,
                "small_hub": config.HUB_PENALTY_SMALL,
                "nonhub": config.HUB_PENALTY_NONHUB,
                "unknown": config.HUB_PENALTY_UNKNOWN,
            },
        },
        "contractor_assignment_scope": contractor_scope,
        "contractor_policy": {
            "travel_cost_policy": config.TRAVEL_COST_POLICY_CONTRACTOR,
            "assignment_scope_mode_default": config.ASSIGNMENT_SCOPE_MODE_NATIONAL,
            "cost_multiplier_default": config.CONTRACTOR_COST_MULTIPLIER,
            "cost_cap_usd_default": config.CONTRACTOR_COST_CAP_USD,
            "dispatch_surcharge_usd_default": config.CONTRACTOR_DISPATCH_SURCHARGE_USD,
            "zone_policy_default": config.ZONE_POLICY_CONTRACTOR_SOFT,
        },
        "special_tech_constraints": special_tech_constraints,
        "operational_zone_policy": {
            "anchor": "airport_based_operational_zone_buckets",
            "standard_rule": "0-1 free, 2 penalized, 3+ blocked",
            "contractor_rule": "0-1 free, 2 penalized, 3+ heavily penalized",
            "arizona_handling": "Mapped to a fixed Mountain operational bucket; DST ignored.",
        },
        "full_cost_model": True,
        "flight_cost_model": "BTS Q2 2025 lookup × corporate premium",
        "full_cost_model_constants": {
            "bts_national_fallback_usd": config.BTS_NATIONAL_FALLBACK,
            "corporate_travel_premium": config.CORPORATE_TRAVEL_PREMIUM,
            "irs_mileage_rate_usd_per_mi": config.IRS_MILEAGE_RATE_USD_PER_MI,
            "rental_car_daily_rate_usd": config.RENTAL_CAR_DAILY_RATE_USD,
            "personal_vehicle_max_one_way_mi": config.PERSONAL_VEHICLE_MAX_ONE_WAY_MI,
            "hotel_nightly_rate_usd": config.HOTEL_NIGHTLY_RATE_USD,
            "hotel_avg_nights": config.HOTEL_AVG_NIGHTS,
            "hotel_avg_usd_legacy": config.HOTEL_AVG_USD,
            "same_day_drive_threshold_mi": config.SAME_DAY_DRIVE_THRESHOLD_MI,
            "overnight_drive_threshold_mi": config.OVERNIGHT_DRIVE_THRESHOLD_MI,
            "baseline_canceled_voided_usd": config.BASELINE_CANCELED_VOIDED_USD,
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
    print(f"Saved: {contractor_out}")
    print(f"Saved: {assumptions_out}")
    print("\nScenario summary:")
    print(summary_df.to_string(index=False))
    print("Step 8 complete.")


if __name__ == "__main__":
    main()
