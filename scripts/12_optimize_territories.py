"""Step 12: Fixed-roster territory optimization for current technicians.

This mode answers a different question than the hire-scenario solver:
"Given the real current roster and fixed home bases, who should mainly cover
which 2025 demand so territories tighten and travel goes down?"
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import config
from optimization_utils import US_STATE_ABBR, canonicalize_tech_name, haversine_miles


STATE_ABBR_TO_NAME = {abbr: name.title() for name, abbr in US_STATE_ABBR.items()}


def load_step08_module():
    """Load the existing optimization module so this mode can reuse its solver."""
    script_path = Path(__file__).resolve().parent / "08_optimize_locations.py"
    spec = importlib.util.spec_from_file_location("step08_optimize_locations", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


STEP08 = load_step08_module()


def optimize_territories_dir(base_dir: Path) -> Path:
    """Return the fixed-roster territory-optimization output directory."""
    return base_dir / getattr(config, "OPTIMIZE_TERRITORIES_SUBDIR", "optimize_territories")


def require_inputs(root_dir: Path) -> dict[str, pd.DataFrame]:
    """Load the shared optimization inputs required for territory optimization."""
    files = {
        "tech": root_dir / "tech_master.csv",
        "demand": root_dir / "demand_appointments.csv",
        "full_cost": root_dir / "full_cost_table.csv",
    }
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        joined = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "Territory optimization is missing prerequisite files. Run steps 06 and 11 first.\n"
            + joined
        )
    return {
        "tech": pd.read_csv(files["tech"]),
        "demand": pd.read_csv(files["demand"]),
        "full_cost": pd.read_csv(files["full_cost"]),
    }


def available_mode_specs() -> list[dict]:
    """Return configured territory-optimization mode specs."""
    return list(getattr(config, "OPTIMIZE_TERRITORIES_MODES", []))


def filter_roster(tech_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Keep the fixed real roster requested for territory optimization."""
    tech = tech_df.copy()
    tech["tech_name"] = tech["tech_name"].apply(canonicalize_tech_name)
    tech["employment_type"] = tech["employment_type"].fillna("").astype(str).str.strip().str.lower()
    tech["base_country"] = tech["base_country"].fillna("").astype(str).str.strip()
    tech["availability_fte"] = pd.to_numeric(tech["availability_fte"], errors="coerce").fillna(0.0)

    excluded_names = {
        canonicalize_tech_name(name)
        for name in getattr(config, "OPTIMIZE_TERRITORIES_EXCLUDED_TECH_NAMES", set())
    }
    excluded_mask = (
        tech["employment_type"].eq("contractor")
        | tech["tech_name"].isin(excluded_names)
        | tech["base_country"].str.upper().eq("CANADA")
        | tech["availability_fte"].le(0)
    )
    excluded = tech.loc[excluded_mask, "tech_name"].astype(str).tolist()
    kept = tech.loc[~excluded_mask].copy().reset_index(drop=True)
    return kept, sorted(excluded)


def filter_2025_demand(demand_df: pd.DataFrame, mode_key: str) -> tuple[pd.DataFrame, dict]:
    """Filter demand to the 2025 territory-design view requested by the business."""
    demand = demand_df.copy()
    demand["scheduled_start"] = pd.to_datetime(demand["scheduled_start"], errors="coerce")
    demand["state_norm"] = demand["state_norm"].fillna("").astype(str).str.strip().str.upper()
    demand["country"] = demand["country"].fillna("").astype(str).str.strip().str.upper()
    demand["skill_class"] = demand["skill_class"].fillna("").astype(str).str.strip().str.lower()
    demand["required_ls"] = pd.to_numeric(demand["required_ls"], errors="coerce").fillna(0).astype(int)
    demand["required_hps"] = pd.to_numeric(demand["required_hps"], errors="coerce").fillna(0).astype(int)

    demand_year = int(getattr(config, "OPTIMIZE_TERRITORIES_YEAR", 2025))
    excluded_states = {
        str(state).strip().upper()
        for state in getattr(config, "OPTIMIZE_TERRITORIES_EXCLUDED_STATES", set())
    }

    year_mask = demand["scheduled_start"].dt.year.eq(demand_year)
    usa_mask = demand["country"].eq("USA")
    state_mask = ~demand["state_norm"].isin(excluded_states)
    filtered = demand.loc[year_mask & usa_mask & state_mask].copy()

    if mode_key == "patient_sim":
        filtered = filtered.loc[filtered["skill_class"].eq("regular")].copy()
    elif mode_key == "learning_space":
        filtered = filtered.loc[filtered["required_ls"].eq(1)].copy()
    elif mode_key == "hps":
        filtered = filtered.loc[filtered["required_hps"].eq(1)].copy()
    elif mode_key == "combined":
        filtered = filtered.copy()
    else:
        raise ValueError(f"Unsupported territory mode: {mode_key}")

    meta = {
        "demand_year": demand_year,
        "excluded_states": sorted(excluded_states),
        "appointments_total_after_year_filter": int((year_mask & usa_mask).sum()),
        "appointments_removed_excluded_states": int((year_mask & usa_mask & ~state_mask).sum()),
        "appointments_in_mode": int(len(filtered)),
        "states_in_mode": int(filtered["state_norm"].nunique()),
    }
    return filtered.reset_index(drop=True), meta


def build_adjusted_cost_lookup(
    full_cost_df: pd.DataFrame,
    tech_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
) -> tuple[dict[tuple[str, str], dict], dict[tuple[str, str], dict]]:
    """Build original + territory-tightened full-cost lookups for the fixed roster."""
    original_lookup = STEP08.build_full_cost_lookup(full_cost_df)
    tech_lookup = tech_df.set_index("tech_id")
    node_lookup = nodes_df.set_index("node_id")
    adjusted_lookup: dict[tuple[str, str], dict] = {}

    one_zone_penalty = float(
        getattr(config, "OPTIMIZE_TERRITORIES_ONE_ZONE_EXTRA_PENALTY_USD", 0.0)
    )
    two_zone_penalty = float(
        getattr(config, "OPTIMIZE_TERRITORIES_TWO_ZONE_EXTRA_PENALTY_USD", 0.0)
    )
    overnight_penalty = float(
        getattr(config, "OPTIMIZE_TERRITORIES_OVERNIGHT_TRIP_PENALTY_USD", 0.0)
    )
    fly_penalty = float(
        getattr(config, "OPTIMIZE_TERRITORIES_FLY_TRIP_PENALTY_USD", 0.0)
    )

    for key, raw_row in original_lookup.items():
        tech_id, node_id = key
        if tech_id not in tech_lookup.index or node_id not in node_lookup.index:
            continue
        row = dict(raw_row)
        trip_mode = str(row.get("trip_mode", "")).strip().lower()
        territory_design_penalty = 0.0
        if trip_mode == "drive_overnight":
            territory_design_penalty += overnight_penalty
        elif trip_mode == "fly":
            territory_design_penalty += fly_penalty

        jump = STEP08.zone_jump_count(
            tech_lookup.at[tech_id, "base_operational_zone_rank"],
            node_lookup.at[node_id, "node_operational_zone_rank"],
        )
        if jump == 1:
            territory_design_penalty += one_zone_penalty
        elif jump is not None and jump >= 2:
            territory_design_penalty += two_zone_penalty

        row["territory_design_penalty_usd"] = round(territory_design_penalty, 2)
        row["effective_unit_cost_usd"] = round(
            float(pd.to_numeric(row.get("effective_unit_cost_usd"), errors="coerce") or 0.0)
            + territory_design_penalty,
            2,
        )
        row["unit_cost_usd"] = round(
            float(pd.to_numeric(row.get("unit_cost_usd"), errors="coerce") or 0.0)
            + territory_design_penalty,
            2,
        )
        adjusted_lookup[key] = row

    return original_lookup, adjusted_lookup


def resolve_appointment_assignments(
    demand_df: pd.DataFrame,
    assignment_df: pd.DataFrame,
    tech_df: pd.DataFrame,
) -> pd.DataFrame:
    """Reconstruct appointment-level ownership from node-level solver quotas."""
    if demand_df.empty or assignment_df.empty or tech_df.empty:
        return pd.DataFrame(
            columns=[
                "appointment_id",
                "tech_id",
                "tech_name",
                "lat",
                "lon",
                "account_name",
                "city",
                "state_norm",
                "skill_class",
                "territory_mode",
            ]
        )

    base_coords = {}
    tech_name_lookup = {}
    for _, row in tech_df.iterrows():
        tech_id = str(row["tech_id"])
        lat = pd.to_numeric(row.get("base_lat"), errors="coerce")
        lon = pd.to_numeric(row.get("base_lon"), errors="coerce")
        if pd.notna(lat) and pd.notna(lon):
            base_coords[tech_id] = (float(lat), float(lon))
        tech_name_lookup[tech_id] = str(row["tech_name"])

    demand = demand_df.copy()
    demand["appointment_id"] = (
        demand["appointment_id"].fillna(demand["Appointment Number"]).astype(str).str.strip()
    )
    demand["lat"] = pd.to_numeric(demand.get("lat"), errors="coerce")
    demand["lon"] = pd.to_numeric(demand.get("lon"), errors="coerce")
    demand["node_id"] = (
        demand["state_norm"].astype(str).str.strip().str.upper()
        + "__"
        + demand["skill_class"].astype(str).str.strip().str.lower()
    )

    node_appointments: dict[str, list[dict]] = defaultdict(list)
    for _, row in demand.dropna(subset=["lat", "lon"]).iterrows():
        node_appointments[str(row["node_id"])].append(
            {
                "appointment_id": str(row["appointment_id"]),
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "account_name": str(row.get("Account: Account Name", "")),
                "city": str(row.get("city", "")),
                "state_norm": str(row.get("state_norm", "")),
                "skill_class": str(row.get("skill_class", "")),
            }
        )

    records: list[dict] = []
    mode_value = (
        assignment_df["territory_mode"].iloc[0]
        if "territory_mode" in assignment_df.columns and not assignment_df.empty
        else ""
    )

    for node_id, node_group in assignment_df.groupby("node_id"):
        appts = sorted(node_appointments.get(str(node_id), []), key=lambda item: item["appointment_id"])
        if not appts:
            continue

        quotas = []
        for _, row in node_group.iterrows():
            tech_id = str(row["tech_id"])
            quota = float(pd.to_numeric(row.get("assigned_appointments"), errors="coerce") or 0.0)
            if quota <= 0 or tech_id not in base_coords:
                continue
            quotas.append((tech_id, quota))
        if not quotas:
            continue

        rounded = [(tech_id, int(round(quota))) for tech_id, quota in quotas]
        total_appts = len(appts)
        total_rounded = sum(quota for _, quota in rounded)
        if rounded and total_rounded != total_appts:
            idx_max = max(range(len(rounded)), key=lambda idx: rounded[idx][1])
            tech_id, quota = rounded[idx_max]
            rounded[idx_max] = (tech_id, max(0, quota + (total_appts - total_rounded)))

        remaining = {tech_id: quota for tech_id, quota in rounded}
        active_techs = [tech_id for tech_id, quota in rounded if quota > 0]
        for appt in appts:
            if not active_techs:
                break
            chosen_id = None
            chosen_dist = float("inf")
            for tech_id in active_techs:
                if remaining.get(tech_id, 0) <= 0:
                    continue
                base_lat, base_lon = base_coords[tech_id]
                dist = haversine_miles(appt["lat"], appt["lon"], base_lat, base_lon)
                if dist < chosen_dist or (
                    dist == chosen_dist and (chosen_id is None or tech_id < chosen_id)
                ):
                    chosen_id = tech_id
                    chosen_dist = dist
            if chosen_id is None:
                continue
            remaining[chosen_id] -= 1
            if remaining[chosen_id] <= 0:
                active_techs = [tech_id for tech_id in active_techs if remaining.get(tech_id, 0) > 0]

            records.append(
                {
                    "appointment_id": appt["appointment_id"],
                    "tech_id": chosen_id,
                    "tech_name": tech_name_lookup.get(chosen_id, chosen_id),
                    "lat": appt["lat"],
                    "lon": appt["lon"],
                    "account_name": appt["account_name"],
                    "city": appt["city"],
                    "state_norm": appt["state_norm"],
                    "skill_class": appt["skill_class"],
                    "territory_mode": mode_value,
                }
            )

    return pd.DataFrame.from_records(records)


def enrich_assignment_costs(
    assignments_df: pd.DataFrame,
    original_lookup: dict[tuple[str, str], dict],
) -> pd.DataFrame:
    """Add raw-travel and design-penalty fields to fixed-roster assignment rows."""
    if assignments_df.empty:
        return assignments_df.copy()

    rows = []
    for _, row in assignments_df.iterrows():
        key = (str(row["tech_id"]), str(row["node_id"]))
        original = original_lookup.get(key, {})
        raw_unit = float(pd.to_numeric(original.get("unit_cost_usd"), errors="coerce") or 0.0)
        adjusted_unit = float(pd.to_numeric(row.get("unit_travel_cost_usd"), errors="coerce") or 0.0)
        assigned_appointments = float(pd.to_numeric(row.get("assigned_appointments"), errors="coerce") or 0.0)
        item = row.to_dict()
        item["raw_unit_travel_cost_usd"] = round(raw_unit, 2)
        item["raw_total_travel_cost_usd"] = round(raw_unit * assigned_appointments, 2)
        item["territory_design_penalty_usd"] = round(max(adjusted_unit - raw_unit, 0.0), 2)
        item["total_territory_design_penalty_usd"] = round(
            max(adjusted_unit - raw_unit, 0.0) * assigned_appointments,
            2,
        )
        rows.append(item)
    return pd.DataFrame(rows)


def build_tech_summary(
    mode_key: str,
    tech_df: pd.DataFrame,
    assignments_df: pd.DataFrame,
    appointment_assignments_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build business-readable per-tech territory summaries."""
    if assignments_df.empty:
        assignment_group = pd.DataFrame(
            columns=[
                "tech_id",
                "tech_name",
                "assigned_appointments",
                "assigned_hours",
                "modeled_objective_cost_usd",
                "raw_travel_cost_usd",
                "territory_design_penalty_usd",
                "timezone_penalty_usd",
                "hub_penalty_usd",
                "out_of_region_penalty_usd",
            ]
        ).set_index("tech_id")
    else:
        assignment_group = (
            assignments_df.groupby(["tech_id", "tech_name"], as_index=False)
            .agg(
                assigned_appointments=("assigned_appointments", "sum"),
                assigned_hours=("assigned_hours", "sum"),
                modeled_objective_cost_usd=("total_travel_cost_usd", "sum"),
                raw_travel_cost_usd=("raw_total_travel_cost_usd", "sum"),
                territory_design_penalty_usd=("total_territory_design_penalty_usd", "sum"),
                timezone_penalty_usd=("total_timezone_penalty_usd", "sum"),
                hub_penalty_usd=("total_hub_penalty_usd", "sum"),
                out_of_region_penalty_usd=("total_out_region_penalty_usd", "sum"),
            )
            .set_index("tech_id")
        )

    state_counts: dict[str, dict[str, int]] = defaultdict(dict)
    if not appointment_assignments_df.empty:
        counts = (
            appointment_assignments_df.groupby(["tech_id", "state_norm"])
            .size()
            .reset_index(name="appointment_count")
        )
        for _, row in counts.iterrows():
            state_counts[str(row["tech_id"])][str(row["state_norm"])] = int(row["appointment_count"])

    zone_jump_by_tech = {}
    two_zone_plus_share = {}
    if not assignments_df.empty:
        zone_metrics = assignments_df.copy()
        zone_metrics["assigned_appointments_num"] = pd.to_numeric(
            zone_metrics["assigned_appointments"], errors="coerce"
        ).fillna(0.0)
        zone_metrics["zone_jump_count_num"] = pd.to_numeric(
            zone_metrics["zone_jump_count"], errors="coerce"
        ).fillna(0.0)
        zone_metrics["weighted_zone_jump"] = (
            zone_metrics["zone_jump_count_num"] * zone_metrics["assigned_appointments_num"]
        )
        zone_metrics["two_zone_plus_appointments"] = zone_metrics["assigned_appointments_num"].where(
            zone_metrics["zone_jump_count_num"] >= 2,
            0.0,
        )
        grouped_zone_metrics = zone_metrics.groupby("tech_id", as_index=False).agg(
            total_appointments=("assigned_appointments_num", "sum"),
            weighted_zone_jump=("weighted_zone_jump", "sum"),
            two_zone_plus_appointments=("two_zone_plus_appointments", "sum"),
        )
        for _, metric_row in grouped_zone_metrics.iterrows():
            tech_id = str(metric_row["tech_id"])
            total_appointments = max(
                float(pd.to_numeric(metric_row["total_appointments"], errors="coerce") or 0.0),
                1.0,
            )
            zone_jump_by_tech[tech_id] = (
                float(pd.to_numeric(metric_row["weighted_zone_jump"], errors="coerce") or 0.0)
                / total_appointments
            )
            two_zone_plus_share[tech_id] = (
                float(
                    pd.to_numeric(metric_row["two_zone_plus_appointments"], errors="coerce") or 0.0
                )
                / total_appointments
            )

    rows = []
    for _, tech_row in tech_df.sort_values("tech_name").iterrows():
        tech_id = str(tech_row["tech_id"])
        state_map = state_counts.get(tech_id, {})
        sorted_states = sorted(state_map.items(), key=lambda item: (-item[1], item[0]))
        primary_states = [state for state, _ in sorted_states[:3]]
        metric_row = assignment_group.loc[tech_id] if tech_id in assignment_group.index else None
        covered_skill_labels = []
        if int(pd.to_numeric(tech_row.get("skill_patient"), errors="coerce") or 0):
            covered_skill_labels.append("Patient Sim")
        if int(pd.to_numeric(tech_row.get("skill_ls"), errors="coerce") or 0):
            covered_skill_labels.append("LearningSpace")
        if int(pd.to_numeric(tech_row.get("skill_hps"), errors="coerce") or 0):
            covered_skill_labels.append("HPS")
        rows.append(
            {
                "territory_mode": mode_key,
                "tech_id": tech_id,
                "tech_name": str(tech_row["tech_name"]),
                "base_city": str(tech_row.get("base_city", "")),
                "base_state": str(tech_row.get("base_state", "")),
                "base_airport_iata": str(tech_row.get("base_airport_iata", "")),
                "availability_fte": float(pd.to_numeric(tech_row.get("availability_fte"), errors="coerce") or 0.0),
                "assigned_appointments": float(metric_row["assigned_appointments"]) if metric_row is not None else 0.0,
                "assigned_hours": float(metric_row["assigned_hours"]) if metric_row is not None else 0.0,
                "modeled_objective_cost_usd": float(metric_row["modeled_objective_cost_usd"]) if metric_row is not None else 0.0,
                "raw_travel_cost_usd": float(metric_row["raw_travel_cost_usd"]) if metric_row is not None else 0.0,
                "territory_design_penalty_usd": float(metric_row["territory_design_penalty_usd"]) if metric_row is not None else 0.0,
                "timezone_penalty_usd": float(metric_row["timezone_penalty_usd"]) if metric_row is not None else 0.0,
                "hub_penalty_usd": float(metric_row["hub_penalty_usd"]) if metric_row is not None else 0.0,
                "out_of_region_penalty_usd": float(metric_row["out_of_region_penalty_usd"]) if metric_row is not None else 0.0,
                "distinct_states_owned": int(len(state_map)),
                "primary_states": ";".join(primary_states),
                "owned_states": ";".join([state for state, _ in sorted_states]),
                "avg_zone_jump_weighted": round(float(zone_jump_by_tech.get(tech_id, 0.0)), 3),
                "share_two_zone_plus": round(float(two_zone_plus_share.get(tech_id, 0.0)), 3),
                "covered_skill_labels": ";".join(covered_skill_labels),
            }
        )
    return pd.DataFrame(rows)


def build_gap_summary(
    mode_key: str,
    tech_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    assignments_df: pd.DataFrame,
    appointment_assignments_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build node-level and state-level coverage-gap summaries."""
    if nodes_df.empty:
        empty_node = pd.DataFrame(
            columns=[
                "territory_mode",
                "node_id",
                "state_norm",
                "state_name",
                "skill_class",
                "appointment_count",
                "dominant_owner_tech_id",
                "dominant_owner_tech_name",
                "dominant_owner_base_state",
                "dominant_owner_zone_jump",
                "same_state_qualified_tech_count",
                "same_zone_qualified_tech_count",
                "coverage_gap_flag",
                "training_gap_flag",
                "gap_reason",
            ]
        )
        empty_state = pd.DataFrame(
            columns=[
                "territory_mode",
                "state_norm",
                "state_name",
                "total_appointments",
                "dominant_owner_tech_id",
                "dominant_owner_tech_name",
                "dominant_owner_share",
                "coverage_gap_flag",
                "training_gap_flag",
                "dominant_owner_color_key",
            ]
        )
        return empty_node, empty_state

    assignment_counts = (
        appointment_assignments_df.groupby(["state_norm", "tech_id", "tech_name"])
        .size()
        .reset_index(name="appointment_count")
        if not appointment_assignments_df.empty
        else pd.DataFrame(columns=["state_norm", "tech_id", "tech_name", "appointment_count"])
    )

    node_rows = []
    for _, node_row in nodes_df.iterrows():
        node_id = str(node_row["node_id"])
        state_norm = str(node_row["state_norm"])
        skill_class = str(node_row["skill_class"])
        assigned_subset = assignments_df.loc[assignments_df["node_id"].astype(str) == node_id].copy()
        assigned_subset["assigned_appointments"] = pd.to_numeric(
            assigned_subset["assigned_appointments"], errors="coerce"
        ).fillna(0.0)
        assigned_subset = assigned_subset.sort_values(
            ["assigned_appointments", "tech_name"], ascending=[False, True]
        )

        dominant_owner = assigned_subset.iloc[0] if not assigned_subset.empty else None
        eligible_techs = tech_df[
            tech_df.apply(
                lambda tech_row: STEP08.tech_eligible_for_node(tech_row, node_row, "anywhere"),
                axis=1,
            )
        ].copy()
        same_state_count = int(
            eligible_techs["base_state"].fillna("").astype(str).str.upper().eq(state_norm).sum()
        )
        same_zone_count = int(
            pd.to_numeric(eligible_techs["base_operational_zone_rank"], errors="coerce")
            .fillna(np.nan)
            .eq(pd.to_numeric(node_row.get("node_operational_zone_rank"), errors="coerce"))
            .sum()
        )
        dominant_zone_jump = (
            int(pd.to_numeric(dominant_owner.get("zone_jump_count"), errors="coerce"))
            if dominant_owner is not None and not pd.isna(pd.to_numeric(dominant_owner.get("zone_jump_count"), errors="coerce"))
            else None
        )
        reasons = []
        coverage_gap = False
        training_gap = False
        if same_zone_count == 0:
            coverage_gap = True
            reasons.append("no same-zone qualified owner")
        if same_state_count == 0:
            reasons.append("no same-state qualified owner")
        if skill_class in {"ls", "hps"} and same_state_count == 0:
            training_gap = True
        if dominant_zone_jump is not None and dominant_zone_jump >= 2:
            reasons.append("assigned owner is 2+ zones away")

        node_rows.append(
            {
                "territory_mode": mode_key,
                "node_id": node_id,
                "state_norm": state_norm,
                "state_name": STATE_ABBR_TO_NAME.get(state_norm, state_norm),
                "skill_class": skill_class,
                "appointment_count": float(pd.to_numeric(node_row["appointment_count"], errors="coerce") or 0.0),
                "dominant_owner_tech_id": str(dominant_owner["tech_id"]) if dominant_owner is not None else "",
                "dominant_owner_tech_name": str(dominant_owner["tech_name"]) if dominant_owner is not None else "",
                "dominant_owner_base_state": str(dominant_owner["base_state"]) if dominant_owner is not None else "",
                "dominant_owner_zone_jump": dominant_zone_jump,
                "same_state_qualified_tech_count": same_state_count,
                "same_zone_qualified_tech_count": same_zone_count,
                "coverage_gap_flag": coverage_gap,
                "training_gap_flag": training_gap,
                "gap_reason": "; ".join(reasons),
            }
        )

    node_gap_df = pd.DataFrame(node_rows)
    state_rows = []
    for state_norm, state_group in node_gap_df.groupby("state_norm"):
        owner_counts = assignment_counts.loc[
            assignment_counts["state_norm"].astype(str) == str(state_norm)
        ].sort_values(["appointment_count", "tech_name"], ascending=[False, True])
        dominant_owner = owner_counts.iloc[0] if not owner_counts.empty else None
        total_assigned = float(owner_counts["appointment_count"].sum()) if not owner_counts.empty else 0.0
        dominant_share = (
            float(dominant_owner["appointment_count"]) / total_assigned
            if dominant_owner is not None and total_assigned > 0
            else np.nan
        )
        dominant_owner_id = str(dominant_owner["tech_id"]) if dominant_owner is not None else ""
        state_rows.append(
            {
                "territory_mode": mode_key,
                "state_norm": str(state_norm),
                "state_name": STATE_ABBR_TO_NAME.get(str(state_norm), str(state_norm)),
                "total_appointments": float(pd.to_numeric(state_group["appointment_count"], errors="coerce").sum()),
                "dominant_owner_tech_id": dominant_owner_id,
                "dominant_owner_tech_name": str(dominant_owner["tech_name"]) if dominant_owner is not None else "",
                "dominant_owner_share": dominant_share,
                "coverage_gap_flag": bool(state_group["coverage_gap_flag"].any()),
                "training_gap_flag": bool(state_group["training_gap_flag"].any()),
                "gap_reason": "; ".join(sorted({reason for reason in state_group["gap_reason"] if str(reason).strip()})),
                "dominant_owner_color_key": dominant_owner_id,
            }
        )
    return node_gap_df, pd.DataFrame(state_rows)


def mode_payload(
    mode_spec: dict,
    demand_meta: dict,
    tech_df: pd.DataFrame,
    tech_summary_df: pd.DataFrame,
    node_gap_df: pd.DataFrame,
    result_summary: dict,
) -> dict:
    """Build one mode payload for the map side panel."""
    active_tech_count = int((tech_summary_df["assigned_appointments"] > 0).sum()) if not tech_summary_df.empty else 0
    return {
        "key": str(mode_spec["key"]),
        "label": str(mode_spec["label"]),
        "description": str(mode_spec["description"]),
        "demand_year": int(demand_meta["demand_year"]),
        "appointments_plotted": int(demand_meta["appointments_in_mode"]),
        "states_in_mode": int(demand_meta["states_in_mode"]),
        "fixed_roster_tech_count": int(len(tech_df)),
        "active_owner_tech_count": active_tech_count,
        "excluded_florida_appointments": int(demand_meta["appointments_removed_excluded_states"]),
        "modeled_objective_cost_usd": float(result_summary.get("modeled_total_cost_usd", 0.0)),
        "raw_travel_cost_usd": float(tech_summary_df["raw_travel_cost_usd"].sum()) if not tech_summary_df.empty else 0.0,
        "territory_design_penalty_usd": float(tech_summary_df["territory_design_penalty_usd"].sum()) if not tech_summary_df.empty else 0.0,
        "timezone_penalty_usd": float(result_summary.get("timezone_penalty_usd", 0.0)),
        "hub_penalty_usd": float(result_summary.get("hub_penalty_usd", 0.0)),
        "unmet_appointments": float(result_summary.get("unmet_appointments", 0.0)),
        "coverage_gap_count": int(node_gap_df["coverage_gap_flag"].sum()) if not node_gap_df.empty else 0,
        "training_gap_count": int(node_gap_df["training_gap_flag"].sum()) if not node_gap_df.empty else 0,
    }


def empty_candidates_frame() -> pd.DataFrame:
    """Return an empty candidates frame acceptable to the shared solver."""
    return pd.DataFrame(
        columns=[
            "candidate_id",
            "candidate_type",
            "city",
            "state",
            "airport_iata",
            "hub_tier",
            "operational_zone_rank",
            "operational_zone_label",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize fixed current-roster territories.")
    parser.add_argument(
        "--output-dir",
        default=config.OPTIMIZATION_DIR,
        help="Base optimization directory containing steps 06 and 11 outputs.",
    )
    parser.add_argument(
        "--time-limit-sec",
        type=int,
        default=getattr(config, "OPTIMIZE_TERRITORIES_TIME_LIMIT_SEC", 120),
        help="MILP time limit per territory mode.",
    )
    args = parser.parse_args()

    root_dir = Path(args.output_dir)
    out_dir = optimize_territories_dir(root_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = require_inputs(root_dir)
    tech_df, excluded_names = filter_roster(inputs["tech"])
    tech_df = tech_df.reset_index(drop=True)

    mode_summaries: dict[str, dict] = {}
    all_assignments: list[pd.DataFrame] = []
    all_appointment_assignments: list[pd.DataFrame] = []
    all_tech_summaries: list[pd.DataFrame] = []
    all_node_gaps: list[pd.DataFrame] = []
    all_state_gaps: list[pd.DataFrame] = []

    full_cost_df = inputs["full_cost"]
    for mode_spec in available_mode_specs():
        mode_key = str(mode_spec["key"])
        print(f"Optimizing fixed territories for mode '{mode_key}'...")
        filtered_demand, demand_meta = filter_2025_demand(inputs["demand"], mode_key)
        nodes = STEP08.build_demand_nodes(filtered_demand)

        if filtered_demand.empty or nodes.empty:
            mode_summaries[mode_key] = {
                "summary": {
                    "scenario_hires": 0,
                    "modeled_total_cost_usd": 0.0,
                    "timezone_penalty_usd": 0.0,
                    "hub_penalty_usd": 0.0,
                    "unmet_appointments": 0.0,
                },
                "panel": mode_payload(
                    mode_spec,
                    demand_meta,
                    tech_df,
                    pd.DataFrame(),
                    pd.DataFrame(),
                    {},
                ),
            }
            continue

        original_lookup, adjusted_lookup = build_adjusted_cost_lookup(full_cost_df, tech_df, nodes)
        solve_result = STEP08.solve_scenario(
            hire_count=0,
            tech=tech_df.copy(),
            nodes=nodes.copy(),
            candidates=empty_candidates_frame(),
            full_cost_lookup=adjusted_lookup,
            contractor_scope="anywhere",
            target_utilization=float(
                getattr(config, "OPTIMIZE_TERRITORIES_TARGET_UTILIZATION", 0.85)
            ),
            out_of_region_penalty=float(
                getattr(config, "OPTIMIZE_TERRITORIES_OUT_OF_REGION_PENALTY_USD", 0.0)
            ),
            unmet_penalty=float(
                getattr(config, "OPTIMIZE_TERRITORIES_UNMET_PENALTY_USD", config.DEFAULT_UNMET_PENALTY_USD)
            ),
            annual_hire_cost_usd=0.0,
            max_hires_per_base=1,
            time_limit_sec=int(args.time_limit_sec),
            blank_slate=False,
            optimized_max_appointments_per_person=None,
        )

        assignments = enrich_assignment_costs(solve_result["existing_assignments"], original_lookup)
        assignments["territory_mode"] = mode_key
        appointment_assignments = resolve_appointment_assignments(filtered_demand, assignments, tech_df)
        tech_summary = build_tech_summary(mode_key, tech_df, assignments, appointment_assignments)
        node_gap_df, state_gap_df = build_gap_summary(
            mode_key,
            tech_df,
            nodes,
            assignments,
            appointment_assignments,
        )
        panel_summary = mode_payload(
            mode_spec,
            demand_meta,
            tech_df,
            tech_summary,
            node_gap_df,
            solve_result["summary"],
        )

        all_assignments.append(assignments)
        all_appointment_assignments.append(appointment_assignments)
        all_tech_summaries.append(tech_summary)
        all_node_gaps.append(node_gap_df)
        all_state_gaps.append(state_gap_df)
        mode_summaries[mode_key] = {
            "summary": solve_result["summary"],
            "panel": panel_summary,
        }

    assignments_df = (
        pd.concat(all_assignments, ignore_index=True)
        if all_assignments
        else pd.DataFrame()
    )
    appointment_assignments_df = (
        pd.concat(all_appointment_assignments, ignore_index=True)
        if all_appointment_assignments
        else pd.DataFrame()
    )
    tech_summary_df = (
        pd.concat(all_tech_summaries, ignore_index=True)
        if all_tech_summaries
        else pd.DataFrame()
    )
    node_gap_df = (
        pd.concat(all_node_gaps, ignore_index=True)
        if all_node_gaps
        else pd.DataFrame()
    )
    state_gap_df = (
        pd.concat(all_state_gaps, ignore_index=True)
        if all_state_gaps
        else pd.DataFrame()
    )

    assumptions = {
        "mode_kind": "fixed_current_roster_territory_optimization",
        "demand_year": int(getattr(config, "OPTIMIZE_TERRITORIES_YEAR", 2025)),
        "excluded_tech_names": excluded_names,
        "excluded_state_codes": sorted(
            str(state).strip().upper()
            for state in getattr(config, "OPTIMIZE_TERRITORIES_EXCLUDED_STATES", set())
        ),
        "fixed_base_note": (
            "Technician bases stay fixed to the current tech_master roster; this mode does not place synthetic hires."
        ),
        "territory_design_penalties": {
            "out_of_region_penalty_usd": float(
                getattr(config, "OPTIMIZE_TERRITORIES_OUT_OF_REGION_PENALTY_USD", 0.0)
            ),
            "one_zone_extra_penalty_usd": float(
                getattr(config, "OPTIMIZE_TERRITORIES_ONE_ZONE_EXTRA_PENALTY_USD", 0.0)
            ),
            "two_zone_extra_penalty_usd": float(
                getattr(config, "OPTIMIZE_TERRITORIES_TWO_ZONE_EXTRA_PENALTY_USD", 0.0)
            ),
            "overnight_trip_penalty_usd": float(
                getattr(config, "OPTIMIZE_TERRITORIES_OVERNIGHT_TRIP_PENALTY_USD", 0.0)
            ),
            "fly_trip_penalty_usd": float(
                getattr(config, "OPTIMIZE_TERRITORIES_FLY_TRIP_PENALTY_USD", 0.0)
            ),
        },
        "roster_note": (
            "Contractors, James Sanchez, Elier Martin, Damion Lyn, and Hakim Mouazer are excluded from this mode."
        ),
        "florida_note": (
            "Florida demand rows are intentionally removed from this mode because the source dataset under-represents Florida for this business question."
        ),
    }

    summary_payload = {
        "available_modes": [str(spec["key"]) for spec in available_mode_specs()],
        "default_mode": str(
            getattr(config, "OPTIMIZE_TERRITORIES_DEFAULT_MODE", "patient_sim")
        ),
        "mode_panels": {
            mode_key: details["panel"] for mode_key, details in mode_summaries.items()
        },
        "assumptions": assumptions,
    }

    assignments_out = out_dir / "territory_assignments.csv"
    appointment_assignments_out = out_dir / "territory_appointment_assignments.csv"
    tech_summary_out = out_dir / "territory_tech_summary.csv"
    node_gap_out = out_dir / "territory_gap_summary.csv"
    state_gap_out = out_dir / "territory_state_summary.csv"
    summary_out = out_dir / "territory_summary.json"

    assignments_df.to_csv(assignments_out, index=False)
    appointment_assignments_df.to_csv(appointment_assignments_out, index=False)
    tech_summary_df.to_csv(tech_summary_out, index=False)
    node_gap_df.to_csv(node_gap_out, index=False)
    state_gap_df.to_csv(state_gap_out, index=False)
    with open(summary_out, "w") as handle:
        json.dump(summary_payload, handle, indent=2)

    print(f"Saved: {assignments_out}")
    print(f"Saved: {appointment_assignments_out}")
    print(f"Saved: {tech_summary_out}")
    print(f"Saved: {node_gap_out}")
    print(f"Saved: {state_gap_out}")
    print(f"Saved: {summary_out}")
    for mode_key, mode_info in summary_payload["mode_panels"].items():
        print(
            f"  {mode_key}: {mode_info['appointments_plotted']} appointments, "
            f"{mode_info['active_owner_tech_count']} active tech owners, "
            f"{mode_info['coverage_gap_count']} gap nodes"
        )


if __name__ == "__main__":
    main()
