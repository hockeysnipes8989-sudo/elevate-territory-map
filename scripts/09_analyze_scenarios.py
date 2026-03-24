"""Step 9: Summarize optimization scenarios and patient-sim install upside."""

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
from install_mix_model import build_patient_sim_install_model


def require_file(path: Path) -> None:
    """Raise clear error if file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def df_records(df: pd.DataFrame) -> list[dict]:
    """Convert DataFrame rows to JSON-safe dicts."""
    if df.empty:
        return []
    clean = df.copy()
    clean = clean.replace({np.nan: None})
    for col in clean.columns:
        if np.issubdtype(clean[col].dtype, np.datetime64):
            clean[col] = clean[col].apply(lambda v: None if pd.isna(v) else str(v))
    return clean.to_dict(orient="records")


def scalar_or_none(value: object) -> object:
    """Convert pandas/numpy scalars into JSON-safe Python values."""
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return str(value)
    if isinstance(value, (np.floating, float)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def json_safe(value: object) -> object:
    """Recursively replace NaN/NaT with None before writing JSON."""
    if isinstance(value, dict):
        return {key: json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return scalar_or_none(value)


def markdown_table(df: pd.DataFrame, columns: list[str], headers: list[str]) -> list[str]:
    """Render a simple markdown table from a DataFrame."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for _, row in df[columns].iterrows():
        rendered = []
        for value in row:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                rendered.append("-")
            elif isinstance(value, float):
                rendered.append(f"{value:,.2f}")
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return lines


def apply_diminishing_returns(
    linear_installs: pd.Series,
    alpha: float,
    ceiling: float | None,
    reference_linear: float | None = None,
) -> pd.Series:
    """Apply two-stage diminishing returns to linear install estimates.

    Stage 1 — Power-law friction:
        adjusted = a * linear^alpha
        where a is calibrated so that the smallest non-zero linear value
        maps to itself (preserving the N=1 baseline).

    Stage 2 — Annual ceiling:
        final = min(adjusted, ceiling)

    Parameters
    ----------
    linear_installs : pd.Series
        Raw linear install-unit estimates per scenario.
    alpha : float
        Power-law exponent (0 < alpha <= 1). alpha=1.0 disables Stage 1.
    ceiling : float or None
        Maximum annual installs. None or 0 disables Stage 2.
    reference_linear : float or None
        The linear value to use as calibration anchor. If None, uses the
        smallest positive value in the series.

    Returns
    -------
    pd.Series
        Adjusted install estimates with diminishing returns applied.
    """
    if alpha >= 1.0 and (ceiling is None or ceiling <= 0):
        return linear_installs.copy()

    result = linear_installs.copy()

    # Stage 1: power-law
    if alpha < 1.0:
        positive_mask = result > 0
        if reference_linear is None or reference_linear <= 0:
            positives = result[positive_mask]
            reference_linear = float(positives.min()) if not positives.empty else 1.0
        # a = ref^(1 - alpha), so that a * ref^alpha = ref
        a_coeff = reference_linear ** (1.0 - alpha)
        result[positive_mask] = a_coeff * result[positive_mask] ** alpha

    # Stage 2: ceiling
    if ceiling is not None and ceiling > 0:
        result = result.clip(upper=ceiling)

    return result


def load_data_span_years(out_dir: Path) -> float:
    """Load optimization data span years."""
    input_summary_path = out_dir / "optimization_input_summary.json"
    if input_summary_path.exists():
        with open(input_summary_path) as f:
            input_summary = json.load(f)
        return float(input_summary.get("data_span_years", 1.0))
    return 1.0


def build_active_family_table(
    forward_mix: pd.DataFrame,
    family_economics: pd.DataFrame,
) -> pd.DataFrame:
    """Join active forward mix rows to per-family economics for weighted math."""
    if forward_mix.empty or family_economics.empty:
        return pd.DataFrame()

    active_mix = forward_mix[forward_mix["forward_share"] > 0].copy()
    if active_mix.empty:
        return pd.DataFrame()

    economics_cols = [
        "family",
        "install_revenue_usd",
        "install_margin",
        "install_calendar_days_used",
        "install_calendar_days_source",
        "revenue_source_note",
    ]
    return active_mix.merge(
        family_economics[economics_cols],
        on="family",
        how="left",
        validate="one_to_one",
    )


def build_scenario_family_breakdown(
    summary: pd.DataFrame,
    forward_mix: pd.DataFrame,
    family_economics: pd.DataFrame,
) -> pd.DataFrame:
    """Allocate enabled installs, revenue, and profit by family per scenario."""
    family_table = build_active_family_table(forward_mix, family_economics)
    if family_table.empty:
        return pd.DataFrame(
            columns=[
                "scenario_hires",
                "family",
                "forward_share",
                "install_units_enabled",
                "install_revenue_enabled_usd",
                "install_profit_enabled_usd",
                "install_calendar_days_used",
                "install_revenue_usd",
                "install_margin",
            ]
        )

    records: list[dict] = []
    for _, scenario_row in summary.iterrows():
        installs_enabled_raw = pd.to_numeric(
            scenario_row.get("install_units_enabled"), errors="coerce"
        )
        if pd.isna(installs_enabled_raw):
            continue
        installs_enabled = float(installs_enabled_raw)
        scenario_hires = int(scenario_row["scenario_hires"])
        for _, family_row in family_table.iterrows():
            family = family_row["family"]
            share = float(family_row["forward_share"] or 0.0)
            revenue_per_install = float(family_row["install_revenue_usd"])
            margin = float(family_row["install_margin"])
            calendar_days = float(family_row["install_calendar_days_used"])
            family_installs = installs_enabled * share
            records.append(
                {
                    "scenario_hires": scenario_hires,
                    "family": family,
                    "forward_share": share,
                    "install_units_enabled": family_installs,
                    "install_revenue_enabled_usd": family_installs * revenue_per_install,
                    "install_profit_enabled_usd": family_installs * revenue_per_install * margin,
                    "install_calendar_days_used": calendar_days,
                    "install_revenue_usd": revenue_per_install,
                    "install_margin": margin,
                }
            )
    return pd.DataFrame.from_records(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze MILP scenario outputs.")
    parser.add_argument(
        "--output-dir",
        default=config.OPTIMIZATION_DIR,
        help="Optimization output directory.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    summary_path = out_dir / "scenario_summary.csv"
    placements_path = out_dir / "scenario_placements.csv"
    util_path = out_dir / "scenario_tech_utilization.csv"
    contractor_usage_path = out_dir / "scenario_contractor_usage.csv"
    assumptions_path = out_dir / "model_assumptions.json"

    require_file(summary_path)
    require_file(placements_path)
    require_file(util_path)
    require_file(assumptions_path)

    summary = pd.read_csv(summary_path).sort_values("scenario_hires").reset_index(drop=True)
    util = pd.read_csv(util_path)
    contractor_usage = (
        pd.read_csv(contractor_usage_path) if contractor_usage_path.exists() else pd.DataFrame()
    )
    with open(assumptions_path, "r") as f:
        assumptions = json.load(f)
    input_summary_path = out_dir / "optimization_input_summary.json"
    input_provenance = {}
    if input_summary_path.exists():
        with open(input_summary_path, "r") as f:
            input_summary = json.load(f)
        input_provenance = {
            "appointments_source": input_summary.get("appointments_source"),
            "appointments_source_kind": input_summary.get("appointments_source_kind"),
            "tech_source": input_summary.get("tech_source"),
            "tech_source_kind": input_summary.get("tech_source_kind"),
            "tech_source_is_cached": bool(input_summary.get("tech_source_is_cached", False)),
            "source_warnings": list(input_summary.get("source_warnings", [])),
        }

    data_span_years = load_data_span_years(out_dir)
    print(f"Data span: {data_span_years:.2f} years — all figures will be annualized")

    install_model = build_patient_sim_install_model(
        appointments_workbook=config.SERVICE_APPTS_DISPATCH,
        demand_appointments_csv=str(out_dir / "demand_appointments.csv"),
    )
    history_rows = install_model.history_rows.copy()
    historical_mix_events = install_model.historical_mix_events.copy()
    historical_mix_units = install_model.historical_mix_units.copy()
    forward_mix = install_model.forward_mix.copy()
    family_economics = install_model.family_economics.copy()

    history_out = out_dir / "patient_sim_install_history_clean.csv"
    historical_events_out = out_dir / "patient_sim_historical_mix_events.csv"
    historical_units_out = out_dir / "patient_sim_historical_mix_units.csv"
    forward_mix_out = out_dir / "patient_sim_forward_mix.csv"
    family_econ_out = out_dir / "patient_sim_family_economics_assumptions.csv"

    history_rows.to_csv(history_out, index=False)
    historical_mix_events.to_csv(historical_events_out, index=False)
    historical_mix_units.to_csv(historical_units_out, index=False)
    forward_mix.to_csv(forward_mix_out, index=False)
    family_economics.to_csv(family_econ_out, index=False)

    active_family_table = build_active_family_table(forward_mix, family_economics)
    if active_family_table.empty:
        weighted_avg_install_calendar_days = float("nan")
        weighted_avg_install_revenue_usd = float("nan")
        weighted_avg_install_profit_per_install_usd = float("nan")
        weighted_avg_install_margin = float("nan")
    else:
        weighted_avg_install_calendar_days = float(
            (active_family_table["forward_share"] * active_family_table["install_calendar_days_used"]).sum()
        )
        weighted_avg_install_revenue_usd = float(
            (active_family_table["forward_share"] * active_family_table["install_revenue_usd"]).sum()
        )
        weighted_avg_install_profit_per_install_usd = float(
            (
                active_family_table["forward_share"]
                * active_family_table["install_revenue_usd"]
                * active_family_table["install_margin"]
            ).sum()
        )
        weighted_avg_install_margin = (
            weighted_avg_install_profit_per_install_usd / weighted_avg_install_revenue_usd
            if weighted_avg_install_revenue_usd > 0
            else float("nan")
        )

    # Annualize from full-period to per-year.
    period_cost_cols = [
        "travel_cost_usd",
        "out_of_region_penalty_usd",
        "timezone_penalty_usd",
        "hub_penalty_usd",
        "hire_cost_usd",
        "unmet_penalty_usd",
        "modeled_total_cost_usd",
        "baseline_canceled_voided_usd",
        "economic_total_with_overhead_usd",
    ]
    for col in period_cost_cols:
        if col in summary.columns:
            summary[col] = summary[col] / data_span_years
    if not contractor_usage.empty and "total_travel_cost_usd" in contractor_usage.columns:
        contractor_usage["total_travel_cost_usd"] = (
            contractor_usage["total_travel_cost_usd"] / data_span_years
        )
    if not contractor_usage.empty and "total_hub_penalty_usd" in contractor_usage.columns:
        contractor_usage["total_hub_penalty_usd"] = (
            contractor_usage["total_hub_penalty_usd"] / data_span_years
        )

    summary["solver_status"] = pd.to_numeric(summary.get("solver_status"), errors="coerce")
    feasible_mask = summary["solver_status"] != -1

    base_row = summary.loc[summary["scenario_hires"] == 0]
    if base_row.empty:
        raise ValueError("Scenario summary must include N=0 baseline.")
    base_cost_raw = pd.to_numeric(
        base_row.iloc[0].get("economic_total_with_overhead_usd"), errors="coerce"
    )
    base_row_feasible = bool(base_row.iloc[0]["solver_status"] != -1)
    base_cost = float(base_cost_raw) if base_row_feasible and not pd.isna(base_cost_raw) else np.nan
    baseline_solver = {
        "scenario_hires": 0,
        "solver_proven_optimal": bool(
            pd.to_numeric(base_row.iloc[0].get("solver_proven_optimal"), errors="coerce") == 1
        ),
        "solver_status": int(
            pd.to_numeric(base_row.iloc[0].get("solver_status"), errors="coerce")
            if not pd.isna(pd.to_numeric(base_row.iloc[0].get("solver_status"), errors="coerce"))
            else 0
        ),
        "solver_mip_gap": scalar_or_none(
            pd.to_numeric(base_row.iloc[0].get("solver_mip_gap"), errors="coerce")
        ),
        "solver_message": str(base_row.iloc[0].get("solver_message", "")).strip(),
    }

    summary["savings_vs_n0_usd"] = np.where(
        feasible_mask & np.isfinite(base_cost),
        base_cost - summary["economic_total_with_overhead_usd"],
        np.nan,
    )
    summary["savings_vs_n0_pct"] = np.where(
        feasible_mask & np.isfinite(base_cost) & (base_cost > 0),
        (summary["savings_vs_n0_usd"] / base_cost) * 100.0,
        np.nan,
    )
    prev_cost = summary["economic_total_with_overhead_usd"].shift(1)
    prev_feasible = feasible_mask.shift(1, fill_value=False)
    summary["marginal_savings_from_prev_usd"] = np.where(
        feasible_mask & prev_feasible,
        prev_cost - summary["economic_total_with_overhead_usd"],
        np.nan,
    )

    baseline_existing_hours = (
        float(util.loc[util["scenario_hires"] == 0, "assigned_hours"].sum())
        if base_row_feasible
        else np.nan
    )
    hours_freed_list = []
    for _, row in summary.iterrows():
        n = int(row["scenario_hires"])
        scenario_feasible = pd.to_numeric(row.get("solver_status"), errors="coerce") != -1
        if not scenario_feasible or not np.isfinite(baseline_existing_hours):
            hours_freed_list.append(np.nan)
            continue
        existing_hours_at_n = float(util.loc[util["scenario_hires"] == n, "assigned_hours"].sum())
        hours_freed_list.append(baseline_existing_hours - existing_hours_at_n)
    summary["hours_freed_existing_techs"] = pd.Series(hours_freed_list, dtype="float64") / data_span_years

    util_factor = float(config.FREED_CAPACITY_UTILIZATION_FACTOR)
    summary["freed_calendar_days_total"] = summary["hours_freed_existing_techs"] / 24.0
    summary["freed_calendar_days_available"] = summary["freed_calendar_days_total"] * util_factor

    if np.isnan(weighted_avg_install_calendar_days) or weighted_avg_install_calendar_days <= 0:
        summary["weighted_avg_install_calendar_days"] = np.nan
        summary["theoretical_max_installations"] = np.nan
        summary["install_units_enabled"] = np.nan
    else:
        summary["weighted_avg_install_calendar_days"] = weighted_avg_install_calendar_days
        summary["theoretical_max_installations"] = (
            summary["freed_calendar_days_total"] / weighted_avg_install_calendar_days
        )
        summary["install_units_enabled"] = (
            summary["freed_calendar_days_available"] / weighted_avg_install_calendar_days
        )

    # Apply diminishing returns to the linear install estimate.
    # Stage 1: power-law friction (coordination, ramp-up, scheduling overhead)
    # Stage 2: annual ceiling (market/pipeline saturation)
    dr_alpha = float(getattr(config, "INSTALL_UPSIDE_DIMINISHING_RETURNS_ALPHA", 1.0))
    dr_ceiling = getattr(config, "INSTALL_UPSIDE_ANNUAL_CEILING", None)
    dr_reference = getattr(config, "INSTALL_UPSIDE_REFERENCE_LINEAR", None)

    summary["linear_install_units_enabled"] = summary["install_units_enabled"].copy()
    summary["install_units_enabled"] = apply_diminishing_returns(
        summary["linear_install_units_enabled"],
        alpha=dr_alpha,
        ceiling=dr_ceiling,
        reference_linear=dr_reference,
    )

    summary["realistic_installations_enabled"] = summary["install_units_enabled"]
    summary["weighted_avg_install_revenue_usd"] = weighted_avg_install_revenue_usd
    summary["weighted_avg_install_margin"] = weighted_avg_install_margin
    summary["weighted_avg_install_profit_per_install_usd"] = weighted_avg_install_profit_per_install_usd
    summary["install_revenue_enabled_usd"] = summary["install_units_enabled"] * weighted_avg_install_revenue_usd
    summary["install_profit_enabled_usd"] = (
        summary["install_units_enabled"] * weighted_avg_install_profit_per_install_usd
    )
    summary["net_cost_increase_usd"] = np.where(
        feasible_mask & np.isfinite(base_cost),
        summary["economic_total_with_overhead_usd"] - base_cost,
        np.nan,
    )
    summary["net_economic_value_install_usd"] = np.where(
        feasible_mask & np.isfinite(summary["install_profit_enabled_usd"]) & np.isfinite(summary["net_cost_increase_usd"]),
        summary["install_profit_enabled_usd"] - summary["net_cost_increase_usd"],
        np.nan,
    )
    summary["roi_install_pct"] = np.where(
        feasible_mask & (summary["net_cost_increase_usd"] > 0),
        (summary["net_economic_value_install_usd"] / summary["net_cost_increase_usd"]) * 100.0,
        np.nan,
    )
    summary["break_even_install_units"] = np.where(
        feasible_mask
        & (summary["net_cost_increase_usd"] > 0)
        & (weighted_avg_install_profit_per_install_usd > 0),
        summary["net_cost_increase_usd"] / weighted_avg_install_profit_per_install_usd,
        np.nan,
    )

    summary.loc[~feasible_mask, [
        "hours_freed_existing_techs",
        "freed_calendar_days_total",
        "freed_calendar_days_available",
        "theoretical_max_installations",
        "linear_install_units_enabled",
        "install_units_enabled",
        "realistic_installations_enabled",
        "install_revenue_enabled_usd",
        "install_profit_enabled_usd",
        "net_cost_increase_usd",
        "net_economic_value_install_usd",
        "roi_install_pct",
        "break_even_install_units",
    ]] = np.nan
    if not np.isfinite(base_cost):
        summary.loc[:, [
            "hours_freed_existing_techs",
            "freed_calendar_days_total",
            "freed_calendar_days_available",
            "theoretical_max_installations",
            "linear_install_units_enabled",
            "install_units_enabled",
            "realistic_installations_enabled",
            "install_revenue_enabled_usd",
            "install_profit_enabled_usd",
            "net_cost_increase_usd",
            "net_economic_value_install_usd",
            "roi_install_pct",
            "break_even_install_units",
            "savings_vs_n0_usd",
            "savings_vs_n0_pct",
        ]] = np.nan

    # Compatibility aliases for existing outputs and map payload.
    summary["gross_revenue_moderate_usd"] = summary["install_revenue_enabled_usd"]
    summary["total_profit_enabled_moderate_usd"] = summary["install_profit_enabled_usd"]
    summary["net_economic_value_moderate_usd"] = summary["net_economic_value_install_usd"]
    summary["break_even_installations_moderate"] = summary["break_even_install_units"]
    summary["roi_moderate_pct"] = summary["roi_install_pct"]
    summary["break_even_installations"] = summary["break_even_install_units"]

    for label in ["conservative", "moderate", "aggressive"]:
        summary[f"net_cost_increase_{label}_usd"] = summary["net_cost_increase_usd"]
        summary[f"installation_profit_{label}_usd"] = summary["install_profit_enabled_usd"]
        summary[f"service_contract_profit_{label}_usd"] = 0.0
        summary[f"total_profit_enabled_{label}_usd"] = summary["install_profit_enabled_usd"]
        summary[f"net_economic_value_{label}_usd"] = summary["net_economic_value_install_usd"]
        summary[f"roi_{label}_pct"] = summary["roi_install_pct"]
        summary[f"break_even_installations_{label}"] = summary["break_even_install_units"]

    scenario_family = build_scenario_family_breakdown(summary, forward_mix, family_economics)
    scenario_family_out = out_dir / "scenario_install_upside_by_family.csv"
    scenario_family.to_csv(scenario_family_out, index=False)

    excluded_counts = (
        history_rows["history_exclusion_reason"]
        .dropna()
        .value_counts()
        .rename_axis("reason")
        .reset_index(name="rows")
    )
    special_tech_constraints = assumptions.get("special_tech_constraints", [])
    scenario_hires_list = [int(v) for v in summary["scenario_hires"].tolist()]
    scenario_labels = ", ".join(f"N={scenario}" for scenario in scenario_hires_list)
    base_cost_label = (
        f"${base_cost:,.2f}" if np.isfinite(base_cost) else "unavailable (N=0 infeasible)"
    )
    utilization_metrics_by_scenario = []
    for scenario_hires in scenario_hires_list:
        util_scenario = util[util["scenario_hires"] == scenario_hires].copy()
        utilization_metrics_by_scenario.append(
            {
                "scenario_hires": scenario_hires,
                "mean_utilization": float(util_scenario["utilization"].mean())
                if not util_scenario.empty
                else np.nan,
                "max_utilization": float(util_scenario["utilization"].max())
                if not util_scenario.empty
                else np.nan,
                "num_over_95pct": int((util_scenario["utilization"] > 0.95).sum())
                if not util_scenario.empty
                else 0,
            }
        )

    report = {
        "data_span_years": data_span_years,
        "annualization_note": f"All figures annualized from {data_span_years:.2f}-year data period",
        "scenario_hires_analyzed": scenario_hires_list,
        "full_cost_model_active": bool(assumptions.get("full_cost_model", False)),
        "input_provenance": input_provenance,
        "baseline_n0_solver": baseline_solver,
        "baseline_n0_cost_with_overhead_usd": base_cost,
        "utilization_metrics_by_scenario": utilization_metrics_by_scenario,
        "assumptions": assumptions,
        "special_tech_constraints": special_tech_constraints,
        "install_model_assumptions": install_model.assumptions,
        "install_model_source": install_model.source_metadata,
        "utilization_metric": {
            "legacy_field_names": [
                "utilization",
                "mean_existing_utilization",
                "max_existing_utilization",
                "scenario_tech_utilization.csv",
            ],
            "workload_basis": (
                "The optimizer uses appointment duration_hours as calendar-window "
                "workload, not pure hands-on labor time."
            ),
            "capacity_basis": (
                "Technician capacity is normalized against that same demand pool "
                "using availability_fte and target_utilization."
            ),
            "interpretation": (
                "These utilization outputs are modeled load ratios / calendar-"
                "based capacity proxies, not payroll-style or weekday-only labor "
                "utilization percentages."
            ),
            "future_note": (
                "A separate operational utilization metric could be added later "
                "if cleaner labor, travel, or timesheet data becomes available."
            ),
        },
        "capacity_model_time_unit": config.PATIENT_SIM_CAPACITY_TIME_UNIT,
        "contractor_usage_by_scenario": df_records(contractor_usage),
        "weighted_avg_install_calendar_days": weighted_avg_install_calendar_days,
        "weighted_avg_install_revenue_usd": weighted_avg_install_revenue_usd,
        "weighted_avg_install_margin": weighted_avg_install_margin,
        "weighted_avg_install_profit_per_install_usd": weighted_avg_install_profit_per_install_usd,
        "historical_mix": {
            "clean_history_rows": int(len(history_rows)),
            "included_history_component_rows": int(history_rows["include_in_history"].sum()),
            "excluded_rows_by_reason": df_records(excluded_counts),
            "events": df_records(historical_mix_events),
            "units": df_records(historical_mix_units),
        },
        "forward_mix": {
            "basis": "unit_equivalents",
            "rows": df_records(forward_mix),
        },
        "family_economics": df_records(family_economics),
        "capacity_freed_all_scenarios": [],
        "diminishing_returns_model": {
            "alpha": dr_alpha,
            "annual_ceiling": dr_ceiling,
            "reference_linear": dr_reference,
            "stage_1_description": (
                "Power-law friction: freed capacity converts to installs at a "
                "decreasing marginal rate (coordination, ramp-up, scheduling overhead)."
            ),
            "stage_2_description": (
                "Annual ceiling: maximum net-new installs the sales pipeline and "
                "market can absorb per year, based on historical install rate plus "
                "reasonable growth headroom."
            ),
        },
        "legacy_tier_alias_mode": (
            "Legacy conservative/moderate/aggressive install columns are compatibility "
            "aliases to the single family-weighted install-only model."
        ),
    }

    for _, scenario_row in summary.iterrows():
        scenario_hires = int(scenario_row["scenario_hires"])
        family_breakdown = scenario_family[scenario_family["scenario_hires"] == scenario_hires].copy()
        report["capacity_freed_all_scenarios"].append(
            {
                "scenario_hires": scenario_hires,
                "hours_freed_existing_techs": float(scenario_row["hours_freed_existing_techs"]),
                "freed_calendar_days_total": float(scenario_row["freed_calendar_days_total"]),
                "freed_calendar_days_available": float(scenario_row["freed_calendar_days_available"]),
                "theoretical_max_installations": (
                    float(scenario_row["theoretical_max_installations"])
                    if not np.isnan(scenario_row["theoretical_max_installations"])
                    else None
                ),
                "linear_install_units_enabled": (
                    float(scenario_row["linear_install_units_enabled"])
                    if not np.isnan(scenario_row["linear_install_units_enabled"])
                    else None
                ),
                "install_units_enabled": (
                    float(scenario_row["install_units_enabled"])
                    if not np.isnan(scenario_row["install_units_enabled"])
                    else None
                ),
                "install_revenue_enabled_usd": float(scenario_row["install_revenue_enabled_usd"]),
                "install_profit_enabled_usd": float(scenario_row["install_profit_enabled_usd"]),
                "timezone_penalty_usd": float(scenario_row.get("timezone_penalty_usd", 0.0)),
                "hub_penalty_usd": float(scenario_row.get("hub_penalty_usd", 0.0)),
                "net_cost_increase_usd": float(scenario_row["net_cost_increase_usd"]),
                "net_economic_value_install_usd": float(scenario_row["net_economic_value_install_usd"]),
                "roi_install_pct": (
                    float(scenario_row["roi_install_pct"])
                    if not np.isnan(scenario_row["roi_install_pct"])
                    else None
                ),
                "break_even_install_units": float(scenario_row["break_even_install_units"]),
                "family_breakdown": df_records(family_breakdown),
            }
        )

    summary_out = out_dir / "scenario_summary_enhanced.csv"
    recommended_out = out_dir / "recommended_hire_locations.csv"
    report_out = out_dir / "analysis_report.json"
    markdown_out = out_dir / "analysis_report.md"

    summary.to_csv(summary_out, index=False)
    removed_recommendation_file = False
    if recommended_out.exists():
        recommended_out.unlink()
        removed_recommendation_file = True
    with open(report_out, "w") as f:
        json.dump(json_safe(report), f, indent=2, allow_nan=False)

    lines = [
        "# Optimization Scenario Analysis",
        "",
        f"- Capacity model time unit: **{config.PATIENT_SIM_CAPACITY_TIME_UNIT}**",
        f"- Scenarios analyzed: **{scenario_labels}**",
        f"- Baseline (N=0) cost with overhead: **{base_cost_label}**",
        f"- Data period: **{data_span_years:.2f} years**",
        f"- Baseline N=0 proven optimal: **{baseline_solver['solver_proven_optimal']}**",
        "",
        "## Capacity Model",
        "",
        (
            "- Utilization framing: **legacy utilization fields are modeled load "
            "ratios under a calendar-window demand framework, not literal "
            "timesheet or Monday-through-Friday labor utilization.**"
        ),
        (
            "- Workload basis: **appointment `duration_hours` are treated as "
            "calendar-window workload, not pure hands-on labor time.**"
        ),
        (
            "- Capacity basis: **technician capacity is normalized against that "
            "same demand pool using `availability_fte` and the target "
            "utilization setting.**"
        ),
        (
            "- Operational zone rule: **standard employees/new hires are free at "
            "0-1 zone jumps, penalized at 2, and blocked at 3+; contractors use "
            "a softer penalty-only rule.**"
        ),
        f"- Freed-capacity utilization factor: **{util_factor:.0%}**",
        f"- Diminishing-returns power-law alpha: **{dr_alpha:.2f}**",
        f"- Diminishing-returns annual ceiling: **{dr_ceiling}**" if dr_ceiling else "- Diminishing-returns annual ceiling: **disabled**",
        f"- Weighted average install calendar days: **{weighted_avg_install_calendar_days:,.2f}**",
        f"- Weighted average install revenue: **${weighted_avg_install_revenue_usd:,.0f}**",
        f"- Weighted average install profit per install: **${weighted_avg_install_profit_per_install_usd:,.0f}**",
        (
            "- File note: **`scenario_tech_utilization.csv` keeps the legacy name "
            "for compatibility, but its values are modeled load ratios.**"
        ),
        "",
        "## Special Technician Constraints",
        "",
    ]
    if not special_tech_constraints:
        lines.append("- No special technician constraints configured.")
    else:
        for item in special_tech_constraints:
            tech_name = str(item.get("tech_name", "Unknown"))
            anchor_site = str(item.get("anchor_site_name", "")).strip() or "Not specified"
            reserved = item.get("anchor_reserved_fte")
            external = item.get("external_field_fte")
            allowed_states = str(item.get("assignment_scope_states", "")).strip()
            notes = str(item.get("anchor_notes", "")).strip()
            parts = [
                f"- **{tech_name}**",
                f"anchor site: **{anchor_site}**",
            ]
            if reserved is not None:
                parts.append(f"reserved duty: **{float(reserved):.0%}**")
            if external is not None:
                parts.append(f"external field capacity: **{float(external):.0%}**")
            if allowed_states:
                parts.append(f"external assignment region: **{allowed_states.replace(';', ' / ')}**")
            if notes:
                parts.append(f"note: {notes}")
            lines.append(", ".join(parts))

    lines.extend(
        [
            "",
        "## Contractor Usage by Scenario",
        "",
        ]
    )
    if contractor_usage.empty:
        lines.append("- No contractor usage rows across the analyzed scenarios.")
    else:
        contractor_usage_md = contractor_usage.sort_values(
            ["scenario_hires", "tech_name"], ascending=[True, True]
        ).copy()
        lines.extend(
            markdown_table(
                contractor_usage_md,
                [
                    "scenario_hires",
                    "tech_name",
                    "assigned_appointments",
                    "assigned_hours",
                    "avg_zone_jump",
                    "share_two_zone_plus",
                    "share_three_zone_plus",
                    "states_served",
                ],
                [
                    "Scenario",
                    "Technician",
                    "Assigned Appointments",
                    "Assigned Hours",
                    "Avg Zone Jump",
                    "Share 2+ Zones",
                    "Share 3+ Zones",
                    "States Served",
                ],
            )
        )

    lines.extend(
        [
            "",
        "## Historical Mix (Events)",
        "",
        ]
    )
    if historical_mix_events.empty:
        lines.append("- No cleaned historical event mix available.")
    else:
        lines.extend(
            markdown_table(
                historical_mix_events,
                ["family", "event_equivalent_count", "event_share"],
                ["Family", "Event Eq. Count", "Event Share"],
            )
        )

    lines.extend(
        [
            "",
            "## Historical Mix (Units)",
            "",
        ]
    )
    if historical_mix_units.empty:
        lines.append("- No cleaned historical unit mix available.")
    else:
        lines.extend(
            markdown_table(
                historical_mix_units,
                ["family", "units_inferred", "unit_share"],
                ["Family", "Units Inferred", "Unit Share"],
            )
        )

    lines.extend(
        [
            "",
            "## Forward Mix",
            "",
        ]
    )
    if forward_mix.empty:
        lines.append("- No forward mix available.")
    else:
        lines.extend(
            markdown_table(
                forward_mix,
                [
                    "family",
                    "historical_units",
                    "forward_units_after_adjustments",
                    "forward_share",
                    "forward_mix_exclusion_reason",
                ],
                [
                    "Family",
                    "Historical Units",
                    "Forward Units",
                    "Forward Share",
                    "Forward Exclusion",
                ],
            )
        )

    lines.extend(
        [
            "",
            "## Family Economics",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            family_economics,
            [
                "family",
                "install_revenue_usd",
                "install_margin",
                "install_calendar_days_used",
                "install_calendar_days_source",
                "revenue_source_note",
            ],
            [
                "Family",
                "Revenue",
                "Margin",
                "Calendar Days Used",
                "Calendar-Day Source",
                "Revenue Source Note",
            ],
        )
    )

    lines.extend(
        [
            "",
            "## Scenario Install Upside",
            "",
        ]
    )
    scenario_summary_md = summary[
        [
            "scenario_hires",
            "freed_calendar_days_available",
            "linear_install_units_enabled",
            "install_units_enabled",
            "install_revenue_enabled_usd",
            "install_profit_enabled_usd",
            "net_cost_increase_usd",
            "net_economic_value_install_usd",
            "break_even_install_units",
        ]
    ].copy()
    scenario_summary_md = scenario_summary_md.rename(
        columns={
            "scenario_hires": "Scenario",
            "freed_calendar_days_available": "Freed Calendar Days Available",
            "linear_install_units_enabled": "Linear Installs (Pre-DR)",
            "install_units_enabled": "Install Units Enabled",
            "install_revenue_enabled_usd": "Install Revenue Enabled",
            "install_profit_enabled_usd": "Install Profit Enabled",
            "net_cost_increase_usd": "Net Cost Increase",
            "net_economic_value_install_usd": "Net Economic Value",
            "break_even_install_units": "Break-Even Install Units",
        }
    )
    lines.extend(
        markdown_table(
            scenario_summary_md,
            list(scenario_summary_md.columns),
            list(scenario_summary_md.columns),
        )
    )

    with open(markdown_out, "w") as f:
        f.write("\n".join(lines))

    print(f"Saved: {summary_out}")
    if removed_recommendation_file:
        print(f"Removed stale recommendation file: {recommended_out}")
    else:
        print(f"Recommendation file not written: {recommended_out}")
    print(f"Saved: {report_out}")
    print(f"Saved: {markdown_out}")
    print(f"Saved: {history_out}")
    print(f"Saved: {historical_events_out}")
    print(f"Saved: {historical_units_out}")
    print(f"Saved: {forward_mix_out}")
    print(f"Saved: {family_econ_out}")
    print(f"Saved: {scenario_family_out}")
    print("\nInstall-only patient-sim upside model:")
    print(f"  Capacity time unit: {config.PATIENT_SIM_CAPACITY_TIME_UNIT}")
    print(f"  Weighted avg install calendar days: {weighted_avg_install_calendar_days:,.2f}")
    print(f"  Weighted avg install revenue: ${weighted_avg_install_revenue_usd:,.0f}")
    print(f"  Weighted avg install profit/install: ${weighted_avg_install_profit_per_install_usd:,.0f}")
    print(f"  HPS excluded from forward mix: {config.PATIENT_SIM_EXCLUDE_HPS_FROM_FUTURE_MIX}")
    print(f"  History source: {install_model.source_metadata['source_kind']}")
    print("Step 9 complete.")


if __name__ == "__main__":
    main()
