"""Step 9: Summarize optimization scenarios and recommendations."""

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


def require_file(path: Path) -> None:
    """Raise clear error if file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


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
    assumptions_path = out_dir / "model_assumptions.json"

    require_file(summary_path)
    require_file(placements_path)
    require_file(util_path)
    require_file(assumptions_path)

    summary = pd.read_csv(summary_path).sort_values("scenario_hires").reset_index(drop=True)
    placements = pd.read_csv(placements_path)
    util = pd.read_csv(util_path)
    with open(assumptions_path, "r") as f:
        assumptions = json.load(f)

    # --- Block A: Compute avg calendar hours per installation ---
    appts_path = Path(config.CLEAN_APPTS_CSV)
    installation_types = ["ISO", "AVS ISO", "AVS"]
    if appts_path.exists():
        appts_df = pd.read_csv(appts_path)
        install_rows = appts_df[appts_df["Service Type"].isin(installation_types)].copy()
        # Use Duration Hours (calendar hours) to match the MILP's assigned_hours unit.
        # Fallback to Duration Days * 24 if Duration Hours is missing (same as Step 06).
        install_hours = pd.to_numeric(install_rows["Duration Hours"], errors="coerce")
        install_hours = install_hours.fillna(install_rows["Duration Days"] * 24)
        install_rows["calendar_hours"] = install_hours
        if len(install_rows) > 0:
            avg_calendar_hours_per_installation = float(install_rows["calendar_hours"].mean())
            avg_duration_days_per_installation = float(install_rows["Duration Days"].mean())
            install_type_breakdown = {}
            for stype in installation_types:
                sub = install_rows[install_rows["Service Type"] == stype]
                if len(sub) > 0:
                    install_type_breakdown[stype] = {
                        "count": int(len(sub)),
                        "avg_calendar_hours": round(float(sub["calendar_hours"].mean()), 2),
                        "avg_duration_days": round(float(sub["Duration Days"].mean()), 2),
                        "share": round(len(sub) / len(install_rows), 4),
                    }
        else:
            avg_calendar_hours_per_installation = float("nan")
            avg_duration_days_per_installation = float("nan")
            install_type_breakdown = {}
    else:
        avg_calendar_hours_per_installation = float("nan")
        avg_duration_days_per_installation = float("nan")
        install_type_breakdown = {}

    base_row = summary.loc[summary["scenario_hires"] == 0]
    if base_row.empty:
        raise ValueError("Scenario summary must include N=0 baseline.")
    base_cost = float(base_row.iloc[0]["economic_total_with_overhead_usd"])

    summary["savings_vs_n0_usd"] = base_cost - summary["economic_total_with_overhead_usd"]
    summary["savings_vs_n0_pct"] = np.where(
        base_cost > 0,
        (summary["savings_vs_n0_usd"] / base_cost) * 100.0,
        0.0,
    )
    summary["marginal_savings_from_prev_usd"] = summary["economic_total_with_overhead_usd"].shift(
        1
    ) - summary["economic_total_with_overhead_usd"]
    summary["marginal_savings_from_prev_usd"] = summary["marginal_savings_from_prev_usd"].fillna(0.0)

    # --- Block B: Capacity freed columns ---
    baseline_existing_hours = float(
        util.loc[util["scenario_hires"] == 0, "assigned_hours"].sum()
    )
    hours_freed_list = []
    for _, row in summary.iterrows():
        n = int(row["scenario_hires"])
        existing_hours_at_n = float(
            util.loc[util["scenario_hires"] == n, "assigned_hours"].sum()
        )
        hours_freed_list.append(baseline_existing_hours - existing_hours_at_n)
    summary["hours_freed_existing_techs"] = hours_freed_list

    summary["theoretical_max_installations"] = np.where(
        np.isnan(avg_calendar_hours_per_installation) | (avg_calendar_hours_per_installation == 0),
        np.nan,
        summary["hours_freed_existing_techs"] / avg_calendar_hours_per_installation,
    )

    # Realistic installation estimate: convert freed calendar hours → days,
    # apply utilization factor, divide by (avg duration days + travel overhead).
    summary["freed_duration_days"] = summary["hours_freed_existing_techs"] / 24.0

    travel_days = config.TRAVEL_DAYS_PER_INSTALLATION
    util_factor = config.FREED_CAPACITY_UTILIZATION_FACTOR
    effective_days_per_install = avg_duration_days_per_installation + travel_days

    summary["realistic_installations_enabled"] = np.where(
        np.isnan(avg_duration_days_per_installation) | (effective_days_per_install == 0),
        np.nan,
        (summary["freed_duration_days"] * util_factor) / effective_days_per_install,
    )

    # --- Block C: Revenue-from-freed-capacity analysis ---
    revenue_scenarios = {
        "conservative": config.REVENUE_PER_INSTALLATION_CONSERVATIVE_USD,
        "moderate": config.REVENUE_PER_INSTALLATION_MODERATE_USD,
        "aggressive": config.REVENUE_PER_INSTALLATION_AGGRESSIVE_USD,
    }
    service_contract_annual = config.AVG_ANNUAL_SERVICE_CONTRACT_USD

    for label, rev_per_install in revenue_scenarios.items():
        # Net cost increase vs N=0 (positive = costs more to hire)
        col_net_cost = f"net_cost_increase_{label}_usd"
        summary[col_net_cost] = summary["economic_total_with_overhead_usd"] - base_cost

        # Installation revenue (realistic installs × revenue per install)
        col_install_rev = f"installation_revenue_{label}_usd"
        summary[col_install_rev] = summary["realistic_installations_enabled"] * rev_per_install

        # Service contract revenue (realistic installs × annual contract)
        col_svc_rev = f"service_contract_revenue_{label}_usd"
        summary[col_svc_rev] = summary["realistic_installations_enabled"] * service_contract_annual

        # Total revenue enabled
        col_total_rev = f"total_revenue_enabled_{label}_usd"
        summary[col_total_rev] = summary[col_install_rev] + summary[col_svc_rev]

        # Net economic value = total revenue - net cost increase
        col_net_value = f"net_economic_value_{label}_usd"
        summary[col_net_value] = summary[col_total_rev] - summary[col_net_cost]

        # ROI = net economic value / net cost increase (only where cost > 0)
        col_roi = f"roi_{label}_pct"
        summary[col_roi] = np.where(
            summary[col_net_cost] > 0,
            (summary[col_net_value] / summary[col_net_cost]) * 100.0,
            np.nan,
        )

        # Break-even installations for this revenue scenario
        col_be = f"break_even_installations_{label}"
        rev_plus_svc = rev_per_install + service_contract_annual
        summary[col_be] = np.where(
            summary[col_net_cost] > 0,
            summary[col_net_cost] / rev_plus_svc,
            0.0,
        )

    # Keep a unified break_even column using the moderate scenario
    summary["break_even_installations"] = summary["break_even_installations_moderate"]

    if "solver_proven_optimal" in summary.columns:
        # Use pd.to_numeric to handle string "0"/"1" values correctly before bool conversion.
        proven_mask = pd.to_numeric(summary["solver_proven_optimal"], errors="coerce").eq(1)
        proven = summary[proven_mask].copy()
    else:
        proven = pd.DataFrame()

    if not proven.empty:
        selection_mode = "proven_optimal_only"
        best_idx = proven["economic_total_with_overhead_usd"].idxmin()
        best_row = proven.loc[best_idx]
    else:
        selection_mode = "all_scenarios_no_proven_optimal"
        best_idx = summary["economic_total_with_overhead_usd"].idxmin()
        best_row = summary.loc[best_idx]
    best_hires = int(best_row["scenario_hires"])

    best_placements = placements[placements["scenario_hires"] == best_hires].copy()
    if best_placements.empty:
        recommended = pd.DataFrame(
            columns=[
                "scenario_hires",
                "candidate_id",
                "city",
                "state",
                "airport_iata",
                "hires_allocated",
                "assigned_appointments",
                "assigned_hours",
                "share_of_total_newhire_hours",
            ]
        )
    else:
        total_new_hours = float(best_placements["assigned_hours"].sum())
        best_placements["share_of_total_newhire_hours"] = np.where(
            total_new_hours > 0,
            best_placements["assigned_hours"] / total_new_hours,
            0.0,
        )
        recommended = best_placements.sort_values(
            ["hires_allocated", "assigned_hours"], ascending=[False, False]
        )

    util_best = util[util["scenario_hires"] == best_hires].copy()
    util_metrics = {
        "mean_utilization": float(util_best["utilization"].mean()) if not util_best.empty else np.nan,
        "max_utilization": float(util_best["utilization"].max()) if not util_best.empty else np.nan,
        "num_over_95pct": int((util_best["utilization"] > 0.95).sum()) if not util_best.empty else 0,
    }

    full_cost_model_active = bool(assumptions.get("full_cost_model", False))
    report = {
        "best_scenario_hires": best_hires,
        "selection_mode": selection_mode,
        "full_cost_model_active": full_cost_model_active,
        "best_total_cost_with_overhead_usd": float(best_row["economic_total_with_overhead_usd"]),
        "baseline_n0_cost_with_overhead_usd": base_cost,
        "best_savings_vs_n0_usd": float(base_cost - best_row["economic_total_with_overhead_usd"]),
        "best_savings_vs_n0_pct": float(
            (base_cost - best_row["economic_total_with_overhead_usd"]) / base_cost * 100.0
        )
        if base_cost > 0
        else 0.0,
        "utilization_metrics_best_scenario": util_metrics,
        "assumptions": assumptions,
        "avg_calendar_hours_per_installation": round(avg_calendar_hours_per_installation, 2)
        if not np.isnan(avg_calendar_hours_per_installation)
        else None,
        "avg_duration_days_per_installation": round(avg_duration_days_per_installation, 2)
        if not np.isnan(avg_duration_days_per_installation)
        else None,
        "travel_days_per_installation": travel_days,
        "freed_capacity_utilization_factor": util_factor,
        "installation_type_breakdown": install_type_breakdown,
        "revenue_scenarios": {
            "conservative_per_install_usd": config.REVENUE_PER_INSTALLATION_CONSERVATIVE_USD,
            "moderate_per_install_usd": config.REVENUE_PER_INSTALLATION_MODERATE_USD,
            "aggressive_per_install_usd": config.REVENUE_PER_INSTALLATION_AGGRESSIVE_USD,
        },
        "avg_annual_service_contract_usd": service_contract_annual,
        "capacity_freed_all_scenarios": [
            {
                "scenario_hires": int(r["scenario_hires"]),
                "hours_freed_existing_techs": round(float(r["hours_freed_existing_techs"]), 2),
                "freed_duration_days": round(float(r["freed_duration_days"]), 2),
                "theoretical_max_installations": round(float(r["theoretical_max_installations"]), 1)
                if not np.isnan(r["theoretical_max_installations"])
                else None,
                "realistic_installations_enabled": round(float(r["realistic_installations_enabled"]), 1)
                if not np.isnan(r["realistic_installations_enabled"])
                else None,
                "break_even_installations": round(float(r["break_even_installations"]), 1)
                if not np.isnan(r["break_even_installations"])
                else None,
                "revenue_analysis": {
                    lbl: {
                        "installation_revenue_usd": round(float(r[f"installation_revenue_{lbl}_usd"]), 2),
                        "service_contract_revenue_usd": round(float(r[f"service_contract_revenue_{lbl}_usd"]), 2),
                        "total_revenue_enabled_usd": round(float(r[f"total_revenue_enabled_{lbl}_usd"]), 2),
                        "net_cost_increase_usd": round(float(r[f"net_cost_increase_{lbl}_usd"]), 2),
                        "net_economic_value_usd": round(float(r[f"net_economic_value_{lbl}_usd"]), 2),
                        "roi_pct": round(float(r[f"roi_{lbl}_pct"]), 1)
                        if not np.isnan(r[f"roi_{lbl}_pct"])
                        else None,
                        "break_even_installations": round(float(r[f"break_even_installations_{lbl}"]), 1),
                    }
                    for lbl in ["conservative", "moderate", "aggressive"]
                },
            }
            for _, r in summary.iterrows()
        ],
    }

    summary_out = out_dir / "scenario_summary_enhanced.csv"
    recommended_out = out_dir / "recommended_hire_locations.csv"
    report_out = out_dir / "analysis_report.json"
    markdown_out = out_dir / "analysis_report.md"

    summary.to_csv(summary_out, index=False)
    recommended.to_csv(recommended_out, index=False)
    with open(report_out, "w") as f:
        json.dump(report, f, indent=2)

    cost_model_label = (
        "Full cost model active (drive/fly + rental + hotel)"
        if full_cost_model_active
        else "Flight-cost-only model (no rental/hotel)"
    )
    lines = [
        "# Optimization Scenario Analysis",
        "",
        f"- Cost model: **{cost_model_label}**",
        f"- Selection mode: **{selection_mode}**",
        f"- Best scenario: **{best_hires}** new hires",
        f"- Baseline (N=0) cost with overhead: **${base_cost:,.2f}**",
        f"- Best scenario cost with overhead: **${best_row['economic_total_with_overhead_usd']:,.2f}**",
        f"- Savings vs N=0: **${report['best_savings_vs_n0_usd']:,.2f} ({report['best_savings_vs_n0_pct']:.2f}%)**",
        "- Cost totals include annual burdened payroll for incremental new hires.",
        "",
        "## Utilization (Best Scenario)",
        f"- Mean existing-tech utilization: {util_metrics['mean_utilization']:.3f}",
        f"- Max existing-tech utilization: {util_metrics['max_utilization']:.3f}",
        f"- Existing techs above 95% utilization: {util_metrics['num_over_95pct']}",
        "",
        "## Recommended New-Hire Bases (Best Scenario)",
    ]
    if recommended.empty:
        lines.append("- No new-hire placements in best scenario.")
    else:
        for _, row in recommended.iterrows():
            lines.append(
                "- "
                f"{row['city']}, {row['state']} ({row['airport_iata']}): "
                f"hires={int(row['hires_allocated'])}, "
                f"assigned_hours={row['assigned_hours']:.1f}, "
                f"share={row['share_of_total_newhire_hours']:.2%}"
            )

    # --- Block D: Capacity Freed markdown section ---
    lines.append("")
    lines.append("## Capacity Freed by Hiring Scenario")
    lines.append("")
    if not np.isnan(avg_duration_days_per_installation):
        type_counts = ", ".join(
            f"{k}: {v['count']}" for k, v in install_type_breakdown.items()
        )
        lines.append(
            f"- Avg duration days per installation: **{avg_duration_days_per_installation:.2f}** ({type_counts})"
        )
        lines.append(
            f"- Travel overhead per installation: **{travel_days:.1f} day**"
        )
        lines.append(
            f"- Practical utilization factor: **{util_factor:.0%}** (accounts for scheduling gaps, PTO, non-installation work)"
        )
        lines.append(
            f"- Effective days per installation: **{effective_days_per_install:.2f}** ({avg_duration_days_per_installation:.2f} + {travel_days:.1f} travel)"
        )
    else:
        lines.append("- Avg duration days per installation: **N/A** (no installation data found)")
    if not np.isnan(avg_calendar_hours_per_installation):
        lines.append(
            f"- Avg calendar hours per installation: **{avg_calendar_hours_per_installation:.1f}** (theoretical max reference)"
        )
    lines.append("")
    lines.append("| Scenario | Days Freed | Usable Days | Realistic Installs | Theoretical Max | Break-Even (Mod) |")
    lines.append("|----------|----------:|-----------:|-------------------:|----------------:|-----------------:|")
    for _, r in summary.iterrows():
        n = int(r["scenario_hires"])
        df = r["freed_duration_days"]
        ud = df * util_factor
        ri = r["realistic_installations_enabled"]
        tm = r["theoretical_max_installations"]
        be = r["break_even_installations"]
        ri_str = f"{ri:,.1f}" if not np.isnan(ri) else "N/A"
        tm_str = f"{tm:,.0f}" if not np.isnan(tm) else "N/A"
        be_str = f"{be:,.1f}" if not np.isnan(be) else "N/A"
        lines.append(f"| N={n} | {df:,.1f} | {ud:,.1f} | {ri_str} | {tm_str} | {be_str} |")

    # --- Revenue-from-Freed-Capacity Analysis ---
    lines.append("")
    lines.append("## Revenue-from-Freed-Capacity Analysis")
    lines.append("")
    lines.append("> **Framing note:** Below 15% volume reduction, the value of hiring should be")
    lines.append("> understood as capacity for revenue, not cost savings. These estimates represent")
    lines.append("> what freed technician time could enable, not guaranteed revenue.")
    lines.append("")
    lines.append("### Assumptions")
    lines.append(f"- Conservative: ${config.REVENUE_PER_INSTALLATION_CONSERVATIVE_USD:,}/install (small systems)")
    lines.append(f"- Moderate: ${config.REVENUE_PER_INSTALLATION_MODERATE_USD:,}/install (mid-range systems)")
    lines.append(f"- Aggressive: ${config.REVENUE_PER_INSTALLATION_AGGRESSIVE_USD:,}/install (large/HPS systems)")
    lines.append(f"- Annual service contract: ${service_contract_annual:,}/system")
    lines.append("- Revenue is Year 1 MSRP gross, not profit margin")
    lines.append("")
    lines.append("### Revenue Summary by Scenario")
    lines.append("")
    lines.append("| Hires | Realistic Installs | Rev Scenario | Install Rev | Svc Contract Rev | Total Rev | Net Cost Increase | Net Economic Value | ROI | Break-Even Installs |")
    lines.append("|------:|-------------------:|:-------------|------------:|-----------------:|----------:|------------------:|-------------------:|----:|--------------------:|")
    for _, r in summary.iterrows():
        n = int(r["scenario_hires"])
        ri = r["realistic_installations_enabled"]
        ri_str = f"{ri:,.1f}" if not np.isnan(ri) else "N/A"
        for lbl in ["conservative", "moderate", "aggressive"]:
            ir = r[f"installation_revenue_{lbl}_usd"]
            sr = r[f"service_contract_revenue_{lbl}_usd"]
            tr = r[f"total_revenue_enabled_{lbl}_usd"]
            nc = r[f"net_cost_increase_{lbl}_usd"]
            nv = r[f"net_economic_value_{lbl}_usd"]
            roi = r[f"roi_{lbl}_pct"]
            be = r[f"break_even_installations_{lbl}"]
            roi_str = f"{roi:,.0f}%" if not np.isnan(roi) else "N/A"
            lines.append(
                f"| {n} | {ri_str} | {lbl} | ${ir:,.0f} | ${sr:,.0f} | ${tr:,.0f} | ${nc:,.0f} | ${nv:,.0f} | {roi_str} | {be:,.1f} |"
            )

    lines.append("")
    lines.append("### Caveats")
    lines.append("1. Revenue figures represent **capacity enabled**, not guaranteed bookings — actual revenue depends on sales pipeline and market demand.")
    lines.append("2. Revenue is **MSRP-based gross revenue**, not profit margin or P&L impact.")
    lines.append("3. Service contract revenue assumes each new installation generates an annual contract.")
    lines.append("4. Estimates are **Year 1 only** — multi-year NPV would require discount rate assumptions.")
    lines.append("5. The MILP optimizer recommendation (N=0) is unchanged — this analysis is supplementary.")

    with open(markdown_out, "w") as f:
        f.write("\n".join(lines))

    print(f"Saved: {summary_out}")
    print(f"Saved: {recommended_out}")
    print(f"Saved: {report_out}")
    print(f"Saved: {markdown_out}")
    print("\nTop-level report:")
    print(json.dumps(report, indent=2))
    print("Step 9 complete.")

    # --- Block E: Console capacity summary ---
    print("\nCapacity Freed by Hiring Scenario:")
    print(f"  Avg duration days per installation: {avg_duration_days_per_installation:.2f}")
    print(f"  Travel overhead: {travel_days:.1f} day | Utilization factor: {util_factor:.0%}")
    print(f"  Effective days per installation: {effective_days_per_install:.2f}")
    print(f"  {'Scenario':<10} {'Days Freed':>11} {'Usable Days':>12} {'Realistic':>10} {'Theoretical':>12} {'Break-Even':>11}")
    print(f"  {'-'*10} {'-'*11} {'-'*12} {'-'*10} {'-'*12} {'-'*11}")
    for _, r in summary.iterrows():
        n = int(r["scenario_hires"])
        df = r["freed_duration_days"]
        ud = df * util_factor
        ri = r["realistic_installations_enabled"]
        tm = r["theoretical_max_installations"]
        be = r["break_even_installations"]
        ri_str = f"{ri:,.1f}" if not np.isnan(ri) else "N/A"
        tm_str = f"{tm:,.0f}" if not np.isnan(tm) else "N/A"
        be_str = f"{be:,.1f}" if not np.isnan(be) else "N/A"
        print(f"  N={n:<7} {df:>11,.1f} {ud:>12,.1f} {ri_str:>10} {tm_str:>12} {be_str:>11}")

    # --- Block F: Console revenue summary ---
    print(f"\nRevenue-from-Freed-Capacity Analysis:")
    print(f"  Revenue per install: Conservative=${config.REVENUE_PER_INSTALLATION_CONSERVATIVE_USD//1000:.0f}K | Moderate=${config.REVENUE_PER_INSTALLATION_MODERATE_USD//1000:.0f}K | Aggressive=${config.REVENUE_PER_INSTALLATION_AGGRESSIVE_USD//1000:.0f}K")
    print(f"  Annual service contract: ${service_contract_annual:,}/system")
    print()
    print(f"  {'Scenario':<10} {'Installs':>9} {'Net Cost Incr':>14}    {'Conservative':>14} {'Moderate':>14} {'Aggressive':>14}    {'BE (Mod)':>9}")
    print(f"  {'':<10} {'':>9} {'':>14}    {'--- Net Economic Value ---':^44}    {'':>9}")
    print(f"  {'-'*10} {'-'*9} {'-'*14}    {'-'*14} {'-'*14} {'-'*14}    {'-'*9}")
    for _, r in summary.iterrows():
        n = int(r["scenario_hires"])
        ri = r["realistic_installations_enabled"]
        ri_str = f"{ri:,.1f}" if not np.isnan(ri) else "N/A"
        nc = r["net_cost_increase_moderate_usd"]
        nv_c = r["net_economic_value_conservative_usd"]
        nv_m = r["net_economic_value_moderate_usd"]
        nv_a = r["net_economic_value_aggressive_usd"]
        be = r["break_even_installations"]
        be_str = f"{be:,.1f}" if not np.isnan(be) else "N/A"
        print(f"  N={n:<7} {ri_str:>9} {'${:>,.0f}'.format(nc):>14}    {'${:>,.0f}'.format(nv_c):>14} {'${:>,.0f}'.format(nv_m):>14} {'${:>,.0f}'.format(nv_a):>14}    {be_str:>9}")


if __name__ == "__main__":
    main()
