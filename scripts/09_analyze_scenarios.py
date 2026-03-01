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
            install_type_breakdown = {}
            for stype in installation_types:
                sub = install_rows[install_rows["Service Type"] == stype]
                if len(sub) > 0:
                    install_type_breakdown[stype] = {
                        "count": int(len(sub)),
                        "avg_calendar_hours": round(float(sub["calendar_hours"].mean()), 2),
                        "share": round(len(sub) / len(install_rows), 4),
                    }
        else:
            avg_calendar_hours_per_installation = float("nan")
            install_type_breakdown = {}
    else:
        avg_calendar_hours_per_installation = float("nan")
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

    summary["potential_installations_enabled"] = np.where(
        np.isnan(avg_calendar_hours_per_installation) | (avg_calendar_hours_per_installation == 0),
        np.nan,
        summary["hours_freed_existing_techs"] / avg_calendar_hours_per_installation,
    )

    rev_per_install = config.DEFAULT_AVG_REVENUE_PER_INSTALLATION_USD
    if rev_per_install is not None and rev_per_install > 0:
        summary["break_even_installations"] = np.where(
            summary["savings_vs_n0_usd"] >= 0,
            0.0,
            (-summary["savings_vs_n0_usd"]) / rev_per_install,
        )
    else:
        summary["break_even_installations"] = np.nan

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
        "installation_type_breakdown": install_type_breakdown,
        "avg_revenue_per_installation_usd": rev_per_install,
        "capacity_freed_all_scenarios": [
            {
                "scenario_hires": int(r["scenario_hires"]),
                "hours_freed_existing_techs": round(float(r["hours_freed_existing_techs"]), 2),
                "potential_installations_enabled": round(float(r["potential_installations_enabled"]), 1)
                if not np.isnan(r["potential_installations_enabled"])
                else None,
                "break_even_installations": round(float(r["break_even_installations"]), 1)
                if not np.isnan(r["break_even_installations"])
                else None,
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
    if not np.isnan(avg_calendar_hours_per_installation):
        type_counts = ", ".join(
            f"{k}: {v['count']}" for k, v in install_type_breakdown.items()
        )
        lines.append(
            f"- Avg calendar hours per installation: **{avg_calendar_hours_per_installation:.1f}** ({type_counts})"
        )
    else:
        lines.append("- Avg calendar hours per installation: **N/A** (no installation data found)")
    if rev_per_install is not None:
        lines.append(f"- Avg revenue per installation: **${rev_per_install:,.0f}**")
    else:
        lines.append("- Avg revenue per installation: **not configured** (set DEFAULT_AVG_REVENUE_PER_INSTALLATION_USD in config.py)")
    lines.append("")
    lines.append("| Scenario | Hours Freed | Potential Installs | Break-Even Installs |")
    lines.append("|----------|------------:|-------------------:|--------------------:|")
    for _, r in summary.iterrows():
        n = int(r["scenario_hires"])
        hf = r["hours_freed_existing_techs"]
        pi = r["potential_installations_enabled"]
        be = r["break_even_installations"]
        pi_str = f"{pi:,.0f}" if not np.isnan(pi) else "N/A"
        be_str = f"{be:,.1f}" if not np.isnan(be) else "N/A"
        lines.append(f"| N={n} | {hf:,.0f} | {pi_str} | {be_str} |")

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
    print(f"  Avg calendar hours per installation: {avg_calendar_hours_per_installation:.1f}")
    print(f"  {'Scenario':<10} {'Hours Freed':>12} {'Potential Installs':>19} {'Break-Even':>12}")
    print(f"  {'-'*10} {'-'*12} {'-'*19} {'-'*12}")
    for _, r in summary.iterrows():
        n = int(r["scenario_hires"])
        hf = r["hours_freed_existing_techs"]
        pi = r["potential_installations_enabled"]
        be = r["break_even_installations"]
        pi_str = f"{pi:,.0f}" if not np.isnan(pi) else "N/A"
        be_str = f"{be:,.1f}" if not np.isnan(be) else "N/A"
        print(f"  N={n:<7} {hf:>12,.0f} {pi_str:>19} {be_str:>12}")


if __name__ == "__main__":
    main()
