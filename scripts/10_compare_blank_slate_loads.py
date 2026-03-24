"""Build a 10-vs-11 Blank Slate load comparison without disturbing live outputs."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


REQUIRED_INPUT_FILES = [
    "tech_master.csv",
    "demand_appointments.csv",
    "candidate_bases.csv",
    "full_cost_table.csv",
    "optimization_input_summary.json",
]


def blank_slate_dir(base_dir: Path) -> Path:
    return base_dir / getattr(config, "BLANK_SLATE_SUBDIR", "blank_slate")


def load_data_span_years(blank_dir: Path, fallback_root: Path) -> float:
    assumptions_path = blank_dir / "model_assumptions.json"
    if assumptions_path.exists():
        with assumptions_path.open() as f:
            assumptions = json.load(f)
        value = assumptions.get("data_span_years")
        if value:
            return float(value)

    summary_path = fallback_root / "optimization_input_summary.json"
    if summary_path.exists():
        with summary_path.open() as f:
            summary = json.load(f)
        value = summary.get("data_span_years")
        if value:
            return float(value)

    return 1.0


def prepare_compare_root(source_root: Path, compare_root: Path) -> None:
    compare_root.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_INPUT_FILES:
        src = source_root / name
        dst = compare_root / name
        if not src.exists():
            raise FileNotFoundError(f"Missing required comparison input: {src}")
        shutil.copy2(src, dst)


def run_blank_slate(compare_root: Path, hires: int, time_limit_sec: int) -> None:
    script_path = Path(__file__).resolve().parent / "08_optimize_locations.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--output-dir",
        str(compare_root),
        "--blank-slate",
        "--min-new-hires",
        str(hires),
        "--max-new-hires",
        str(hires),
        "--max-hires-per-base",
        "1",
        "--time-limit-sec",
        str(time_limit_sec),
    ]
    subprocess.run(cmd, check=True)


def build_detail_table(
    placements_df: pd.DataFrame,
    scenario_hires: int,
    data_span_years: float,
    annual_underload_threshold: float,
) -> pd.DataFrame:
    details = placements_df.copy()
    details["scenario_hires"] = int(scenario_hires)
    details["assigned_appointments"] = pd.to_numeric(
        details.get("assigned_appointments"), errors="coerce"
    ).fillna(0.0)
    details["assigned_hours"] = pd.to_numeric(
        details.get("assigned_hours"), errors="coerce"
    ).fillna(0.0)
    annualized = details["assigned_appointments"] if data_span_years <= 0 else (
        details["assigned_appointments"] / float(data_span_years)
    )
    details["annualized_assigned_appointments"] = annualized
    details["underloaded_flag"] = details["annualized_assigned_appointments"] < float(
        annual_underload_threshold
    )
    keep_cols = [
        "scenario_hires",
        "candidate_id",
        "city",
        "state",
        "airport_iata",
        "assigned_appointments",
        "annualized_assigned_appointments",
        "assigned_hours",
        "underloaded_flag",
    ]
    return details[keep_cols].sort_values(
        ["scenario_hires", "assigned_appointments", "city"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


def build_summary_table(details_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        details_df.groupby("scenario_hires", as_index=False)
        .agg(
            placement_count=("candidate_id", "count"),
            min_assigned_appointments=("assigned_appointments", "min"),
            max_assigned_appointments=("assigned_appointments", "max"),
            min_annualized_appointments=("annualized_assigned_appointments", "min"),
            max_annualized_appointments=("annualized_assigned_appointments", "max"),
            underloaded_placements_count=("underloaded_flag", "sum"),
        )
        .sort_values("scenario_hires")
        .reset_index(drop=True)
    )
    grouped["underloaded_placements_count"] = grouped["underloaded_placements_count"].astype(int)
    return grouped


def load_placements(blank_dir: Path, expected_hires: int) -> pd.DataFrame:
    placements = pd.read_csv(blank_dir / "scenario_placements.csv")
    placements["scenario_hires"] = pd.to_numeric(
        placements.get("scenario_hires"), errors="coerce"
    ).fillna(-1).astype(int)
    placements = placements[placements["scenario_hires"] == int(expected_hires)].copy()
    if placements.empty:
        raise RuntimeError(f"No scenario_placements rows found for scenario_hires={expected_hires} in {blank_dir}")
    return placements


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare blank-slate placement loads for 11 vs 10 hires.")
    parser.add_argument(
        "--source-root",
        default=config.OPTIMIZATION_DIR,
        help="Root optimization directory containing the live 11-hire blank_slate outputs.",
    )
    parser.add_argument(
        "--compare-root",
        default=str(Path(config.OPTIMIZATION_DIR) / "blank_slate_compare_10"),
        help="Separate root directory for the 10-hire comparison run.",
    )
    parser.add_argument(
        "--baseline-hires",
        type=int,
        default=11,
        help="Live blank-slate scenario_hires value to treat as the baseline.",
    )
    parser.add_argument(
        "--compare-hires",
        type=int,
        default=10,
        help="Comparison blank-slate scenario_hires value to solve into compare-root.",
    )
    parser.add_argument(
        "--annual-underload-threshold",
        type=float,
        default=75.0,
        help="Annualized appointment count below which a placement is flagged as underloaded.",
    )
    parser.add_argument("--time-limit-sec", type=int, default=600)
    args = parser.parse_args()

    source_root = Path(args.source_root)
    compare_root = Path(args.compare_root)
    live_blank_dir = blank_slate_dir(source_root)
    if not (live_blank_dir / "scenario_placements.csv").exists():
        raise FileNotFoundError(
            f"Missing live blank-slate placements at {live_blank_dir / 'scenario_placements.csv'}"
        )

    prepare_compare_root(source_root, compare_root)
    run_blank_slate(compare_root, hires=args.compare_hires, time_limit_sec=args.time_limit_sec)

    compare_blank_dir = blank_slate_dir(compare_root)
    baseline_years = load_data_span_years(live_blank_dir, source_root)
    compare_years = load_data_span_years(compare_blank_dir, compare_root)

    baseline_details = build_detail_table(
        load_placements(live_blank_dir, expected_hires=args.baseline_hires),
        scenario_hires=args.baseline_hires,
        data_span_years=baseline_years,
        annual_underload_threshold=args.annual_underload_threshold,
    )
    compare_details = build_detail_table(
        load_placements(compare_blank_dir, expected_hires=args.compare_hires),
        scenario_hires=args.compare_hires,
        data_span_years=compare_years,
        annual_underload_threshold=args.annual_underload_threshold,
    )

    all_details = pd.concat([baseline_details, compare_details], ignore_index=True)
    summary = build_summary_table(all_details)

    details_out = compare_root / "blank_slate_load_comparison_details.csv"
    summary_out = compare_root / "blank_slate_load_comparison_summary.csv"
    all_details.to_csv(details_out, index=False)
    summary.to_csv(summary_out, index=False)

    print(f"Comparison details saved to {details_out}")
    print(f"Comparison summary saved to {summary_out}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
