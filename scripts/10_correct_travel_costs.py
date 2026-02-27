"""Step 10: BTS-calibrated travel cost correction.

Replaces speculative airport benchmarks in travel_cost_matrix.csv with
BTS government fare data (×1.22 Elevate corporate premium), blended with
Navan actuals by data density.

Outputs:
  data/processed/optimization/travel_cost_matrix_bts_corrected.csv
  data/processed/optimization/cost_correction_log.csv
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
OPT_DIR = PROJECT_ROOT / "data" / "processed" / "optimization"

MATRIX_IN = OPT_DIR / "travel_cost_matrix.csv"
AUDIT_XLSX = OPT_DIR / "travel_cost_audit_report.xlsx"
MATRIX_OUT = OPT_DIR / "travel_cost_matrix_bts_corrected.csv"
LOG_OUT = OPT_DIR / "cost_correction_log.csv"

# ---------------------------------------------------------------------------
# BTS corporate-adjusted one-way fares  (BTS Q2 2025 itin ÷ 2 × 1.22)
# Tier 1 = verified BTS H1 2025 data; Tier 2 = BTS-estimated midpoints.
# Canadian airports additionally apply ×2.0 cross-border multiplier.
# ---------------------------------------------------------------------------
BTS_CORP_ADJ_OW: dict[str, float] = {
    # Tier 1 — verified BTS H1 2025
    "ATL": 262.0,
    "BOS": 227.0,
    "BWI": 229.0,
    "CLT": 271.0,
    "DCA": 230.0,
    "DEN": 224.0,
    "DTW": 270.0,
    "EWR": 261.0,
    "IAD": 290.0,
    "LAS": 177.0,
    "LAX": 251.0,
    "LGA": 212.0,
    "MCO": 176.0,
    "MDW": 199.0,
    "MIA": 218.0,
    "MSP": 262.0,
    "ORD": 240.0,
    "PHL": 258.0,
    "PHX": 234.0,
    "SEA": 242.0,
    "SFO": 268.0,
    "SLC": 281.0,
    "TPA": 205.0,
    "ANC": 334.0,
    "CMH": 252.0,
    "CVG": 226.0,
    "IND": 240.0,
    "MCI": 269.0,
    "MSY": 212.0,
    "PDX": 237.0,
    "PIT": 230.0,
    "RDU": 223.0,
    "SAT": 255.0,
    "SJC": 203.0,
    "SMF": 238.0,
    "STL": 260.0,
    # Tier 2 — BTS-estimated midpoints
    "ABQ": 241.0,
    "BOI": 235.0,
    "BUF": 223.0,
    "CHS": 256.0,
    "JAX": 232.0,
    "MKE": 244.0,
    "OMA": 256.0,
    "ONT": 220.0,
    "BHM": 275.0,
    "DSM": 262.0,
    "FAR": 271.0,
    "GRR": 262.0,
    "LIT": 262.0,
    "MEM": 275.0,
    "OKC": 250.0,
    "RIC": 250.0,
    "RNO": 238.0,
    "SDF": 250.0,
    "TUL": 256.0,
    "TUS": 238.0,
    "BIL": 296.0,
    "BIS": 296.0,
    "BTR": 284.0,
}

# National average fallback: $386/2 × 1.22 = $235
BTS_NATIONAL_FALLBACK = 235.0

# Canadian airports: base OW est (USD) × 1.22 × 2.0 cross-border multiplier
# YUL overridden below after blending with Navan actuals.
CANADIAN_BTS_CORP_ADJ_OW: dict[str, float] = {
    "YEG": 405.0,   # ~$166 × 1.22 × 2.0
    "YQR": 471.0,   # ~$193 × 1.22 × 2.0
    "YUL": 447.0,   # ~$183 × 1.22 × 2.0  (pure BTS component; blended separately)
    "YVR": 491.0,   # ~$201 × 1.22 × 2.0
    "YYC": 405.0,   # ~$166 × 1.22 × 2.0
    "YYZ": 447.0,   # ~$183 × 1.22 × 2.0
}


def get_bts_fare(airport: str) -> float:
    """Return BTS corporate-adjusted one-way fare for the given airport code."""
    code = airport.strip().upper()
    if code in CANADIAN_BTS_CORP_ADJ_OW:
        return CANADIAN_BTS_CORP_ADJ_OW[code]
    return BTS_CORP_ADJ_OW.get(code, BTS_NATIONAL_FALLBACK)


def compute_corrected_benchmark(
    airport: str,
    navan_count: int,
    navan_mean: float,
    bts_corporate: float,
) -> float:
    """Apply blending table and return corrected airport-level benchmark."""
    # Special case: YUL with 10 Navan flights → 60/40 blend
    if airport.upper() == "YUL" and navan_count >= 10:
        return 0.60 * navan_mean + 0.40 * bts_corporate

    if navan_count >= 20:
        return navan_mean
    elif navan_count >= 10:
        return 0.60 * navan_mean + 0.40 * bts_corporate
    elif navan_count >= 5:
        return 0.40 * navan_mean + 0.60 * bts_corporate
    elif navan_count >= 1:
        return 0.20 * navan_mean + 0.80 * bts_corporate
    else:  # 0 flights
        return bts_corporate


def main() -> None:
    print("=" * 60)
    print("Step 10: BTS-Calibrated Travel Cost Correction")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load inputs
    # ------------------------------------------------------------------
    print("\n[1/6] Loading travel cost matrix...")
    if not MATRIX_IN.exists():
        sys.exit(f"ERROR: {MATRIX_IN} not found. Run steps 6–7 first.")
    matrix = pd.read_csv(MATRIX_IN)
    print(f"  Matrix rows: {len(matrix):,}  |  Columns: {list(matrix.columns)}")

    print("\n[2/6] Loading audit report Excel sheets...")
    if not AUDIT_XLSX.exists():
        sys.exit(f"ERROR: {AUDIT_XLSX} not found.")

    airport_summary = pd.read_excel(AUDIT_XLSX, sheet_name=0)
    route_detail = pd.read_excel(AUDIT_XLSX, sheet_name=1)
    print(f"  Airport summary rows: {len(airport_summary)}")
    print(f"  Route detail rows:    {len(route_detail)}")

    # Normalise column names (strip whitespace)
    airport_summary.columns = airport_summary.columns.str.strip()
    route_detail.columns = route_detail.columns.str.strip()

    # Column name aliases: support both naming conventions
    # Airport Summary sheet uses: num_actual_flights, mean_actual_cost
    # Plan described them as: navan_flight_count, navan_mean_cost_usd
    def normalise_summary(df: pd.DataFrame) -> pd.DataFrame:
        rename = {
            "num_actual_flights": "navan_flight_count",
            "mean_actual_cost": "navan_mean_cost_usd",
            "mean_model_cost": "model_mean_cost_usd",
        }
        return df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    def normalise_route(df: pd.DataFrame) -> pd.DataFrame:
        rename = {
            "destination_state": "state_norm",
            "num_actual_flights": "navan_flight_count",
            "mean_actual_cost": "navan_mean_cost_usd",
        }
        return df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    airport_summary = normalise_summary(airport_summary)
    route_detail = normalise_route(route_detail)

    # ------------------------------------------------------------------
    # Build airport-level lookup from audit report
    # ------------------------------------------------------------------
    print("\n[3/6] Computing corrected benchmarks per airport...")

    # Expect: origin_airport, navan_flight_count, navan_mean_cost_usd
    required_summary_cols = {"origin_airport", "navan_flight_count", "navan_mean_cost_usd"}
    missing_cols = required_summary_cols - set(airport_summary.columns)
    if missing_cols:
        sys.exit(f"ERROR: Airport Summary sheet missing columns: {missing_cols}")

    # Airport-level current mean from the matrix itself
    matrix_airport_means = (
        matrix.groupby("origin_airport")["expected_cost_usd"]
        .mean()
        .rename("airport_current_mean")
    )

    log_rows = []
    airport_corrections: dict[str, float] = {}  # airport → correction ratio

    for _, row in airport_summary.iterrows():
        airport = str(row["origin_airport"]).strip().upper()
        navan_count = int(row["navan_flight_count"]) if pd.notna(row["navan_flight_count"]) else 0
        navan_mean = float(row["navan_mean_cost_usd"]) if pd.notna(row["navan_mean_cost_usd"]) and navan_count > 0 else 0.0

        bts_corporate = get_bts_fare(airport)
        corrected_benchmark = compute_corrected_benchmark(airport, navan_count, navan_mean, bts_corporate)

        airport_current_mean = float(matrix_airport_means.get(airport, np.nan))

        if np.isnan(airport_current_mean) or airport_current_mean <= 0:
            correction_ratio = 1.0
            print(f"  WARN: {airport} has no routes in matrix — skipping ratio correction.")
        else:
            correction_ratio = corrected_benchmark / airport_current_mean

        airport_corrections[airport] = correction_ratio

        # Determine tier
        if airport in CANADIAN_BTS_CORP_ADJ_OW:
            tier = "canadian_crossborder"
        elif airport in BTS_CORP_ADJ_OW:
            tier = "tier1_bts_verified" if airport not in {
                "ABQ", "BOI", "BUF", "CHS", "JAX", "MKE", "OMA", "ONT",
                "BHM", "DSM", "FAR", "GRR", "LIT", "MEM", "OKC", "RIC",
                "RNO", "SDF", "TUL", "TUS", "BIL", "BIS", "BTR",
            } else "tier2_bts_estimated"
        else:
            tier = "national_fallback"

        log_rows.append({
            "origin_airport": airport,
            "navan_flight_count": navan_count,
            "navan_mean_cost_usd": round(navan_mean, 2),
            "bts_fare": round(bts_corporate / 1.22, 2),  # back-calculate pre-corporate fare
            "bts_corporate": round(bts_corporate, 2),
            "corrected_benchmark": round(corrected_benchmark, 2),
            "airport_current_mean": round(airport_current_mean, 2) if not np.isnan(airport_current_mean) else None,
            "correction_ratio": round(correction_ratio, 4),
            "route_override_count": 0,  # filled in below
            "correction_tier": tier,
        })

    print(f"  Computed corrections for {len(log_rows)} airports.")

    # ------------------------------------------------------------------
    # Apply ratio correction to all routes
    # ------------------------------------------------------------------
    print("\n[4/6] Applying ratio corrections to routes...")

    corrected = matrix.copy()
    corrected["expected_cost_usd_original"] = corrected["expected_cost_usd"]
    corrected["correction_applied"] = False

    for airport, ratio in airport_corrections.items():
        mask = corrected["origin_airport"].str.upper() == airport
        if mask.sum() == 0:
            continue
        corrected.loc[mask, "expected_cost_usd"] = (
            corrected.loc[mask, "expected_cost_usd"] * ratio
        )
        corrected.loc[mask, "correction_applied"] = True

    # ------------------------------------------------------------------
    # Per-route Navan override: routes with ≥5 Navan flights use actuals
    # ------------------------------------------------------------------
    print("\n[5/6] Applying per-route Navan overrides (≥5 flights)...")

    required_route_cols = {"origin_airport", "state_norm", "navan_flight_count", "navan_mean_cost_usd"}
    missing_route_cols = required_route_cols - set(route_detail.columns)
    if missing_route_cols:
        print(f"  WARN: Route Detail sheet missing columns: {missing_route_cols} — skipping per-route overrides.")
        override_count_total = 0
    else:
        strong_routes = route_detail[
            pd.to_numeric(route_detail["navan_flight_count"], errors="coerce").fillna(0) >= 5
        ].copy()
        strong_routes["origin_airport"] = strong_routes["origin_airport"].str.upper()

        override_count_total = 0
        override_counts_by_airport: dict[str, int] = {}

        for _, r in strong_routes.iterrows():
            apt = str(r["origin_airport"]).upper()
            st = str(r["state_norm"])
            navan_actual = float(r["navan_mean_cost_usd"])
            if pd.isna(navan_actual) or navan_actual <= 0:
                continue
            route_mask = (
                (corrected["origin_airport"].str.upper() == apt) &
                (corrected["state_norm"] == st)
            )
            if route_mask.sum() > 0:
                corrected.loc[route_mask, "expected_cost_usd"] = navan_actual
                corrected.loc[route_mask, "correction_applied"] = True
                override_count_total += int(route_mask.sum())
                override_counts_by_airport[apt] = override_counts_by_airport.get(apt, 0) + int(route_mask.sum())

        # Update log with override counts
        for log_row in log_rows:
            apt = log_row["origin_airport"]
            log_row["route_override_count"] = override_counts_by_airport.get(apt, 0)

        print(f"  Per-route overrides applied to {override_count_total} route rows.")

    # ------------------------------------------------------------------
    # Clip and validate
    # ------------------------------------------------------------------
    print("\n  Clipping values to [$50, $2000]...")
    before_clip = corrected["expected_cost_usd"].copy()
    corrected["expected_cost_usd"] = corrected["expected_cost_usd"].clip(lower=50.0, upper=2000.0)
    clipped = (corrected["expected_cost_usd"] != before_clip).sum()
    if clipped:
        print(f"  WARNING: {clipped} values clipped to bounds.")

    assert corrected["expected_cost_usd"].isna().sum() == 0, "NaN values in corrected matrix!"
    assert (corrected["expected_cost_usd"] < 0).sum() == 0, "Negative values in corrected matrix!"

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    mean_before = corrected["expected_cost_usd_original"].mean()
    mean_after = corrected["expected_cost_usd"].mean()
    changed = (corrected["expected_cost_usd"] != corrected["expected_cost_usd_original"]).sum()
    corrected["_abs_diff"] = (corrected["expected_cost_usd"] - corrected["expected_cost_usd_original"]).abs()
    top_changes = (
        corrected.nlargest(5, "_abs_diff")[
            ["origin_airport", "state_norm", "expected_cost_usd_original", "expected_cost_usd"]
        ]
        if changed > 0 else None
    )
    corrected.drop(columns=["_abs_diff"], inplace=True)

    print("\n--- Correction Summary ---")
    print(f"  Routes changed:      {changed:,} of {len(corrected):,}")
    print(f"  Mean cost (before):  ${mean_before:,.2f}")
    print(f"  Mean cost (after):   ${mean_after:,.2f}")
    print(f"  Mean change:         {((mean_after - mean_before) / mean_before * 100):+.1f}%")

    # Key airport sanity checks
    for apt in ["MDW", "TPA", "IND", "BOI", "YUL"]:
        mask = corrected["origin_airport"].str.upper() == apt
        if mask.sum() > 0:
            mean_b = corrected.loc[mask, "expected_cost_usd_original"].mean()
            mean_a = corrected.loc[mask, "expected_cost_usd"].mean()
            print(f"  {apt}: ${mean_b:,.0f} → ${mean_a:,.0f}  ({((mean_a - mean_b) / mean_b * 100):+.1f}%)")

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    print(f"\n[6/6] Writing outputs...")

    # Drop internal helper column before saving
    out_cols = [c for c in corrected.columns if c != "expected_cost_usd_original"]
    corrected[out_cols].to_csv(MATRIX_OUT, index=False)
    print(f"  Corrected matrix → {MATRIX_OUT}")
    print(f"  Rows: {len(corrected):,} | Columns: {len(out_cols)}")

    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(LOG_OUT, index=False)
    print(f"  Correction log    → {LOG_OUT}")
    print(f"  Rows: {len(log_df)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
