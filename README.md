# Elevate Healthcare - Interactive Service Territory Map

Interactive US/Canada map and optimization model for service coverage, travel cost, and hiring scenarios.

- Live map: https://hockeysnipes8989-sudo.github.io/elevate-territory-map/
- Repo: https://github.com/hockeysnipes8989-sudo/elevate-territory-map

## What This Includes

1. Map layers for active-contract assets, appointments, technicians, territories, and airports.
2. Optimization scenario panel on the map for `N=0..4` new hires.
3. End-to-end MILP pipeline for travel + hiring economics.
4. Hybrid travel-cost engine built from Navan flight data.
5. BTS-calibrated cost matrix (Step 10) replacing speculative airport benchmarks with government fare data + 1.22× corporate premium.

## Install

```bash
pip install -r requirements.txt
```

Recommended runtime in this environment:

```bash
/opt/miniconda3/bin/python3 ...
```

## Standard Map Pipeline (Steps 1-5)

```bash
python scripts/01_clean_data.py
python scripts/02_geocode.py
python scripts/03_match_install_base.py
python scripts/04_build_territories.py
python scripts/05_generate_map.py
```

Source files for steps 1-4 are expected in `data/raw/`. Geocoding cache lives in `data/geocode_cache.json`.

## Optimization Pipeline (Steps 6-10)

Outputs are written to `data/processed/optimization/`.

```bash
python scripts/06_build_optimization_inputs.py
python scripts/07_build_travel_cost_model.py --engine hybrid --min-direct-route-n 5 --shrinkage-k 10
python scripts/10_correct_travel_costs.py
python scripts/08_optimize_locations.py --min-new-hires 0 --max-new-hires 4 --max-hires-per-base 1 --time-limit-sec 600
python scripts/09_analyze_scenarios.py
python scripts/05_generate_map.py
```

Step 10 (BTS calibration) only needs to re-run when `travel_cost_matrix.csv` changes. Toggle `BTS_CORRECTED_MATRIX` in `scripts/config.py` to switch between matrices without re-running.

Default external workbook paths are in `scripts/config.py` (overridable via env vars
`ELEVATE_APPTS_SOURCE`, `ELEVATE_TECH_SOURCE`, `ELEVATE_NAVAN_SOURCE`):

- `EXTERNAL_APPOINTMENTS_XLSX`
- `EXTERNAL_TECH_ROSTER_XLSX`
- `EXTERNAL_NAVAN_XLSX`

## Current Model Rules (Important)

- Annual burdened planning cost per incremental new hire: `146,640` USD.
- Unmet demand penalty: `5,000` USD per appointment (`DEFAULT_UNMET_PENALTY_USD`).
- Out-of-region soft penalty default: `0.0` USD (disabled by default).
- Canada coverage rule:
  - Techs flagged `constraint_canada_wide=1` (Hakim policy) are Canada-only.
  - Canada demand nodes can only be assigned to those Canada-wide techs.
  - New hires are blocked from Canada demand in current model.
- Canceled/voided travel handling:
  - Not optimized per scenario.
  - Added as a fixed baseline constant from Navan `Report` tab to every scenario.
- Contractor scope defaults to `texas_only` unless explicitly overridden.
- New-hire concentration cap defaults to `1` per base (`--max-hires-per-base 1`).
- Current verified technician roster is 16 total (including both HTX contractors).
- Technician markers are grouped by shared base location (popup lists all names at that base).

## Travel Cost Engine (Steps 7 + 10)

**Step 7** builds the raw hybrid matrix (`travel_cost_matrix.csv`). Default mode is `--engine hybrid`.

- Inputs: Navan `Clean Flights` + `Report`.
- Training rows: non-management, `TICKETED`, valid origin/destination, positive paid amount.
- Hybrid route estimate logic:
  - Use empirical direct-route information when support is strong.
  - Blend empirical + model prediction using shrinkage for strong direct routes.
  - For sparse routes, blend model prediction with heuristic estimate using support-based weights.
  - Apply guardrails so sparse predictions cannot collapse to unrealistic near-zero fares.
  - Optional BTS prior for low-confidence US gaps.
  - Final fallback to legacy heuristic if needed.
- Diagnostics written:
  - `travel_model_metrics.json`
  - `travel_model_feature_importance.csv`
  - `travel_matrix_coverage_report.json`
  - `bts_prior_coverage_report.json`
  - `travel_matrix_origin_anomaly_report.json`
  - `dropped_zero_fare_flights.csv`

Legacy mode remains available: `--engine heuristic`.

**Step 10** corrects the raw matrix with BTS government fares (`travel_cost_matrix_bts_corrected.csv`).

- Embeds BTS Q2 2025 itinerary fares ÷ 2 × 1.22 corporate premium for 59 US airports + 6 Canadian airports (×2.0 cross-border multiplier).
- Blends BTS benchmarks with Navan actuals by data density (0–4 flights → 80% BTS; 5–9 → 60% BTS; 10–19 → 40% BTS; 20+ → Navan only).
- Per-route override: routes with ≥5 Navan flights use observed actuals directly.
- Audit log written to `cost_correction_log.csv` (one row per origin airport).

## Scenario Cost Formula

Step 8 computes:

- `modeled_total_cost_usd = travel_cost_usd + out_of_region_penalty_usd + hire_cost_usd + unmet_penalty_usd`
- `economic_total_with_overhead_usd = modeled_total_cost_usd + baseline_canceled_voided_usd`

Where:

- `hire_cost_usd = (number of incremental new hires) * 146,640`
- `baseline_canceled_voided_usd` is fixed from Navan baseline data

## Latest Baseline Snapshot (Current Repo Outputs)

From the latest committed optimization artifacts (BTS-corrected matrix active):

- Navan canceled/voided baseline constant: `35,632.02` USD
- Scenario window: `N=0..4`
- Best scenario: `N=0`
- Best total with overhead: `700,408.18` USD *(up from `612,807.52` on original matrix — reflects TPA route correction +34.5%)*
- N=0 travel cost: `664,776.16` USD *(was `577,175` on original matrix)*
- Hard cap result: no scenario allocates more than 1 hire to the same base
- Solver MIP gap: ~3e-5 (near-optimal; 600s time limit reached)

## Key Output Files

- `data/processed/optimization/tech_master.csv`
- `data/processed/optimization/demand_appointments.csv`
- `data/processed/optimization/candidate_bases.csv`
- `data/processed/optimization/travel_cost_matrix.csv` — raw hybrid model matrix
- `data/processed/optimization/travel_cost_matrix_bts_corrected.csv` — BTS-calibrated matrix (active)
- `data/processed/optimization/cost_correction_log.csv` — per-airport BTS correction audit trail
- `data/processed/optimization/scenario_summary.csv`
- `data/processed/optimization/scenario_summary_enhanced.csv`
- `data/processed/optimization/recommended_hire_locations.csv`
- `data/processed/optimization/analysis_report.json`
- `data/processed/optimization/travel_matrix_origin_anomaly_report.json`
- `docs/index.html`

## Deeper Documentation

For full context (business rules, caveats, and runbook details), see `CLAUDE.md`.
