# CLAUDE.md - Elevate Territory Map Operating Context

## Purpose

This repository supports Elevate Healthcare service planning across the US and Canada with:

1. An interactive map (`docs/index.html`) for assets, appointments, technicians, territories, airports, and scenario overlays.
2. A cost optimization pipeline (`scripts/06` to `09`) for evaluating incremental hiring scenarios.

This file is the canonical context handoff for future chats.

## Canonical Links

- Repo: https://github.com/hockeysnipes8989-sudo/elevate-territory-map
- Live map: https://hockeysnipes8989-sudo.github.io/elevate-territory-map/

## Current High-Level State

- Hybrid travel-cost engine is implemented and active in Step 7.
- **BTS-calibrated cost matrix is active** (Step 10). The optimizer reads `travel_cost_matrix_bts_corrected.csv` when `config.BTS_CORRECTED_MATRIX = True`. Original matrix is preserved as `travel_cost_matrix.csv`.
- Burdened new-hire payroll is modeled in Step 8 (`146,640` USD per incremental hire).
- Default out-of-region penalty is disabled (`0` USD).
- Hakim-only Canada servicing rule is implemented (Canada is restricted to Canada-wide specialist techs).
- Simulation panel reads optimization outputs and shows scenario KPIs for `N=0..4`.
- Technician markers are grouped by shared coordinates so all 16 roster members are visible via popup rosters.
- New-hire allocation is hard-capped at 1 hire per base by default.

## Repository Structure (Important Paths)

```text
elevate-territory-map/
  scripts/
    01_clean_data.py
    02_geocode.py
    03_match_install_base.py
    04_build_territories.py
    05_generate_map.py
    06_build_optimization_inputs.py
    07_build_travel_cost_model.py
    08_optimize_locations.py
    09_analyze_scenarios.py
    travel_cost_modeling.py
    optimization_utils.py
    config.py
  data/
    raw/                           # local source files (not for public sharing)
    processed/
      ...                          # map pipeline outputs
      optimization/                # optimization outputs
  docs/
    index.html                     # deployed map artifact
  README.md
  CLAUDE.md
```

## Environment and Runtime

- Python dependencies are in `requirements.txt` (includes `scikit-learn` for hybrid model).
- In this workstation, prefer:
  - `/opt/miniconda3/bin/python3`
- Reason: avoids mixed interpreter issues (system Python may miss `openpyxl`/`sklearn`).

## Source Data Inputs

### Map Pipeline Raw Inputs

Expected in `data/raw/`:

- UIUC service appointments workbook
- service appointments report workbook
- install base workbook

### Optimization External Inputs

Configured in `scripts/config.py`. Paths default to machine-specific locations but can be
overridden via environment variables:

- `EXTERNAL_APPOINTMENTS_XLSX` (env: `ELEVATE_APPTS_SOURCE`)
- `EXTERNAL_TECH_ROSTER_XLSX` (env: `ELEVATE_TECH_SOURCE`)
- `EXTERNAL_NAVAN_XLSX` (env: `ELEVATE_NAVAN_SOURCE`)

Do not commit sensitive external files.

## Script-by-Script Runbook

### Steps 1-5 (Map)

1. `01_clean_data.py`
2. `02_geocode.py`
3. `03_match_install_base.py`
4. `04_build_territories.py`
5. `05_generate_map.py`

Typical UI-only changes require Step 5 only.

### Steps 6-10 (Optimization)

1. `06_build_optimization_inputs.py`
2. `07_build_travel_cost_model.py --engine hybrid --min-direct-route-n 5 --shrinkage-k 10`
3. `10_correct_travel_costs.py` — BTS calibration; produces `travel_cost_matrix_bts_corrected.csv`
4. `08_optimize_locations.py --min-new-hires 0 --max-new-hires 4 --max-hires-per-base 1 --time-limit-sec 600`
5. `09_analyze_scenarios.py`
6. `05_generate_map.py` to refresh scenario panel in map output.

Step 10 only needs to re-run when the raw travel cost matrix (`travel_cost_matrix.csv`) changes. Toggle `config.BTS_CORRECTED_MATRIX` to switch which matrix the optimizer uses without re-running Step 10.

## Optimization Model: Exact Logic

### Step 6: Build Inputs

- Builds:
  - `tech_master.csv`
  - `demand_appointments.csv`
  - `candidate_bases.csv`
- Skills are parsed from appointment text and roster columns.
- Special technician constraints are derived from roster comments:
  - `constraint_florida_only`
  - `constraint_canada_wide`
- Candidate bases combine major airports plus top demand-city candidates.

### Step 7: Travel Cost Matrix (Hybrid Engine)

Inputs:

- Navan `Clean Flights` tab (flight-level rows).
- Navan `Report` tab (for baseline ticketed/canceled/voided totals).

Training rows for the model are filtered to:

- non-management travelers
- `Booking Status == TICKETED`
- valid origin/destination
- positive numeric paid amount (`USD Total Paid > 0`)

Hybrid route cost logic:

1. Compute model prediction for `(origin_airport, destination_airport, destination_state)`.
2. If direct empirical route support is strong:
   - blend empirical and model cost using shrinkage:
   - `w_empirical = n_direct / (n_direct + shrinkage_k)`
3. For sparse/non-direct routes:
   - blend model and heuristic costs using support-based model weight.
   - apply heuristic-relative guardrails to prevent implausible near-zero route costs.
4. If confidence is low on US route and BTS prior is available:
   - apply BTS state-pair prior.
5. If cost is still invalid:
   - fallback to heuristic estimator.

Outputs:

- `travel_cost_matrix.csv`
- `baseline_kpis.json`
- `travel_model_metrics.json`
- `travel_model_feature_importance.csv`
- `travel_matrix_coverage_report.json`
- `bts_prior_coverage_report.json`
- `travel_matrix_origin_anomaly_report.json`

Current note:

- BTS prior file is optional. If missing, model runs without it.

### Step 8: MILP Scenarios

Scenarios are solved for each `N` in `[min_new_hires, max_new_hires]`.

Decision variables:

- existing-tech appointment assignments
- candidate/new-hire appointment assignments
- integer hire allocations by candidate base
- unmet demand assignments

Objective (modeled):

- minimize travel cost + out-of-region penalties + hire payroll + unmet penalties
- enforce `max_hires_per_base` hard cap across candidate bases (default `1`)

Formally:

- `modeled_total_cost_usd = travel_cost_usd + out_of_region_penalty_usd + hire_cost_usd + unmet_penalty_usd`

Then economic total shown to users:

- `economic_total_with_overhead_usd = modeled_total_cost_usd + baseline_canceled_voided_usd`

Where:

- `baseline_canceled_voided_usd` comes from Navan baseline report and is fixed across all scenarios.

### Step 9: Analysis

- Computes savings vs `N=0`.
- Computes marginal savings from previous `N`.
- Picks best scenario using proven-optimal solutions first (`selection_mode = proven_optimal_only`).
- Writes:
  - `scenario_summary_enhanced.csv`
  - `recommended_hire_locations.csv`
  - `analysis_report.json`
  - `analysis_report.md`

## Business Rules and Assumptions (Current)

### Payroll Burden

- `DEFAULT_ANNUAL_HIRE_COST_USD = 146640.0`
- Interpretation: burdened company planning cost per incremental new hire (not take-home pay).
- Applied only to new hires in each scenario.

### Out-of-Region Friction

- `DEFAULT_OUT_OF_REGION_PENALTY_USD = 0.0`
- Current model effectively uses flight costs only, with no extra state-crossing surcharge.

### Unmet Demand Penalty

- `DEFAULT_UNMET_PENALTY_USD = 5000.0`
- Per-appointment penalty for unmet demand in the MILP objective.

### Canceled/Voided Handling

- Fixed baseline constant from Navan `Report` tab.
- It is not scaled by hires and not re-estimated per scenario.
- Intent: preserve known historical overhead as a static add-on to scenario totals.

### Canada Coverage Rule (Hakim Policy)

- Canada nodes can only be assigned to techs with `constraint_canada_wide = 1`.
- Those techs are restricted to Canada nodes only.
- New hires are currently blocked from serving Canada nodes.

### Contractor Scope

- Default scope is `texas_only` unless explicitly overridden.

### Current Technician Roster Baseline

- Expected current technician count is 16 (includes both HTX contractors).
- If count diverges from expectation in pipeline/map reads, code now emits a warning for data-gap triage.

## Map UI and KPI Interpretation

Simulation panel (left side) reads scenario files and shows:

- `Total Cost`: `economic_total_with_overhead_usd`
- `Cost Change vs N=0`
- `Marginal Cost Change` vs previous hire count
- `Unmet Appointments` (rendered only if any scenario has unmet > 0)
- `Annual Hire Payroll` (incremental hires only)
- Mean/max existing-tech utilization
- Recommended base placements

This means `Total Cost` already includes fixed canceled/voided baseline overhead.

## Latest Validated Run Snapshot

From current optimization artifacts in this repo (BTS-corrected matrix active):

- Scenario range: `N=0..4`
- Selection mode: `all_scenarios_no_proven_optimal` (solver hit 600s time limit; MIP gap ~3e-5, near-optimal)
- Best scenario: `N=0`
- Best total with overhead: `700,408.18` USD  *(was `612,807.52` on original matrix — increase driven by TPA correction +34.5%)*
- N=0 travel cost: `664,776.16` USD  *(was `577,175` on original matrix)*
- Baseline canceled/voided constant: `35,632.02` USD
- Burdened annual per-hire planning cost: `146,640.00` USD
- Hybrid model valid MAE improvement vs heuristic: `16.20%`
- Navan flight date window used in Step 7: `2025-07-09` to `2026-03-13`
- No scenario allocates more than one hire to a single base (`max_hires_per_base=1`).
- BTS-corrected matrix airport benchmarks (key sanity checks):
  - MDW: $471 → $199 (-57.7%) — now grounded in BTS data, not model speculation
  - TPA: $418 → $563 (+34.5%) — 6 TPA-based techs now realistically priced
  - IND: $267 → $225 (-15.9%)
  - BOI: $322 → $220 (-31.5%)
  - YUL: $676 → $754 (+11.6%) — 60/40 blend of 10 Navan actuals + BTS cross-border

## Important File Outputs to Check First

- `data/processed/optimization/travel_model_metrics.json`
- `data/processed/optimization/travel_matrix_coverage_report.json`
- `data/processed/optimization/travel_matrix_origin_anomaly_report.json`
- `data/processed/optimization/baseline_kpis.json`
- `data/processed/optimization/scenario_summary.csv`
- `data/processed/optimization/analysis_report.json`
- `docs/index.html`

## Known Limitations

1. Navan coverage window is shorter than full appointment history, so route learning is partially sparse.
2. Canceled/voided overhead is fixed, not behaviorally modeled.
3. BTS prior is optional and currently inactive if the prior CSV is missing.
4. Solver runtime for `N=0..4` hits the 600s time limit; MIP gap ~3e-5 (effectively optimal in practice).
5. YUL (Montreal-Trudeau) raw hybrid model estimated ~$676 vs observed ~$959 (29% underestimate).
   The BTS correction blends 60% Navan actual ($959) + 40% BTS cross-border ($447) → $754. The
   remaining ~$205 gap vs observed reflects that cross-border fares vary by specific routing.
6. BTS fares embedded in Step 10 are Q2 2025 data. Tier 2 airports (~24 of 66) use estimated
   midpoints rather than directly verified BTS figures. Re-run Step 10 if fare data is refreshed.
7. 57 of 66 origin airports had fewer than 10 Navan flights; 25 had zero. BTS correction addresses
   the resulting bias but does not replace the need for broader Navan flight data over time.

## Recommended Defaults for Re-Runs

Use these commands unless a test requires deviation:

```bash
/opt/miniconda3/bin/python3 scripts/06_build_optimization_inputs.py
/opt/miniconda3/bin/python3 scripts/07_build_travel_cost_model.py --engine hybrid --min-direct-route-n 5 --shrinkage-k 10
/opt/miniconda3/bin/python3 scripts/10_correct_travel_costs.py
/opt/miniconda3/bin/python3 scripts/08_optimize_locations.py --min-new-hires 0 --max-new-hires 4 --max-hires-per-base 1 --time-limit-sec 600
/opt/miniconda3/bin/python3 scripts/09_analyze_scenarios.py
/opt/miniconda3/bin/python3 scripts/05_generate_map.py
```

To revert to the original (uncorrected) matrix without re-running, set `BTS_CORRECTED_MATRIX = False` in `scripts/config.py` and re-run steps 8–9–5.

## If Starting a New Chat

State these immediately to avoid context drift:

1. Hybrid travel-cost engine is already implemented and in use.
2. Burdened hire cost is `146,640` per incremental hire.
3. Out-of-region penalty default is `0`.
4. Canceled/voided cost is fixed baseline overhead, not scenario-variable.
5. Hakim-only Canada rule is active.
6. Scenario panel `Total Cost` includes canceled/voided overhead.
7. Technician map points are grouped by base; roster details are in marker popup.
