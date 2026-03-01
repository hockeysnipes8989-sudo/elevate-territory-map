# CLAUDE.md - Elevate Territory Map Operating Context

## Purpose

This repository supports Elevate Healthcare service planning across the US and Canada with:

1. An interactive map (`docs/index.html`) for assets, appointments, technicians, territories, airports, and scenario overlays.
2. A cost optimization pipeline (`scripts/06` to `11`) for evaluating incremental hiring scenarios.

This file is the canonical context handoff for future chats.

## Canonical Links

- Repo: https://github.com/hockeysnipes8989-sudo/elevate-territory-map
- Live map: https://hockeysnipes8989-sudo.github.io/elevate-territory-map/

## Current High-Level State

- **Annualization is active.** The appointment dataset spans 2.08 years (Jan 2, 2024 → Jan 29, 2026, 758 days). Step 06 computes `data_span_years` (2.0753). Step 08 scales hire cost to match the data period. Step 09 divides all period costs and freed hours by `data_span_years` so every output figure is an annual equivalent. All figures labeled in the map and reports are annualized.
- Hybrid travel-cost engine is implemented and active in Step 7.
- **BTS-calibrated cost matrix is active** (Step 10). The optimizer reads `travel_cost_matrix_bts_corrected.csv` when `config.BTS_CORRECTED_MATRIX = True`. Original matrix is preserved as `travel_cost_matrix.csv`.
- **Full cost model is active** (Step 11). `config.FULL_COST_MODEL = True`. Hotel cost is **duration-scaled**: `HOTEL_NIGHTLY_RATE_USD` ($159) × hotel nights (derived from per-node avg appointment duration). Day-trip logic zeros out hotel for short drive trips (≤150 mi + ≤1 day avg). Each fly trip also includes rental car ($235). Drive trips use IRS mileage ($0.70/mi × round-trip) + duration-scaled hotel. Drive/fly classified by 300-mile haversine threshold. Step 11 pre-computes a per-(tech/candidate, node) cost table (`full_cost_table.csv`) used by the optimizer.
- Burdened new-hire payroll is modeled in Step 8 (`146,640` USD per incremental hire per year).
- Default out-of-region penalty is disabled (`0` USD).
- Hakim-only Canada servicing rule is implemented (Canada is restricted to Canada-wide specialist techs).
- Simulation panel reads optimization outputs and shows scenario KPIs for `N=0..4`.
- Technician markers are grouped by shared coordinates so all 16 roster members are visible via popup rosters.
- New-hire allocation is hard-capped at 1 hire per base by default.
- **Revenue-from-freed-capacity analysis is active** (Step 09). Three profit-margin-based revenue scenarios ($50K/$120K/$250K per install at 15%/25%/40% margins) + $7K annual service contracts (70% margin) quantify the net economic value of freed technician capacity. This is supplementary analysis — the MILP optimizer and N=0 recommendation are unchanged.

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
    10_correct_travel_costs.py
    11_build_full_cost_table.py        # per-(tech/candidate, node) drive/fly cost table
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

### Steps 6-11 (Optimization)

1. `06_build_optimization_inputs.py` — also computes `data_span_years` for annualization
2. `07_build_travel_cost_model.py --engine hybrid --min-direct-route-n 5 --shrinkage-k 10`
3. `10_correct_travel_costs.py` — BTS calibration; produces `travel_cost_matrix_bts_corrected.csv`
4. `11_build_full_cost_table.py` — pre-computes `full_cost_table.csv` (drive/fly + rental + duration-scaled hotel). Computes per-node avg appointment duration from `demand_appointments.csv` and maps to hotel nights. Re-run when `demand_appointments.csv`, `tech_master.csv`, or cost matrix changes.
5. `08_optimize_locations.py --min-new-hires 0 --max-new-hires 4 --max-hires-per-base 1 --time-limit-sec 600` — reads `data_span_years`, scales hire cost to match data period
6. `09_analyze_scenarios.py` — reads `data_span_years`, annualizes all period costs and freed hours
7. `05_generate_map.py` to refresh scenario panel in map output.

Step 10 only needs to re-run when the raw travel cost matrix (`travel_cost_matrix.csv`) changes. Toggle `config.BTS_CORRECTED_MATRIX` to switch which matrix the optimizer uses without re-running Step 10.

## Optimization Model: Exact Logic

### Step 6: Build Inputs

- Builds:
  - `tech_master.csv`
  - `demand_appointments.csv`
  - `candidate_bases.csv`
  - `optimization_input_summary.json` (includes `data_span_years`, `data_span_days`, date range)
- Skills are parsed from appointment text and roster columns.
- Special technician constraints are derived from roster comments:
  - `constraint_florida_only`
  - `constraint_canada_wide`
- Candidate bases combine major airports plus top demand-city candidates.
- **Data span computation:** `data_span_years = max(date_span_days / 365.25, 0.5)`. Currently 2.0753 years (Jan 2, 2024 → Jan 29, 2026, 758 days).

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

**Annualization in the MILP:** Travel costs in the MILP cover the full data period (all 1,480 appointments across 2.08 years). To make hire cost commensurable, Step 08 reads `data_span_years` and passes `hire_cost_for_period = annual_hire_cost × data_span_years` ($304,322) to the solver. This ensures the MILP compares like-for-like costs over the same time period. The annualization back to per-year happens in Step 09.

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

- **Annualizes all period costs:** divides `travel_cost_usd`, `hire_cost_usd`, `baseline_canceled_voided_usd`, `modeled_total_cost_usd`, `economic_total_with_overhead_usd`, and other period-total columns by `data_span_years`. Also annualizes `hours_freed_existing_techs`.
- Computes savings vs `N=0`.
- Computes marginal savings from previous `N`.
- **Capacity-freed analysis:** converts freed existing-tech hours → realistic installation estimates using avg duration days (3.25), travel overhead (1.0 day), and 75% utilization factor. Hours are annualized before conversion.
- **Revenue-from-freed-capacity analysis:** for each hiring scenario, computes net economic value across 3 profit-margin tiers (conservative/moderate/aggressive) plus annual service contract revenue. Includes ROI and break-even installations per tier.
- Picks best scenario using proven-optimal solutions first (`selection_mode = proven_optimal_only`).
- Writes:
  - `scenario_summary_enhanced.csv` (all figures annualized; includes 21 revenue columns: 7 metrics × 3 tiers)
  - `recommended_hire_locations.csv`
  - `analysis_report.json` (includes `data_span_years`, `annualization_note`, `revenue_scenarios`, per-scenario `revenue_analysis`)
  - `analysis_report.md` (includes annualization note, revenue summary table, and caveats)

## Business Rules and Assumptions (Current)

### Annualization

- The appointment dataset spans **2.08 years** (758 days, Jan 2 2024 → Jan 29 2026, 1,480 appointments).
- `data_span_years` = 2.0753, computed in Step 06, stored in `optimization_input_summary.json`.
- The MILP solves over the full data period (all 1,480 appointments). Hire cost is scaled to match: `$146,640 × 2.0753 = $304,322`.
- Step 09 divides all period-total costs and freed hours by `data_span_years` before analysis.
- The annualization preserves MILP solution quality (no re-solving needed) and makes hire cost cancel out correctly: `$304,322 / 2.0753 = $146,640`.
- Per-appointment metrics (avg hours/install, utilization ratios, revenue per install) are NOT annualized — they are time-period agnostic.

### Payroll Burden

- `DEFAULT_ANNUAL_HIRE_COST_USD = 146640.0`
- Interpretation: burdened company planning cost per incremental new hire per year (not take-home pay).
- Applied only to new hires in each scenario.

### Out-of-Region Friction

- `DEFAULT_OUT_OF_REGION_PENALTY_USD = 0.0`
- No extra state-crossing surcharge. Travel cost alone drives assignment decisions.

### Unmet Demand Penalty

- `DEFAULT_UNMET_PENALTY_USD = 5000.0`
- Per-appointment penalty for unmet demand in the MILP objective.

### Canceled/Voided Handling

- Fixed baseline constant from Navan `Report` tab ($35,632.02 for the full 2.08-year period).
- It is not scaled by hires and not re-estimated per scenario.
- Annualized to ~$17,170/year in Step 09 output.
- Intent: preserve known historical overhead as a static add-on to scenario totals.

### Canada Coverage Rule (Hakim Policy)

- Canada nodes can only be assigned to techs with `constraint_canada_wide = 1`.
- Those techs are restricted to Canada nodes only.
- New hires are currently blocked from serving Canada nodes.

### Skill Constraints (HPS / LS)

- Existing techs must have `skill_hps=1` to serve HPS nodes and `skill_ls=1` to serve LS nodes.
- **New hires cannot serve HPS nodes** — hard variable bound (`ub=0.0`) in Step 8. This is a policy assumption, not a data-driven constraint. If new hires can be HPS-trained, the model underestimates their value.
- New hires can serve LS and regular nodes with no restriction.
- 115 HPS appointments in the demand pool are all served by existing HPS-certified techs.

### Capacity Model

- Capacity is **demand-normalized**, not calendar-based. There is no fixed "2,080 annual hours" concept.
- Formula: `hours_per_unit = total_demand_hours / (total_FTE × target_utilization)`
- Each tech's capacity: `availability_fte × hours_per_unit`
- `target_utilization` defaults to `0.85` (Step 08 CLI argument, not in config.py). This means the fleet targets 85% utilization at N=0, leaving 15% buffer for scheduling friction.
- Techs with `availability_fte=0.0` (e.g., James Sanchez / Alex Rondero) get zero capacity and zero assignments.
- Current computed values: total_FTE=13.25, total_demand=86,760 hrs, hours_per_unit=7,703.44

### Revenue-from-Freed-Capacity Model (Step 09)

- **Framing:** Below 15% volume reduction (Shannon Drew directive), the value of hiring should be understood as capacity for revenue, not cost savings.
- Three profit-margin-based revenue scenarios (per installation):
  - Conservative: `$50,000` × 15% margin = $7,500 profit (small systems — Aria, Apollo)
  - Moderate: `$120,000` × 25% margin = $30,000 profit (mid-range — Lucina, Evo)
  - Aggressive: `$250,000` × 40% margin = $100,000 profit (large systems — HPS full suite)
- Annual recurring service contract: `$7,000/system` × 70% margin = $4,900 profit.
- Revenue figures are **capacity enabled, not guaranteed** — actual revenue depends on sales pipeline and market demand.
- Profit margins are applied (not raw MSRP) — the analysis shows actual P&L impact.
- Estimates are per year (annualized) — no multi-year NPV.
- The MILP optimizer recommendation (N=0) is unchanged; revenue analysis is purely supplementary.
- Config constants: `REVENUE_PER_INSTALLATION_CONSERVATIVE_USD`, `REVENUE_PER_INSTALLATION_MODERATE_USD`, `REVENUE_PER_INSTALLATION_AGGRESSIVE_USD`, `AVG_ANNUAL_SERVICE_CONTRACT_USD`, `INSTALLATION_PROFIT_MARGIN_*`, `SERVICE_CONTRACT_PROFIT_MARGIN`.

### Capacity-Freed Model Parameters (Step 09)

- `TRAVEL_DAYS_PER_INSTALLATION = 1.0` — travel overhead per installation (days).
- `FREED_CAPACITY_UTILIZATION_FACTOR = 0.75` — fraction of freed days practically usable (accounts for scheduling gaps, PTO, non-installation work).
- Avg duration days per installation: `3.25` (computed from appointment data: ISO 2.1d, AVS ISO 3.6d, AVS 1.1d).
- Effective days per installation: `4.25` (3.25 duration + 1.0 travel).
- Realistic installations = (freed days × 0.75) / 4.25.
- All freed-hours metrics are annualized (divided by `data_span_years`) before conversion.

### Contractor Scope

- Default scope is `texas_only` unless explicitly overridden.

### Current Technician Roster Baseline

- Expected current technician count is 16 (includes both HTX contractors).
- If count diverges from expectation in pipeline/map reads, code now emits a warning for data-gap triage.

## Map UI and KPI Interpretation

Simulation panel (left side) reads scenario files and shows:

- `Total Cost`: `economic_total_with_overhead_usd` (annualized)
- `Cost Change vs N=0`
- `Marginal Cost Change` vs previous hire count
- `Unmet Appointments` (rendered only if any scenario has unmet > 0)
- `Annual Hire Payroll` (incremental hires only)
- Mean/max existing-tech utilization
- Recommended base placements

All figures in the simulation panel are annualized. The subtitle reads "Cost-first optimization — all figures annualized."

## Latest Validated Run Snapshot

From current optimization artifacts (BTS-corrected matrix + full cost model + annualization active):

- Data span: **2.0753 years** (Jan 2, 2024 → Jan 29, 2026, 758 days, 1,480 appointments)
- Annualized appointment count: ~713/year
- Scenario range: `N=0..4`
- Selection mode: `proven_optimal_only` (all 5 scenarios solved to proven optimality)
- Best scenario: `N=0`
- **N=0 annualized travel cost: `$593,472` USD**
- **N=0 annualized overhead: `$17,170` USD**
- **N=0 annualized total: `$610,642` USD**
- Burdened annual per-hire planning cost: `$146,640` USD (period-scaled to `$304,322` in MILP)
- Full cost model constants: IRS $0.70/mi, rental $235/trip, hotel $159/night (duration-scaled), drive threshold 300 mi, day-trip ≤150 mi + ≤1 day
- Full cost table: 8,008 rows (16 techs × 77 nodes + 88 candidates × 77 nodes)
- Drive/fly split in cost table: 8.5% drive, 91.5% fly
- Hotel nights distribution: 1 night (2.6%), 2 nights (44.2%), 3 nights (46.8%), 4 nights (6.5%). Mean hotel cost: $409/trip. No day trips (min node avg = 1.18 days > 1.0 threshold).
- Navan flight date window used in Step 7: `2025-07-09` to `2026-03-13`
- No scenario allocates more than one hire to a single base (`max_hires_per_base=1`).
- All 1,480 appointments served across all scenarios (zero unmet).
- Active techs at N=0: 15 (excludes James Sanchez / Rondero, `availability_fte=0.0`).
- Mean utilization at N=0: 85.7%. Max: 99.99% (one tech near ceiling — workforce is tightly loaded).

### Scenario Cost Summary (Annualized)

| N | Annual Travel | Annual Payroll | Annual Overhead | Annual Total |
|---|--------------|----------------|-----------------|-------------|
| 0 | $593,472 | $0 | $17,170 | **$610,642** |
| 1 | $551,329 | $146,640 | $17,170 | $715,139 |
| 2 | $518,260 | $293,280 | $17,170 | $828,710 |
| 3 | $489,870 | $439,920 | $17,170 | $946,959 |
| 4 | $467,033 | $586,560 | $17,170 | $1,070,763 |

Marginal annual travel savings diminish: $42K (N=0→1), $33K (N=1→2), $28K (N=2→3), $23K (N=3→4).

### Revenue-from-Freed-Capacity Summary (Annualized)

| N | Realistic Installs/yr | Net Cost Increase | Net Value (Conservative) | Net Value (Moderate) | Net Value (Aggressive) | Break-Even (Mod) |
|---|----------------------:|------------------:|-------------------------:|---------------------:|-----------------------:|-----------------:|
| 0 | 0.0 | $0 | $0 | $0 | $0 | 0.0 |
| 1 | 27.3 | $104,497 | $234,043 | $848,329 | $2,759,441 | 3.0 |
| 2 | 54.6 | $218,068 | $459,197 | $1,688,104 | $5,511,373 | 6.2 |
| 3 | 81.9 | $336,318 | $679,348 | $2,522,289 | $8,255,883 | 9.6 |
| 4 | 109.2 | $460,121 | $893,940 | $3,350,905 | $10,994,798 | 13.2 |

Key takeaway: Even at conservative estimates, N=1 enables ~$234K in annual profit for ~$104K incremental cost (break-even at 8.4 installs conservative, 3.0 moderate). The cost-only optimizer says N=0 is cheapest, but the profit lens shows substantial economic upside from hiring.

Note: Revenue figures are unchanged from previous run because capacity-freed hours (driven by MILP assignments) are stable. The ~$10K drop in total cost (from $620K to $611K) flows through travel cost, not capacity metrics.

### Hiring Placements by Scenario

| N | Recommended Bases |
|---|-------------------|
| 1 | CLE (Cleveland, OH) |
| 2 | CLE, MKE (Milwaukee, WI) |
| 3 | BOS (Boston, MA), ORD (Chicago, IL), CLE |
| 4 | BOS, ORD, CLE, Fort Smith AR (→ LIT airport) |

- BTS-corrected matrix airport benchmarks (key sanity checks):
  - MDW: $471 → $199 (-57.7%) — now grounded in BTS data, not model speculation
  - TPA: $418 → $563 (+34.5%) — 6 TPA-based techs now realistically priced
  - IND: $267 → $225 (-15.9%)
  - BOI: $322 → $220 (-31.5%)
  - YUL: $676 → $754 (+11.6%) — 60/40 blend of 10 Navan actuals + BTS cross-border

## Important File Outputs to Check First

- `data/processed/optimization/optimization_input_summary.json` (includes `data_span_years`)
- `data/processed/optimization/model_assumptions.json` (includes `data_span_years`, `hire_cost_for_optimization_period`)
- `data/processed/optimization/travel_model_metrics.json`
- `data/processed/optimization/travel_matrix_coverage_report.json`
- `data/processed/optimization/travel_matrix_origin_anomaly_report.json`
- `data/processed/optimization/baseline_kpis.json`
- `data/processed/optimization/scenario_summary.csv`
- `data/processed/optimization/scenario_summary_enhanced.csv`
- `data/processed/optimization/scenario_placements.csv`
- `data/processed/optimization/scenario_assignments_existing.csv`
- `data/processed/optimization/scenario_assignments_newhires.csv`
- `data/processed/optimization/scenario_tech_utilization.csv`
- `data/processed/optimization/full_cost_table.csv`
- `data/processed/optimization/analysis_report.json`
- `docs/index.html`

## Known Limitations and Caveats

### Data Sparsity
1. Navan coverage window is shorter than full appointment history, so route learning is partially sparse.
2. 57 of 68 origin airports had fewer than 10 Navan flights; 27 had zero. BTS correction addresses the resulting bias but does not replace the need for broader Navan flight data over time.
3. BTS fares embedded in Step 10 are Q2 2025 data. Tier 2 airports (~24 of 68) use estimated midpoints rather than directly verified BTS figures. Re-run Step 10 if fare data is refreshed.

### Cost Model Simplifications
4. Hotel cost is **duration-scaled** using per-node average appointment duration (not per-appointment). Nodes with mixed short/long appointments get a blended average. The nightly rate ($159) and day-trip thresholds are Navan-derived constants. Day-trip logic (≤150 mi + ≤1 day avg → $0 hotel) currently triggers on zero nodes because the minimum node avg is 1.18 days.
5. Same-city trip bundling is not modeled — each of 1,480 appointments is treated as a separate trip. In practice, techs bundle nearby appointments.
6. Great-circle distance (not road distance) for drive/fly classification. Road distance is typically 10–25% longer, meaning some trips classified as "drive" might actually exceed 300 road-miles.
7. Canceled/voided overhead ($35,632 full-period / $17,170 annualized) is fixed across all scenarios, not re-estimated per hiring level.
8. Full cost model hotel nightly rate ($159) and rental car ($235) are 2025 Navan actuals. Re-update in `config.py` if Navan benchmarks change meaningfully.

### Model Assumptions
9. **No seasonality** — the model treats all appointments as equivalent regardless of when they occur during the year.
10. **New hires cannot serve HPS nodes** — this is a policy assumption. If new hires can be HPS-trained, the model underestimates their value.
11. Capacity model is demand-normalized (not calendar-based). See "Capacity Model" section above.
12. BTS prior is optional and currently inactive if the prior CSV is missing.
13. **Annualization assumes uniform distribution** — dividing by `data_span_years` assumes costs are evenly distributed across the 2.08-year period. If demand or travel patterns shifted significantly within the period, annualized figures may not perfectly represent a single future year.

### Proxy and Approximation Notes
14. **SHV and ICT travel costs are proxy-based, not BTS-calibrated.** Their matrix rows were generated from LIT (for SHV) and TUL (for ICT) as proxies. SHV mean cost ($476) and ICT mean cost ($370) are reasonable for regional airports but are not BTS-grounded.
15. **Fort Smith AR maps to LIT (Little Rock, ~157 mi).** No closer airport is in the 68-airport list. Fort Smith has a small regional airport (FSM) not in our candidate pool.
16. YUL (Montreal-Trudeau) raw hybrid model estimated ~$676 vs observed ~$959 (29% underestimate). The BTS correction blends 60% Navan actual ($959) + 40% BTS cross-border ($447) → $754. The remaining ~$205 gap vs observed reflects that cross-border fares vary by specific routing.

### Revenue Model Caveats
17. Revenue figures represent **capacity enabled**, not guaranteed bookings — actual revenue depends on sales pipeline and market demand.
18. Profit margins are applied (15%/25%/40% on installations, 70% on service contracts) — actual margins vary by product line and deal structure.
19. Service contract revenue assumes each new installation generates an annual $7K contract. Fleet mix may shift this up (Apex-tier) or down (Peak-tier).
20. Estimates are **annualized from the 2.08-year data period** — actual future-year results depend on demand trends.
21. Revenue analysis is supplementary — the MILP optimizer recommendation (N=0) is unchanged and based purely on cost minimization.

### Solver
22. All 5 scenarios solve to proven optimality (MIP gap = 0.0). Max existing-tech utilization is 99.99% at N=0, indicating the workforce is very tightly loaded.

## Recommended Defaults for Re-Runs

Use these commands unless a test requires deviation:

```bash
/opt/miniconda3/bin/python3 scripts/06_build_optimization_inputs.py
/opt/miniconda3/bin/python3 scripts/07_build_travel_cost_model.py --engine hybrid --min-direct-route-n 5 --shrinkage-k 10
/opt/miniconda3/bin/python3 scripts/10_correct_travel_costs.py
/opt/miniconda3/bin/python3 scripts/11_build_full_cost_table.py
/opt/miniconda3/bin/python3 scripts/08_optimize_locations.py --min-new-hires 0 --max-new-hires 4 --max-hires-per-base 1 --time-limit-sec 600
/opt/miniconda3/bin/python3 scripts/09_analyze_scenarios.py
/opt/miniconda3/bin/python3 scripts/05_generate_map.py
```

To revert to the original (uncorrected) matrix without re-running, set `BTS_CORRECTED_MATRIX = False` in `scripts/config.py` and re-run steps 11–8–9–5.
To revert to the flight-cost-only model, set `FULL_COST_MODEL = False` in `scripts/config.py` and re-run steps 8–9–5 (Step 11 can be skipped).

## If Starting a New Chat

State these immediately to avoid context drift:

1. Hybrid travel-cost engine is already implemented and in use.
2. **Annualization is active.** Data spans 2.08 years (1,480 appts). All output figures are annualized. Step 06 computes `data_span_years` (2.0753). Step 08 scales hire cost for MILP period. Step 09 divides all costs/hours by `data_span_years`.
3. **Full cost model is active** (Step 11): duration-scaled hotel ($159/night × node-avg nights) + rental on fly trips; IRS mileage + duration-scaled hotel on drive trips; day-trip logic (≤150 mi + ≤1 day → $0 hotel); drive/fly classified by 300-mile haversine threshold.
4. **BTS-calibrated cost matrix is active** (Step 10): 68 airports, 1.22× corporate premium, 2.0× Canadian cross-border multiplier.
5. Burdened hire cost is `$146,640`/year per incremental hire ($304,322 in MILP period).
6. Out-of-region penalty default is `0`.
7. Canceled/voided cost is fixed baseline overhead (`$35,632` full-period / `$17,170` annualized), not scenario-variable.
8. Hakim-only Canada rule is active. New hires blocked from Canada nodes.
9. **New hires cannot serve HPS nodes** (policy constraint, hard variable bound).
10. Capacity model is demand-normalized with `target_utilization=0.85`.
11. Scenario panel `Total Cost` shows annualized `economic_total_with_overhead_usd`.
12. Technician map points are grouped by base; roster details are in marker popup.
13. N=0 annualized total: `$610,642`. Best scenario is N=0 — all hiring scenarios cost more.
14. **Revenue-from-freed-capacity analysis is active** in Step 09: 3 profit tiers + $7K service contracts. N=1 moderate net value: ~$848K/year. Break-even: 3.0 installs. This is supplementary — MILP recommendation unchanged.
15. Pipeline order: 06 → 07 → 10 → 11 → 08 → 09 → 05.
