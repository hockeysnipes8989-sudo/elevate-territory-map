# CLAUDE.md - Elevate Territory Map Operating Context

## Purpose

This repository supports Elevate Healthcare service planning across the US with:

1. An interactive map (`docs/index.html`) for assets, appointments, technicians, territories, airports, and scenario overlays.
2. A cost optimization pipeline (`scripts/06` to `11`) for evaluating incremental hiring scenarios.

This file is the canonical context handoff for future chats.

## Canonical Links

- Repo: https://github.com/hockeysnipes8989-sudo/elevate-territory-map
- Live map: https://hockeysnipes8989-sudo.github.io/elevate-territory-map/

## Current High-Level State

- **Annualization is active.** The appointment dataset spans 2.08 years (Jan 2, 2024 → Jan 29, 2026, 758 days). Step 06 computes `data_span_years` (2.0753). Step 08 scales hire cost to match the data period. Step 09 divides all period costs and freed hours by `data_span_years` so every output figure is an annual equivalent. All figures labeled in the map and reports are annualized.
- **BTS Q2 2025 lookup table with 1.6x corporate premium** is active. Flight cost = `BTS_RAW_ITINERARY_FARES[origin_airport] × 1.6`. Origin-only — no destination dependency. 63 US airports with direct BTS fares; national fallback ($386) for unlisted airports. Steps 07 and 10 (ML model + BTS correction) are deprecated and removed from the pipeline.
- **Full cost model is active** (Step 11). Three-tier distance-based trip classification: same-day drive (<100 mi, mileage only), overnight drive (100–300 mi, mileage + 1 hotel night), fly (≥300 mi, flight + duration-scaled hotel + rental car). IRS mileage $0.70/mi round-trip, hotel $159/night, rental $235/fly trip. Step 11 pre-computes a per-(tech/candidate, node) cost table (`full_cost_table.csv`) used by the optimizer.
- Burdened new-hire payroll is modeled in Step 8 (`146,640` USD per incremental hire per year).
- Default out-of-region penalty is disabled (`0` USD).
- Canada is excluded from optimization scope. Hakim Mouazer (Montreal) has availability_fte=0.0 (visible on map only).
- Simulation panel reads optimization outputs and shows scenario KPIs for `N=0..4`.
- Technician markers are grouped by shared coordinates so all 16 roster members are visible via popup rosters.
- New-hire allocation is hard-capped at 1 hire per base by default.
- **AVS = Learning Space (LS)** — confirmed by Shannon/Isabelle. All AVS appointments are reclassified as LS-requiring in Step 06 post-processing. ~315 AVS appointments now require LS-certified techs.
- **Revenue-from-freed-capacity analysis is active** (Step 09). Patient simulator installations only (ISO). Three system-size scenarios ($50K/$120K/$250K per install) at uniform 40% margin + $7K annual service contracts (70% margin). Learning Space (AVS/LS) installations excluded per Shannon — too variable ($6.5K–$106K). This is supplementary analysis — the MILP optimizer recommendation is unchanged.

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
    07_build_travel_cost_model.py    # DEPRECATED — left on disk, not in pipeline
    08_optimize_locations.py
    09_analyze_scenarios.py
    10_correct_travel_costs.py       # DEPRECATED — left on disk, not in pipeline
    11_build_full_cost_table.py      # per-(tech/candidate, node) drive/fly cost table
    travel_cost_modeling.py          # DEPRECATED — left on disk, not in pipeline
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

- Python dependencies are in `requirements.txt`.
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

Pipeline order: **06 → 11 → 08 → 09 → 05** (Steps 07 and 10 are deprecated).

1. `06_build_optimization_inputs.py` — also computes `data_span_years` for annualization
2. `11_build_full_cost_table.py` — pre-computes `full_cost_table.csv` using three-tier trip model (drive_day / drive_overnight / fly) with BTS Q2 2025 fares × 1.6 for flight costs. Re-run when `demand_appointments.csv`, `tech_master.csv`, or BTS fare data changes.
3. `08_optimize_locations.py --min-new-hires 0 --max-new-hires 4 --max-hires-per-base 1 --time-limit-sec 600` — reads `data_span_years`, scales hire cost to match data period
4. `09_analyze_scenarios.py` — reads `data_span_years`, annualizes all period costs and freed hours
5. `05_generate_map.py` to refresh scenario panel in map output.

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
- Candidate bases combine major airports plus top demand-city candidates.
- **Data span computation:** `data_span_years = max(date_span_days / 365.25, 0.5)`. Currently 2.0753 years (Jan 2, 2024 → Jan 29, 2026, 758 days).

### Step 7: DEPRECATED (Travel Cost Matrix)

Step 7 (`07_build_travel_cost_model.py`) is deprecated and no longer part of the pipeline. The ML-based flight cost engine (GradientBoostingRegressor trained on Navan flights) has been replaced by a direct BTS Q2 2025 fare lookup table in `config.py`. The script file remains on disk for historical reference.

### Step 10: DEPRECATED (BTS Correction Layer)

Step 10 (`10_correct_travel_costs.py`) is deprecated and no longer part of the pipeline. The BTS correction layer that post-processed the ML model output is no longer needed — flight costs now come directly from BTS data via `config.BTS_RAW_ITINERARY_FARES`. The script file remains on disk for historical reference.

### Step 11: Full Cost Table (BTS Lookup)

Flight cost = `BTS_RAW_ITINERARY_FARES[origin_airport] × CORPORATE_TRAVEL_PREMIUM (1.6)`. Origin-only — cost does not vary by destination. 63 US airports have direct BTS fares; airports not in the table fall back to `BTS_NATIONAL_FALLBACK` ($386).

The 1.6× corporate premium is calibrated from Navan median actual cost ($633) divided by BTS average domestic fare ($386).

Three-tier trip cost per (tech/candidate, node) pair:
- **Same-day drive** (<100 mi): IRS mileage ($0.70/mi × round-trip), no hotel, no rental
- **Overnight drive** (100–300 mi): IRS mileage ($0.70/mi × round-trip) + 1 hotel night ($159), no rental
- **Fly** (≥300 mi): flight cost + rental car ($235) + duration-scaled hotel ($159/night × node avg duration)

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

- `baseline_canceled_voided_usd` = `config.BASELINE_CANCELED_VOIDED_USD` ($0.00). Set to zero because the Navan export covers only ~2 of 16 technicians, making the canceled/voided overhead unrepresentative. Fixed cost, does not affect relative scenario comparison.

### Step 9: Analysis

- **Annualizes all period costs:** divides `travel_cost_usd`, `hire_cost_usd`, `baseline_canceled_voided_usd`, `modeled_total_cost_usd`, `economic_total_with_overhead_usd`, and other period-total columns by `data_span_years`. Also annualizes `hours_freed_existing_techs`.
- Computes savings vs `N=0`.
- Computes marginal savings from previous `N`.
- **Capacity-freed analysis:** converts freed existing-tech hours → realistic patient simulator installation estimates using avg duration days (2.1 for ISO-only), travel overhead (1.0 day), and 75% utilization factor. Hours are annualized before conversion. Learning Space (AVS/LS) installations excluded from revenue conversion per Shannon — too variable ($6.5K–$106K).
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

- `BASELINE_CANCELED_VOIDED_USD = 0.0` in `config.py`.
- Set to zero because the Navan export covers only ~2 of 16 technicians, making the canceled/voided overhead unrepresentative of the full fleet.
- This is a fixed cost that does not vary by scenario and does not affect the relative comparison between hiring levels.
- Previously $35,632 (full-period) from Navan Report tab, now excluded.

### Skill Constraints (HPS / LS / AVS)

- Existing techs must have `skill_hps=1` to serve HPS nodes and `skill_ls=1` to serve LS nodes.
- **AVS = Learning Space:** All AVS appointments are reclassified as LS-requiring in Step 06. ~315 AVS appointments + 5 original LS = 320 LS-requiring appointments. 7 of 16 techs have LS certification.
- **New hires cannot serve HPS nodes** — hard variable bound (`ub=0.0`) in Step 8. Shannon/Isabelle confirmed new hires would NOT be HPS-trained (product line discontinuing).
- New hires can serve LS and regular nodes with no restriction.
- 115 HPS appointments in the demand pool are all served by existing HPS-certified techs.

### Capacity Model

- Capacity is **demand-normalized**, not calendar-based. There is no fixed "2,080 annual hours" concept.
- Formula: `hours_per_unit = total_demand_hours / (total_FTE × target_utilization)`
- Each tech's capacity: `availability_fte × hours_per_unit`
- `target_utilization` defaults to `0.85` (Step 08 CLI argument, not in config.py). This means the fleet targets 85% utilization at N=0, leaving 15% buffer for scheduling friction.
- James Sanchez: `availability_fte=1.0` — full field tech; temporarily on phones (injury) but model targets ideal state.
- Damion Lyn: `availability_fte=0.20` — repair center hybrid, 20-25% field when fully staffed (conservative). Clay's team.
- Elier Martin: `availability_fte=0.10` — phone tech, confirmed by Shannon/Isabelle at ~10% field. Clay's team.
- Hakim Mouazer: `availability_fte=0.0` — Canada excluded (visible on map only).
- Current computed values: total_FTE=12.55, total_demand=86,760 hrs, hours_per_unit=8,751

### Revenue-from-Freed-Capacity Model (Step 09)

- **Framing:** Below 15% volume reduction (Shannon Drew directive), the value of hiring should be understood as capacity for revenue, not cost savings.
- **Patient simulator installations only** (ISO). Learning Space (AVS/LS) excluded per Shannon — install values "too variable" ($6.5K–$106K).
- Three system-size revenue scenarios at **uniform 40% gross margin** (Shannon: company targets 60%, "wouldn't go any less than 40%"):
  - Conservative: `$50,000` × 40% margin = $20,000 profit (small patient sims — Aria, Apollo)
  - Moderate: `$120,000` × 40% margin = $48,000 profit (mid-range patient sims — Lucina, Evo)
  - Aggressive: `$250,000` × 40% margin = $100,000 profit (large patient sims — HPS full suite)
- Annual recurring service contract: `$7,000/system` × 70% margin = $4,900 profit.
- Revenue figures are **capacity enabled, not guaranteed** — actual revenue depends on sales pipeline and market demand.
- Profit margins are applied (not raw MSRP) — the analysis shows actual P&L impact.
- Estimates are per year (annualized) — no multi-year NPV.
- The MILP optimizer recommendation (N=0) is unchanged; revenue analysis is purely supplementary.
- Config constants: `REVENUE_PER_INSTALLATION_CONSERVATIVE_USD`, `REVENUE_PER_INSTALLATION_MODERATE_USD`, `REVENUE_PER_INSTALLATION_AGGRESSIVE_USD`, `AVG_ANNUAL_SERVICE_CONTRACT_USD`, `INSTALLATION_PROFIT_MARGIN_*`, `SERVICE_CONTRACT_PROFIT_MARGIN`.

### Capacity-Freed Model Parameters (Step 09)

- `TRAVEL_DAYS_PER_INSTALLATION = 1.0` — travel overhead per installation (days).
- `FREED_CAPACITY_UTILIZATION_FACTOR = 0.30` — fraction of freed days practically usable. Conservative estimate accounting for scheduling friction, non-installation tasks (phone coverage, training, admin), sales pipeline constraints, ramp-up time, and general real-world overhead.
- Avg duration days per installation: `2.1` (computed from ISO-only appointment data). Shannon confirmed patient sim on-site time is ~5 hours (4hr assembly + sub-1hr orientation); the 2.1 days represents the full calendar window (travel + setup + on-site + teardown).
- Effective days per installation: `3.1` (2.1 duration + 1.0 travel).
- Realistic installations = (freed days × 0.30) / 3.1.
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

From current optimization artifacts (BTS Q2 2025 lookup + three-tier cost model + annualization active + AVS=LS + Shannon/Isabelle FTE updates):

- Data span: **2.0753 years** (Jan 2, 2024 → Jan 29, 2026, 758 days, 1,471 US appointments)
- Annualized appointment count: ~709/year
- Skill breakdown: 115 HPS, 320 LS (including 315 AVS→LS), 1,036 regular
- Scenario range: `N=0..4`
- Selection mode: `proven_optimal_only` (N=1..4 solved to proven optimality; N=0 hit time limit with MIP gap 0.004% — effectively optimal)
- Best scenario: `N=1` (proven optimal; N=0 not proven optimal due to time limit)
- **N=0 annualized travel cost: `$540,747` USD** (0 unmet appointments)
- **N=0 annualized overhead: `$0` USD** (canceled/voided excluded)
- **N=0 annualized total: `$540,747` USD**
- Burdened annual per-hire planning cost: `$146,640` USD (period-scaled to `$304,322` in MILP)
- Full cost model constants: IRS $0.70/mi, rental $235/fly trip, hotel $159/night. Same-day drive <100 mi, overnight drive 100–300 mi, fly ≥300 mi.
- Flight cost: BTS Q2 2025 × 1.6 corporate premium. Mean flight cost $633, range $455–$783.
- Full cost table: 10,961 rows (15 techs × 113 nodes + 82 candidates × 113 nodes). Hakim Mouazer (no lat/lon) excluded.
- Trip type split: 1.8% same-day drive, 7.1% overnight drive, 91.1% fly
- Hotel nights distribution: 0 nights (1.8%), 1 night (7.9%), 2 nights (36.1%), 3 nights (31.6%), 4 nights (22.7%). Mean hotel cost: $422/trip.
- No scenario allocates more than one hire to a single base (`max_hires_per_base=1`).
- N=0: 1,471 served, 0 unmet. N=1..4: all 1,471 served, 0 unmet.
- Active techs at N=0: 16 (all techs active). James Sanchez at 1.0 FTE, Damion Lyn at 0.20 FTE, Elier Martin at 0.10 FTE. Total effective FTE: 12.55.
- Mean utilization at N=0: 85.1%. Max: 100.00%.

### Scenario Cost Summary (Annualized)

| N | Annual Travel | Annual Payroll | Annual Overhead | Annual Total | Served | Unmet |
|---|--------------|----------------|-----------------|-------------|--------|-------|
| 0 | $540,747 | $0 | $0 | **$540,747** | 1,471 | 0 |
| 1 | $466,463 | $146,640 | $0 | $613,103 | 1,471 | 0 |
| 2 | $411,008 | $293,280 | $0 | $704,288 | 1,471 | 0 |
| 3 | $366,397 | $439,920 | $0 | $806,317 | 1,471 | 0 |
| 4 | $324,352 | $586,560 | $0 | $910,912 | 1,471 | 0 |

Marginal annual travel savings diminish: $74K (N=0→1), $55K (N=1→2), $45K (N=2→3), $42K (N=3→4).

### Revenue-from-Freed-Capacity Summary (Annualized)

| N | Installs/yr | Net Cost Increase | Net Value (Conservative) | Net Value (Moderate) | Net Value (Aggressive) | Break-Even (Mod) |
|---|------------:|------------------:|-------------------------:|---------------------:|-----------------------:|-----------------:|
| 0 | 0.0 | $0 | $0 | $0 | $0 | 0.0 |
| 1 | 17.0 | $72,355 | $351,296 | $827,691 | $1,712,425 | 1.4 |
| 2 | 34.0 | $163,540 | $682,642 | $1,634,172 | $3,401,299 | 3.1 |
| 3 | 51.0 | $265,570 | $1,003,941 | $2,431,504 | $5,082,691 | 5.0 |
| 4 | 67.8 | $370,164 | $1,318,092 | $3,216,533 | $6,742,208 | 7.0 |

Revenue assumptions: $50K/$120K/$250K per patient sim install × uniform 40% margin + $7K×70% annual service contract. ISO installations only (LS excluded). Profit margins applied — figures represent P&L impact. Utilization factor 30% (conservative — accounts for scheduling friction, admin overhead, sales pipeline constraints, ramp-up time).

Key takeaway: N=0 serves all 1,471 US appointments with 0 unmet. N=1 frees capacity for 17.0 patient sim installs/yr at $72K incremental cost (break-even at 1.4 moderate installs, ROI 1,144%). Revenue analysis supports hiring even though cost-only optimization selects N=0.

### Hiring Placements by Scenario

| N | Recommended Bases |
|---|-------------------|
| 1 | CLE (Cleveland, OH) |
| 2 | CLE, Janesville WI |
| 3 | ORD (Chicago, IL), CLE, Fort Smith AR |
| 4 | ATL (Atlanta, GA), ORD, CLE, Fort Smith AR |


## Important File Outputs to Check First

- `data/processed/optimization/optimization_input_summary.json` (includes `data_span_years`)
- `data/processed/optimization/model_assumptions.json` (includes `data_span_years`, `hire_cost_for_optimization_period`)
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

### Flight Cost Data
1. BTS fares in `config.BTS_RAW_ITINERARY_FARES` are Q2 2025 data. SHV, BIL, BIS, FAR, and ANC use the national fallback ($386) as their BTS-specific fares were unavailable. Update the dict in `config.py` if fare data is refreshed.
2. The 1.6× corporate premium is calibrated from Navan median ($633) / BTS average ($386). If Navan booking patterns change, recalibrate.

### Cost Model Simplifications
3. Three-tier drive model: same-day (<100 mi, no hotel), overnight (100–300 mi, 1 hotel night), fly (≥300 mi, duration-scaled hotel). Hotel cost for fly trips uses per-node average appointment duration (not per-appointment). The nightly rate ($159) is a Navan-derived constant. Drive trips use fixed hotel nights (0 or 1) regardless of appointment duration.
4. Same-city trip bundling is not modeled — each of 1,480 appointments is treated as a separate trip. In practice, techs bundle nearby appointments.
5. Great-circle distance (not road distance) for drive/fly classification. Road distance is typically 10–25% longer, meaning some trips classified as "drive" might actually exceed 300 road-miles.
6. Canceled and voided booking overhead is excluded (`BASELINE_CANCELED_VOIDED_USD = 0.0`) due to incomplete Navan data coverage (~2 of 16 techs). This is a fixed cost that does not vary by scenario and does not affect the relative comparison between hiring levels.
7. Full cost model hotel nightly rate ($159) and rental car ($235) are 2025 Navan actuals. Re-update in `config.py` if Navan benchmarks change meaningfully.

### Model Assumptions
8. **No seasonality** — the model treats all appointments as equivalent regardless of when they occur during the year.
9. **New hires cannot serve HPS nodes** — this is a policy assumption. If new hires can be HPS-trained, the model underestimates their value.
10. Capacity model is demand-normalized (not calendar-based). See "Capacity Model" section above.
11. **Annualization assumes uniform distribution** — dividing by `data_span_years` assumes costs are evenly distributed across the 2.08-year period. If demand or travel patterns shifted significantly within the period, annualized figures may not perfectly represent a single future year.
12. Flight cost is origin-only (no destination dependency). All flights from a given airport cost the same regardless of where the technician is going. This simplification is reasonable because BTS averages are itinerary-level (round-trip) and vary primarily by origin market.

### Proxy and Approximation Notes
13. **Fort Smith AR maps to LIT (Little Rock, ~157 mi).** No closer airport is in the 62-airport list. Fort Smith has a small regional airport (FSM) not in our candidate pool.

### Revenue Model Caveats
14. Revenue figures represent **capacity enabled**, not guaranteed bookings — actual revenue depends on sales pipeline and market demand.
15. Profit margins are uniform 40% on installations (confirmed by Shannon Drew), 70% on service contracts — actual margins vary by product line and deal structure.
16. Service contract revenue assumes each new installation generates an annual $7K contract. Fleet mix may shift this up (Apex-tier) or down (Peak-tier).
17. Estimates are **annualized from the 2.08-year data period** — actual future-year results depend on demand trends.
18. Revenue analysis is supplementary — the MILP optimizer recommendation (N=0) is unchanged and based purely on cost minimization.

### Solver
19. N=1..4 solve to proven optimality (MIP gap = 0.0). N=0 hits time limit with MIP gap 0.004% (effectively optimal). Max existing-tech utilization is 100.00% at N=0.
20. **Learning Space (AVS/LS) excluded from revenue model** — Shannon confirmed install values range $6.5K to $106K, too variable to model meaningfully. LS appointments remain in demand data for dispatch/utilization modeling.

## Recommended Defaults for Re-Runs

Use these commands unless a test requires deviation:

```bash
/opt/miniconda3/bin/python3 scripts/06_build_optimization_inputs.py
/opt/miniconda3/bin/python3 scripts/11_build_full_cost_table.py
/opt/miniconda3/bin/python3 scripts/08_optimize_locations.py --min-new-hires 0 --max-new-hires 4 --max-hires-per-base 1 --time-limit-sec 600
/opt/miniconda3/bin/python3 scripts/09_analyze_scenarios.py
/opt/miniconda3/bin/python3 scripts/05_generate_map.py
```

To update BTS fares, edit `BTS_RAW_ITINERARY_FARES` in `scripts/config.py` and re-run steps 11–08–09–05.

## If Starting a New Chat

State these immediately to avoid context drift:

1. **BTS Q2 2025 lookup table with 1.6× corporate premium** is the flight cost engine. `BTS_RAW_ITINERARY_FARES[origin] × 1.6`. Steps 07 and 10 are deprecated.
2. **Annualization is active.** Data spans 2.08 years (1,471 US appts). All output figures are annualized. Step 06 computes `data_span_years` (2.0753). Step 08 scales hire cost for MILP period. Step 09 divides all costs/hours by `data_span_years`.
3. **Full cost model is active** (Step 11): three-tier distance-based — same-day drive (<100 mi, mileage only), overnight drive (100–300 mi, mileage + 1 hotel night), fly (≥300 mi, flight + duration-scaled hotel + rental). IRS $0.70/mi, hotel $159/night, rental $235/fly trip.
4. Burdened hire cost is `$146,640`/year per incremental hire ($304,322 in MILP period).
5. Out-of-region penalty default is `0`.
6. Canceled/voided overhead excluded (`$0`) — Navan covers only ~2/16 techs. Fixed cost, doesn't affect scenario comparison.
7. Canada excluded from optimization. Hakim Mouazer (Montreal) at availability_fte=0.0 (map only).
8. **New hires cannot serve HPS nodes** (policy constraint, hard variable bound).
9. Capacity model is demand-normalized with `target_utilization=0.85`.
10. Scenario panel `Total Cost` shows annualized `economic_total_with_overhead_usd`.
11. Technician map points are grouped by base; roster details are in marker popup.
12. **AVS = Learning Space (LS)** — all AVS appointments reclassified as LS-requiring in Step 06. ~320 LS appointments require LS-certified techs (7 of 16).
13. **Revenue-from-freed-capacity analysis is active** in Step 09: patient sim only (ISO), uniform 40% margin, 3 system-size tiers + $7K service contracts. Supplementary — MILP recommendation unchanged.
14. Pipeline order: **06 → 11 → 08 → 09 → 05**.
