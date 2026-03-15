# CLAUDE.md - Elevate Territory Map Operating Context

## Purpose

This repo supports Elevate Healthcare service planning with:

1. An interactive map in `docs/index.html`
2. A travel + hiring optimization pipeline
3. A patient-simulator install-upside layer built on top of freed capacity

This file is the working context handoff for future chats.

## Canonical Links

- Repo: https://github.com/hockeysnipes8989-sudo/elevate-territory-map
- Live map: https://hockeysnipes8989-sudo.github.io/elevate-territory-map/

## Current High-Level State

- Annualization is active. The current dataset spans `2.078` years.
- The active pipeline order is **06 → 11 → 08 → 09 → 05**.
- Step 11 uses the three-tier trip model:
  - same-day drive
  - overnight drive
  - fly
- Step 11 now uses:
  - `$0.60/mi` mileage
  - `$165/night` hotel
  - `$40/day` rental
  - a `125` one-way median-mile threshold for switching drive trips from personal mileage to rental-car economics
- Step 08 now layers in airport-based operational zones:
  - standard employees/new hires: `0-1` free, `2` penalized, `3+` blocked
  - HTX contractors: softer penalty-only treatment, not full exemption
- Tameka is now modeled as an anchored technician:
  - `75%` reserved at **Morgan State University**
  - `25%` external field capacity
  - external assignments limited to the configured nearby state set
- Step 05 now defaults to a **stakeholder** UI mode:
  - keeps the scenario coverage dots, hire markers, and technician bases front and center
  - exposes one quiet `Flight hubs` chip in the panel header for airport hubs
  - hides the heavier diagnostic map layers and floating legend boxes
  - supports `ELEVATE_MAP_UI_MODE=debug` for the old inspection-heavy layer set
- Step 05 now ships three stakeholder views:
  - `Optimized`
  - `Historical`
  - `Blank Slate`
  - and it surfaces provisional-solver + source-provenance warnings directly in the panel
- AVS is treated as Learning Space for dispatch skill logic.
- Normal scenarios keep the `new hires cannot serve HPS` rule.
- Blank Slate lifts that HPS restriction because it represents fully trained hypothetical hires.
- Step 08 now also applies a flight-only hub-connectivity penalty based on the origin airport tier.
- Step 09 uses the newer **patient-sim family mix** install-upside model.
- Step 09 is now **install-only** for primary upside. Service-contract profit is not in the main model.

## The Most Important Framing: Capacity and Utilization

The current Step 08 math is intentional and should not be described like a literal payroll-utilization model.

What the solver uses:
- appointment `duration_hours`
- technician `availability_fte`
- a `target_utilization` setting

What `duration_hours` means in this repo:
- It is the model's **calendar-window workload** input.
- It is not a pure hands-on labor field.
- It is not a clean timesheet or clock-in / clock-out field.

What the capacity side means:
- Technician capacity is normalized against that same demand pool.
- The model computes:
  - `hours_per_capacity_unit = total_demand_hours / (total_FTE × target_utilization)`
  - each tech capacity = `availability_fte × hours_per_capacity_unit`

How to interpret the legacy utilization fields:
- `utilization`
- `mean_existing_utilization`
- `max_existing_utilization`
- `scenario_tech_utilization.csv`

These should be read as:
- modeled load ratios
- optimization load metrics
- calendar-based capacity usage proxies

They should not be read as:
- exact payroll utilization
- exact weekday labor utilization
- exact Monday-through-Friday work-hour usage
- exact clocked labor usage

Why the repo uses this framing:
- travel matters in the real job
- appointment duration is not the same thing as pure labor
- some work can happen outside a simple weekday window
- the repo does not currently have a clean, defensible labor-time denominator

If better labor, travel, or timesheet data becomes available later, a separate operational utilization metric could be added. That is outside the scope of the current solver.

## Repository Structure

```text
elevate-territory-map/
  scripts/
    01_clean_data.py
    02_geocode.py
    03_match_install_base.py
    04_build_territories.py
    05_generate_map.py
    06_build_optimization_inputs.py
    08_optimize_locations.py
    09_analyze_scenarios.py
    11_build_full_cost_table.py
    install_mix_model.py
    optimization_utils.py
    config.py
  data/
    raw/
    processed/
      optimization/
  docs/
    index.html
  README.md
  CLAUDE.md
```

## Environment

- Dependencies are in `requirements.txt`
- Preferred interpreter on this machine:
  - `/opt/miniconda3/bin/python3`

## Source Inputs

### Map Pipeline Inputs

Expected in `data/raw/`:
- UIUC service appointments workbook
- service appointments report workbook
- install base workbook
- `technician_anchor_allocations.csv` for anchored / reserved-duty technicians

### Optimization Inputs

Configured in `scripts/config.py` and overridable with env vars:
- `EXTERNAL_APPOINTMENTS_XLSX` (`ELEVATE_APPTS_SOURCE`)
- `EXTERNAL_TECH_ROSTER_XLSX` (`ELEVATE_TECH_SOURCE`)

Do not commit sensitive external files.

## Pipeline Runbook

### Map Steps

```bash
/opt/miniconda3/bin/python3 scripts/01_clean_data.py
/opt/miniconda3/bin/python3 scripts/02_geocode.py
/opt/miniconda3/bin/python3 scripts/03_match_install_base.py
/opt/miniconda3/bin/python3 scripts/04_build_territories.py
/opt/miniconda3/bin/python3 scripts/05_generate_map.py
```

### Optimization Steps

```bash
/opt/miniconda3/bin/python3 scripts/06_build_optimization_inputs.py
/opt/miniconda3/bin/python3 scripts/11_build_full_cost_table.py
/opt/miniconda3/bin/python3 scripts/08_optimize_locations.py --min-new-hires 0 --max-new-hires 4 --max-hires-per-base 1 --time-limit-sec 600
/opt/miniconda3/bin/python3 scripts/09_analyze_scenarios.py
/opt/miniconda3/bin/python3 scripts/08_optimize_locations.py --blank-slate --min-new-hires 16 --max-new-hires 16 --max-hires-per-base 1 --time-limit-sec 600
/opt/miniconda3/bin/python3 scripts/05_generate_map.py
```

## Step-by-Step Logic

### Step 06

Builds:
- `tech_master.csv`
- `demand_appointments.csv`
- `candidate_bases.csv`
- `optimization_input_summary.json`

Important notes:
- `duration_hours` is carried forward as the model's calendar-window workload field
- AVS appointments are remapped to Learning Space skill requirements
- Canada and excluded territories stay out of optimization scope
- Step 06 prefers `data/processed/technicians.csv` as the current-state roster source when no external roster workbook is configured
- missing current-roster skill columns are derived from roster comments plus historical dispatch activity
- tech, demand, and candidate outputs now also carry airport-based operational-zone fields
- HTX contractor rows now carry per-tech scope, travel-policy, and zone-policy fields
- anchored-tech rows can also carry:
  - anchor-site metadata
  - reserved-duty FTE
  - external-field FTE
  - explicit allowed external states

### Step 11

Builds `full_cost_table.csv`.

Cost model:
- same-day drive: mileage only
- overnight drive: mileage or rental + 1 hotel night
- fly: BTS Q2 2025 fare × corporate premium + duration-scaled rental + duration-scaled hotel

Ground transport detail:
- `median_dist_mi` is the one-way median base-to-appointment distance inside each node
- drive trips below the `125` one-way threshold use mileage reimbursement
- drive trips at or above the threshold use rental-car economics
- rental days follow `trip_span_days = max(1, ceil(node_avg_days))`

The hotel scaling still uses the same calendar-window duration framing as the rest of the repo.

### Step 08

Runs the MILP scenarios.

Objective:
- travel cost
- out-of-region penalties
- operational-zone penalties
- hub-connectivity penalties on `fly` assignments
- hire payroll
- unmet penalties

Capacity setup:
- workload = appointment `duration_hours`
- total capacity is normalized using `availability_fte` and `target_utilization`
- resulting utilization outputs are modeled load ratios

Phase 1 operational realism now added on top of the same cost-first MILP shape:
- demand nodes carry an airport-based operational zone label/rank
- techs and candidates carry airport-based operational zone label/rank
- standard employees and new hires:
  - `0-1` zone jumps are free
  - `2` zone jumps stay feasible but carry a penalty
  - `3+` zone jumps are blocked
- HTX contractors:
  - remain in Step 08 as assignable capacity
  - are no longer Texas-only
  - use a softer zone rule
  - use a compressed-and-capped travel-cost proxy plus dispatch surcharge so they are flexible but not free
- Anchored technicians:
  - keep reserved duty out of the flexible field-capacity pool through `availability_fte`
  - can use explicit state-set scope rules for the remaining field work
  - do not require synthetic demand rows in Phase 1

### Step 09

Analyzes scenarios and writes the install-upside outputs.

Current Step 09 model:
- annualizes the scenario outputs
- computes freed existing-tech calendar time
- builds cleaned patient-sim install history from the raw workbook first
- separates:
  - cleaned historical mix
  - forward-looking mix
- maps product text to normalized families:
  - Apollo
  - Lucina
  - Juno
  - Ares
  - Aria
  - Evo
  - Luna
  - HPS
- excludes AVS / AVS ISO / Learning Space from patient-sim revenue modeling
- excludes obvious non-net-new install rows
- excludes HPS from the forward mix by default
- uses calendar-day install effort end to end
- applies per-family revenue, margin, and install-day assumptions
- keeps the primary upside view install-only

Important Step 09 framing:
- freed capacity is enabled install capacity, not guaranteed bookings
- primary upside excludes service-contract profit on purpose
- old conservative/moderate/aggressive install outputs are compatibility aliases only
- analysis outputs now also surface special-tech constraints such as anchored-duty assumptions

## Current Assumptions and Business Rules

### Core Optimization

- `DEFAULT_ANNUAL_HIRE_COST_USD = 146640.0`
- `DEFAULT_UNMET_PENALTY_USD = 5000.0`
- `DEFAULT_OUT_OF_REGION_PENALTY_USD = 0.0`
- new-hire concentration cap defaults to `1`
- current verified technician roster count is `16`
- total modeled FTE is `12.55`
- Hakim Mouazer remains visible on the map only with `availability_fte = 0.0`

### Skill Rules

- AVS = Learning Space for dispatch modeling
- existing techs need HPS skill for HPS nodes
- existing techs need LS skill for LS nodes
- normal scenarios: new hires cannot serve HPS nodes
- blank slate: new hires can serve HPS nodes

### Capacity / Load Rules

- `target_utilization` defaults to `0.85`
- this is a modeled load target, not a payroll target
- there is no literal `2,080-hour` denominator in the current solver

### Travel / Zone Rules

- `IRS_MILEAGE_RATE_USD_PER_MI = 0.60`
- `HOTEL_NIGHTLY_RATE_USD = 165.0`
- `RENTAL_CAR_DAILY_RATE_USD = 40.0`
- `PERSONAL_VEHICLE_MAX_ONE_WAY_MI = 125.0`
- drive/fly trip buckets stay:
  - `<100` same-day drive
  - `100-300` overnight drive
  - `>=300` fly
- airport-based operational zone buckets are the active Phase 1 time-zone realism proxy
- Arizona stays pinned to a fixed Mountain operational bucket for this rule
- hub-connectivity penalties apply only on `fly` assignments:
  - large hub: `$0`
  - medium hub: `$75`
  - small hub: `$150`

### Contractor Rules

- HTX contractors stay in the solve as real assignable coverage capacity
- default contractor scope is national, not Texas-only
- default contractor travel policy is `contractor_compressed_capped`
- default contractor cost multiplier = `0.65`
- default contractor cost cap = `$900`
- default contractor dispatch surcharge = `$125`
- default contractor zone policy = `contractor_soft`

### Install-Upside Rules

- `FREED_CAPACITY_UTILIZATION_FACTOR = 0.30`
- capacity model time unit for install upside = `calendar_days`
- patient-sim forward mix excludes HPS by default
- per-family install margin defaults to `0.40`
- per-family revenue values in config are provisional planning assumptions with source notes

## Current Repo Snapshot

From the current generated outputs:

- Data span: `2.078` years
- All `1,466` US appointments are served in every scenario
- Scenario range: `N=0..4`
- Lowest annualized modeled cost is currently `N=0`, but that row is still provisional because the solver hit the time limit before proving optimality
- `recommended_hire_locations.csv` still points to `Cleveland, OH` for the current `N=1` placement reference
- Current `N=4` placement set: `Washington, DC`, `Atlanta, GA`, `Bossier City, LA`, `Janesville, WI`
- N=0 annualized total cost: `$464,668.06`
- Mean existing-tech legacy utilization field at N=0: `0.8600`
- Max existing-tech legacy utilization field at N=0: `1.0000`
- Weighted average install effort: `1.52` calendar days
- Weighted average install revenue: `$47,277`
- Weighted average install profit per install: `$18,911`
- Blank Slate annualized total cost: `$2,480,867`
- Blank Slate annualized travel cost: `$134,627`
- Install history source: raw workbook merge from:
  - `report1770130594436`
  - `Derived Fields`

### Current Annualized Cost Table

| N | Annual Travel | Annual Time-Zone Penalty | Annual Hub Penalty | Annual Payroll | Annual Total |
|---|--------------:|-------------------------:|-------------------:|---------------:|-------------:|
| 0 | $462,863 | $0 | $1,805 | $0 | $464,668 |
| 1 | $390,603 | $0 | $36 | $146,640 | $537,279 |
| 2 | $340,376 | $0 | $36 | $293,280 | $633,692 |
| 3 | $297,084 | $0 | $36 | $439,920 | $737,040 |
| 4 | $255,809 | $361 | $36 | $586,560 | $842,766 |

### Current Annualized Install-Upside Table

| N | Install Units Enabled | Net Cost Increase | Net Economic Value | Break-Even Install Units |
|---|----------------------:|------------------:|-------------------:|-------------------------:|
| 0 | 0.0 | $0 | $0 | 0.0 |
| 1 | 25.0 | $72,611 | $399,421 | 3.8 |
| 2 | 40.6 | $169,024 | $597,894 | 8.9 |
| 3 | 53.8 | $272,372 | $745,232 | 14.4 |
| 4 | 55.0 | $378,098 | $661,992 | 20.0 |

## Important Output Files

- `data/processed/optimization/optimization_input_summary.json`
- `data/processed/optimization/model_assumptions.json`
- `data/processed/optimization/historical_roster.csv`
- `data/processed/optimization/full_cost_table.csv`
- `data/processed/optimization/scenario_summary.csv`
- `data/processed/optimization/scenario_summary_enhanced.csv`
- `data/processed/optimization/scenario_placements.csv`
- `data/processed/optimization/scenario_assignments_existing.csv`
- `data/processed/optimization/scenario_assignments_newhires.csv`
- `data/processed/optimization/scenario_contractor_usage.csv`
- `data/processed/optimization/scenario_tech_utilization.csv`
  Legacy filename; values are modeled load ratios.
- `data/processed/optimization/patient_sim_install_history_clean.csv`
- `data/processed/optimization/patient_sim_historical_mix_events.csv`
- `data/processed/optimization/patient_sim_historical_mix_units.csv`
- `data/processed/optimization/patient_sim_forward_mix.csv`
- `data/processed/optimization/patient_sim_family_economics_assumptions.csv`
- `data/processed/optimization/scenario_install_upside_by_family.csv`
- `data/processed/optimization/recommended_hire_locations.csv`
- `data/processed/optimization/analysis_report.json`
- `data/processed/optimization/analysis_report.md`
- `data/processed/optimization/blank_slate/`
- `docs/index.html`

## Known Caveats

1. Load-ratio outputs are modeled capacity proxies, not literal labor utilization.
2. Same-city trip bundling is not modeled.
3. Great-circle distance is used for trip classification.
4. Operational zones are broad airport-based buckets, not exact clock math.
5. HTX contractor economics use a compressed-and-capped proxy, not a literal reimbursement ledger.
6. The flight model is origin-based, not destination-specific.
7. Some scenario rows, including the current `N=0` baseline and Blank Slate, can remain provisional if the solver reaches the time limit before proving optimality.
8. AVS / Learning Space stays in dispatch demand but is excluded from patient-sim install revenue modeling.
9. Install upside is enabled capacity, not guaranteed sales.
10. Per-family install economics remain provisional until finance replaces them.

## If Starting a New Chat

State these first:

1. The current utilization fields are modeled load ratios, not literal timesheet utilization.
2. The workload basis is appointment `duration_hours` as calendar-window time.
3. Capacity is normalized against that same demand pool using `availability_fte` and `target_utilization`.
4. The active pipeline is **06 → 11 → 08 → 09 → 05**.
5. Step 08 now uses airport-based operational zone buckets as a realism layer.
6. Step 08 also adds a flight-only hub-connectivity penalty based on the origin airport tier.
7. HTX contractors are national flexible resources with compressed/capped travel-cost friction, not Texas-only and not free.
8. Step 09 uses the family-based patient-sim install-only upside model.
9. HPS is kept in historical reporting but excluded from the forward mix by default.
10. Old moderate/conservative/aggressive install fields still exist only as compatibility aliases.
