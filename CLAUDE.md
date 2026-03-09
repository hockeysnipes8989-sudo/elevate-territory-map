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
- AVS is treated as Learning Space for dispatch skill logic.
- New hires cannot serve HPS nodes.
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

### Step 11

Builds `full_cost_table.csv`.

Cost model:
- same-day drive: mileage only
- overnight drive: mileage + 1 hotel night
- fly: BTS Q2 2025 fare × corporate premium + rental + duration-scaled hotel

The hotel scaling still uses the same calendar-window duration framing as the rest of the repo.

### Step 08

Runs the MILP scenarios.

Objective:
- travel cost
- out-of-region penalties
- hire payroll
- unmet penalties

Capacity setup:
- workload = appointment `duration_hours`
- total capacity is normalized using `availability_fte` and `target_utilization`
- resulting utilization outputs are modeled load ratios

The solver math should stay unchanged unless a true bug is found.

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
- new hires cannot serve HPS nodes

### Capacity / Load Rules

- `target_utilization` defaults to `0.85`
- this is a modeled load target, not a payroll target
- there is no literal `2,080-hour` denominator in the current solver

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
- Best scenario under the repo's proven-optimal rule: `N=1`
- N=0 annualized total cost: `$556,548.61`
- Mean existing-tech legacy utilization field at N=0: `0.8505`
- Max existing-tech legacy utilization field at N=0: `0.99999`
- Weighted average install effort: `1.52` calendar days
- Weighted average install revenue: `$47,277`
- Weighted average install profit per install: `$18,911`
- Install history source: raw workbook merge from:
  - `report1770130594436`
  - `Derived Fields`

### Current Annualized Cost Table

| N | Annual Travel | Annual Payroll | Annual Total |
|---|--------------:|---------------:|-------------:|
| 0 | $556,549 | $0 | $556,549 |
| 1 | $480,342 | $146,640 | $626,982 |
| 2 | $423,357 | $293,280 | $716,637 |
| 3 | $375,018 | $439,920 | $814,938 |
| 4 | $329,658 | $586,560 | $916,218 |

### Current Annualized Install-Upside Table

| N | Install Units Enabled | Net Cost Increase | Net Economic Value | Break-Even Install Units |
|---|----------------------:|------------------:|-------------------:|-------------------------:|
| 0 | 0.0 | $0 | $0 | 0.0 |
| 1 | 34.5 | $70,433 | $582,336 | 3.7 |
| 2 | 69.0 | $160,088 | $1,145,620 | 8.5 |
| 3 | 103.3 | $258,389 | $1,695,067 | 13.7 |
| 4 | 137.8 | $359,669 | $2,246,128 | 19.0 |

## Important Output Files

- `data/processed/optimization/optimization_input_summary.json`
- `data/processed/optimization/model_assumptions.json`
- `data/processed/optimization/full_cost_table.csv`
- `data/processed/optimization/scenario_summary.csv`
- `data/processed/optimization/scenario_summary_enhanced.csv`
- `data/processed/optimization/scenario_placements.csv`
- `data/processed/optimization/scenario_assignments_existing.csv`
- `data/processed/optimization/scenario_assignments_newhires.csv`
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
- `docs/index.html`

## Known Caveats

1. Load-ratio outputs are modeled capacity proxies, not literal labor utilization.
2. Same-city trip bundling is not modeled.
3. Great-circle distance is used for trip classification.
4. The flight model is origin-based, not destination-specific.
5. AVS / Learning Space stays in dispatch demand but is excluded from patient-sim install revenue modeling.
6. Install upside is enabled capacity, not guaranteed sales.
7. Per-family install economics remain provisional until finance replaces them.

## If Starting a New Chat

State these first:

1. The current utilization fields are modeled load ratios, not literal timesheet utilization.
2. The workload basis is appointment `duration_hours` as calendar-window time.
3. Capacity is normalized against that same demand pool using `availability_fte` and `target_utilization`.
4. The active pipeline is **06 → 11 → 08 → 09 → 05**.
5. Step 09 uses the family-based patient-sim install-only upside model.
6. HPS is kept in historical reporting but excluded from the forward mix by default.
7. Old moderate/conservative/aggressive install fields still exist only as compatibility aliases.
