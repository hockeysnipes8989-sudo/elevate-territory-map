# Elevate Healthcare Territory Map

Interactive US/Canada service map plus a hiring and dispatch optimization pipeline for Elevate field service planning.

- Live map: https://hockeysnipes8989-sudo.github.io/elevate-territory-map/
- Repo: https://github.com/hockeysnipes8989-sudo/elevate-territory-map

## What This Repo Does

1. Cleans technician, appointment, and install-base data.
2. Builds map layers for assets, appointments, technicians, territories, and airports.
3. Builds optimization inputs and a trip-cost table.
4. Runs Step 08 hiring scenarios for `N=0..4` new hires.
5. Translates freed capacity into patient-simulator install upside.
6. Publishes the optimized, historical, and blank-slate views in `docs/index.html`.

Step 05 now builds a **stakeholder-first** map by default:
- visible by default: scenario coverage dots, selected-scenario hire markers, technician home bases
- visible control: one small `Flight hubs` chip in the panel header for airport hubs
- hidden from the stakeholder build: the heavier diagnostic layers and legend boxes
- optional internal debug view: set `ELEVATE_MAP_UI_MODE=debug` before running `scripts/05_generate_map.py`
- anchored tech sites such as **Morgan State University** can now be shown directly on the map

## Capacity / Utilization Framing

This repo keeps the current Step 08 capacity math.

What the current model does:
- The optimizer uses appointment `duration_hours` as **calendar-window workload**.
- That means the demand side reflects the appointment time window the model is covering, not pure hands-on labor time.
- Technician capacity is then normalized against that same demand pool using `availability_fte` and the Step 08 `target_utilization` setting.

How to interpret the output:
- The `utilization` fields in code and CSV outputs are **legacy field names**.
- They should be read as a **modeled load ratio** or **calendar-based capacity proxy**.
- They should **not** be read as exact payroll utilization, exact weekday work-hour utilization, or exact clocked labor usage.

Why the model is framed this way:
- Travel matters in this business.
- Appointment duration is not the same thing as pure wrench time.
- Some work can happen outside a simple Monday-Friday window.
- The repo does not currently have a clean timesheet-style labor denominator.

So the current framing is deliberate: it is a practical optimization load model, not a true labor-time model.

If better labor, travel, or timesheet data becomes available later, a separate operational utilization metric could be added without changing the current solver framing.

## Install

```bash
pip install -r requirements.txt
```

Recommended runtime in this environment:

```bash
/opt/miniconda3/bin/python3 ...
```

## Standard Map Pipeline

```bash
python scripts/01_clean_data.py
python scripts/02_geocode.py
python scripts/03_match_install_base.py
python scripts/04_build_territories.py
python scripts/05_generate_map.py
```

Source files for steps 1-4 are expected in `data/raw/`. Geocoding cache lives in `data/geocode_cache.json`.

## Optimization Pipeline

Outputs are written to `data/processed/optimization/`.

Pipeline order: **06 → 11 → 08 → 09 → 05**

```bash
python scripts/06_build_optimization_inputs.py
python scripts/11_build_full_cost_table.py
python scripts/08_optimize_locations.py --min-new-hires 0 --max-new-hires 4 --max-hires-per-base 1 --time-limit-sec 600
python scripts/09_analyze_scenarios.py
python scripts/08_optimize_locations.py --blank-slate --min-new-hires 16 --max-new-hires 16 --max-hires-per-base 1 --time-limit-sec 600
python scripts/05_generate_map.py
```

Step 11 uses a three-tier trip model:
- Same-day drive: `<100` miles
- Overnight drive: `100-300` miles
- Fly: `>=300` miles

Ground-transport detail inside the drive tiers:
- drive trips below `125` one-way median miles use personal-vehicle mileage
- drive trips at or above `125` one-way median miles switch to rental-car economics
- rental days follow the model's trip-span proxy, not a flat one-time rental fee

Special-tech inputs:
- `data/raw/technician_anchor_allocations.csv` stores anchor-site / reserved-duty metadata for technicians who are not fully general field resources

Step 06 current-roster sourcing:
- prefers an external roster workbook if one is configured
- otherwise uses `data/processed/technicians.csv` as the current-state roster source
- only falls back to the old `Service Appointments report - final.xlsx` roster sheet or cached `tech_master.csv` when needed
- derives missing skill flags from roster comments plus historical dispatch activity when the source roster does not carry explicit skill columns

## Current Model Rules

- Annual burdened planning cost per incremental new hire: `$146,640`
- Unmet demand penalty: `$5,000` per appointment
- Out-of-region soft penalty default: `$0`
- Mileage reimbursement: `$0.60` per mile
- Hotel proxy: `$165` per night
- Rental-car proxy: `$40` per day
- Canada excluded from optimization scope
- Current verified technician roster: `16`
- Total effective FTE in the current modeled roster: `12.55`
- Normal scenarios keep the `new hires cannot serve HPS` rule
- Blank Slate lifts that HPS restriction because it represents fully trained hypothetical hires
- AVS is treated as Learning Space for dispatch skill logic
- New-hire concentration cap defaults to `1` per base
- HTX contractors remain in Step 08 as real capacity
- HTX contractors are modeled as **national** flexible resources, not Texas-only resources
- HTX contractors use a compressed-and-capped travel-cost proxy plus a dispatch surcharge
- Standard employees and new hires are free at `0-1` operational zone jumps, penalized at `2`, and blocked at `3+`
- Contractors use a softer operational-zone rule: free at `0-1`, penalized at `2`, heavily penalized at `3+`
- Step 08 also adds a small airport-connectivity hub penalty on `fly` assignments from weaker origin airports
- Anchored technicians can reserve part of their FTE at a named site while limiting their remaining field work to an explicit nearby state set

## Step 08 Capacity Logic

Step 08 keeps a normalized capacity setup:

- `total_demand_hours = sum(duration_hours)`
- `hours_per_capacity_unit = total_demand_hours / (total_FTE × target_utilization)`
- each tech capacity = `availability_fte × hours_per_capacity_unit`
- legacy `utilization = assigned_hours / capacity_hours`

Important interpretation:
- `duration_hours` is a calendar-window demand measure
- `capacity_hours` is a normalized model capacity measure
- the resulting utilization value is a **modeled load ratio**

There is no literal fixed `2,080-hour` payroll-style denominator in the current solver.

## Step 08 Operational Realism Rules

Phase 1 added a small realism layer without changing the cost-first structure of the solver:

- Every tech base, candidate base, and demand node now gets an **operational zone** bucket from its airport anchor.
- The solver works with broad zone jumps like Eastern → Central, not precise wall-clock math.
- Arizona is kept in a fixed **Mountain** bucket so DST quirks do not create weird behavior.
- Step 08 now carries a separate `timezone_penalty_usd` term in the modeled cost output.
- HTX contractors stay in the solve, but they are no longer treated as Texas-only.
- Contractors are also **not** treated like free national resources. They still carry assignment friction.
- Tameka is now modeled as an anchored technician:
  - `75%` reserved at **Morgan State University**
  - `25%` available for external field work
  - external work limited to the configured nearby state set instead of broad national same-zone assignments

## Step 09 Patient-Sim Install Upside

Step 09 now uses a patient-simulator family mix model instead of generic low/mid/high install buckets.

Current approach:
- cleans historical patient-sim install history from the raw workbook first
- excludes AVS / AVS ISO / Learning Space from patient-sim install revenue modeling
- excludes obvious non-net-new rows such as returns, replacements, and reinstalls
- normalizes families such as:
  - `APP` / `APN` / generic Apollo → `Apollo`
  - `MFS` / `Lucina` → `Lucina`
  - `Juno`, `Ares`, `Aria`, `Evo`, `Luna`, `HPS`
- keeps **historical mix** separate from **forward mix**
- excludes `HPS` from the forward mix by default
- uses **calendar days end to end** for install effort and freed-capacity conversion
- uses per-family revenue, margin, and install-calendar-day assumptions
- keeps the primary upside view **install-only**

What it does not do:
- it does not add service-contract profit to the primary model
- it does not treat historical raw install-like rows as automatically equal to net-new future units

## Annualization

The current appointment dataset spans **2.078 years**.

- Step 06 computes `data_span_years`
- Step 08 scales hire cost to that same time span for the MILP
- Step 09 annualizes the outputs back to per-year figures

All scenario KPIs shown in the reports and map are annualized.

## Latest Repo Snapshot

From the current checked-in optimization outputs:

- Data span: **2.078 years**
- Scenario window: `N=0..4`
- All `1,466` US appointments are served in every scenario
- Lowest annualized modeled cost is currently **N=0**, but that baseline is still **provisional** because the solver hit the time limit before proving optimality
- `recommended_hire_locations.csv` still provides the current `N=1` placement reference: **Cleveland, OH**
- Current `N=4` placement set: **Washington, DC; Atlanta, GA; Bossier City, LA; Janesville, WI**
- N=0 annualized total cost: **$464,668**
- Mean existing-tech legacy utilization field at N=0: **0.8600**
- Max existing-tech legacy utilization field at N=0: **1.0000**
- Weighted average install effort: **1.52 calendar days**
- Weighted average install revenue: **$47,277**
- Weighted average install profit per install: **$18,911**
- History source for the install model: **raw workbook merge** (`report1770130594436` + `Derived Fields`)
- Blank Slate annualized total cost: **$2,480,867** across **16** placements
- Blank Slate annualized travel cost: **$134,627**

### Scenario Cost Summary

| N | Annual Travel | Annual Time-Zone Penalty | Annual Hub Penalty | Annual Payroll | Annual Total |
|---|--------------:|-------------------------:|-------------------:|---------------:|-------------:|
| 0 | $462,863 | $0 | $1,805 | $0 | $464,668 |
| 1 | $390,603 | $0 | $36 | $146,640 | $537,279 |
| 2 | $340,376 | $0 | $36 | $293,280 | $633,692 |
| 3 | $297,084 | $0 | $36 | $439,920 | $737,040 |
| 4 | $255,809 | $361 | $36 | $586,560 | $842,766 |

### Scenario Install-Upside Summary

| N | Install Units Enabled | Net Cost Increase | Net Economic Value | Break-Even Install Units |
|---|----------------------:|------------------:|-------------------:|-------------------------:|
| 0 | 0.0 | $0 | $0 | 0.0 |
| 1 | 25.0 | $72,611 | $399,421 | 3.8 |
| 2 | 40.6 | $169,024 | $597,894 | 8.9 |
| 3 | 53.8 | $272,372 | $745,232 | 14.4 |
| 4 | 55.0 | $378,098 | $661,992 | 20.0 |

## Key Caveats

- The current utilization outputs are modeled load ratios, not literal timesheet utilization.
- Travel and appointment duration are simplified into the current calendar-window workload model.
- Same-city trip bundling is not modeled.
- Great-circle distance is used for trip classification thresholds.
- Operational zone jumps are broad airport-based buckets, not exact time arithmetic.
- Contractor travel cost is a compressed-and-capped proxy, not a literal expense-reimbursement ledger.
- AVS / Learning Space stays in the dispatch demand model but is excluded from patient-sim install revenue modeling.
- Install upside represents **enabled capacity**, not guaranteed bookings.
- Per-family install economics are provisional planning assumptions until finance replaces them.

## Key Output Files

- `data/processed/optimization/optimization_input_summary.json`
- `data/processed/optimization/model_assumptions.json`
- `data/processed/optimization/tech_master.csv`
- `data/processed/optimization/demand_appointments.csv`
- `data/processed/optimization/candidate_bases.csv`
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

## Deeper Documentation

See `CLAUDE.md` for the fuller operating context, current assumptions, and rerun notes.
