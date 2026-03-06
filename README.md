# Elevate Healthcare - Interactive Service Territory Map

Interactive US/Canada map and optimization model for service coverage, travel cost, and hiring scenarios.

- Live map: https://hockeysnipes8989-sudo.github.io/elevate-territory-map/
- Repo: https://github.com/hockeysnipes8989-sudo/elevate-territory-map

## What This Includes

1. Map layers for active-contract assets, appointments, technicians, territories, and airports. Split-state territories use reference markers instead of invented full-state polygons.
2. Optimization scenario panel on the map for `N=0..4` new hires (all figures annualized).
3. End-to-end MILP pipeline for travel + hiring economics with automatic annualization.
4. BTS Q2 2025 fare lookup table with 1.6× corporate premium for flight costs.
5. AVS = Learning Space product mapping (all AVS appointments require LS-certified techs).
6. Revenue-from-freed-capacity analysis: patient sim only (ISO), uniform 40% margin across three system-size tiers.

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

If the external technician workbook is unavailable, Step 1 falls back to the tracked canonical roster at `data/processed/optimization/tech_master.csv` before using the legacy `Resources` sheet.

## Optimization Pipeline (Steps 6-11)

Outputs are written to `data/processed/optimization/`.

Pipeline order: **06 → 11 → 08 → 09 → 05** (Steps 07 and 10 are deprecated).

```bash
python scripts/06_build_optimization_inputs.py
python scripts/11_build_full_cost_table.py
python scripts/08_optimize_locations.py --min-new-hires 0 --max-new-hires 4 --max-hires-per-base 1 --time-limit-sec 600
python scripts/09_analyze_scenarios.py
python scripts/05_generate_map.py
```

Step 11 uses a three-tier distance-based trip model per (tech/candidate, node) pair:
- **Same-day drive** (<100 mi): IRS mileage ($0.70/mi × round-trip), no hotel, no rental
- **Overnight drive** (100–300 mi): IRS mileage + 1 hotel night ($159), no rental
- **Fly** (≥300 mi): BTS Q2 2025 fare × 1.6 corporate premium + duration-scaled hotel ($159/night) + rental car ($235)

Default external workbook paths are in `scripts/config.py` (overridable via env vars
`ELEVATE_APPTS_SOURCE`, `ELEVATE_TECH_SOURCE`):

- `EXTERNAL_APPOINTMENTS_XLSX`
- `EXTERNAL_TECH_ROSTER_XLSX`

## Annualization

The appointment dataset spans **2.08 years** (Jan 2, 2024 → Jan 29, 2026, 759 days, 1,466 US appointments). The pipeline automatically detects this and annualizes all output figures:

- **Step 06** computes `data_span_years` (2.0780) from the appointment date range and writes it to `optimization_input_summary.json`.
- **Step 08** reads `data_span_years` and scales hire cost to match the data period ($146,640/yr × 2.0780 = $304,718) so the MILP compares travel and hire costs over the same time span.
- **Step 09** divides all period-total costs (travel, hire, overhead) and freed capacity hours by `data_span_years` to produce annual equivalents. All figures in reports and the map are per-year.

This means the MILP solution quality is fully preserved (same solver, same appointments, same optimality) while reported numbers accurately represent one year of operations.

## Current Model Rules (Important)

- Annual burdened planning cost per incremental new hire: `$146,640` (scaled to `$304,718` in MILP to match the 2.08-year data period).
- Unmet demand penalty: `$5,000` per appointment (`DEFAULT_UNMET_PENALTY_USD`).
- Out-of-region soft penalty default: `$0.0` (disabled by default).
- Canada excluded from optimization scope. Hakim Mouazer (Montreal) at `availability_fte=0.0` (visible on map only).
- James Sanchez: `availability_fte=1.0` (full field tech; temporarily on phones due to injury but model targets ideal state).
- Damion Lyn: `availability_fte=0.20` (repair center hybrid, 20-25% field when fully staffed).
- Elier Martin: `availability_fte=0.10` (phone tech, ~10% field).
- **AVS = Learning Space (LS):** All AVS appointments reclassified as LS-requiring. 319 LS appointments require LS-certified techs (7 of 16).
- Canceled/voided overhead excluded (`$0`) — Navan data covers only ~2/16 techs. Fixed cost, does not affect scenario comparison.
- Contractor scope defaults to `texas_only` unless explicitly overridden.
- New-hire concentration cap defaults to `1` per base (`--max-hires-per-base 1`).
- Current verified technician roster is 16 total (including both HTX contractors). Total effective FTE: 12.55.
- Technician markers are grouped by shared base location (popup lists all names at that base).
- **New hires cannot serve HPS nodes** — policy constraint (hard variable bound in MILP). 114 HPS appointments are served by existing certified techs. Shannon/Isabelle confirmed new hires would NOT be HPS-trained (product line discontinuing).

## Flight Cost Model

Flight cost = `BTS_RAW_ITINERARY_FARES[origin_airport] × 1.6`. Origin-only — cost does not vary by destination.

- 63 US airports with direct BTS Q2 2025 itinerary fares in `config.py`.
- Airports not in the table fall back to `BTS_NATIONAL_FALLBACK` ($386).
- The 1.6× corporate premium is calibrated from Navan median actual cost ($633) / BTS average domestic fare ($386).
- Mean flight cost: $633. Range: $455–$783.

Steps 07 and 10 (ML flight cost model and BTS correction layer) were removed — replaced by the BTS Q2 2025 lookup table in Step 11.

## Scenario Cost Formula

Step 8 computes (over the full 2.08-year data period):

- `modeled_total_cost_usd = travel_cost_usd + out_of_region_penalty_usd + hire_cost_usd + unmet_penalty_usd`
- `economic_total_with_overhead_usd = modeled_total_cost_usd + baseline_canceled_voided_usd`

Where:

- `hire_cost_usd = (number of incremental new hires) × $304,718` (annual $146,640 × 2.0780 years)
- `baseline_canceled_voided_usd = $0` (excluded — incomplete Navan data coverage)

Step 9 then divides all costs by `data_span_years` (2.0780) to produce annual equivalents shown in reports and the map.

## Latest Baseline Snapshot (Annualized — Current Repo Outputs)

From the latest optimization artifacts (BTS Q2 2025 lookup + three-tier cost model + annualization + AVS=LS + Shannon/Isabelle FTE updates):

- Data span: **2.0780 years** (Jan 2, 2024 → Jan 29, 2026, 759 days)
- Annualized appointment count: ~706/year (1,466 US total)
- Skill breakdown: 114 HPS, 319 LS (including AVS→LS remaps), 1,033 regular
- Scenario window: `N=0..4`
- Best scenario: `N=1` under the repo's proven-optimal selection rule (`N=0` was cheapest observed but did not prove optimal before the time limit)
- N=0 annualized total: **$556,549** (travel only, overhead excluded)
- Hard cap result: no scenario allocates more than 1 hire to the same base
- All 1,466 appointments served across all scenarios (zero unmet)
- Active techs at N=0: 16. Total effective FTE: 12.55
- Trip type split: 1.8% same-day drive, 7.1% overnight drive, 91.1% fly
- Target utilization: 85% (demand-normalized capacity model)
- Mean utilization at N=0: 85.1%. Max: 100.00%.

### Scenario Results (Annualized)

| N | Annual Travel | Annual Payroll | Annual Total |
|---|--------------|----------------|-------------|
| 0 | $556,549 | $0 | **$556,549** |
| 1 | $480,342 | $146,640 | $626,982 |
| 2 | $423,357 | $293,280 | $716,637 |
| 3 | $375,018 | $439,920 | $814,938 |
| 4 | $329,658 | $586,560 | $916,218 |

All scenarios cost more than N=0. Marginal annual travel savings diminish with each additional hire.

### Revenue-from-Freed-Capacity (Annualized Profit Analysis)

| N | Installs/yr | Net Cost Increase | Net Value (Conservative) | Net Value (Moderate) | Net Value (Aggressive) | Break-Even (Mod) |
|---|------------:|------------------:|-------------------------:|---------------------:|-----------------------:|-----------------:|
| 0 | 0.0 | $0 | $0 | $0 | $0 | 0.0 |
| 1 | 16.9 | $70,433 | $351,010 | $824,921 | $1,705,043 | 1.3 |
| 2 | 33.9 | $160,088 | $682,907 | $1,630,854 | $3,391,327 | 3.0 |
| 3 | 50.7 | $258,389 | $1,002,808 | $2,421,021 | $5,054,845 | 4.9 |
| 4 | 67.6 | $359,669 | $1,322,694 | $3,214,508 | $6,727,876 | 6.8 |

Revenue assumptions: $50K/$120K/$250K per patient sim install × uniform 40% margin + $7K×70% annual service contract per system. ISO installations only — Learning Space (AVS/LS) excluded per Shannon. Profit margins applied — figures represent P&L impact. Utilization factor 30% (conservative — accounts for scheduling friction, admin overhead, sales pipeline constraints, ramp-up time).

### Hiring Recommendations by Scenario

| N | Recommended Bases |
|---|-------------------|
| 1 | CLE (Cleveland, OH) |
| 2 | CLE, Janesville WI |
| 3 | ORD (Chicago, IL), CLE, Fort Smith AR |
| 4 | ATL (Atlanta, GA), ORD, CLE, Fort Smith AR |

## Key Caveats

- Three-tier drive model: same-day (<100 mi, no hotel), overnight (100–300 mi, 1 hotel night), fly (≥300 mi, duration-scaled hotel). Trip bundling is not modeled.
- Great-circle distance (not road distance) for trip classification thresholds.
- Territory polygons are only drawn for unique-state coverage. Split-state territories (for example FL/TX/CA/NY/NJ/PA splits) are shown with reference markers rather than invented full-state boundaries.
- No seasonality — all appointments treated equivalently regardless of time of year.
- New hires cannot serve HPS nodes (policy constraint). 114 HPS appointments are served by existing certified techs. HPS product line discontinuing.
- AVS = Learning Space mapping reclassifies ~315 appointments as LS-requiring, constraining dispatch to LS-certified techs (7 of 16).
- BTS fares are Q2 2025 data. A few airports (SHV, BIL, BIS, FAR, ANC) use the national fallback ($386) where specific BTS data was unavailable.
- Flight cost is origin-only (no destination dependency) — a simplification justified by BTS itinerary-level averages varying primarily by origin market.
- Annualization assumes uniform distribution of costs across the 2.08-year data period.
- Revenue figures represent capacity enabled, not guaranteed bookings. Uniform 40% margin confirmed by Shannon Drew (VP Service).
- Revenue model covers patient simulator installations only (ISO). Learning Space (AVS/LS) excluded — install values too variable ($6.5K–$106K).

## Key Output Files

- `data/processed/optimization/optimization_input_summary.json` — includes `data_span_years`
- `data/processed/optimization/model_assumptions.json` — includes `data_span_years`, `hire_cost_for_optimization_period`
- `data/processed/optimization/tech_master.csv`
- `data/processed/optimization/demand_appointments.csv`
- `data/processed/optimization/candidate_bases.csv`
- `data/processed/optimization/full_cost_table.csv` — per-(tech/candidate, node) drive/fly cost table (10,961 rows)
- `data/processed/optimization/scenario_summary.csv` — raw MILP output (full-period costs)
- `data/processed/optimization/scenario_summary_enhanced.csv` — annualized with revenue analysis
- `data/processed/optimization/scenario_placements.csv`
- `data/processed/optimization/scenario_assignments_existing.csv`
- `data/processed/optimization/scenario_assignments_newhires.csv`
- `data/processed/optimization/scenario_tech_utilization.csv`
- `data/processed/optimization/recommended_hire_locations.csv`
- `data/processed/optimization/analysis_report.json`
- `data/processed/optimization/analysis_report.md`
- `docs/index.html`

## Deeper Documentation

For full context (business rules, caveats, and runbook details), see `CLAUDE.md`.
