# Elevate Healthcare — Interactive Service Territory Map

Interactive map showing service territories, active contract simulators, service appointments, and technician home bases across the US and Canada.

**[View the live map](https://patricklipinski.github.io/elevate-territory-map/)**

## Map Layers (toggleable)

1. **Active Contract Simulators** — Territory-level choropleth colored by number of active assets (3,087 assets across 19 territories), with point-level markers for matched accounts
2. **Service Appointments** — 1,480 clustered markers color-coded by service type (PM/Repair/Install)
3. **Technician Home Bases** — 16 markers (green=active, orange=special)
4. **Territory Boundaries** — 19 semi-transparent polygons computed via convex hull
5. **Simulation Panel (N=0..4)** — scenario buttons, KPI cards, and star-marked recommended hire locations (requires optimization outputs)

## Setup

```bash
pip install -r requirements.txt
```

## Pipeline

```bash
python scripts/01_clean_data.py          # Load Excel, clean data
python scripts/02_geocode.py             # Geocode ~500 city/state pairs (~8 min first run)
python scripts/03_match_install_base.py  # Match install base accounts to locations
python scripts/04_build_territories.py   # Build territory boundary polygons
python scripts/05_generate_map.py        # Generate docs/index.html
```

Place source Excel files in `data/raw/` before running. Geocode results are cached in `data/geocode_cache.json`.

## Optimization Pipeline (MILP)

Optimization runs from external source-of-truth files and writes outputs to
`data/processed/optimization/`.

```bash
python scripts/06_build_optimization_inputs.py
python scripts/07_build_travel_cost_model.py
python scripts/08_optimize_locations.py --min-new-hires 0 --max-new-hires 4
python scripts/09_analyze_scenarios.py
python scripts/05_generate_map.py          # Rebuild docs/index.html with simulation panel
```

Default external workbook paths are configured in `scripts/config.py`:

- `EXTERNAL_APPOINTMENTS_XLSX` (service appointments with Description field)
- `EXTERNAL_TECH_ROSTER_XLSX` (technician source-of-truth skills and locations)
- `EXTERNAL_NAVAN_XLSX` (Navan travel data)

### Burdened Hire Cost Assumption

- Default annual burdened company planning cost per new field-tech hire:
  `DEFAULT_ANNUAL_HIRE_COST_USD = 146640.0`
- This value is applied to **incremental new hires only** in Step 8 scenario economics.
- Default out-of-region soft penalty is `0.0` (`DEFAULT_OUT_OF_REGION_PENALTY_USD`) so
  baseline optimization reflects direct travel costs only.
- Override on demand:

```bash
python scripts/08_optimize_locations.py --min-new-hires 0 --max-new-hires 4 --annual-hire-cost-usd 146640 --out-of-region-penalty 0
```

- This is a company planning burden input, not technician take-home pay.

Key optimization outputs:

- `data/processed/optimization/tech_master.csv`
- `data/processed/optimization/demand_appointments.csv`
- `data/processed/optimization/candidate_bases.csv`
- `data/processed/optimization/travel_cost_matrix.csv`
- `data/processed/optimization/scenario_summary.csv`
- `data/processed/optimization/recommended_hire_locations.csv`

## Data Sources

- Service Appointments (1,480 records with city/state)
- Install Base (8,528 assets, 3,087 with active contracts)
- Technician Resources (16 active/special technicians with home bases)
