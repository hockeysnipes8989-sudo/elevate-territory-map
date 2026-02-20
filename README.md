# Elevate Healthcare — Interactive Service Territory Map

Interactive map showing service territories, active contract simulators, service appointments, and technician home bases across the US and Canada.

**[View the live map](https://patricklipinski.github.io/elevate-territory-map/)**

## Map Layers (toggleable)

1. **Active Contract Simulators** — Territory-level choropleth colored by number of active assets (3,087 assets across 19 territories), with point-level markers for matched accounts
2. **Service Appointments** — 1,480 clustered markers color-coded by service type (PM/Repair/Install)
3. **Technician Home Bases** — 16 markers (green=active, gray=former, orange=special)
4. **Territory Boundaries** — 19 semi-transparent polygons computed via convex hull

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

## Data Sources

- Service Appointments (1,480 records with city/state)
- Install Base (8,528 assets, 3,087 with active contracts)
- Technician Resources (16 technicians with home bases)
