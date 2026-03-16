"""Configuration for Elevate Healthcare Territory Map pipeline."""
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
REFERENCE_DIR = os.path.join(PROJECT_ROOT, "data", "reference")
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")
OPTIMIZATION_DIR = os.path.join(PROCESSED_DIR, "optimization")

# Optimization defaults
# Annual burdened company planning cost per new field-tech hire (not take-home wages).
DEFAULT_ANNUAL_HIRE_COST_USD = 146640.0
# Soft penalty applied when assigning work outside a tech's home state.
# Keep at 0.0 for pure travel-cost-first optimization.
DEFAULT_OUT_OF_REGION_PENALTY_USD = 0.0
# Verified current field-tech roster count (includes HTX contractors).
EXPECTED_CURRENT_TECH_COUNT = 16

# Source Excel files
SERVICE_APPTS_DISPATCH = os.path.join(
    RAW_DIR, "UIUC service appointments with dipsatch date - final.xlsx"
)
SERVICE_APPTS_REPORT = os.path.join(
    RAW_DIR, "Service Appointments report - final.xlsx"
)
INSTALL_BASE = os.path.join(
    RAW_DIR, "Install Base (all patient sim assets) - final.xlsx"
)
TECHNICIAN_ANCHOR_ALLOCATIONS_CSV = os.path.join(
    RAW_DIR, "technician_anchor_allocations.csv"
)

# External source-of-truth workbooks used by optimization scripts.
# Set env vars (ELEVATE_APPTS_SOURCE, ELEVATE_TECH_SOURCE) to override the
# machine-specific fallback paths below.
EXTERNAL_APPOINTMENTS_XLSX = os.environ.get(
    "ELEVATE_APPTS_SOURCE",
    "/Users/patricklipinski/Desktop/opus 4.6 final excel sheets/"
    "UIUC service appointments with dipsatch date - final.xlsx",
)
EXTERNAL_TECH_ROSTER_XLSX = os.environ.get(
    "ELEVATE_TECH_SOURCE",
    "/Users/patricklipinski/Downloads/Tech location and product experience (1).xlsx",
)

# Processed outputs
CLEAN_APPTS_CSV = os.path.join(PROCESSED_DIR, "service_appointments_clean.csv")
CLEAN_TECHS_CSV = os.path.join(PROCESSED_DIR, "technicians.csv")
CLEAN_INSTALL_CSV = os.path.join(PROCESSED_DIR, "install_base_clean.csv")
GEOCODE_CACHE = os.path.join(PROJECT_ROOT, "data", "geocode_cache.json")
GEOCODED_APPTS_CSV = os.path.join(PROCESSED_DIR, "appointments_geocoded.csv")
GEOCODED_TECHS_CSV = os.path.join(PROCESSED_DIR, "technicians_geocoded.csv")
INSTALL_MATCHED_CSV = os.path.join(PROCESSED_DIR, "install_base_matched.csv")
INSTALL_ALL_MATCHED_CSV = os.path.join(PROCESSED_DIR, "install_base_all_matched.csv")
INSTALL_NONACTIVE_MATCHED_CSV = os.path.join(PROCESSED_DIR, "install_base_nonactive_matched.csv")
TERRITORY_SUMMARY_CSV = os.path.join(PROCESSED_DIR, "territory_summary.csv")
TERRITORIES_GEOJSON = os.path.join(PROCESSED_DIR, "territories.geojson")
MAP_OUTPUT = os.path.join(DOCS_DIR, "index.html")
FAA_HUB_CLASS_REFERENCE_CSV = os.path.join(
    REFERENCE_DIR, "faa_cy2024_airport_hub_classes.csv"
)
FAA_HUB_SOURCE_YEAR = 2024
FAA_HUB_SOURCE_URL = (
    "https://www.faa.gov/airports/planning_capacity/passenger_allcargo_stats/"
    "passenger/arp-cy2024-commercial-service-enplanements.xlsx"
)

# ---------------------------------------------------------------------------
# Excel sheet names
# ---------------------------------------------------------------------------
APPTS_RAW_SHEET = "report1770130594436"
APPTS_DISPATCH_SHEET = "Derived Fields"
APPTS_REPORT_RESOURCES_SHEET = "Resources"
INSTALL_BASE_SHEET = "report1769446081737"

# ---------------------------------------------------------------------------
# Technician name mapping (service appointment names → canonical names)
# ---------------------------------------------------------------------------
TECH_NAME_MAP = {
    "Clarence Bonner, Jr": "Clarence Bonner",
    "[S] Alex Rondero": "Alex Rondero",
    "Elier Alvarez Martin": "Elier Martin",
}

LOCATION_AIRPORT_OVERRIDES = {
    "charlotte nc": "CLT",
    "st louis mo": "STL",
    "st louis, mo": "STL",
    "ontario ca": "ONT",
    "tampa fl": "TPA",
    "phoenix az": "PHX",
    "philadelphia pa": "PHL",
    "baltimore md": "BWI",
    "houston tx": "IAH",
    "sarasota fl": "TPA",
    "sarasota, fl": "TPA",
}

# Historical view: show actual past assignments alongside MILP scenarios.
HISTORICAL_VIEW_ENABLED = True

# Historical base overrides are only used when the archived Resources sheet is
# incomplete or only gives a region-level location.
HISTORICAL_BASE_MANUAL_OVERRIDES = {}

HISTORICAL_BASE_REGION_CANDIDATES = {
    "Alberta, Canada": [
        {
            "city": "Calgary",
            "state": "AB",
            "country": "Canada",
            "airport_iata": "YYC",
        },
        {
            "city": "Edmonton",
            "state": "AB",
            "country": "Canada",
            "airport_iata": "YEG",
        },
    ],
}

# Former technicians to exclude from current-state maps/outputs
INACTIVE_TECH_NAMES = {
    "David Bazany",
    "John Aleksa",
    "Trent Osborne",
    "Alex Rondero",
}

# Technician name aliases (canonical form lookup used in optimization pipeline).
# Normalized via optimization_utils.normalize_name() before lookup.
TECH_NAME_ALIASES = {
    "rob cohen": "Robert Cohen",
    "blad torres": "Bladimir Torres",
    "damion": "Damion Lyn",
    "elier": "Elier Martin",
    "josh brown": "Josh Brown",
}

# Optimization cost defaults
DEFAULT_UNMET_PENALTY_USD = 5000.0

# ---------------------------------------------------------------------------
# BTS Q2 2025 average domestic itinerary fares (raw, per round-trip).
# Source: Bureau of Transportation Statistics, Table 1 — Average Domestic
# Itinerary Fares By Origin Airport.
# 58 airports have real Q2 2025 data; 5 (ANC, BIL, BIS, FAR, SHV) use the
# $386 national fallback (not tracked by BTS or stale data).
# Multiply by CORPORATE_TRAVEL_PREMIUM at runtime for final flight cost.
# ---------------------------------------------------------------------------
BTS_RAW_ITINERARY_FARES = {
    "ABQ": 412.80, "ANC": 386.00, "ATL": 418.60, "AUS": 393.72,
    "BHM": 489.07, "BIL": 386.00, "BIS": 386.00, "BNA": 363.95,
    "BOI": 391.38, "BOS": 377.15, "BUF": 331.26, "BWI": 364.51,
    "CLE": 370.08, "CLT": 439.81, "CMH": 393.45, "CVG": 364.95,
    "DCA": 374.49, "DEN": 363.47, "DFW": 418.34, "DSM": 380.88,
    "DTW": 419.50, "EWR": 427.44, "FAR": 386.00, "IAD": 462.97,
    "IAH": 422.67, "ICT": 430.40, "IND": 378.72, "JAX": 410.39,
    "JFK": 431.52, "LAS": 284.74, "LAX": 413.68, "LGA": 358.83,
    "LIT": 457.83, "MCI": 423.86, "MCO": 284.54, "MDW": 310.93,
    "MEM": 402.24, "MIA": 346.50, "MKE": 414.72, "MSP": 401.09,
    "MSY": 333.69, "OKC": 447.30, "OMA": 434.91, "ONT": 375.80,
    "ORD": 371.91, "PDX": 383.85, "PHL": 417.63, "PHX": 378.81,
    "PIT": 367.10, "RDU": 355.40, "SAN": 375.91, "SAT": 407.61,
    "SEA": 400.25, "SFO": 445.13, "SHV": 386.00, "SJC": 341.37,
    "SLC": 450.03, "SMF": 395.67, "STL": 412.71, "TPA": 331.55,
    "TUL": 435.94, "TUS": 434.60,
}
BTS_NATIONAL_FALLBACK = 386.00       # Q2 2025 US domestic average itinerary fare
CORPORATE_TRAVEL_PREMIUM = 1.6       # Navan median $633 / BTS avg $386

# Canceled/voided booking overhead. Set to $0 because Navan export covers only
# ~2 of 16 technicians — the observed $35,632 is not representative of the full
# fleet. This is a fixed cost that does not vary by scenario and does not affect
# the relative comparison between hiring levels.
BASELINE_CANCELED_VOIDED_USD = 0.0

IRS_MILEAGE_RATE_USD_PER_MI = 0.60   # Updated planning assumption from stakeholder review
RENTAL_CAR_DAILY_RATE_USD = 40.0     # Updated planning assumption: approximate daily rental rate
HOTEL_NIGHTLY_RATE_USD = 165.0       # Updated planning assumption from stakeholder review
HOTEL_AVG_NIGHTS = 2.5                # Navan average stay length (fly trip duration fallback)
PERSONAL_VEHICLE_MAX_ONE_WAY_MI = 125.0  # Applies to Step 11 median_dist_mi (one-way)

# Three-tier trip cost model (Step 11):
#   < 100 mi:  same-day drive — mileage only, no hotel, no rental
#   100–300 mi: overnight drive — mileage + 1 hotel night, no rental
#   >= 300 mi:  fly — flight + duration-scaled hotel + rental car
SAME_DAY_DRIVE_THRESHOLD_MI = 100.0
OVERNIGHT_DRIVE_THRESHOLD_MI = 300.0

# Legacy reference (fly-trip average only — use HOTEL_NIGHTLY_RATE_USD × nights instead)
HOTEL_AVG_USD = round(HOTEL_NIGHTLY_RATE_USD * HOTEL_AVG_NIGHTS)  # $398 ≈ $399
RENTAL_CAR_AVG_USD = round(RENTAL_CAR_DAILY_RATE_USD * HOTEL_AVG_NIGHTS)

# Operational zone policy (Phase 1 realism upgrade).
# These are intentionally broad operational buckets, not literal DST-aware
# wall-clock offsets. The current solver works at strategic assignment level.
OPERATIONAL_ZONE_DEFINITIONS = {
    "Eastern": {"rank": 0, "utc_offset_standard": -5},
    "Central": {"rank": 1, "utc_offset_standard": -6},
    "Mountain": {"rank": 2, "utc_offset_standard": -7},
    "Pacific": {"rank": 3, "utc_offset_standard": -8},
    "Alaska": {"rank": 4, "utc_offset_standard": -9},
    "Hawaii": {"rank": 5, "utc_offset_standard": -10},
}

ZONE_POLICY_STANDARD = "standard_hard_3plus"
ZONE_POLICY_CONTRACTOR_SOFT = "contractor_soft"
EMPLOYEE_TWO_ZONE_JUMP_PENALTY_USD = 450.0
CONTRACTOR_TWO_ZONE_JUMP_PENALTY_USD = 250.0
CONTRACTOR_THREE_PLUS_ZONE_JUMP_PENALTY_USD = 1500.0

# Hub connectivity penalty: surcharge per appointment for weak origin-airport
# connectivity. Applied only to fly trips in Step 08.
HUB_TIER_LARGE = "large_hub"
HUB_TIER_MEDIUM = "medium_hub"
HUB_TIER_SMALL = "small_hub"
HUB_TIER_NONHUB = "nonhub"
HUB_TIER_UNKNOWN = "unknown"
HUB_PENALTY_LARGE = 0.0
HUB_PENALTY_MEDIUM = 150.0
HUB_PENALTY_SMALL = 300.0
HUB_PENALTY_NONHUB = 450.0
HUB_PENALTY_UNKNOWN = HUB_PENALTY_MEDIUM
HUB_PENALTY_MAP = {
    HUB_TIER_LARGE: HUB_PENALTY_LARGE,
    HUB_TIER_MEDIUM: HUB_PENALTY_MEDIUM,
    HUB_TIER_SMALL: HUB_PENALTY_SMALL,
    HUB_TIER_NONHUB: HUB_PENALTY_NONHUB,
}

ASSIGNMENT_SCOPE_MODE_NATIONAL = "national"
ASSIGNMENT_SCOPE_MODE_STATE_LIMITED = "state_limited"
ASSIGNMENT_SCOPE_MODE_STATE_SET_LIMITED = "state_set_limited"
TRAVEL_COST_POLICY_EMPLOYEE = "employee_standard"
TRAVEL_COST_POLICY_CONTRACTOR = "contractor_compressed_capped"
CONTRACTOR_COST_MULTIPLIER = 0.65
CONTRACTOR_COST_CAP_USD = 900.0
CONTRACTOR_DISPATCH_SURCHARGE_USD = 125.0

# Realistic installation estimate parameters (Step 09 capacity-freed analysis).
# The freed time here comes from the repo's calendar-window demand model, so it
# should be read as modeled capacity available, not literal payroll-utilization
# time that can be re-booked one-for-one.
# 30% of freed calendar time realistically converts to installations.
# Conservative estimate accounting for: scheduling friction, non-installation
# tasks (phone coverage, training, admin), sales pipeline constraints
# (installations require customers who have already purchased), ramp-up time,
# and general real-world overhead. Previous value was 0.75.
FREED_CAPACITY_UTILIZATION_FACTOR = 0.30
PATIENT_SIM_CAPACITY_TIME_UNIT = "calendar_days"

# Patient simulator install upside model (Step 09).
# These are provisional planning assumptions and should stay centralized so the
# scenario outputs remain easy to audit and update later.
PATIENT_SIM_EXCLUDE_HPS_FROM_FUTURE_MIX = True
PATIENT_SIM_RECENCY_WEIGHTING_ENABLED = False
PATIENT_SIM_RECENCY_HALFLIFE_YEARS = 1.5
PATIENT_SIM_FORWARD_MIX_MANUAL_MULTIPLIERS = {}

# Diminishing-returns parameters for install-upside capacity conversion (Step 09).
# Stage 1 — power-law friction: freed capacity converts to installs at a
# decreasing marginal rate.  alpha < 1 encodes coordination friction, ramp-up
# time, and scheduling overhead that grow with scale.  The formula is:
#   a * linear_installs^alpha
# where a is calibrated so the N=1 baseline scenario is unchanged.
# Set alpha = 1.0 to disable Stage 1.
INSTALL_UPSIDE_DIMINISHING_RETURNS_ALPHA = 0.70

# Stage 2 — annual ceiling: maximum net-new installs the sales pipeline and
# market can absorb per year, regardless of available technician capacity.
# Based on ~52 historical installs/year + modest growth headroom.
# Set to None or 0 to disable the ceiling.
INSTALL_UPSIDE_ANNUAL_CEILING = 55.0

# Reference linear install count used to calibrate the power-law coefficient
# so that the first non-zero scenario is unaffected.  Set to None to auto-detect
# from the smallest positive scenario value.
INSTALL_UPSIDE_REFERENCE_LINEAR = None

PATIENT_SIM_FAMILY_ASSUMPTIONS = {
    "Apollo": {
        "install_revenue_usd": 50_000,
        "install_margin": 0.40,
        "install_calendar_days_override": 2.1,
        "revenue_range_low_usd": 40_000,
        "revenue_range_high_usd": 60_000,
        "revenue_source_note": (
            "Provisional planning estimate from user-provided public ballpark "
            "range for the combined Apollo family (APP + APN + generic Apollo); "
            "replace with finance-approved value when available."
        ),
    },
    "Lucina": {
        "install_revenue_usd": 80_000,
        "install_margin": 0.40,
        "install_calendar_days_override": 2.1,
        "revenue_range_low_usd": 67_000,
        "revenue_range_high_usd": 90_000,
        "revenue_source_note": (
            "Provisional planning estimate from user-provided public ballpark "
            "range for the Lucina family (MFS + Lucina); replace with "
            "finance-approved value when available."
        ),
    },
    "Juno": {
        "install_revenue_usd": 19_000,
        "install_margin": 0.40,
        "install_calendar_days_override": 2.1,
        "revenue_range_low_usd": 15_000,
        "revenue_range_high_usd": 22_000,
        "revenue_source_note": (
            "Provisional planning estimate from user-provided public ballpark "
            "range for Juno; replace with finance-approved value when available."
        ),
    },
    "Ares": {
        "install_revenue_usd": 33_000,
        "install_margin": 0.40,
        "install_calendar_days_override": 2.1,
        "revenue_range_low_usd": 30_000,
        "revenue_range_high_usd": 36_000,
        "revenue_source_note": (
            "Provisional planning estimate from user-provided public ballpark "
            "range for Ares; replace with finance-approved value when available."
        ),
    },
    "Aria": {
        "install_revenue_usd": 38_000,
        "install_margin": 0.40,
        "install_calendar_days_override": 2.1,
        "revenue_range_low_usd": 32_000,
        "revenue_range_high_usd": 43_000,
        "revenue_source_note": (
            "Provisional planning estimate from user-provided public ballpark "
            "range for Aria; replace with finance-approved value when available."
        ),
    },
    "Evo": {
        "install_revenue_usd": 45_000,
        "install_margin": 0.40,
        "install_calendar_days_override": 2.1,
        "revenue_range_low_usd": 40_000,
        "revenue_range_high_usd": 50_000,
        "revenue_source_note": (
            "Provisional planning estimate near Apollo/Aria per user direction "
            "until a finance-approved Evo value is available."
        ),
    },
    "Luna": {
        "install_revenue_usd": 46_000,
        "install_margin": 0.40,
        "install_calendar_days_override": 2.1,
        "revenue_range_low_usd": 46_000,
        "revenue_range_high_usd": 46_000,
        "revenue_source_note": (
            "Provisional planning estimate from user direction for Luna; replace "
            "with finance-approved value when available."
        ),
    },
    "HPS": {
        "install_revenue_usd": None,
        "install_margin": 0.40,
        "install_calendar_days_override": None,
        "revenue_range_low_usd": None,
        "revenue_range_high_usd": None,
        "revenue_source_note": (
            "Excluded from the forward-looking mix by default. If HPS is later "
            "reactivated, supply an explicit revenue value before using it in the "
            "install upside model."
        ),
    },
}

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

# Technician marker colors
TECH_COLORS = {
    "active": "green",
    "special": "orange",
}

# Stakeholder-facing map UI mode.
# "stakeholder" keeps only the highest-value scenario storytelling layers.
# "debug" preserves the broader internal inspection layers and controls.
MAP_UI_MODE = os.environ.get("ELEVATE_MAP_UI_MODE", "stakeholder").strip().lower()
if MAP_UI_MODE not in {"stakeholder", "debug"}:
    MAP_UI_MODE = "stakeholder"

MAP_PANEL_FONT_FAMILY = '"Public Sans", "Segoe UI", "Helvetica Neue", Arial, sans-serif'
MAP_PANEL_WIDTH_PX = 360
MAP_PANEL_RADIUS_PX = 18
MAP_PANEL_SHADOW = "0 18px 40px rgba(15, 23, 42, 0.16)"
MAP_CARD_RADIUS_PX = 14
MAP_COVERAGE_ASSIGNMENTS_DEFAULT_COUNT = 6
MAP_STAKEHOLDER_SHOW_HUB_TOGGLE = True
MAP_STAKEHOLDER_HUBS_DEFAULT_VISIBLE = False
MAP_HUB_TOGGLE_LABEL = "Airports"

MAP_TECH_MARKER_COLORS = {
    "active": "#2E6F5E",
    "special": "#B46A2B",
    "mixed": "#51606E",
}
MAP_NEW_HIRE_MARKER_COLOR = "#C5662D"
MAP_AIRPORT_MARKER_COLOR = "#6B7280"
MAP_AIRPORT_MARKER_FILL = "#F8FAFC"
MAP_AIRPORT_MARKER_BORDER = "#CBD5E1"
MAP_ANCHOR_MARKER_COLOR = "#7C3E66"
MAP_ANCHOR_MARKER_FILL = "#FDF2F8"
MAP_ANCHOR_MARKER_BORDER = "#D8B4C6"

MAP_ASSIGNMENT_DOT_LIGHT_STROKE = "#FFFFFF"
MAP_ASSIGNMENT_DOT_DARK_STROKE = "#1F2937"
MAP_ASSIGNMENT_DOT_STROKE_WIDTH = 1.5
MAP_ASSIGNMENT_DOT_STAKEHOLDER_RADIUS = 5
MAP_ASSIGNMENT_DOT_STAKEHOLDER_OPACITY = 0.82
MAP_ASSIGNMENT_DOT_DEBUG_RADIUS = 6
MAP_ASSIGNMENT_DOT_DEBUG_OPACITY = 0.85

# Service appointment type colors
SERVICE_TYPE_COLORS = {
    "PM": "blue",
    "Repair": "red",
    "Install": "green",
    "Other": "purple",
}

# Territory choropleth gradient (light yellow → dark red)
CHOROPLETH_COLORS = [
    "#ffffb2",  # light yellow
    "#fecc5c",
    "#fd8d3c",
    "#f03b20",
    "#bd0026",  # dark red
]

# 19 distinct territory border colors
TERRITORY_COLORS = {
    "New England": "#1f77b4",
    "NY/NJ": "#ff7f0e",
    "Upstate NY/Western PA": "#2ca02c",
    "Mid Atlantic": "#d62728",
    "Carolinas/Virginias": "#9467bd",
    "Great Lakes": "#8c564b",
    "Illinois": "#e377c2",
    "Northern Plains": "#7f7f7f",
    "North Florida": "#bcbd22",
    "South Florida": "#17becf",
    "Gulf Coast": "#aec7e8",
    "North Texas": "#ffbb78",
    "South Texas": "#98df8a",
    "West Plains": "#ff9896",
    "Southwest": "#c5b0d5",
    "Northern California": "#c49c94",
    "Southern California": "#f7b6d2",
    "Pacific Northwest": "#c7c7c7",
    "Canada": "#dbdb8d",
}

# ---------------------------------------------------------------------------
# Map settings
# ---------------------------------------------------------------------------
MAP_CENTER = [39.8283, -98.5795]  # Geographic center of contiguous US
MAP_ZOOM = 4
MAP_TILES = "CartoDB positron"
GEOCODE_USER_AGENT = "elevate_healthcare_uiuc"
GEOCODE_DELAY = 1.1  # seconds between Nominatim requests

# ---------------------------------------------------------------------------
# Per-tech territory visualization (scenario-based assignment coloring)
# ---------------------------------------------------------------------------
# Explicit fixed assignee colors so high-usage technicians do not end up with
# near-lookalike colors in the stakeholder map.
TECH_ASSIGNMENT_COLOR_MAP = {
    "ben_walker": "#D81B60",            # strong magenta
    "bladimir_torres": "#1E88E5",      # bright blue
    "clarence_bonner": "#2E7D32",      # green
    "curt_corder": "#A16207",          # dark gold / mustard
    "damion_lyn": "#7B1FA2",           # purple
    "elier_martin": "#795548",         # taupe / brown
    "eric_olinger": "#00ACC1",         # cyan
    "hector_arias": "#EC407A",         # pink
    "htx_contractor_alex": "#C62828",  # true red
    "htx_contractor_robert": "#1E3A8A",# navy
    "james_sanchez": "#455A64",        # charcoal slate
    "josh_brown": "#EF6C00",           # true orange
    "robert_cohen": "#00897B",         # jade / teal-green
    "scott_fogo": "#4338CA",           # indigo
    "tameka_gongs": "#FDD835",         # yellow
    "airport_atl": "#A44A3F",          # terracotta
    "airport_cle": "#8D4E2D",          # sienna
    "airport_iad": "#9CCB19",          # lime
    "demand_bossier_city_la": "#C2185B",  # rose
    "demand_janesville_wi": "#7C4DFF",    # violet
    # Former techs (historical view only)
    "_hist_david_bazany": "#FF6F00",      # amber orange
    "_hist_john_aleksa": "#00BFA5",       # bright teal
    "_hist_trent_osborne": "#AA00FF",     # vivid purple
    "_hist_alex_rondero": "#5C6BC0",      # slate indigo
}

# Reserve palette only for future unknown assignee ids.
TECH_ASSIGNMENT_FALLBACK_PALETTE = [
    "#0F766E",
    "#2563EB",
    "#B91C1C",
    "#9333EA",
    "#CA8A04",
    "#BE185D",
    "#1D4ED8",
    "#15803D",
    "#7C2D12",
    "#334155",
    "#0EA5E9",
    "#4D7C0F",
]

TECH_TERRITORY_PALETTE = list(TECH_ASSIGNMENT_FALLBACK_PALETTE)
STAKEHOLDER_TERRITORY_PALETTE = list(TECH_ASSIGNMENT_FALLBACK_PALETTE)
BLANK_SLATE_ASSIGNMENT_PALETTE = [
    "#0F766E",
    "#2563EB",
    "#B91C1C",
    "#9333EA",
    "#CA8A04",
    "#BE185D",
    "#1D4ED8",
    "#15803D",
    "#7C2D12",
    "#334155",
    "#0EA5E9",
    "#4D7C0F",
    "#C2410C",
    "#A21CAF",
    "#0369A1",
    "#B45309",
]
TERRITORY_DOT_RADIUS = 6            # CircleMarker radius for assignment dots
TERRITORY_DOT_OPACITY = 0.85        # CircleMarker fill opacity
ENABLE_SIMULATION_UI = True
BLANK_SLATE_ENABLED = True
BLANK_SLATE_SUBDIR = "blank_slate"
SIM_SCENARIO_MIN = 0
SIM_SCENARIO_MAX = 4

# Known typo/mis-geocode overrides (applied every run)
GEOCODE_OVERRIDES = {
    "Pheonix, AZ, USA": {"lat": 33.4484367, "lon": -112.0741410},
    "Ashville, NC, USA": {"lat": 35.5953630, "lon": -82.5508407},
    "Belleville, TX, USA": {"lat": 29.9502253, "lon": -96.2571858},
    "Clarksville, GA, USA": {"lat": 34.6125971, "lon": -83.5248933},
    "Gainsville, GA, USA": {"lat": 34.2978794, "lon": -83.8240663},
    "Bismark, ND, USA": {"lat": 46.8083270, "lon": -100.7837390},
    # Explicitly pin these to avoid future ambiguity
    "London, Ontario, Canada": {"lat": 42.9836747, "lon": -81.2496068},
    "Ontario, CA, USA": {"lat": 34.0658460, "lon": -117.6484300},
    "Kingshill, US Virgin Islands, USA": {"lat": 17.7225219, "lon": -64.7826756},
    "Alberta, Canada": {"lat": 54.0000000, "lon": -114.0000000},
    "Abottsford, Canada": {"lat": 49.0521162, "lon": -122.3294790},
    "Brooklin, NY, USA": {"lat": 40.6526006, "lon": -73.9497211},
    "Jaskcon, MS, USA": {"lat": 32.2998686, "lon": -90.1830408},
    "Pikesville, KY, USA": {"lat": 37.4793000, "lon": -82.5188000},
    "Vallhalla, NY, USA": {"lat": 41.0752130, "lon": -73.7750061},
    "Winnimucca, NV, USA": {"lat": 40.9724295, "lon": -117.7348020},
    "Clarkesville, GA, USA": {"lat": 34.6125971, "lon": -83.5248933},
}

# Exact account-level point overrides used when a known site should not inherit a
# broader city geocode or centroid fallback.
ACCOUNT_EXACT_COORD_OVERRIDES = {
    "Morgan State University": {"lat": 39.3438, "lon": -76.5844},
}

# ---------------------------------------------------------------------------
# Major US and Canadian airports (for travel optimization layer)
# ---------------------------------------------------------------------------
AIRPORT_HUB_MANUAL_OVERRIDES = {
    "YYC": {
        "faa_hub_class": HUB_TIER_LARGE,
        "hub_tier": HUB_TIER_LARGE,
        "hub_source": "manual_canada_override",
        "hub_source_year": FAA_HUB_SOURCE_YEAR,
    },
    "YEG": {
        "faa_hub_class": HUB_TIER_MEDIUM,
        "hub_tier": HUB_TIER_MEDIUM,
        "hub_source": "manual_canada_override",
        "hub_source_year": FAA_HUB_SOURCE_YEAR,
    },
    "YUL": {
        "faa_hub_class": HUB_TIER_LARGE,
        "hub_tier": HUB_TIER_LARGE,
        "hub_source": "manual_canada_override",
        "hub_source_year": FAA_HUB_SOURCE_YEAR,
    },
}

# U.S. airport hub tiers here are fallback labels for human readability.
# build_airports_df() overrides them from the FAA CY2024 reference table when
# a U.S. airport code exists there. Canada stays on the manual override path.
MAJOR_AIRPORTS = [
    # --- NORTHEAST ---
    {"code": "BOS", "name": "Boston Logan",           "city": "Boston, MA",          "lat": 42.3656, "lon": -71.0096, "hub_tier": HUB_TIER_LARGE},
    {"code": "JFK", "name": "New York JFK",           "city": "New York, NY",        "lat": 40.6413, "lon": -73.7781, "hub_tier": HUB_TIER_LARGE},
    {"code": "LGA", "name": "New York LaGuardia",     "city": "New York, NY",        "lat": 40.7769, "lon": -73.8740, "hub_tier": HUB_TIER_LARGE},
    {"code": "EWR", "name": "Newark Liberty",         "city": "Newark, NJ",          "lat": 40.6895, "lon": -74.1745, "hub_tier": HUB_TIER_LARGE},
    {"code": "PHL", "name": "Philadelphia Intl",      "city": "Philadelphia, PA",    "lat": 39.8744, "lon": -75.2424, "hub_tier": HUB_TIER_LARGE},
    {"code": "PIT", "name": "Pittsburgh Intl",        "city": "Pittsburgh, PA",      "lat": 40.4915, "lon": -80.2329, "hub_tier": HUB_TIER_MEDIUM},
    {"code": "BUF", "name": "Buffalo Niagara",        "city": "Buffalo, NY",         "lat": 42.9405, "lon": -78.7322, "hub_tier": HUB_TIER_MEDIUM},
    # --- MID ATLANTIC ---
    {"code": "DCA", "name": "Washington Reagan",      "city": "Washington, DC",      "lat": 38.8512, "lon": -77.0402, "hub_tier": HUB_TIER_LARGE},
    {"code": "IAD", "name": "Washington Dulles",      "city": "Washington, DC",      "lat": 38.9531, "lon": -77.4565, "hub_tier": HUB_TIER_LARGE},
    {"code": "BWI", "name": "Baltimore/Washington",   "city": "Baltimore, MD",       "lat": 39.1754, "lon": -76.6682, "hub_tier": HUB_TIER_LARGE},
    # --- SOUTHEAST ---
    {"code": "CLT", "name": "Charlotte Douglas",      "city": "Charlotte, NC",       "lat": 35.2140, "lon": -80.9431, "hub_tier": HUB_TIER_LARGE},
    {"code": "RDU", "name": "Raleigh-Durham",         "city": "Raleigh, NC",         "lat": 35.8776, "lon": -78.7875, "hub_tier": HUB_TIER_MEDIUM},
    {"code": "ATL", "name": "Atlanta Hartsfield",     "city": "Atlanta, GA",         "lat": 33.6407, "lon": -84.4277, "hub_tier": HUB_TIER_LARGE},
    {"code": "MCO", "name": "Orlando Intl",           "city": "Orlando, FL",         "lat": 28.4312, "lon": -81.3081, "hub_tier": HUB_TIER_LARGE},
    {"code": "TPA", "name": "Tampa Intl",             "city": "Tampa, FL",           "lat": 27.9755, "lon": -82.5332, "hub_tier": HUB_TIER_LARGE},
    {"code": "MIA", "name": "Miami Intl",             "city": "Miami, FL",           "lat": 25.7959, "lon": -80.2870, "hub_tier": HUB_TIER_LARGE},
    {"code": "JAX", "name": "Jacksonville Intl",      "city": "Jacksonville, FL",    "lat": 30.4941, "lon": -81.6879, "hub_tier": HUB_TIER_MEDIUM},
    # --- MIDWEST ---
    {"code": "ORD", "name": "Chicago O'Hare",         "city": "Chicago, IL",         "lat": 41.9742, "lon": -87.9073, "hub_tier": HUB_TIER_LARGE},
    {"code": "MDW", "name": "Chicago Midway",         "city": "Chicago, IL",         "lat": 41.7868, "lon": -87.7522, "hub_tier": HUB_TIER_LARGE},
    {"code": "DTW", "name": "Detroit Metro",          "city": "Detroit, MI",         "lat": 42.2124, "lon": -83.3534, "hub_tier": HUB_TIER_LARGE},
    {"code": "CLE", "name": "Cleveland Hopkins",      "city": "Cleveland, OH",       "lat": 41.4117, "lon": -81.8498, "hub_tier": HUB_TIER_MEDIUM},
    {"code": "CMH", "name": "Columbus Intl",          "city": "Columbus, OH",        "lat": 39.9980, "lon": -82.8919, "hub_tier": HUB_TIER_MEDIUM},
    {"code": "CVG", "name": "Cincinnati/N. Kentucky", "city": "Cincinnati, OH",      "lat": 39.0488, "lon": -84.6678, "hub_tier": HUB_TIER_MEDIUM},
    {"code": "IND", "name": "Indianapolis Intl",      "city": "Indianapolis, IN",    "lat": 39.7173, "lon": -86.2944, "hub_tier": HUB_TIER_MEDIUM},
    {"code": "MKE", "name": "Milwaukee Mitchell",     "city": "Milwaukee, WI",       "lat": 42.9472, "lon": -87.8966, "hub_tier": HUB_TIER_MEDIUM},
    {"code": "MSP", "name": "Minneapolis-St. Paul",   "city": "Minneapolis, MN",     "lat": 44.8848, "lon": -93.2223, "hub_tier": HUB_TIER_LARGE},
    {"code": "STL", "name": "St. Louis Lambert",      "city": "St. Louis, MO",       "lat": 38.7487, "lon": -90.3700, "hub_tier": HUB_TIER_MEDIUM},
    {"code": "MCI", "name": "Kansas City Intl",       "city": "Kansas City, MO",     "lat": 39.2976, "lon": -94.7139, "hub_tier": HUB_TIER_MEDIUM},
    {"code": "ICT", "name": "Wichita Eisenhower",     "city": "Wichita, KS",         "lat": 37.6499, "lon": -97.4331, "hub_tier": HUB_TIER_SMALL},
    {"code": "OMA", "name": "Omaha Eppley",           "city": "Omaha, NE",           "lat": 41.3032, "lon": -95.8940, "hub_tier": HUB_TIER_MEDIUM},
    {"code": "DSM", "name": "Des Moines Intl",        "city": "Des Moines, IA",      "lat": 41.5340, "lon": -93.6631, "hub_tier": HUB_TIER_SMALL},
    # --- SOUTH / GULF COAST ---
    {"code": "BNA", "name": "Nashville Intl",         "city": "Nashville, TN",       "lat": 36.1245, "lon": -86.6782, "hub_tier": HUB_TIER_LARGE},
    {"code": "MEM", "name": "Memphis Intl",           "city": "Memphis, TN",         "lat": 35.0424, "lon": -89.9767, "hub_tier": HUB_TIER_SMALL},
    {"code": "BHM", "name": "Birmingham Shuttlesworth","city": "Birmingham, AL",     "lat": 33.5629, "lon": -86.7527, "hub_tier": HUB_TIER_SMALL},
    {"code": "MSY", "name": "New Orleans Intl",       "city": "New Orleans, LA",     "lat": 29.9934, "lon": -90.2580, "hub_tier": HUB_TIER_MEDIUM},
    {"code": "SHV", "name": "Shreveport Regional",    "city": "Shreveport, LA",      "lat": 32.4466, "lon": -93.8261, "hub_tier": HUB_TIER_NONHUB},
    {"code": "IAH", "name": "Houston Intercontinental","city": "Houston, TX",        "lat": 29.9902, "lon": -95.3368, "hub_tier": HUB_TIER_LARGE},
    {"code": "SAT", "name": "San Antonio Intl",       "city": "San Antonio, TX",     "lat": 29.5337, "lon": -98.4698, "hub_tier": HUB_TIER_MEDIUM},
    {"code": "AUS", "name": "Austin-Bergstrom",       "city": "Austin, TX",          "lat": 30.1975, "lon": -97.6664, "hub_tier": HUB_TIER_LARGE},
    {"code": "DFW", "name": "Dallas/Fort Worth",      "city": "Dallas, TX",          "lat": 32.8998, "lon": -97.0403, "hub_tier": HUB_TIER_LARGE},
    {"code": "OKC", "name": "Oklahoma City Will Rogers","city": "Oklahoma City, OK", "lat": 35.3931, "lon": -97.6007, "hub_tier": HUB_TIER_SMALL},
    {"code": "TUL", "name": "Tulsa Intl",             "city": "Tulsa, OK",           "lat": 36.1984, "lon": -95.8881, "hub_tier": HUB_TIER_SMALL},
    {"code": "LIT", "name": "Little Rock National",   "city": "Little Rock, AR",     "lat": 34.7294, "lon": -92.2242, "hub_tier": HUB_TIER_SMALL},
    # --- PLAINS / MOUNTAIN ---
    {"code": "DEN", "name": "Denver Intl",            "city": "Denver, CO",          "lat": 39.8561, "lon": -104.6737, "hub_tier": HUB_TIER_LARGE},
    {"code": "ABQ", "name": "Albuquerque Sunport",    "city": "Albuquerque, NM",     "lat": 35.0402, "lon": -106.6090, "hub_tier": HUB_TIER_MEDIUM},
    {"code": "SLC", "name": "Salt Lake City Intl",    "city": "Salt Lake City, UT",  "lat": 40.7899, "lon": -111.9791, "hub_tier": HUB_TIER_LARGE},
    {"code": "BIL", "name": "Billings Logan",         "city": "Billings, MT",        "lat": 45.8077, "lon": -108.5428, "hub_tier": HUB_TIER_NONHUB},
    {"code": "FAR", "name": "Fargo Hector",           "city": "Fargo, ND",           "lat": 46.9207, "lon": -96.8158, "hub_tier": HUB_TIER_SMALL},
    {"code": "BIS", "name": "Bismarck Municipal",     "city": "Bismarck, ND",        "lat": 46.7727, "lon": -100.7467, "hub_tier": HUB_TIER_NONHUB},
    {"code": "YYC", "name": "Calgary Intl",           "city": "Calgary, AB",         "lat": 51.1315, "lon": -114.0106, "hub_tier": HUB_TIER_LARGE},
    {"code": "YEG", "name": "Edmonton Intl",          "city": "Edmonton, AB",        "lat": 53.3097, "lon": -113.5800, "hub_tier": HUB_TIER_MEDIUM},
    {"code": "YUL", "name": "Montreal-Trudeau",       "city": "Montreal, QC",        "lat": 45.4706, "lon": -73.7408, "hub_tier": HUB_TIER_LARGE},
    {"code": "ONT", "name": "Ontario Intl",           "city": "Ontario, CA",         "lat": 34.0558, "lon": -117.6009, "hub_tier": HUB_TIER_MEDIUM},
    # --- WEST COAST ---
    {"code": "PHX", "name": "Phoenix Sky Harbor",     "city": "Phoenix, AZ",         "lat": 33.4373, "lon": -112.0078, "hub_tier": HUB_TIER_LARGE},
    {"code": "TUS", "name": "Tucson Intl",            "city": "Tucson, AZ",          "lat": 32.1161, "lon": -110.9410, "hub_tier": HUB_TIER_SMALL},
    {"code": "LAS", "name": "Las Vegas Harry Reid",   "city": "Las Vegas, NV",       "lat": 36.0840, "lon": -115.1537, "hub_tier": HUB_TIER_LARGE},
    {"code": "LAX", "name": "Los Angeles Intl",       "city": "Los Angeles, CA",     "lat": 33.9425, "lon": -118.4081, "hub_tier": HUB_TIER_LARGE},
    {"code": "SAN", "name": "San Diego Intl",         "city": "San Diego, CA",       "lat": 32.7336, "lon": -117.1897, "hub_tier": HUB_TIER_LARGE},
    {"code": "SFO", "name": "San Francisco Intl",     "city": "San Francisco, CA",   "lat": 37.6213, "lon": -122.3790, "hub_tier": HUB_TIER_LARGE},
    {"code": "SJC", "name": "San Jose Mineta",        "city": "San Jose, CA",        "lat": 37.3626, "lon": -121.9290, "hub_tier": HUB_TIER_MEDIUM},
    {"code": "SMF", "name": "Sacramento Intl",        "city": "Sacramento, CA",      "lat": 38.6954, "lon": -121.5908, "hub_tier": HUB_TIER_MEDIUM},
    # --- PACIFIC NORTHWEST ---
    {"code": "SEA", "name": "Seattle-Tacoma",         "city": "Seattle, WA",         "lat": 47.4502, "lon": -122.3088, "hub_tier": HUB_TIER_LARGE},
    {"code": "PDX", "name": "Portland Intl",          "city": "Portland, OR",        "lat": 45.5898, "lon": -122.5951, "hub_tier": HUB_TIER_MEDIUM},
    {"code": "BOI", "name": "Boise Airport",          "city": "Boise, ID",           "lat": 43.5644, "lon": -116.2228, "hub_tier": HUB_TIER_MEDIUM},
    {"code": "ANC", "name": "Anchorage Intl",         "city": "Anchorage, AK",       "lat": 61.1744, "lon": -149.9963, "hub_tier": HUB_TIER_MEDIUM},
]

AIRPORT_OPERATIONAL_ZONE_LABELS = {
    "ABQ": "Mountain",
    "ANC": "Alaska",
    "ATL": "Eastern",
    "AUS": "Central",
    "BHM": "Central",
    "BIL": "Mountain",
    "BIS": "Central",
    "BNA": "Central",
    "BOI": "Mountain",
    "BOS": "Eastern",
    "BUF": "Eastern",
    "BWI": "Eastern",
    "CLE": "Eastern",
    "CLT": "Eastern",
    "CMH": "Eastern",
    "CVG": "Eastern",
    "DCA": "Eastern",
    "DEN": "Mountain",
    "DFW": "Central",
    "DSM": "Central",
    "DTW": "Eastern",
    "EWR": "Eastern",
    "FAR": "Central",
    "IAD": "Eastern",
    "IAH": "Central",
    "ICT": "Central",
    "IND": "Eastern",
    "JAX": "Eastern",
    "JFK": "Eastern",
    "LAS": "Pacific",
    "LAX": "Pacific",
    "LGA": "Eastern",
    "LIT": "Central",
    "MCI": "Central",
    "MCO": "Eastern",
    "MDW": "Central",
    "MEM": "Central",
    "MIA": "Eastern",
    "MKE": "Central",
    "MSP": "Central",
    "MSY": "Central",
    "OKC": "Central",
    "OMA": "Central",
    "ONT": "Pacific",
    "ORD": "Central",
    "PDX": "Pacific",
    "PHL": "Eastern",
    "PHX": "Mountain",
    "PIT": "Eastern",
    "RDU": "Eastern",
    "SAN": "Pacific",
    "SAT": "Central",
    "SEA": "Pacific",
    "SFO": "Pacific",
    "SHV": "Central",
    "SJC": "Pacific",
    "SLC": "Mountain",
    "SMF": "Pacific",
    "STL": "Central",
    "TPA": "Eastern",
    "TUL": "Central",
    "TUS": "Mountain",
    "YEG": "Mountain",
    "YUL": "Eastern",
    "YYC": "Mountain",
}

AIRPORT_OPERATIONAL_ZONES = {
    code: {
        "label": label,
        **OPERATIONAL_ZONE_DEFINITIONS[label],
    }
    for code, label in AIRPORT_OPERATIONAL_ZONE_LABELS.items()
}
