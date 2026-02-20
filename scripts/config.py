"""Configuration for Elevate Healthcare Territory Map pipeline."""
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")

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

# Processed outputs
CLEAN_APPTS_CSV = os.path.join(PROCESSED_DIR, "service_appointments_clean.csv")
CLEAN_TECHS_CSV = os.path.join(PROCESSED_DIR, "technicians.csv")
CLEAN_INSTALL_CSV = os.path.join(PROCESSED_DIR, "install_base_clean.csv")
GEOCODE_CACHE = os.path.join(PROJECT_ROOT, "data", "geocode_cache.json")
GEOCODED_APPTS_CSV = os.path.join(PROCESSED_DIR, "appointments_geocoded.csv")
GEOCODED_TECHS_CSV = os.path.join(PROCESSED_DIR, "technicians_geocoded.csv")
INSTALL_MATCHED_CSV = os.path.join(PROCESSED_DIR, "install_base_matched.csv")
TERRITORY_SUMMARY_CSV = os.path.join(PROCESSED_DIR, "territory_summary.csv")
TERRITORIES_GEOJSON = os.path.join(PROCESSED_DIR, "territories.geojson")
MAP_OUTPUT = os.path.join(DOCS_DIR, "index.html")

# ---------------------------------------------------------------------------
# Excel sheet names
# ---------------------------------------------------------------------------
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

# Technician status
TECH_STATUS = {
    "Alex Rondero": "special",
    "Ben Walker": "active",
    "Bladimir Torres": "active",
    "Clarence Bonner": "active",
    "Damion Lyn": "special",
    "David Bazany": "former",
    "Elier Martin": "special",
    "Eric Olinger": "active",
    "Hector Arias": "active",
    "James Sanchez": "active",
    "John Aleksa": "former",
    "Josh Brown": "active",
    "Robert Cohen": "active",
    "Scott Fogo": "active",
    "Tameka Gongs": "active",
    "Trent Osborne": "former",
}

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

# Technician marker colors
TECH_COLORS = {
    "active": "green",
    "former": "gray",
    "special": "orange",
}

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
TERRITORY_BUFFER_DEG = 0.3  # buffer around convex hull in degrees
