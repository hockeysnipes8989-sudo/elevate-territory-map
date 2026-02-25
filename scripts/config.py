"""Configuration for Elevate Healthcare Territory Map pipeline."""
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")
OPTIMIZATION_DIR = os.path.join(PROCESSED_DIR, "optimization")

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

# External source-of-truth workbooks used by optimization scripts
EXTERNAL_APPOINTMENTS_XLSX = (
    "/Users/patricklipinski/Desktop/opus 4.6 final excel sheets/"
    "UIUC service appointments with dipsatch date - final.xlsx"
)
EXTERNAL_TECH_ROSTER_XLSX = (
    "/Users/patricklipinski/Downloads/Tech location and product experience (1).xlsx"
)
EXTERNAL_NAVAN_XLSX = (
    "/Users/patricklipinski/Downloads/"
    "REPORT_2026_02_25__13_27_12_948ET-CONTAINS-SENSITIVE-DATA-REMOVE-AFTER-USE.xlsx"
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
    "Elier Martin": "special",
    "Eric Olinger": "active",
    "Hector Arias": "active",
    "James Sanchez": "active",
    "Josh Brown": "active",
    "Robert Cohen": "active",
    "Scott Fogo": "active",
    "Tameka Gongs": "active",
}

# Former technicians to exclude from current-state maps/outputs
INACTIVE_TECH_NAMES = {
    "David Bazany",
    "John Aleksa",
    "Trent Osborne",
}

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

# Technician marker colors
TECH_COLORS = {
    "active": "green",
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
ENABLE_SIMULATION_UI = True
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
    "Gainesville, GA, USA": {"lat": 34.2978794, "lon": -83.8240663},
    "Bismarck, ND, USA": {"lat": 46.8083270, "lon": -100.7837390},
}

# ---------------------------------------------------------------------------
# Major US and Canadian airports (for travel optimization layer)
# ---------------------------------------------------------------------------
MAJOR_AIRPORTS = [
    # --- NORTHEAST ---
    {"code": "BOS", "name": "Boston Logan",          "city": "Boston, MA",          "lat": 42.3656, "lon": -71.0096},
    {"code": "JFK", "name": "New York JFK",           "city": "New York, NY",         "lat": 40.6413, "lon": -73.7781},
    {"code": "LGA", "name": "New York LaGuardia",     "city": "New York, NY",         "lat": 40.7769, "lon": -73.8740},
    {"code": "EWR", "name": "Newark Liberty",         "city": "Newark, NJ",           "lat": 40.6895, "lon": -74.1745},
    {"code": "PHL", "name": "Philadelphia Intl",      "city": "Philadelphia, PA",     "lat": 39.8744, "lon": -75.2424},
    {"code": "PIT", "name": "Pittsburgh Intl",        "city": "Pittsburgh, PA",       "lat": 40.4915, "lon": -80.2329},
    {"code": "BUF", "name": "Buffalo Niagara",        "city": "Buffalo, NY",          "lat": 42.9405, "lon": -78.7322},
    # --- MID ATLANTIC ---
    {"code": "DCA", "name": "Washington Reagan",      "city": "Washington, DC",       "lat": 38.8512, "lon": -77.0402},
    {"code": "IAD", "name": "Washington Dulles",      "city": "Washington, DC",       "lat": 38.9531, "lon": -77.4565},
    {"code": "BWI", "name": "Baltimore/Washington",   "city": "Baltimore, MD",        "lat": 39.1754, "lon": -76.6682},
    # --- SOUTHEAST ---
    {"code": "CLT", "name": "Charlotte Douglas",      "city": "Charlotte, NC",        "lat": 35.2140, "lon": -80.9431},
    {"code": "RDU", "name": "Raleigh-Durham",         "city": "Raleigh, NC",          "lat": 35.8776, "lon": -78.7875},
    {"code": "ATL", "name": "Atlanta Hartsfield",     "city": "Atlanta, GA",          "lat": 33.6407, "lon": -84.4277},
    {"code": "MCO", "name": "Orlando Intl",           "city": "Orlando, FL",          "lat": 28.4312, "lon": -81.3081},
    {"code": "TPA", "name": "Tampa Intl",             "city": "Tampa, FL",            "lat": 27.9755, "lon": -82.5332},
    {"code": "MIA", "name": "Miami Intl",             "city": "Miami, FL",            "lat": 25.7959, "lon": -80.2870},
    {"code": "JAX", "name": "Jacksonville Intl",      "city": "Jacksonville, FL",     "lat": 30.4941, "lon": -81.6879},
    # --- MIDWEST ---
    {"code": "ORD", "name": "Chicago O'Hare",         "city": "Chicago, IL",          "lat": 41.9742, "lon": -87.9073},
    {"code": "MDW", "name": "Chicago Midway",         "city": "Chicago, IL",          "lat": 41.7868, "lon": -87.7522},
    {"code": "DTW", "name": "Detroit Metro",          "city": "Detroit, MI",          "lat": 42.2124, "lon": -83.3534},
    {"code": "CLE", "name": "Cleveland Hopkins",      "city": "Cleveland, OH",        "lat": 41.4117, "lon": -81.8498},
    {"code": "CMH", "name": "Columbus Intl",          "city": "Columbus, OH",         "lat": 39.9980, "lon": -82.8919},
    {"code": "CVG", "name": "Cincinnati/N. Kentucky", "city": "Cincinnati, OH",       "lat": 39.0488, "lon": -84.6678},
    {"code": "IND", "name": "Indianapolis Intl",      "city": "Indianapolis, IN",     "lat": 39.7173, "lon": -86.2944},
    {"code": "MKE", "name": "Milwaukee Mitchell",     "city": "Milwaukee, WI",        "lat": 42.9472, "lon": -87.8966},
    {"code": "MSP", "name": "Minneapolis-St. Paul",   "city": "Minneapolis, MN",      "lat": 44.8848, "lon": -93.2223},
    {"code": "STL", "name": "St. Louis Lambert",      "city": "St. Louis, MO",        "lat": 38.7487, "lon": -90.3700},
    {"code": "MCI", "name": "Kansas City Intl",       "city": "Kansas City, MO",      "lat": 39.2976, "lon": -94.7139},
    {"code": "OMA", "name": "Omaha Eppley",           "city": "Omaha, NE",            "lat": 41.3032, "lon": -95.8940},
    {"code": "DSM", "name": "Des Moines Intl",        "city": "Des Moines, IA",       "lat": 41.5340, "lon": -93.6631},
    # --- SOUTH / GULF COAST ---
    {"code": "BNA", "name": "Nashville Intl",         "city": "Nashville, TN",        "lat": 36.1245, "lon": -86.6782},
    {"code": "MEM", "name": "Memphis Intl",           "city": "Memphis, TN",          "lat": 35.0424, "lon": -89.9767},
    {"code": "BHM", "name": "Birmingham Shuttlesworth","city": "Birmingham, AL",      "lat": 33.5629, "lon": -86.7527},
    {"code": "MSY", "name": "New Orleans Intl",       "city": "New Orleans, LA",      "lat": 29.9934, "lon": -90.2580},
    {"code": "IAH", "name": "Houston Intercontinental","city": "Houston, TX",         "lat": 29.9902, "lon": -95.3368},
    {"code": "SAT", "name": "San Antonio Intl",       "city": "San Antonio, TX",      "lat": 29.5337, "lon": -98.4698},
    {"code": "AUS", "name": "Austin-Bergstrom",       "city": "Austin, TX",           "lat": 30.1975, "lon": -97.6664},
    {"code": "DFW", "name": "Dallas/Fort Worth",      "city": "Dallas, TX",           "lat": 32.8998, "lon": -97.0403},
    {"code": "OKC", "name": "Oklahoma City Will Rogers","city": "Oklahoma City, OK",  "lat": 35.3931, "lon": -97.6007},
    {"code": "TUL", "name": "Tulsa Intl",             "city": "Tulsa, OK",            "lat": 36.1984, "lon": -95.8881},
    {"code": "LIT", "name": "Little Rock National",   "city": "Little Rock, AR",      "lat": 34.7294, "lon": -92.2242},
    # --- PLAINS / MOUNTAIN ---
    {"code": "DEN", "name": "Denver Intl",            "city": "Denver, CO",           "lat": 39.8561, "lon": -104.6737},
    {"code": "ABQ", "name": "Albuquerque Sunport",    "city": "Albuquerque, NM",      "lat": 35.0402, "lon": -106.6090},
    {"code": "SLC", "name": "Salt Lake City Intl",    "city": "Salt Lake City, UT",   "lat": 40.7899, "lon": -111.9791},
    {"code": "BIL", "name": "Billings Logan",         "city": "Billings, MT",         "lat": 45.8077, "lon": -108.5428},
    {"code": "FAR", "name": "Fargo Hector",           "city": "Fargo, ND",            "lat": 46.9207, "lon": -96.8158},
    {"code": "BIS", "name": "Bismarck Municipal",     "city": "Bismarck, ND",         "lat": 46.7727, "lon": -100.7467},
    # --- WEST COAST ---
    {"code": "PHX", "name": "Phoenix Sky Harbor",     "city": "Phoenix, AZ",          "lat": 33.4373, "lon": -112.0078},
    {"code": "TUS", "name": "Tucson Intl",            "city": "Tucson, AZ",           "lat": 32.1161, "lon": -110.9410},
    {"code": "LAS", "name": "Las Vegas Harry Reid",   "city": "Las Vegas, NV",        "lat": 36.0840, "lon": -115.1537},
    {"code": "LAX", "name": "Los Angeles Intl",       "city": "Los Angeles, CA",      "lat": 33.9425, "lon": -118.4081},
    {"code": "SAN", "name": "San Diego Intl",         "city": "San Diego, CA",        "lat": 32.7336, "lon": -117.1897},
    {"code": "SFO", "name": "San Francisco Intl",     "city": "San Francisco, CA",    "lat": 37.6213, "lon": -122.3790},
    {"code": "SJC", "name": "San Jose Mineta",        "city": "San Jose, CA",         "lat": 37.3626, "lon": -121.9290},
    {"code": "SMF", "name": "Sacramento Intl",        "city": "Sacramento, CA",       "lat": 38.6954, "lon": -121.5908},
    # --- PACIFIC NORTHWEST ---
    {"code": "SEA", "name": "Seattle-Tacoma",         "city": "Seattle, WA",          "lat": 47.4502, "lon": -122.3088},
    {"code": "PDX", "name": "Portland Intl",          "city": "Portland, OR",         "lat": 45.5898, "lon": -122.5951},
    {"code": "BOI", "name": "Boise Airport",          "city": "Boise, ID",            "lat": 43.5644, "lon": -116.2228},
    {"code": "ANC", "name": "Anchorage Intl",         "city": "Anchorage, AK",        "lat": 61.1744, "lon": -149.9963},
    # --- CANADA ---
    {"code": "YYZ", "name": "Toronto Pearson",        "city": "Toronto, ON",          "lat": 43.6777, "lon": -79.6248},
    {"code": "YUL", "name": "Montreal-Trudeau",       "city": "Montreal, QC",         "lat": 45.4706, "lon": -73.7408},
    {"code": "YYC", "name": "Calgary Intl",           "city": "Calgary, AB",          "lat": 51.1215, "lon": -114.0076},
    {"code": "YEG", "name": "Edmonton Intl",          "city": "Edmonton, AB",         "lat": 53.3097, "lon": -113.5827},
    {"code": "YVR", "name": "Vancouver Intl",         "city": "Vancouver, BC",        "lat": 49.1967, "lon": -123.1815},
    {"code": "YQR", "name": "Regina Intl",            "city": "Regina, SK",           "lat": 50.4319, "lon": -104.6659},
]
