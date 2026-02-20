"""Step 4: Build territory boundary polygons from US states and Canadian provinces."""
import json
import os
import sys
import urllib.request
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import config


US_STATES_URL = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
CANADA_PROVINCES_URL = (
    "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/canada.geojson"
)

US_STATES_CACHE_PATH = os.path.join(config.PROCESSED_DIR, "us-states.json")
CANADA_PROVINCES_CACHE_PATH = os.path.join(config.PROCESSED_DIR, "canada.geojson")


US_ABBR_TO_NAME = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DC": "District of Columbia",
    "DE": "Delaware",
    "FL": "Florida",
    "GA": "Georgia",
    "IA": "Iowa",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "MA": "Massachusetts",
    "MD": "Maryland",
    "ME": "Maine",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MO": "Missouri",
    "MS": "Mississippi",
    "MT": "Montana",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "NE": "Nebraska",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NV": "Nevada",
    "NY": "New York",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VA": "Virginia",
    "VT": "Vermont",
    "WA": "Washington",
    "WI": "Wisconsin",
    "WV": "West Virginia",
}


TERRITORY_TO_US_STATES = {
    "New England": ["CT", "MA", "ME", "NH", "RI", "VT"],
    "NY/NJ": ["NY", "NJ"],
    "Upstate NY/Western PA": ["NY", "PA"],
    "Mid Atlantic": ["DE", "MD", "PA", "NJ"],
    "Carolinas/Virginias": ["DC", "NC", "SC", "VA", "WV"],
    "Great Lakes": ["KY", "MI", "OH", "TN"],
    "Illinois": ["IL", "IN"],
    "Northern Plains": ["IA", "MN", "ND", "SD", "WI"],
    "North Florida": ["FL", "GA"],
    "South Florida": ["FL"],
    "Gulf Coast": ["AL", "LA", "MS"],
    "North Texas": ["AR", "OK", "TX"],
    "South Texas": ["TX"],
    "West Plains": ["CO", "KS", "MO", "MT", "NE"],
    "Southwest": ["AZ", "NM", "UT", "NV"],
    "Northern California": ["CA", "NV"],
    "Southern California": ["CA"],
    "Pacific Northwest": ["AK", "ID", "OR", "WA"],
    "Canada": [],
}


OUTLIER_STATE_NAMES = {
    "AK",
    "Alaska",
    "HI",
    "Hawaii",
    "US Virgin Islands",
    "Virgin Islands",
    "VI",
    "Puerto Rico",
    "PR",
    "Guam",
    "American Samoa",
    "Northern Mariana Islands",
}


def load_geojson_with_cache(url, cache_path):
    """Fetch GeoJSON from URL with on-disk cache fallback."""
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            payload = response.read().decode("utf-8")
        data = json.loads(payload)
        with open(cache_path, "w") as f:
            f.write(payload)
        return data
    except Exception:
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                return json.load(f)
        raise


def geometry_to_multipolygon_coords(geometry):
    """Return a geometry as a list of Polygon coordinate arrays."""
    gtype = geometry.get("type")
    coords = geometry.get("coordinates", [])
    if gtype == "Polygon":
        return [coords]
    if gtype == "MultiPolygon":
        return coords
    return []


def make_outlier_markers(appts):
    """Build per-territory outlier point markers (outside contiguous US)."""
    markers = {territory: [] for territory in config.TERRITORY_COLORS}

    # Static Alaska marker to avoid a giant polygon distorting the map
    markers["Pacific Northwest"].append(
        {
            "lat": 64.2008,
            "lon": -149.4937,
            "label": "Alaska (Pacific Northwest reference)",
        }
    )

    appts = appts.dropna(subset=["lat", "lon", "Territory"]).copy()
    outlier_mask = (
        appts["State/Province"].astype(str).isin(OUTLIER_STATE_NAMES)
        | (appts["lat"] < 24)
        | (appts["lon"] < -130)
        | (appts["lon"] > -66)
    )
    outliers = appts[outlier_mask].copy()
    if outliers.empty:
        return markers

    # One marker per unique location/territory
    for _, row in (
        outliers[["Territory", "City", "State/Province", "lat", "lon"]]
        .drop_duplicates()
        .iterrows()
    ):
        territory = row["Territory"]
        label = f"{row['City']}, {row['State/Province']}"
        markers.setdefault(territory, []).append(
            {
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "label": label,
            }
        )

    return markers


def main():
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)

    appts = pd.read_csv(config.GEOCODED_APPTS_CSV)
    territory_summary = pd.read_csv(config.TERRITORY_SUMMARY_CSV)
    appt_counts = appts["Territory"].value_counts().to_dict()
    asset_counts = dict(zip(territory_summary["Territory"], territory_summary["total_assets"]))
    outlier_markers = make_outlier_markers(appts)

    us_geojson = load_geojson_with_cache(US_STATES_URL, US_STATES_CACHE_PATH)
    canada_geojson = load_geojson_with_cache(CANADA_PROVINCES_URL, CANADA_PROVINCES_CACHE_PATH)

    us_by_name = {}
    for feature in us_geojson["features"]:
        state_name = feature.get("properties", {}).get("name")
        if state_name:
            us_by_name[state_name] = feature

    canada_features = canada_geojson["features"]

    features = []
    for territory in config.TERRITORY_COLORS:
        polygons = []
        sources = []

        if territory == "Canada":
            for feature in canada_features:
                name = feature.get("properties", {}).get("name")
                sources.append(name)
                polygons.extend(geometry_to_multipolygon_coords(feature["geometry"]))
        else:
            state_abbrs = TERRITORY_TO_US_STATES.get(territory, [])
            for abbr in state_abbrs:
                # Keep AK as marker only to avoid map distortion
                if territory == "Pacific Northwest" and abbr == "AK":
                    continue
                state_name = US_ABBR_TO_NAME.get(abbr)
                if not state_name or state_name not in us_by_name:
                    continue
                sources.append(state_name)
                polygons.extend(geometry_to_multipolygon_coords(us_by_name[state_name]["geometry"]))

        feature = {
            "type": "Feature",
            "properties": {
                "name": territory,
                "color": config.TERRITORY_COLORS.get(territory, "#999999"),
                "appointments": int(appt_counts.get(territory, 0)),
                "active_assets": int(asset_counts.get(territory, 0)),
                "source_regions": sources,
                "outlier_markers": outlier_markers.get(territory, []),
            },
            "geometry": {
                "type": "MultiPolygon",
                "coordinates": polygons,
            },
        }
        features.append(feature)
        print(
            f"{territory}: regions={len(sources)} polygons={len(polygons)} "
            f"appointments={appt_counts.get(territory, 0)} "
            f"active_assets={asset_counts.get(territory, 0)}"
        )

    geojson = {"type": "FeatureCollection", "features": features}
    with open(config.TERRITORIES_GEOJSON, "w") as f:
        json.dump(geojson, f, indent=2)

    print(f"\nSaved: {config.TERRITORIES_GEOJSON}")
    print(f"Total territories: {len(features)}")
    print("Step 4 complete.")


if __name__ == "__main__":
    main()
