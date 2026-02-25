"""Shared helpers for optimization scripts."""

from __future__ import annotations

import math
import re
from typing import Optional

import pandas as pd


US_STATE_ABBR = {
    "ALABAMA": "AL",
    "ALASKA": "AK",
    "ARIZONA": "AZ",
    "ARKANSAS": "AR",
    "CALIFORNIA": "CA",
    "COLORADO": "CO",
    "CONNECTICUT": "CT",
    "DELAWARE": "DE",
    "DISTRICT OF COLUMBIA": "DC",
    "FLORIDA": "FL",
    "GEORGIA": "GA",
    "HAWAII": "HI",
    "IDAHO": "ID",
    "ILLINOIS": "IL",
    "INDIANA": "IN",
    "IOWA": "IA",
    "KANSAS": "KS",
    "KENTUCKY": "KY",
    "LOUISIANA": "LA",
    "MAINE": "ME",
    "MARYLAND": "MD",
    "MASSACHUSETTS": "MA",
    "MICHIGAN": "MI",
    "MINNESOTA": "MN",
    "MISSISSIPPI": "MS",
    "MISSOURI": "MO",
    "MONTANA": "MT",
    "NEBRASKA": "NE",
    "NEVADA": "NV",
    "NEW HAMPSHIRE": "NH",
    "NEW JERSEY": "NJ",
    "NEW MEXICO": "NM",
    "NEW YORK": "NY",
    "NORTH CAROLINA": "NC",
    "NORTH DAKOTA": "ND",
    "OHIO": "OH",
    "OKLAHOMA": "OK",
    "OREGON": "OR",
    "PENNSYLVANIA": "PA",
    "RHODE ISLAND": "RI",
    "SOUTH CAROLINA": "SC",
    "SOUTH DAKOTA": "SD",
    "TENNESSEE": "TN",
    "TEXAS": "TX",
    "UTAH": "UT",
    "VERMONT": "VT",
    "VIRGINIA": "VA",
    "WASHINGTON": "WA",
    "WEST VIRGINIA": "WV",
    "WISCONSIN": "WI",
    "WYOMING": "WY",
}

CANADA_PROV_ABBR = {
    "ALBERTA": "AB",
    "BRITISH COLUMBIA": "BC",
    "MANITOBA": "MB",
    "NEW BRUNSWICK": "NB",
    "NEWFOUNDLAND AND LABRADOR": "NL",
    "NOVA SCOTIA": "NS",
    "ONTARIO": "ON",
    "PRINCE EDWARD ISLAND": "PE",
    "QUEBEC": "QC",
    "SASKATCHEWAN": "SK",
    "NORTHWEST TERRITORIES": "NT",
    "NUNAVUT": "NU",
    "YUKON": "YT",
}

CANADA_ABBR = set(CANADA_PROV_ABBR.values())
US_ABBR = set(US_STATE_ABBR.values()) | {"DC"}


def slugify(value: str) -> str:
    """Return filesystem-safe slug."""
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def normalize_state(value: object) -> Optional[str]:
    """Normalize US state / Canadian province to abbreviation."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    up = text.upper()
    if up in US_ABBR or up in CANADA_ABBR:
        return up
    if up in US_STATE_ABBR:
        return US_STATE_ABBR[up]
    if up in CANADA_PROV_ABBR:
        return CANADA_PROV_ABBR[up]
    return up


def country_from_state(state_abbr: Optional[str]) -> str:
    """Infer country from state/province abbreviation."""
    if not state_abbr:
        return "USA"
    if state_abbr in CANADA_ABBR:
        return "Canada"
    return "USA"


def normalize_name(value: object) -> str:
    """Lightweight canonicalizer for names/labels."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return " ".join(text.split())


def parse_city_state(text: object) -> tuple[Optional[str], Optional[str]]:
    """Parse a free-form city/state string into (city, state_abbr)."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None, None
    raw = str(text).strip()
    if not raw:
        return None, None

    # Handles "City, ST" and "City ST"
    comma_parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(comma_parts) >= 2:
        state = normalize_state(comma_parts[-1])
        city = ", ".join(comma_parts[:-1]).strip()
        return city if city else None, state

    m = re.match(r"^(.*?)[\s\-]+([A-Za-z]{2})$", raw)
    if m:
        city = m.group(1).replace("-", " ").strip()
        state = normalize_state(m.group(2))
        return city if city else None, state

    return raw, None


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance between two points in KM."""
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def nearest_airport_code(
    lat: float,
    lon: float,
    airports_df: pd.DataFrame,
    state_hint: Optional[str] = None,
) -> tuple[Optional[str], Optional[float]]:
    """Find nearest airport code and distance in KM."""
    if pd.isna(lat) or pd.isna(lon):
        return None, None
    subset = airports_df
    if state_hint:
        narrowed = airports_df[airports_df["state_abbr"] == state_hint]
        if not narrowed.empty:
            subset = narrowed
    distances = subset.apply(
        lambda r: haversine_km(float(lat), float(lon), float(r["lat"]), float(r["lon"])),
        axis=1,
    )
    idx = distances.idxmin()
    return str(subset.loc[idx, "airport_code"]), float(distances.loc[idx])


def build_airports_df(major_airports: list[dict]) -> pd.DataFrame:
    """Convert config.MAJOR_AIRPORTS list to dataframe."""
    rows = []
    for ap in major_airports:
        city_raw = str(ap.get("city", "")).strip()
        city_parts = [p.strip() for p in city_raw.split(",")]
        city_name = city_parts[0] if city_parts else city_raw
        state_abbr = normalize_state(city_parts[1]) if len(city_parts) > 1 else None
        rows.append(
            {
                "airport_code": ap.get("code"),
                "airport_name": ap.get("name"),
                "city_name": city_name,
                "state_abbr": state_abbr,
                "country": country_from_state(state_abbr),
                "lat": float(ap.get("lat")),
                "lon": float(ap.get("lon")),
            }
        )
    df = pd.DataFrame(rows)
    df["city_norm"] = df["city_name"].map(normalize_name)
    return df


def coerce_excel_datetime(series: pd.Series) -> pd.Series:
    """Parse datetime from mixed string / Excel serial representation."""
    parsed = pd.to_datetime(series, errors="coerce")
    numeric = pd.to_numeric(series, errors="coerce")
    excel_mask = parsed.isna() & numeric.notna() & (numeric > 20000) & (numeric < 70000)
    if excel_mask.any():
        parsed.loc[excel_mask] = pd.to_datetime("1899-12-30") + pd.to_timedelta(
            numeric.loc[excel_mask], unit="D"
        )
    return parsed
