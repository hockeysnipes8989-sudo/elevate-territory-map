"""Shared helpers for optimization scripts."""

from __future__ import annotations

import math
import os
import re
import warnings
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd

import config


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
US_ABBR = set(US_STATE_ABBR.values())
CANADA_COUNTRY_TOKENS = {"CANADA", "CAN"}
KM_PER_MILE = 1.60934
MILES_PER_KM = 1.0 / KM_PER_MILE


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
    token = str(state_abbr).strip().upper()
    if token in CANADA_ABBR or token in CANADA_COUNTRY_TOKENS:
        return "Canada"
    return "USA"


def normalize_name(value: object) -> str:
    """Lightweight canonicalizer for names/labels."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return " ".join(text.split())


def canonicalize_tech_name(value: object) -> str:
    """Map raw technician/resource names to a canonical display name."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    raw = str(value).strip()
    if not raw:
        return ""
    if raw in config.TECH_NAME_MAP:
        return str(config.TECH_NAME_MAP[raw]).strip()

    normalized = normalize_name(raw)
    normalized_exact_map = {
        normalize_name(src): str(dst).strip()
        for src, dst in getattr(config, "TECH_NAME_MAP", {}).items()
    }
    if normalized in normalized_exact_map:
        return normalized_exact_map[normalized]

    normalized_alias_map = {
        normalize_name(src): str(dst).strip()
        for src, dst in getattr(config, "TECH_NAME_ALIASES", {}).items()
    }
    return normalized_alias_map.get(normalized, raw)


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


def choose_airport_for_location(location: object, airports_df: pd.DataFrame) -> str | None:
    """Map a free-form location string to the closest supported airport choice."""
    if location is None or (isinstance(location, float) and pd.isna(location)):
        return None
    loc_norm = normalize_name(location)
    overrides = getattr(config, "LOCATION_AIRPORT_OVERRIDES", {})
    if loc_norm in overrides:
        return str(overrides[loc_norm]).strip().upper()

    city, state_abbr = parse_city_state(location)
    city_norm = normalize_name(city)
    if city_norm and state_abbr:
        exact = airports_df[
            (airports_df["city_norm"] == city_norm) & (airports_df["state_abbr"] == state_abbr)
        ]
        if not exact.empty:
            return str(exact.iloc[0]["airport_code"])
        same_state = airports_df[airports_df["state_abbr"] == state_abbr]
        if not same_state.empty:
            city_match = (
                same_state[same_state["city_norm"].str.contains(city_norm, na=False)]
                if city_norm
                else pd.DataFrame()
            )
            if not city_match.empty:
                return str(city_match.iloc[0]["airport_code"])
    if city_norm:
        near_city = airports_df[airports_df["city_norm"].str.contains(city_norm, na=False)]
        if not near_city.empty:
            return str(near_city.iloc[0]["airport_code"])
    return None


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance between two points in KM."""
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance between two points in miles."""
    return haversine_km(lat1, lon1, lat2, lon2) * MILES_PER_KM


def nearest_airport_code(
    lat: float,
    lon: float,
    airports_df: pd.DataFrame,
    state_hint: Optional[str] = None,
) -> tuple[Optional[str], Optional[float]]:
    """Find nearest airport code and distance in KM."""
    if pd.isna(lat) or pd.isna(lon):
        return None, None
    state_norm = normalize_state(state_hint) if state_hint else None
    country_hint = country_from_state(state_norm) if state_norm else None

    subset = airports_df
    if country_hint:
        same_country = airports_df[airports_df["country"] == country_hint]
        if not same_country.empty:
            subset = same_country

    ranked = subset.copy()
    ranked["distance_km"] = ranked.apply(
        lambda r: haversine_km(float(lat), float(lon), float(r["lat"]), float(r["lon"])),
        axis=1,
    )
    if state_norm:
        ranked["same_state_preference"] = np.where(
            ranked["state_abbr"].eq(state_norm),
            0,
            1,
        )
    else:
        ranked["same_state_preference"] = 1
    ranked = ranked.sort_values(
        ["distance_km", "same_state_preference", "airport_code"]
    ).reset_index(drop=True)
    best = ranked.iloc[0]
    return str(best["airport_code"]), float(best["distance_km"])


def get_airport_operational_zone(airport_code: object) -> dict:
    """Return broad operational zone metadata for an airport code."""
    code = "" if airport_code is None else str(airport_code).strip().upper()
    zone = config.AIRPORT_OPERATIONAL_ZONES.get(code)
    if zone:
        return dict(zone)
    return {
        "label": "Unknown",
        "rank": None,
        "utc_offset_standard": None,
    }


@lru_cache(maxsize=1)
def load_airport_hub_reference() -> dict[str, dict]:
    """Load FAA-backed airport hub classes for repo airports."""
    path = getattr(config, "FAA_HUB_CLASS_REFERENCE_CSV", "")
    if not path or not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    lookup = {}
    for _, row in df.iterrows():
        code = str(row.get("airport_code", "")).strip().upper()
        if not code:
            continue
        lookup[code] = {
            "faa_hub_class": str(row.get("faa_hub_class", "")).strip() or config.HUB_TIER_UNKNOWN,
            "hub_tier": str(row.get("faa_hub_class", "")).strip() or config.HUB_TIER_UNKNOWN,
            "hub_source": str(row.get("source", "")).strip() or "faa_reference",
            "hub_source_year": pd.to_numeric(row.get("source_year"), errors="coerce"),
            "cy_24_enplanements": pd.to_numeric(row.get("cy_24_enplanements"), errors="coerce"),
        }
    return lookup


def build_airports_df(major_airports: list[dict]) -> pd.DataFrame:
    """Convert config.MAJOR_AIRPORTS list to dataframe."""
    faa_lookup = load_airport_hub_reference()
    manual_lookup = getattr(config, "AIRPORT_HUB_MANUAL_OVERRIDES", {})
    rows = []
    for ap in major_airports:
        code = str(ap.get("code", "")).strip().upper()
        city_raw = str(ap.get("city", "")).strip()
        city_parts = [p.strip() for p in city_raw.split(",")]
        city_name = city_parts[0] if city_parts else city_raw
        state_abbr = normalize_state(city_parts[1]) if len(city_parts) > 1 else None
        hub_meta = faa_lookup.get(code)
        if hub_meta is None:
            manual_meta = manual_lookup.get(code)
            if manual_meta:
                hub_meta = {
                    "faa_hub_class": str(
                        manual_meta.get("faa_hub_class", manual_meta.get("hub_tier", config.HUB_TIER_UNKNOWN))
                    ).strip()
                    or config.HUB_TIER_UNKNOWN,
                    "hub_tier": str(
                        manual_meta.get("hub_tier", config.HUB_TIER_UNKNOWN)
                    ).strip()
                    or config.HUB_TIER_UNKNOWN,
                    "hub_source": str(
                        manual_meta.get("hub_source", "manual_override")
                    ).strip()
                    or "manual_override",
                    "hub_source_year": pd.to_numeric(
                        manual_meta.get("hub_source_year"), errors="coerce"
                    ),
                    "cy_24_enplanements": pd.to_numeric(
                        manual_meta.get("cy_24_enplanements"), errors="coerce"
                    ),
                }
            else:
                hub_tier = str(
                    ap.get("hub_tier", config.HUB_TIER_MEDIUM)
                ).strip() or config.HUB_TIER_MEDIUM
                faa_hub_class = str(ap.get("faa_hub_class", hub_tier)).strip() or hub_tier
                hub_meta = {
                    "faa_hub_class": faa_hub_class,
                    "hub_tier": hub_tier,
                    "hub_source": "airport_list_default",
                    "hub_source_year": np.nan,
                    "cy_24_enplanements": np.nan,
                }
        rows.append(
            {
                "airport_code": code,
                "airport_name": ap.get("name"),
                "city_name": city_name,
                "state_abbr": state_abbr,
                "country": country_from_state(state_abbr),
                "lat": float(ap.get("lat")),
                "lon": float(ap.get("lon")),
                "faa_hub_class": hub_meta["faa_hub_class"],
                "hub_tier": hub_meta["hub_tier"],
                "hub_source": hub_meta["hub_source"],
                "hub_source_year": hub_meta["hub_source_year"],
                "cy_24_enplanements": hub_meta["cy_24_enplanements"],
                "operational_zone_label": get_airport_operational_zone(code)["label"],
                "operational_zone_rank": get_airport_operational_zone(code)["rank"],
                "utc_offset_standard": get_airport_operational_zone(code)["utc_offset_standard"],
            }
        )
    df = pd.DataFrame(rows)
    df["city_norm"] = df["city_name"].map(normalize_name)
    return df


def get_flight_cost(airport: object) -> float:
    """Return origin-airport flight cost using the shared planning model."""
    code = "" if airport is None else str(airport).strip().upper()
    raw = config.BTS_RAW_ITINERARY_FARES.get(code, config.BTS_NATIONAL_FALLBACK)
    return raw * config.CORPORATE_TRAVEL_PREMIUM


def compute_trip_span_days(avg_days: float | None) -> int:
    """Round modeled calendar-window duration up to whole trip-span days."""
    if avg_days is None or pd.isna(avg_days):
        avg_days = config.HOTEL_AVG_NIGHTS
    return max(1, int(np.ceil(float(avg_days))))


def compute_ground_transport(
    median_dist_mi: float,
    trip_span_days: int,
) -> dict:
    """Return drive-mode ground transport costs for a one-way median distance."""
    if median_dist_mi < config.PERSONAL_VEHICLE_MAX_ONE_WAY_MI:
        return {
            "ground_transport_mode": "personal_vehicle",
            "rental_days": 0,
            "mileage_cost_usd": config.IRS_MILEAGE_RATE_USD_PER_MI * 2.0 * median_dist_mi,
            "rental_cost_usd": 0.0,
        }
    return {
        "ground_transport_mode": "rental_car",
        "rental_days": trip_span_days,
        "mileage_cost_usd": 0.0,
        "rental_cost_usd": config.RENTAL_CAR_DAILY_RATE_USD * trip_span_days,
    }


def apply_travel_cost_policy(
    employee_style_unit_cost: float,
    travel_cost_policy: str,
    contractor_cost_multiplier: float,
    contractor_cost_cap_usd: float,
    contractor_dispatch_surcharge_usd: float,
) -> float:
    """Transform employee-style cost into the effective unit cost used by the model."""
    if travel_cost_policy == config.TRAVEL_COST_POLICY_CONTRACTOR:
        effective = (
            employee_style_unit_cost * contractor_cost_multiplier
            + contractor_dispatch_surcharge_usd
        )
        return min(effective, contractor_cost_cap_usd)
    return employee_style_unit_cost


def prepare_demand(demand_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize demand appointments for node-based travel-cost computation."""
    df = demand_df.copy()
    df["state_norm"] = df["state_norm"].map(normalize_state)
    df["skill_class"] = df["skill_class"].astype(str)
    df["node_id"] = df["state_norm"] + "__" + df["skill_class"]
    df = df.dropna(subset=["state_norm", "node_id"])
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])
    return df


def compute_entity_node_costs(
    entities: list[dict],
    demand: pd.DataFrame,
    node_avg_days: dict[str, float] | None = None,
) -> list[dict]:
    """Compute full per-trip costs for all (entity, node) pairs."""
    node_groups: dict[str, pd.DataFrame] = {}
    for node_id, grp in demand.groupby("node_id"):
        node_groups[node_id] = grp

    rows: list[dict] = []
    warned_nodes: set[str] = set()

    for entity in entities:
        entity_id = entity["id"]
        base_lat = float(entity["lat"]) if pd.notna(entity["lat"]) else float("nan")
        base_lon = float(entity["lon"]) if pd.notna(entity["lon"]) else float("nan")
        airport = str(entity["airport"]).strip()
        travel_cost_policy = str(
            entity.get("travel_cost_policy", config.TRAVEL_COST_POLICY_EMPLOYEE)
        ).strip() or config.TRAVEL_COST_POLICY_EMPLOYEE
        contractor_cost_multiplier = float(
            entity.get("contractor_cost_multiplier", 1.0)
            if pd.notna(entity.get("contractor_cost_multiplier", 1.0))
            else 1.0
        )
        contractor_cost_cap_usd = float(
            entity.get("contractor_cost_cap_usd", np.inf)
            if pd.notna(entity.get("contractor_cost_cap_usd", np.inf))
            else np.inf
        )
        contractor_dispatch_surcharge_usd = float(
            entity.get("contractor_dispatch_surcharge_usd", 0.0)
            if pd.notna(entity.get("contractor_dispatch_surcharge_usd", 0.0))
            else 0.0
        )

        if np.isnan(base_lat) or np.isnan(base_lon):
            warnings.warn(
                f"Entity {entity_id} has no valid lat/lon; skipping all node costs.",
                stacklevel=2,
            )
            continue

        for node_id, node_appts in node_groups.items():
            lats = node_appts["lat"].values
            lons = node_appts["lon"].values

            if len(lats) == 0:
                if node_id not in warned_nodes:
                    warned_nodes.add(node_id)
                    print(f"  WARNING: node {node_id} has 0 valid appointments; skipping.")
                continue

            dists_mi = np.array(
                [
                    haversine_miles(base_lat, base_lon, float(la), float(lo))
                    for la, lo in zip(lats, lons)
                ]
            )
            median_dist = float(np.median(dists_mi))
            avg_days = (
                node_avg_days.get(node_id, config.HOTEL_AVG_NIGHTS)
                if node_avg_days
                else config.HOTEL_AVG_NIGHTS
            )
            trip_span_days = compute_trip_span_days(avg_days)

            if median_dist < config.SAME_DAY_DRIVE_THRESHOLD_MI:
                trip_mode = "drive_day"
                ground = compute_ground_transport(median_dist, trip_span_days)
                mileage_cost = ground["mileage_cost_usd"]
                flight_cost = 0.0
                rental_cost = ground["rental_cost_usd"]
                hotel_nights = 0
                hotel_cost = 0.0
                ground_transport_mode = ground["ground_transport_mode"]
                rental_days = ground["rental_days"]
                employee_style_unit_cost = mileage_cost + rental_cost
            elif median_dist < config.OVERNIGHT_DRIVE_THRESHOLD_MI:
                trip_mode = "drive_overnight"
                ground = compute_ground_transport(median_dist, trip_span_days)
                mileage_cost = ground["mileage_cost_usd"]
                flight_cost = 0.0
                rental_cost = ground["rental_cost_usd"]
                hotel_nights = 1
                hotel_cost = config.HOTEL_NIGHTLY_RATE_USD
                ground_transport_mode = ground["ground_transport_mode"]
                rental_days = ground["rental_days"]
                employee_style_unit_cost = mileage_cost + rental_cost + hotel_cost
            else:
                trip_mode = "fly"
                hotel_nights = trip_span_days
                flight_cost = get_flight_cost(airport)
                mileage_cost = 0.0
                rental_days = trip_span_days
                rental_cost = config.RENTAL_CAR_DAILY_RATE_USD * rental_days
                hotel_cost = hotel_nights * config.HOTEL_NIGHTLY_RATE_USD
                ground_transport_mode = "rental_car"
                employee_style_unit_cost = flight_cost + rental_cost + hotel_cost

            effective_unit_cost = apply_travel_cost_policy(
                employee_style_unit_cost=employee_style_unit_cost,
                travel_cost_policy=travel_cost_policy,
                contractor_cost_multiplier=contractor_cost_multiplier,
                contractor_cost_cap_usd=contractor_cost_cap_usd,
                contractor_dispatch_surcharge_usd=contractor_dispatch_surcharge_usd,
            )

            rows.append(
                {
                    "tech_or_candidate_id": entity_id,
                    "node_id": node_id,
                    "trip_mode": trip_mode,
                    "median_dist_mi": round(median_dist, 2),
                    "trip_span_days": trip_span_days,
                    "ground_transport_mode": ground_transport_mode,
                    "flight_cost_usd": round(flight_cost, 2),
                    "mileage_cost_usd": round(mileage_cost, 2),
                    "rental_cost_usd": round(rental_cost, 2),
                    "rental_days": rental_days,
                    "hotel_cost_usd": round(hotel_cost, 2),
                    "hotel_nights": hotel_nights,
                    "travel_cost_policy": travel_cost_policy,
                    "contractor_cost_multiplier": contractor_cost_multiplier,
                    "contractor_cost_cap_usd": (
                        round(contractor_cost_cap_usd, 2)
                        if np.isfinite(contractor_cost_cap_usd)
                        else np.nan
                    ),
                    "contractor_dispatch_surcharge_usd": round(
                        contractor_dispatch_surcharge_usd, 2
                    ),
                    "employee_style_unit_cost_usd": round(employee_style_unit_cost, 2),
                    "effective_unit_cost_usd": round(effective_unit_cost, 2),
                    "unit_cost_usd": round(effective_unit_cost, 2),
                }
            )

    return rows
