"""Step 6: Build optimization inputs from source-of-truth external files."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import config
from optimization_utils import (
    build_airports_df,
    country_from_state,
    get_airport_operational_zone,
    nearest_airport_code,
    normalize_name,
    normalize_state,
    parse_city_state,
    slugify,
)


DEFAULT_APPOINTMENTS_XLSX = config.EXTERNAL_APPOINTMENTS_XLSX
DEFAULT_TECH_XLSX = config.EXTERNAL_TECH_ROSTER_XLSX

RAW_APPOINTMENTS_SHEET = "report1770130594436"
DERIVED_APPOINTMENTS_SHEET = "Derived Fields"

# Tech name aliases are now defined canonically in config.TECH_NAME_ALIASES.
TECH_NAME_ALIASES = config.TECH_NAME_ALIASES

TECH_LOCATION_AIRPORT_OVERRIDES = {
    "charlotte nc": "CLT",
    "st louis mo": "STL",
    "st louis, mo": "STL",
    "ontario ca": "ONT",
    "tampa fl": "TPA",
    "phoenix az": "PHX",
    "philadelphia pa": "PHL",
    "baltimore md": "BWI",
    "houston tx": "IAH",
}

HPS_PATTERNS = [
    r"\bhps[0-9a-z]*\b",
    r"\bphps[0-9a-z]*\b",
    r"\bmeti[ _-]*hps\b",
]

LS_PATTERNS = [
    r"learning\s*space",
    r"\bmlsp\b",
    r"\bls\b",
]

EXCLUDED_US_TERRITORIES = {
    "US VIRGIN ISLANDS",
    "VIRGIN ISLANDS",
    "VI",
}


def resolve_path(candidates: list[str], label: str) -> str:
    """Return first existing file path."""
    for path in candidates:
        if not path:
            continue
        if os.path.exists(path):
            return path
    options = "\n".join(f"- {p}" for p in candidates if p)
    raise FileNotFoundError(f"Could not resolve {label}. Tried:\n{options}")


def yesno_to_bool(value: object) -> bool:
    """Map yes/no text to bool."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return str(value).strip().lower() == "yes"


def choose_airport_for_location(location: str, airports_df: pd.DataFrame) -> str | None:
    """Map location string to nearest/sensible airport code."""
    loc_norm = normalize_name(location)
    if loc_norm in TECH_LOCATION_AIRPORT_OVERRIDES:
        return TECH_LOCATION_AIRPORT_OVERRIDES[loc_norm]

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
            # Prefer a partial city-name match within the state before taking first entry.
            city_match = same_state[same_state["city_norm"].str.contains(city_norm, na=False)] if city_norm else pd.DataFrame()
            if not city_match.empty:
                return str(city_match.iloc[0]["airport_code"])
            print(f"  NOTE: no exact airport match for '{location}'; using first {state_abbr} airport: {same_state.iloc[0]['airport_code']}")
            return str(same_state.iloc[0]["airport_code"])
    if city_norm:
        near_city = airports_df[airports_df["city_norm"].str.contains(city_norm, na=False)]
        if not near_city.empty:
            return str(near_city.iloc[0]["airport_code"])
    return None


def classify_skill(description: object, subject: object) -> dict:
    """Classify skill requirements from text."""
    combined = " | ".join(
        [
            str(x).strip()
            for x in [description, subject]
            if x is not None and not (isinstance(x, float) and pd.isna(x)) and str(x).strip()
        ]
    )
    text = combined.lower()

    hps_hits = [p for p in HPS_PATTERNS if re.search(p, text)]
    ls_hits = [p for p in LS_PATTERNS if re.search(p, text)]
    requires_hps = bool(hps_hits)
    requires_ls = bool(ls_hits)

    if requires_hps and requires_ls:
        skill_class = "hps_ls"
        confidence = "high"
        reason = "matched_hps_and_ls_patterns"
    elif requires_hps:
        skill_class = "hps"
        confidence = "high"
        reason = "matched_hps_patterns"
    elif requires_ls:
        skill_class = "ls"
        confidence = "high"
        reason = "matched_learning_space_patterns"
    else:
        skill_class = "regular"
        confidence = "high"
        reason = "default_regular_no_hps_ls_signal"

    return {
        "required_hps": int(requires_hps),
        "required_ls": int(requires_ls),
        "required_patient": 1,
        "skill_class": skill_class,
        "parse_confidence": confidence,
        "parse_reason": reason,
    }


def parse_model_hint(description: object, subject: object) -> str:
    """Extract broad model family hints for analysis/debugging."""
    combined = " ".join(
        [
            str(x).upper()
            for x in [description, subject]
            if x is not None and not (isinstance(x, float) and pd.isna(x))
        ]
    )
    families = []
    for token in ["HPS", "MFS", "APN", "APP", "ARIA", "JUNO", "APO", "LUCINA"]:
        if re.search(rf"\b{token}[0-9A-Z-]*\b", combined):
            families.append(token)
    if not families:
        return "UNKNOWN"
    return ",".join(sorted(set(families)))


def lookup_airport_zone_fields(airport_code: object, airports_df: pd.DataFrame) -> dict:
    """Return operational-zone fields for an airport code using the airport table."""
    code = "" if airport_code is None else str(airport_code).strip().upper()
    if code:
        match = airports_df[airports_df["airport_code"] == code]
        if not match.empty:
            row = match.iloc[0]
            return {
                "operational_zone_label": row.get("operational_zone_label", "Unknown"),
                "operational_zone_rank": row.get("operational_zone_rank"),
                "utc_offset_standard": row.get("utc_offset_standard"),
            }
    fallback = get_airport_operational_zone(code)
    return {
        "operational_zone_label": fallback["label"],
        "operational_zone_rank": fallback["rank"],
        "utc_offset_standard": fallback["utc_offset_standard"],
    }


def assignment_scope_defaults(is_contractor: bool, contractor_scope: str) -> dict:
    """Return per-tech assignment scope defaults."""
    if not is_contractor:
        return {
            "assignment_scope_mode": config.ASSIGNMENT_SCOPE_MODE_NATIONAL,
            "assignment_scope_state": "",
        }
    if contractor_scope == "texas_only":
        return {
            "assignment_scope_mode": config.ASSIGNMENT_SCOPE_MODE_STATE_LIMITED,
            "assignment_scope_state": "TX",
        }
    return {
        "assignment_scope_mode": config.ASSIGNMENT_SCOPE_MODE_NATIONAL,
        "assignment_scope_state": "",
    }


def enrich_tech_policy_fields(
    tech_master: pd.DataFrame,
    airports_df: pd.DataFrame,
    contractor_scope: str,
) -> pd.DataFrame:
    """Attach operational-zone and policy fields to tech_master rows."""
    if tech_master.empty:
        return tech_master.copy()

    rows = []
    for _, row in tech_master.iterrows():
        enriched = row.to_dict()
        employment_type = str(enriched.get("employment_type", "")).strip().lower()
        is_contractor = employment_type == "contractor"
        scope = assignment_scope_defaults(is_contractor, contractor_scope)
        zone = lookup_airport_zone_fields(enriched.get("base_airport_iata"), airports_df)

        enriched["contractor_assignment_scope"] = (
            contractor_scope if is_contractor else enriched.get("contractor_assignment_scope", "")
        )
        enriched["assignment_scope_mode"] = scope["assignment_scope_mode"]
        enriched["assignment_scope_state"] = scope["assignment_scope_state"]
        enriched["travel_cost_policy"] = (
            config.TRAVEL_COST_POLICY_CONTRACTOR if is_contractor else config.TRAVEL_COST_POLICY_EMPLOYEE
        )
        enriched["contractor_cost_multiplier"] = (
            config.CONTRACTOR_COST_MULTIPLIER if is_contractor else 1.0
        )
        enriched["contractor_cost_cap_usd"] = (
            config.CONTRACTOR_COST_CAP_USD if is_contractor else np.nan
        )
        enriched["contractor_dispatch_surcharge_usd"] = (
            config.CONTRACTOR_DISPATCH_SURCHARGE_USD if is_contractor else 0.0
        )
        enriched["zone_policy"] = (
            config.ZONE_POLICY_CONTRACTOR_SOFT if is_contractor else config.ZONE_POLICY_STANDARD
        )
        enriched["base_operational_zone_label"] = zone["operational_zone_label"]
        enriched["base_operational_zone_rank"] = zone["operational_zone_rank"]
        enriched["base_utc_offset_standard"] = zone["utc_offset_standard"]
        rows.append(enriched)

    return pd.DataFrame(rows).sort_values("tech_name").reset_index(drop=True)


def enrich_candidate_zone_fields(
    candidates: pd.DataFrame,
    airports_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach operational-zone fields to candidate bases."""
    if candidates.empty:
        return candidates.copy()

    rows = []
    for _, row in candidates.iterrows():
        enriched = row.to_dict()
        zone = lookup_airport_zone_fields(enriched.get("airport_iata"), airports_df)
        enriched["operational_zone_label"] = zone["operational_zone_label"]
        enriched["operational_zone_rank"] = zone["operational_zone_rank"]
        enriched["utc_offset_standard"] = zone["utc_offset_standard"]
        rows.append(enriched)

    return pd.DataFrame(rows)


def build_tech_master(
    tech_path: str,
    airports_df: pd.DataFrame,
    contractor_scope: str,
) -> pd.DataFrame:
    """Build canonical technician table."""
    if str(tech_path).lower().endswith(".csv"):
        cached = pd.read_csv(tech_path)
        required_cols = {
            "tech_id",
            "tech_name",
            "source_name",
            "employment_type",
            "base_location_raw",
            "base_city",
            "base_state",
            "base_country",
            "base_airport_iata",
            "base_lat",
            "base_lon",
            "skill_hps",
            "skill_ls",
            "skill_patient",
            "availability_fte",
            "fixed_base",
            "can_relocate",
            "contractor_assignment_scope",
            "constraint_florida_only",
            "notes",
        }
        missing = sorted(required_cols - set(cached.columns))
        if missing:
            raise ValueError(
                f"Cached tech_master CSV at {tech_path} is missing required columns: {missing}"
            )
        valid_airports = set(airports_df["airport_code"].astype(str))
        airport_series = cached["base_airport_iata"].fillna("").astype(str).str.strip()
        invalid_mask = airport_series.ne("") & ~airport_series.isin(valid_airports)
        if invalid_mask.any():
            invalid_values = sorted(set(airport_series[invalid_mask].tolist()))
            cached.loc[invalid_mask, "base_airport_iata"] = ""
            print(
                "  NOTE: cleared unsupported cached airport code(s): "
                + ", ".join(invalid_values)
            )
        print("  NOTE: using cached tech_master CSV fallback.")
        return enrich_tech_policy_fields(
            cached.sort_values("tech_name").reset_index(drop=True),
            airports_df,
            contractor_scope,
        )

    tech = None

    # Preferred "Tech/Location/Comments" workbook layout.
    try:
        raw = pd.read_excel(tech_path, sheet_name=0, header=None)
        if len(raw) >= 3:
            headers = ["" if pd.isna(h) else str(h).strip() for h in raw.iloc[1].tolist()]
            parsed = raw.iloc[2:].copy()
            parsed.columns = headers
            parsed = parsed.dropna(how="all")
            if {"Tech", "Location", "Comments"}.issubset(parsed.columns):
                tech = parsed[["Tech", "Location", "Comments"]].copy()
    except Exception:
        tech = None

    # Fallback legacy layout from Service Appointments report.
    if tech is None:
        fallback = pd.read_excel(tech_path, sheet_name=config.APPTS_REPORT_RESOURCES_SHEET)
        missing_cols = [
            c for c in ["Service Resource Name", "Location", "Comment"] if c not in fallback.columns
        ]
        if missing_cols:
            raise ValueError(
                f"Unsupported technician workbook format at {tech_path}. "
                f"Missing columns: {missing_cols}"
            )
        tech = fallback.rename(
            columns={
                "Service Resource Name": "Tech",
                "Comment": "Comments",
            }
        )[["Tech", "Location", "Comments"]].copy()
        print(
            "  NOTE: technician workbook did not match source-of-truth schema; "
            "using Resources-sheet fallback columns."
        )

    for col in tech.columns:
        tech[col] = tech[col].apply(lambda v: "" if pd.isna(v) else str(v).strip())

    rows = []
    for _, row in tech.iterrows():
        name_raw = row.get("Tech", "").strip()
        if not name_raw:
            continue
        name_norm = normalize_name(name_raw)
        canonical_name = TECH_NAME_ALIASES.get(name_norm, name_raw)

        location = row.get("Location", "").strip()
        city, state_abbr = parse_city_state(location)
        country = country_from_state(state_abbr)

        airport_code = choose_airport_for_location(location, airports_df)
        airport_lat = np.nan
        airport_lon = np.nan
        if airport_code:
            ap = airports_df[airports_df["airport_code"] == airport_code]
            if not ap.empty:
                airport_lat = float(ap.iloc[0]["lat"])
                airport_lon = float(ap.iloc[0]["lon"])
            else:
                print(
                    f"  WARNING: '{canonical_name}' mapped to unknown airport "
                    f"'{airport_code}' (not in MAJOR_AIRPORTS); leaving airport blank."
                )
                airport_code = None

        is_contractor = "contractor" in name_raw.lower()
        availability = 0.5 if is_contractor else 1.0
        if canonical_name == "Tameka Gongs":
            availability = 0.25
        if canonical_name == "James Sanchez":
            availability = 1.0  # Full field tech; temporarily on phones (injury) but model targets ideal state
        if canonical_name == "Hakim Mouazer":
            availability = 0.0
        if canonical_name == "Elier Martin":
            availability = 0.10  # Phone tech, confirmed by Shannon/Isabelle at ~10% field
        if canonical_name == "Damion Lyn":
            availability = 0.20  # Repair center hybrid, 20-25% field when fully staffed (conservative)

        note = row.get("Comments", "").strip()
        florida_only = int("only covers florida" in note.lower())

        rows.append(
            {
                "tech_id": slugify(canonical_name),
                "tech_name": canonical_name,
                "source_name": name_raw,
                "employment_type": "contractor" if is_contractor else "fte",
                "base_location_raw": location,
                "base_city": city,
                "base_state": state_abbr,
                "base_country": country,
                "base_airport_iata": airport_code or "",
                "base_lat": airport_lat,
                "base_lon": airport_lon,
                "skill_hps": int(yesno_to_bool(row.get("HPS"))),
                "skill_ls": int(yesno_to_bool(row.get("LearningSpace"))),
                "skill_patient": int(yesno_to_bool(row.get("All other Patient Sims"))),
                "availability_fte": float(availability),
                "fixed_base": 1,
                "can_relocate": 0,
                "contractor_assignment_scope": contractor_scope if is_contractor else "",
                "constraint_florida_only": florida_only,
                "notes": note,
            }
        )

    tech_master = pd.DataFrame(rows).sort_values("tech_name").reset_index(drop=True)
    return enrich_tech_policy_fields(tech_master, airports_df, contractor_scope)


def build_demand_appointments(
    appts_path: str,
    geocoded_appts_csv: str,
    airports_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build appointment-level demand table with parsed skill requirements."""
    derived = pd.read_excel(appts_path, sheet_name=DERIVED_APPOINTMENTS_SHEET)
    raw = pd.read_excel(appts_path, sheet_name=RAW_APPOINTMENTS_SHEET)

    keep_raw = ["Appointment Number", "Description", "Subject"]
    raw = raw[keep_raw].copy()
    demand = derived.merge(raw, on="Appointment Number", how="left")

    if os.path.exists(geocoded_appts_csv):
        geo = pd.read_csv(geocoded_appts_csv, usecols=["Appointment Number", "lat", "lon"])
        demand = demand.merge(geo, on="Appointment Number", how="left")
    else:
        demand["lat"] = np.nan
        demand["lon"] = np.nan

    demand["appointment_id"] = demand["Appointment Number"].astype(str).str.strip()
    demand["scheduled_start"] = pd.to_datetime(demand["Scheduled Start"], errors="coerce")
    demand["scheduled_end"] = pd.to_datetime(demand["Scheduled End"], errors="coerce")

    # Keep appointment duration in calendar-window hours because that is the
    # workload basis used by the current optimization model. This is not a pure
    # hands-on labor measure or a timesheet-style hours-worked field.
    duration_hours = pd.to_numeric(demand["Duration Hours"], errors="coerce")
    duration_days = pd.to_numeric(demand["Duration Days"], errors="coerce")
    duration_hours = duration_hours.fillna(duration_days * 24)
    duration_hours = duration_hours.fillna(8.0)
    duration_hours = duration_hours.clip(lower=0.25)
    demand["duration_hours"] = duration_hours

    demand["city"] = demand["City"].astype(str).str.strip()
    demand["state_raw"] = demand["State/Province"].astype(str).str.strip()
    demand["state_norm"] = demand["state_raw"].map(normalize_state)
    demand["country"] = demand["state_norm"].map(country_from_state)
    # Exclude Canadian appointments and US territories outside scope.
    n_before = len(demand)
    state_norm_upper = demand["state_norm"].astype(str).str.upper()
    canada_mask = demand["country"] == "Canada"
    excluded_territory_mask = state_norm_upper.isin(EXCLUDED_US_TERRITORIES)
    demand = demand[~(canada_mask | excluded_territory_mask)].copy()
    n_dropped = n_before - len(demand)
    if n_dropped:
        n_canada = int(canada_mask.sum())
        n_territory = int(excluded_territory_mask.sum())
        print(
            "  Filtered out "
            f"{n_dropped} appointment(s) outside optimization scope "
            f"({n_canada} Canada, {n_territory} excluded territory)."
        )
    demand["territory"] = demand["Territory"].astype(str).str.strip()

    parsed = demand.apply(
        lambda r: classify_skill(r.get("Description"), r.get("Subject")),
        axis=1,
        result_type="expand",
    )
    demand = pd.concat([demand, parsed], axis=1)

    # AVS (Audiovisual Solutions) = Learning Space — confirmed by Shannon/Isabelle.
    # AVS was a temporary name; all AVS appointments require LS-certified techs.
    avs_mask = demand["Service Type"].astype(str).str.upper().str.contains("AVS", na=False)
    n_avs = int(avs_mask.sum())
    if n_avs > 0:
        demand.loc[avs_mask & (demand["required_ls"] == 0), "required_ls"] = 1
        demand.loc[avs_mask & (demand["skill_class"] == "regular"), "skill_class"] = "ls"
        demand.loc[avs_mask & (demand["skill_class"] == "hps"), "skill_class"] = "hps_ls"
        demand.loc[avs_mask, "parse_reason"] = demand.loc[avs_mask, "parse_reason"] + "__avs_is_ls"
        print(f"  AVS→LS mapping: {n_avs} AVS appointments reclassified as LS-requiring.")

    demand["model_hint"] = demand.apply(
        lambda r: parse_model_hint(r.get("Description"), r.get("Subject")),
        axis=1,
    )

    # Fill remaining lat/lon from city-state centroid if needed
    city_centroids = (
        demand.dropna(subset=["lat", "lon", "city", "state_norm"])
        .groupby(["city", "state_norm"])[["lat", "lon"]]
        .mean()
        .reset_index()
    )
    demand = demand.merge(
        city_centroids,
        on=["city", "state_norm"],
        how="left",
        suffixes=("", "_city_centroid"),
    )
    demand["lat"] = demand["lat"].fillna(demand["lat_city_centroid"])
    demand["lon"] = demand["lon"].fillna(demand["lon_city_centroid"])
    demand = demand.drop(columns=["lat_city_centroid", "lon_city_centroid"])

    null_location_count = int(demand["lat"].isna().sum())
    if null_location_count:
        print(f"  WARNING: {null_location_count} demand appointments have no geocoded lat/lon and will have no nearest hub airport.")

    nearest = demand.apply(
        lambda r: nearest_airport_code(r["lat"], r["lon"], airports_df, r["state_norm"]),
        axis=1,
    )
    demand["nearest_hub_airport"] = [x[0] for x in nearest]
    demand["nearest_hub_distance_km"] = [x[1] for x in nearest]
    nearest_zone = demand["nearest_hub_airport"].apply(
        lambda code: lookup_airport_zone_fields(code, airports_df)
    )
    demand["nearest_hub_operational_zone_label"] = [item["operational_zone_label"] for item in nearest_zone]
    demand["nearest_hub_operational_zone_rank"] = [item["operational_zone_rank"] for item in nearest_zone]
    demand["nearest_hub_utc_offset_standard"] = [item["utc_offset_standard"] for item in nearest_zone]

    output_cols = [
        "appointment_id",
        "Appointment Number",
        "Account: Account Name",
        "Service Resource: Name",
        "territory",
        "city",
        "state_raw",
        "state_norm",
        "country",
        "scheduled_start",
        "scheduled_end",
        "Dispatched Date/Time",
        "duration_hours",
        "Service Type",
        "Description",
        "Subject",
        "lat",
        "lon",
        "nearest_hub_airport",
        "nearest_hub_distance_km",
        "nearest_hub_operational_zone_label",
        "nearest_hub_operational_zone_rank",
        "nearest_hub_utc_offset_standard",
        "required_hps",
        "required_ls",
        "required_patient",
        "skill_class",
        "parse_confidence",
        "parse_reason",
        "model_hint",
    ]
    return demand[output_cols].copy()


def build_candidate_bases(
    airports_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    top_demand_cities: int,
) -> pd.DataFrame:
    """Build candidate base pool = major airports + top demand cities."""
    airports = airports_df.copy()
    airports["candidate_id"] = airports["airport_code"].map(lambda c: f"airport_{c.lower()}")
    airports["candidate_type"] = "major_airport"
    airports["is_major_airport"] = 1
    airports["demand_rank"] = np.nan
    airports["demand_hours"] = 0.0
    airports["demand_appointments"] = 0
    airports["city"] = airports["city_name"]
    airports["state"] = airports["state_abbr"]
    airports["airport_iata"] = airports["airport_code"]
    major = airports[
        [
            "candidate_id",
            "candidate_type",
            "is_major_airport",
            "city",
            "state",
            "country",
            "lat",
            "lon",
            "airport_iata",
            "demand_rank",
            "demand_hours",
            "demand_appointments",
        ]
    ].copy()

    demand_city = (
        demand_df.dropna(subset=["city", "state_norm", "lat", "lon"])
        .groupby(["city", "state_norm", "country"])
        .agg(
            demand_hours=("duration_hours", "sum"),
            demand_appointments=("appointment_id", "count"),
            lat=("lat", "mean"),
            lon=("lon", "mean"),
        )
        .reset_index()
        .sort_values("demand_hours", ascending=False)
        .head(top_demand_cities)
    )
    demand_city["airport_iata"] = demand_city.apply(
        lambda r: nearest_airport_code(r["lat"], r["lon"], airports_df, r["state_norm"])[0],
        axis=1,
    )
    demand_city["candidate_id"] = demand_city.apply(
        lambda r: "demand_{}".format(slugify("{}_{}".format(r["city"], r["state_norm"]))),
        axis=1,
    )
    demand_city["candidate_type"] = "demand_city"
    demand_city["is_major_airport"] = 0
    demand_city["demand_rank"] = range(1, len(demand_city) + 1)
    demand_city = demand_city.rename(columns={"state_norm": "state"})

    major_city_state = set(zip(major["city"].map(normalize_name), major["state"]))
    demand_city = demand_city[
        ~demand_city.apply(
            lambda r: (normalize_name(r["city"]), r["state"]) in major_city_state,
            axis=1,
        )
    ]
    demand_city = demand_city[
        [
            "candidate_id",
            "candidate_type",
            "is_major_airport",
            "city",
            "state",
            "country",
            "lat",
            "lon",
            "airport_iata",
            "demand_rank",
            "demand_hours",
            "demand_appointments",
        ]
    ]

    candidates = pd.concat([major, demand_city], ignore_index=True)
    candidates = candidates.drop_duplicates(subset=["candidate_id"]).reset_index(drop=True)
    return enrich_candidate_zone_fields(candidates, airports_df)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build optimization input tables.")
    parser.add_argument("--appts-xlsx", default=None, help="Appointments workbook path.")
    parser.add_argument("--tech-xlsx", default=None, help="Technician workbook path.")
    parser.add_argument(
        "--output-dir",
        default=config.OPTIMIZATION_DIR,
        help="Optimization output directory.",
    )
    parser.add_argument(
        "--top-demand-cities",
        type=int,
        default=40,
        help="How many high-demand cities to add to candidate pool.",
    )
    parser.add_argument(
        "--contractor-assignment-scope",
        choices=["texas_only", "anywhere"],
        default="anywhere",
        help="Legacy contractor geography override for generated tech policies.",
    )
    args = parser.parse_args()

    appts_path = resolve_path(
        [
            args.appts_xlsx,
            os.environ.get("ELEVATE_APPTS_SOURCE"),
            DEFAULT_APPOINTMENTS_XLSX,
            config.SERVICE_APPTS_DISPATCH,
        ],
        "appointments workbook",
    )
    tech_path = resolve_path(
        [
            args.tech_xlsx,
            os.environ.get("ELEVATE_TECH_SOURCE"),
            DEFAULT_TECH_XLSX,
            os.path.join(config.OPTIMIZATION_DIR, "tech_master.csv"),
            config.SERVICE_APPTS_REPORT,
        ],
        "technician workbook",
    )

    airports_df = build_airports_df(config.MAJOR_AIRPORTS)
    tech_master = build_tech_master(tech_path, airports_df, args.contractor_assignment_scope)
    demand = build_demand_appointments(appts_path, config.GEOCODED_APPTS_CSV, airports_df)
    candidates = build_candidate_bases(airports_df, demand, args.top_demand_cities)

    # Compute data time span for annualization
    sched_start_dt = pd.to_datetime(demand["scheduled_start"], errors="coerce").dropna()
    date_min = sched_start_dt.min()
    date_max = sched_start_dt.max()
    data_span_days = max((date_max - date_min).days + 1, 1)
    data_span_years = max(data_span_days / 365.25, 0.5)  # floor at 0.5 to avoid division issues

    if data_span_years < 0.8:
        print(f"  WARNING: Data span is only {data_span_years:.2f} years — check date range")
    print(f"  Date range: {date_min.date()} to {date_max.date()} ({data_span_days} days, {data_span_years:.2f} years)")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tech_path_out = out_dir / "tech_master.csv"
    demand_path_out = out_dir / "demand_appointments.csv"
    candidate_path_out = out_dir / "candidate_bases.csv"
    parse_ambiguous_out = out_dir / "skill_parse_ambiguous.csv"
    summary_out = out_dir / "optimization_input_summary.json"

    tech_master.to_csv(tech_path_out, index=False)
    demand.to_csv(demand_path_out, index=False)
    candidates.to_csv(candidate_path_out, index=False)
    demand[demand["parse_confidence"] != "high"].to_csv(parse_ambiguous_out, index=False)

    summary = {
        "appointments_source": appts_path,
        "tech_source": tech_path,
        "rows": {
            "tech_master": int(len(tech_master)),
            "demand_appointments": int(len(demand)),
            "candidate_bases": int(len(candidates)),
        },
        "skills": {
            "hps_required_appointments": int(demand["required_hps"].sum()),
            "ls_required_appointments": int(demand["required_ls"].sum()),
            "hps_ls_required_appointments": int((demand["skill_class"] == "hps_ls").sum()),
        },
        "contractor_assignment_scope": args.contractor_assignment_scope,
        "contractor_policy_defaults": {
            "travel_cost_policy": config.TRAVEL_COST_POLICY_CONTRACTOR,
            "zone_policy": config.ZONE_POLICY_CONTRACTOR_SOFT,
            "contractor_cost_multiplier": config.CONTRACTOR_COST_MULTIPLIER,
            "contractor_cost_cap_usd": config.CONTRACTOR_COST_CAP_USD,
            "contractor_dispatch_surcharge_usd": config.CONTRACTOR_DISPATCH_SURCHARGE_USD,
        },
        "operational_zone_anchor": "airport_based",
        "data_span_days": int(data_span_days),
        "data_span_years": round(float(data_span_years), 4),
        "date_range_start": str(date_min.date()),
        "date_range_end": str(date_max.date()),
        "annualized_appointment_count": round(len(demand) / data_span_years, 1),
    }
    with open(summary_out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved: {tech_path_out}")
    print(f"Saved: {demand_path_out}")
    print(f"Saved: {candidate_path_out}")
    print(f"Saved: {parse_ambiguous_out}")
    print(f"Saved: {summary_out}")
    print("\nInput summary:")
    print(json.dumps(summary, indent=2))
    print("Step 6 complete.")


if __name__ == "__main__":
    main()
