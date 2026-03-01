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
    "montreal qc": "YUL",
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


def build_tech_master(
    tech_path: str,
    airports_df: pd.DataFrame,
    contractor_scope: str,
) -> pd.DataFrame:
    """Build canonical technician table."""
    raw = pd.read_excel(tech_path, sheet_name=0, header=None)
    headers = raw.iloc[1].tolist()
    tech = raw.iloc[2:].copy()
    tech.columns = headers
    tech = tech.dropna(how="all")

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

        is_contractor = "contractor" in name_raw.lower()
        availability = 0.5 if is_contractor else 1.0
        if canonical_name == "Tameka Gongs":
            availability = 0.25
        if canonical_name == "James Sanchez":
            availability = 0.0
        if canonical_name in ("Damion Lyn", "Elier Martin"):
            availability = 0.10

        note = row.get("Comments", "").strip()
        florida_only = int("only covers florida" in note.lower())
        canada_wide = int("covers all of canada" in note.lower())

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
                "base_airport_iata": airport_code,
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
                "constraint_canada_wide": canada_wide,
                "notes": note,
            }
        )

    tech_master = pd.DataFrame(rows).sort_values("tech_name").reset_index(drop=True)
    return tech_master


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
    demand["territory"] = demand["Territory"].astype(str).str.strip()

    parsed = demand.apply(
        lambda r: classify_skill(r.get("Description"), r.get("Subject")),
        axis=1,
        result_type="expand",
    )
    demand = pd.concat([demand, parsed], axis=1)
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
    return candidates


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
        default="texas_only",
        help="Contractor assignment geography assumption.",
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
    data_span_days = (date_max - date_min).days
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
