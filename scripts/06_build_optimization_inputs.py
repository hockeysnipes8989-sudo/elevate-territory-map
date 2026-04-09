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
    canonicalize_tech_name,
    choose_airport_for_location,
    compute_entity_node_costs,
    country_from_state,
    get_airport_operational_zone,
    nearest_airport_code,
    normalize_name,
    normalize_state,
    parse_city_state,
    prepare_demand,
    slugify,
)


DEFAULT_APPOINTMENTS_XLSX = config.EXTERNAL_APPOINTMENTS_XLSX
DEFAULT_TECH_XLSX = config.EXTERNAL_TECH_ROSTER_XLSX

RAW_APPOINTMENTS_SHEET = "report1770130594436"
DERIVED_APPOINTMENTS_SHEET = "Derived Fields"

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

TECH_SKILL_COLUMNS = ["HPS", "LearningSpace", "All other Patient Sims"]


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


def derive_hps_skill_names(appointments_workbook_path: str | None) -> set[str]:
    """Infer which technicians have handled HPS work from historical appointments."""
    if not appointments_workbook_path or not os.path.exists(appointments_workbook_path):
        return set()

    derived = pd.read_excel(
        appointments_workbook_path,
        sheet_name=DERIVED_APPOINTMENTS_SHEET,
        usecols=["Appointment Number", "Service Resource: Name"],
    )
    raw = pd.read_excel(
        appointments_workbook_path,
        sheet_name=RAW_APPOINTMENTS_SHEET,
        usecols=["Appointment Number", "Description", "Subject"],
    )
    merged = derived.merge(raw, on="Appointment Number", how="left")
    parsed = merged.apply(
        lambda r: classify_skill(r.get("Description"), r.get("Subject")),
        axis=1,
        result_type="expand",
    )
    merged = pd.concat([merged, parsed], axis=1)
    names = merged.loc[
        pd.to_numeric(merged["required_hps"], errors="coerce").fillna(0).astype(int) == 1,
        "Service Resource: Name",
    ].apply(canonicalize_tech_name)
    return {name for name in names if name}

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


def lookup_airport_hub_fields(airport_code: object, airports_df: pd.DataFrame) -> dict:
    """Return hub metadata for an airport code using the airport table."""
    default_fields = {
        "faa_hub_class": config.HUB_TIER_UNKNOWN,
        "hub_tier": config.HUB_TIER_UNKNOWN,
        "hub_source": "unknown",
        "hub_source_year": np.nan,
    }
    code = "" if airport_code is None else str(airport_code).strip().upper()
    if code:
        match = airports_df[airports_df["airport_code"] == code]
        if not match.empty:
            row = match.iloc[0]
            fields = default_fields.copy()
            for key in fields:
                value = row.get(key)
                if key == "hub_source_year":
                    fields[key] = pd.to_numeric(value, errors="coerce")
                else:
                    text = "" if value is None else str(value).strip()
                    if text:
                        fields[key] = text
            return fields
    return default_fields


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


ANCHOR_TECH_COLUMNS = {
    "assignment_scope_states": "",
    "anchor_site_name": "",
    "anchor_site_location_raw": "",
    "anchor_site_lat": np.nan,
    "anchor_site_lon": np.nan,
    "anchor_reserved_fte": np.nan,
    "external_field_fte": np.nan,
    "anchor_show_on_map": 0,
    "anchor_notes": "",
}


def normalize_allowed_states(value: object) -> str:
    """Normalize a delimited state list into canonical semicolon-separated abbreviations."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    tokens = re.split(r"[;,|/]+", str(value))
    normalized: list[str] = []
    for token in tokens:
        raw = token.strip()
        if not raw:
            continue
        state = normalize_state(raw)
        if not state or country_from_state(state) != "USA":
            raise ValueError(f"Unsupported state token in anchor allocation: {raw}")
        if state not in normalized:
            normalized.append(state)
    return ";".join(normalized)


def load_anchor_allocations(anchor_csv: str | None) -> pd.DataFrame:
    """Load manual anchor-site technician allocations if configured."""
    required_cols = {
        "tech_name",
        "anchor_site_name",
        "anchor_site_location_raw",
        "anchor_site_lat",
        "anchor_site_lon",
        "anchor_reserved_fte",
        "external_field_fte",
        "external_assignment_mode",
        "external_allowed_states",
        "anchor_show_on_map",
        "anchor_notes",
    }
    if not anchor_csv or not os.path.exists(anchor_csv):
        return pd.DataFrame(columns=sorted(required_cols))

    anchors = pd.read_csv(anchor_csv)
    missing = sorted(required_cols - set(anchors.columns))
    if missing:
        raise ValueError(
            f"Anchor allocations CSV at {anchor_csv} is missing required columns: {missing}"
        )

    anchors = anchors.copy()
    anchors["tech_name"] = anchors["tech_name"].apply(canonicalize_tech_name)
    anchors["anchor_reserved_fte"] = pd.to_numeric(
        anchors["anchor_reserved_fte"], errors="raise"
    )
    anchors["external_field_fte"] = pd.to_numeric(
        anchors["external_field_fte"], errors="raise"
    )
    anchors["anchor_site_lat"] = pd.to_numeric(anchors["anchor_site_lat"], errors="raise")
    anchors["anchor_site_lon"] = pd.to_numeric(anchors["anchor_site_lon"], errors="raise")
    anchors["anchor_show_on_map"] = anchors["anchor_show_on_map"].apply(
        lambda value: int(str(value).strip().lower() in {"1", "true", "yes", "y"})
    )
    anchors["external_assignment_mode"] = (
        anchors["external_assignment_mode"].fillna("").astype(str).str.strip()
    )
    anchors["external_allowed_states"] = anchors["external_allowed_states"].apply(
        normalize_allowed_states
    )

    for _, row in anchors.iterrows():
        reserved = float(row["anchor_reserved_fte"])
        external = float(row["external_field_fte"])
        if reserved < 0 or reserved > 1 or external < 0 or external > 1:
            raise ValueError(
                f"Anchor allocation FTE values must be between 0 and 1 for {row['tech_name']}."
            )
        if reserved + external > 1.000001:
            raise ValueError(
                f"Anchor allocation FTE values exceed 1.0 for {row['tech_name']}."
            )
        mode = str(row["external_assignment_mode"])
        if mode == config.ASSIGNMENT_SCOPE_MODE_STATE_SET_LIMITED and not str(
            row["external_allowed_states"]
        ).strip():
            raise ValueError(
                f"state_set_limited anchor allocation requires external_allowed_states for {row['tech_name']}."
            )
    return anchors


def apply_anchor_allocations(
    tech_master: pd.DataFrame,
    anchor_allocations: pd.DataFrame,
) -> pd.DataFrame:
    """Overlay anchored-tech metadata and reserved-duty availability onto tech_master."""
    tech_master = tech_master.copy()
    for col, default in ANCHOR_TECH_COLUMNS.items():
        if col not in tech_master.columns:
            tech_master[col] = default

    if anchor_allocations.empty:
        tameka_mask = tech_master["tech_name"].eq("Tameka Gongs")
        if tameka_mask.any():
            tech_master.loc[tameka_mask, "availability_fte"] = 0.25
        return tech_master

    anchor_by_name = {
        str(row["tech_name"]).strip(): row for _, row in anchor_allocations.iterrows()
    }

    missing_names = sorted(set(anchor_by_name) - set(tech_master["tech_name"].astype(str)))
    if missing_names:
        raise ValueError(
            "Anchor allocation tech_name values not found in technician roster: "
            + ", ".join(missing_names)
        )

    for idx, row in tech_master.iterrows():
        tech_name = str(row["tech_name"]).strip()
        anchor = anchor_by_name.get(tech_name)
        if anchor is None:
            continue
        tech_master.at[idx, "availability_fte"] = float(anchor["external_field_fte"])
        tech_master.at[idx, "assignment_scope_mode"] = str(anchor["external_assignment_mode"]).strip()
        tech_master.at[idx, "assignment_scope_state"] = ""
        tech_master.at[idx, "assignment_scope_states"] = str(anchor["external_allowed_states"]).strip()
        tech_master.at[idx, "anchor_site_name"] = str(anchor["anchor_site_name"]).strip()
        tech_master.at[idx, "anchor_site_location_raw"] = str(anchor["anchor_site_location_raw"]).strip()
        tech_master.at[idx, "anchor_site_lat"] = float(anchor["anchor_site_lat"])
        tech_master.at[idx, "anchor_site_lon"] = float(anchor["anchor_site_lon"])
        tech_master.at[idx, "anchor_reserved_fte"] = float(anchor["anchor_reserved_fte"])
        tech_master.at[idx, "external_field_fte"] = float(anchor["external_field_fte"])
        tech_master.at[idx, "anchor_show_on_map"] = int(anchor["anchor_show_on_map"])
        tech_master.at[idx, "anchor_notes"] = str(anchor["anchor_notes"]).strip()

    return tech_master


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
        hub_fields = lookup_airport_hub_fields(enriched.get("base_airport_iata"), airports_df)

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
        enriched["base_faa_hub_class"] = hub_fields["faa_hub_class"]
        enriched["base_hub_tier"] = hub_fields["hub_tier"]
        enriched["base_hub_source"] = hub_fields["hub_source"]
        enriched["base_hub_source_year"] = hub_fields["hub_source_year"]
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
        hub_fields = lookup_airport_hub_fields(enriched.get("airport_iata"), airports_df)
        enriched["faa_hub_class"] = hub_fields["faa_hub_class"]
        enriched["hub_tier"] = hub_fields["hub_tier"]
        enriched["hub_source"] = hub_fields["hub_source"]
        enriched["hub_source_year"] = hub_fields["hub_source_year"]
        enriched["operational_zone_label"] = zone["operational_zone_label"]
        enriched["operational_zone_rank"] = zone["operational_zone_rank"]
        enriched["utc_offset_standard"] = zone["utc_offset_standard"]
        rows.append(enriched)

    return pd.DataFrame(rows)


def apply_manual_tech_policy_overrides(tech_master: pd.DataFrame) -> pd.DataFrame:
    """Apply stable manual technician-policy overrides after enrichment."""
    overrides = getattr(config, "OPTIMIZATION_TECH_POLICY_OVERRIDES", {})
    if tech_master.empty or not overrides:
        return tech_master.copy()

    tech_master = tech_master.copy()
    tech_ids = tech_master["tech_id"].astype(str)
    for tech_id, fields in overrides.items():
        mask = tech_ids.eq(str(tech_id))
        if not mask.any():
            continue
        for field, value in dict(fields).items():
            if field not in tech_master.columns:
                tech_master[field] = ""
            tech_master.loc[mask, field] = value
    return tech_master


def build_tech_master(
    tech_path: str,
    airports_df: pd.DataFrame,
    contractor_scope: str,
    anchor_allocations: pd.DataFrame | None = None,
    appointments_workbook_path: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Build canonical technician table."""
    source_meta = {
        "tech_source_path": str(tech_path),
        "tech_source_kind": "unknown",
        "tech_source_is_cached": False,
        "source_warnings": [],
    }
    tech = None
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
        if not missing:
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
                source_meta["source_warnings"].append(
                    "Cleared unsupported cached airport code(s): " + ", ".join(invalid_values)
                )
            print("  NOTE: using cached tech_master CSV fallback.")
            source_meta["tech_source_kind"] = "cached_tech_master_csv"
            source_meta["tech_source_is_cached"] = True
            enriched = enrich_tech_policy_fields(
                cached.sort_values("tech_name").reset_index(drop=True),
                airports_df,
                contractor_scope,
            )
            return (
                apply_anchor_allocations(
                    enriched,
                    anchor_allocations if anchor_allocations is not None else pd.DataFrame(),
                ),
                source_meta,
            )

        roster_cols = {"name", "location", "comment"}
        if roster_cols.issubset(cached.columns):
            tech = cached.rename(
                columns={
                    "name": "Tech",
                    "location": "Location",
                    "comment": "Comments",
                    "status": "Status",
                }
            ).copy()
            keep_cols = ["Tech", "Location", "Comments"]
            if "Status" in tech.columns:
                keep_cols.append("Status")
                tech = tech[
                    tech["Status"].fillna("").astype(str).str.strip().str.lower().isin(
                        {"active", "special"}
                    )
                ].copy()
            tech = tech[keep_cols]
            source_meta["tech_source_kind"] = "cleaned_current_roster_csv"
        else:
            raise ValueError(
                f"Unsupported technician CSV format at {tech_path}. Missing required columns: {missing}"
            )

    # Preferred "Tech/Location/Comments" workbook layout.
    if tech is None:
        try:
            raw = pd.read_excel(tech_path, sheet_name=0, header=None)
            if len(raw) >= 3:
                headers = ["" if pd.isna(h) else str(h).strip() for h in raw.iloc[1].tolist()]
                parsed = raw.iloc[2:].copy()
                parsed.columns = headers
                parsed = parsed.dropna(how="all")
                if {"Tech", "Location", "Comments"}.issubset(parsed.columns):
                    keep_cols = ["Tech", "Location", "Comments"] + [
                        col for col in TECH_SKILL_COLUMNS if col in parsed.columns
                    ]
                    tech = parsed[keep_cols].copy()
                    source_meta["tech_source_kind"] = "source_truth_workbook"
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
        )[
            ["Tech", "Location", "Comments"]
            + [col for col in TECH_SKILL_COLUMNS if col in fallback.columns]
        ].copy()
        print(
            "  NOTE: technician workbook did not match source-of-truth schema; "
            "using Resources-sheet fallback columns."
        )
        source_meta["tech_source_kind"] = "resources_sheet_workbook_fallback"
        source_meta["source_warnings"].append(
            "Technician workbook did not match source-truth schema; used Resources-sheet fallback."
        )

    for col in tech.columns:
        tech[col] = tech[col].apply(lambda v: "" if pd.isna(v) else str(v).strip())

    report_comment_lookup: dict[str, str] = {}
    if source_meta["tech_source_kind"] == "cleaned_current_roster_csv" and os.path.exists(
        config.SERVICE_APPTS_REPORT
    ):
        try:
            report_resources = pd.read_excel(
                config.SERVICE_APPTS_REPORT,
                sheet_name=config.APPTS_REPORT_RESOURCES_SHEET,
                usecols=["Service Resource Name", "Comment"],
            )
            for _, report_row in report_resources.iterrows():
                canonical_name = canonicalize_tech_name(report_row.get("Service Resource Name", ""))
                comment_text = (
                    ""
                    if pd.isna(report_row.get("Comment"))
                    else str(report_row.get("Comment", "")).strip()
                )
                if canonical_name and comment_text:
                    report_comment_lookup[canonical_name] = comment_text
            for idx, row in tech.iterrows():
                if str(row.get("Comments", "")).strip():
                    continue
                canonical_name = canonicalize_tech_name(row.get("Tech", ""))
                fallback_comment = report_comment_lookup.get(canonical_name, "")
                if fallback_comment:
                    tech.at[idx, "Comments"] = fallback_comment
        except Exception as exc:
            warning = (
                "Could not supplement technician comments from "
                f"{config.SERVICE_APPTS_REPORT}: {exc}"
            )
            print(f"  WARNING: {warning}")
            source_meta["source_warnings"].append(warning)

    hps_skill_names = derive_hps_skill_names(appointments_workbook_path)

    missing_skill_cols = [col for col in TECH_SKILL_COLUMNS if col not in tech.columns]
    for col in missing_skill_cols:
        tech[col] = ""

    cached_skill_lookup: dict[str, dict[str, str]] = {}
    cached_skill_path = Path(config.OPTIMIZATION_DIR) / "tech_master.csv"
    if (
        cached_skill_path.exists()
        and source_meta["tech_source_kind"] != "cleaned_current_roster_csv"
    ):
        try:
            cached_master = pd.read_csv(cached_skill_path)
            for _, cached_row in cached_master.iterrows():
                cached_name = canonicalize_tech_name(
                    cached_row.get("tech_name", cached_row.get("source_name", ""))
                )
                if not cached_name:
                    continue
                cached_skill_lookup[cached_name] = {
                    "HPS": "Yes" if int(pd.to_numeric(cached_row.get("skill_hps", 0), errors="coerce") or 0) else "",
                    "LearningSpace": "Yes"
                    if int(pd.to_numeric(cached_row.get("skill_ls", 0), errors="coerce") or 0)
                    else "",
                    "All other Patient Sims": "Yes"
                    if int(pd.to_numeric(cached_row.get("skill_patient", 0), errors="coerce") or 0)
                    else "",
                }
        except Exception as exc:
            warning = (
                "Could not load cached tech_master skill fallback from "
                f"{cached_skill_path}: {exc}"
            )
            print(f"  WARNING: {warning}")
            source_meta["source_warnings"].append(warning)

    supplemented_skill_count = 0
    if cached_skill_lookup:
        for idx, row in tech.iterrows():
            canonical_name = canonicalize_tech_name(row.get("Tech", ""))
            if not canonical_name or canonical_name not in cached_skill_lookup:
                continue
            cached_skills = cached_skill_lookup[canonical_name]
            for col in TECH_SKILL_COLUMNS:
                current_value = str(row.get(col, "")).strip()
                if current_value:
                    continue
                fallback_value = cached_skills.get(col, "")
                if fallback_value:
                    tech.at[idx, col] = fallback_value
                    supplemented_skill_count += 1

    if missing_skill_cols:
        if supplemented_skill_count:
            warning = (
                "Explicit technician skill columns missing from source roster: "
                + ", ".join(missing_skill_cols)
                + f". Backfilled {supplemented_skill_count} value(s) from cached tech_master.csv and derived the rest from comments/history where possible; unresolved blanks default to no."
            )
        else:
            warning = (
                "Explicit technician skill columns missing from source roster: "
                + ", ".join(missing_skill_cols)
                + ". Derived skills from comments/history where possible; unresolved blanks default to no."
            )
        print(f"  WARNING: {warning}")
        source_meta["source_warnings"].append(warning)

    rows = []
    unresolved_airports: list[str] = []
    for _, row in tech.iterrows():
        name_raw = row.get("Tech", "").strip()
        if not name_raw:
            continue
        canonical_name = canonicalize_tech_name(name_raw)

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
        elif location:
            unresolved_airports.append(f"{canonical_name} ({location})")

        is_contractor = "contractor" in name_raw.lower()
        availability = 0.5 if is_contractor else 1.0
        if canonical_name == "James Sanchez":
            availability = 1.0  # Full field tech; temporarily on phones (injury) but model targets ideal state
        if canonical_name == "Hakim Mouazer":
            availability = 0.0
        if canonical_name == "Elier Martin":
            availability = 0.10  # Phone tech, confirmed by Shannon/Isabelle at ~10% field
        if canonical_name == "Damion Lyn":
            availability = 0.20  # Repair center hybrid, 20-25% field when fully staffed (conservative)

        note = row.get("Comments", "").strip()
        supplemental_comment = report_comment_lookup.get(canonical_name, "")
        skill_comment_text = " | ".join(
            part for part in [note, supplemental_comment] if str(part).strip()
        )
        note_lower = skill_comment_text.lower()
        status_raw = str(row.get("Status", "")).strip().lower()
        if status_raw in {"inactive", "former"} or "no longer with elevate" in note_lower:
            continue

        hps_value = str(row.get("HPS", "")).strip()
        if not hps_value and canonical_name in hps_skill_names:
            hps_value = "Yes"
        if not hps_value and "hps" in note_lower:
            hps_value = "Yes"

        ls_value = str(row.get("LearningSpace", "")).strip()
        if not ls_value and ("learningspace" in note_lower or "learning space" in note_lower):
            ls_value = "Yes"

        patient_value = str(row.get("All other Patient Sims", "")).strip()
        if not patient_value and (
            "patient simulator" in note_lower
            or "learningspace" in note_lower
            or "learning space" in note_lower
            or source_meta["tech_source_kind"] == "cleaned_current_roster_csv"
        ):
            patient_value = "Yes"

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
                "skill_hps": int(yesno_to_bool(hps_value)),
                "skill_ls": int(yesno_to_bool(ls_value)),
                "skill_patient": int(yesno_to_bool(patient_value)),
                "availability_fte": float(availability),
                "fixed_base": 1,
                "can_relocate": 0,
                "contractor_assignment_scope": contractor_scope if is_contractor else "",
                "constraint_florida_only": florida_only,
                "notes": note,
            }
        )

    tech_master = pd.DataFrame(rows).sort_values("tech_name").reset_index(drop=True)
    tech_master = enrich_tech_policy_fields(tech_master, airports_df, contractor_scope)
    if unresolved_airports:
        warning = (
            "Could not resolve base airport for: "
            + ", ".join(sorted(unresolved_airports))
        )
        print(f"  WARNING: {warning}")
        source_meta["source_warnings"].append(warning)
    tech_master = apply_anchor_allocations(
        tech_master,
        anchor_allocations if anchor_allocations is not None else pd.DataFrame(),
    )
    tech_master = apply_manual_tech_policy_overrides(tech_master)
    return (tech_master, source_meta)


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
) -> tuple[pd.DataFrame, dict]:
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

    demand_city_raw = (
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
    demand_city = demand_city_raw.copy()
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
    pre_filter_count = int(len(demand_city))
    demand_city = demand_city[
        ~demand_city.apply(
            lambda r: (normalize_name(r["city"]), r["state"]) in major_city_state,
            axis=1,
        )
    ]
    demand_city = demand_city.reset_index(drop=True)
    demand_city["demand_rank"] = range(1, len(demand_city) + 1)
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
    candidate_meta = {
        "requested_top_demand_cities": int(top_demand_cities),
        "selected_pre_dedupe_demand_cities": int(len(demand_city_raw)),
        "dropped_due_major_airport_overlap": int(pre_filter_count - len(demand_city)),
        "final_demand_city_candidates": int(len(demand_city)),
        "final_candidate_bases": int(len(candidates)),
    }
    return enrich_candidate_zone_fields(candidates, airports_df), candidate_meta


def classify_historical_roster_status(comment: object) -> str:
    """Map archived roster comments to a broad status bucket."""
    text = "" if comment is None or pd.isna(comment) else str(comment).strip().lower()
    if "no longer with elevate" in text:
        return "former"
    if "special resource" in text:
        return "special"
    return "active"


def build_historical_cost_context(demand_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Prepare shared demand inputs used to score estimated historical bases."""
    demand_prepared = prepare_demand(demand_df)
    demand_prepared["duration_hours"] = pd.to_numeric(
        demand_prepared["duration_hours"], errors="coerce"
    ).fillna(24.0 * config.HOTEL_AVG_NIGHTS)
    node_avg_duration = demand_prepared.groupby("node_id")["duration_hours"].mean()
    node_avg_days = (node_avg_duration / 24.0).to_dict()
    return demand_prepared, node_avg_days


def build_historical_base_row(
    *,
    canonical_name: str,
    source_name: str,
    status: str,
    comment: str,
    base_location_raw: str,
    base_city: str | None,
    base_state: str | None,
    base_country: str,
    base_airport_iata: str,
    base_lat: float,
    base_lon: float,
    base_source: str,
    base_confidence: str,
) -> dict:
    """Build one historical roster output row."""
    state_display = str(base_state).strip() if base_state else ""
    city_display = str(base_city).strip() if base_city else ""
    base_label = city_display
    if city_display and state_display:
        base_label = f"{city_display}, {state_display}"
    elif state_display:
        base_label = state_display
    return {
        "historical_tech_id": slugify(canonical_name),
        "tech_name": canonical_name,
        "source_name": source_name,
        "status": status,
        "comment": comment,
        "base_location_raw": base_location_raw,
        "base_city": city_display,
        "base_state": state_display,
        "base_country": base_country,
        "base_airport_iata": base_airport_iata,
        "base_lat": base_lat,
        "base_lon": base_lon,
        "base_source": base_source,
        "base_confidence": base_confidence,
        "base_label": base_label or base_location_raw,
    }


def airport_location_fields(
    airports_df: pd.DataFrame,
    airport_code: object,
) -> tuple[float, float]:
    """Return airport coordinates when the code exists in the supported airport set."""
    code = "" if airport_code is None else str(airport_code).strip().upper()
    match = airports_df[airports_df["airport_code"] == code]
    if match.empty:
        return np.nan, np.nan
    row = match.iloc[0]
    return float(row["lat"]), float(row["lon"])


def add_historical_candidate(
    candidates: list[dict],
    seen: set[tuple[str, str, str]],
    *,
    city: str,
    state: str,
    country: str,
    airport_iata: str,
    lat: float,
    lon: float,
    source: str,
    state_appointment_count: int,
) -> None:
    """Append a unique historical base candidate."""
    city_clean = str(city or "").strip()
    state_clean = str(state or "").strip()
    airport_clean = str(airport_iata or "").strip().upper()
    if not city_clean or not state_clean or not airport_clean:
        return
    if pd.isna(lat) or pd.isna(lon):
        return
    dedupe_key = (normalize_name(city_clean), state_clean, airport_clean)
    if dedupe_key in seen:
        return
    seen.add(dedupe_key)
    candidates.append(
        {
            "candidate_id": slugify(f"{city_clean}_{state_clean}_{airport_clean}"),
            "city": city_clean,
            "state": state_clean,
            "country": country,
            "airport_iata": airport_clean,
            "lat": float(lat),
            "lon": float(lon),
            "source": source,
            "state_appointment_count": int(state_appointment_count),
            "candidate_sort": f"{state_clean}_{city_clean}_{airport_clean}",
        }
    )


def build_historical_estimate_candidates(
    canonical_name: str,
    base_location_raw: str,
    tech_appts: pd.DataFrame,
    airports_df: pd.DataFrame,
) -> list[dict]:
    """Build candidate historical bases for an unresolved or region-only tech base."""
    del canonical_name  # kept for future name-specific heuristics
    candidates: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    for override in config.HISTORICAL_BASE_REGION_CANDIDATES.get(base_location_raw, []):
        airport_iata = str(override.get("airport_iata", "")).strip().upper()
        lat, lon = airport_location_fields(airports_df, airport_iata)
        add_historical_candidate(
            candidates,
            seen,
            city=str(override.get("city", "")).strip(),
            state=str(override.get("state", "")).strip(),
            country=str(override.get("country", "")).strip() or country_from_state(override.get("state")),
            airport_iata=airport_iata,
            lat=lat,
            lon=lon,
            source="region_candidate",
            state_appointment_count=0,
        )

    if base_location_raw in config.HISTORICAL_BASE_REGION_CANDIDATES:
        return candidates

    if tech_appts.empty:
        return candidates

    airport_counts = (
        tech_appts["nearest_hub_airport"]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace("nan", "")
    )
    airport_counts = airport_counts[airport_counts.ne("")].value_counts().head(3)
    for airport_iata, _ in airport_counts.items():
        match = airports_df[airports_df["airport_code"] == airport_iata]
        if match.empty:
            continue
        row = match.iloc[0]
        state = str(row["state_abbr"])
        add_historical_candidate(
            candidates,
            seen,
            city=str(row["city_name"]),
            state=state,
            country=str(row["country"]),
            airport_iata=str(row["airport_code"]),
            lat=float(row["lat"]),
            lon=float(row["lon"]),
            source="historical_hub_usage",
            state_appointment_count=int((tech_appts["state_norm"] == state).sum()),
        )

    state_counts = (
        tech_appts["state_norm"]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace("nan", "")
    )
    state_counts = state_counts[state_counts.ne("")].value_counts().head(3)
    for state, state_count in state_counts.items():
        state_appts = tech_appts[tech_appts["state_norm"].astype(str).str.strip() == state]
        city_counts = (
            state_appts.dropna(subset=["city"])
            .assign(city_clean=lambda df: df["city"].astype(str).str.strip())
            .groupby("city_clean")
            .size()
            .sort_values(ascending=False)
            .head(2)
        )
        for city, _ in city_counts.items():
            city_appts = state_appts[state_appts["city"].astype(str).str.strip() == city]
            lat = pd.to_numeric(city_appts["lat"], errors="coerce").mean()
            lon = pd.to_numeric(city_appts["lon"], errors="coerce").mean()
            airport_iata = nearest_airport_code(lat, lon, airports_df, state)[0]
            if not airport_iata:
                airport_iata = choose_airport_for_location(f"{city}, {state}", airports_df)
            if not airport_iata:
                continue
            if pd.isna(lat) or pd.isna(lon):
                lat, lon = airport_location_fields(airports_df, airport_iata)
            add_historical_candidate(
                candidates,
                seen,
                city=city,
                state=state,
                country=country_from_state(state),
                airport_iata=airport_iata,
                lat=lat,
                lon=lon,
                source="historical_service_city",
                state_appointment_count=int(state_count),
            )

    return candidates


def choose_historical_estimated_base(
    canonical_name: str,
    base_location_raw: str,
    tech_appts: pd.DataFrame,
    demand_prepared: pd.DataFrame,
    node_avg_days: dict[str, float],
    airports_df: pd.DataFrame,
) -> dict | None:
    """Pick the lowest-cost historical base estimate for a tech with no exact archived base."""
    candidates = build_historical_estimate_candidates(
        canonical_name, base_location_raw, tech_appts, airports_df
    )
    if not candidates:
        return None

    tech_node_counts = tech_appts["node_id"].value_counts().to_dict()
    if not tech_node_counts:
        return sorted(
            candidates,
            key=lambda candidate: (
                -int(candidate["state_appointment_count"]),
                candidate["candidate_sort"],
            ),
        )[0]

    cost_rows = compute_entity_node_costs(
        [
            {
                "id": candidate["candidate_id"],
                "lat": candidate["lat"],
                "lon": candidate["lon"],
                "airport": candidate["airport_iata"],
                "travel_cost_policy": config.TRAVEL_COST_POLICY_EMPLOYEE,
                "contractor_cost_multiplier": 1.0,
                "contractor_cost_cap_usd": np.nan,
                "contractor_dispatch_surcharge_usd": 0.0,
            }
            for candidate in candidates
        ],
        demand_prepared,
        node_avg_days=node_avg_days,
    )
    cost_lookup: dict[tuple[str, str], float] = {}
    for row in cost_rows:
        cost_lookup[(str(row["tech_or_candidate_id"]), str(row["node_id"]))] = float(
            row["unit_cost_usd"]
        )

    return min(
        candidates,
        key=lambda candidate: (
            round(
                sum(
                    tech_node_counts[node_id]
                    * cost_lookup.get((candidate["candidate_id"], node_id), 0.0)
                    for node_id in tech_node_counts
                ),
                4,
            ),
            -int(candidate["state_appointment_count"]),
            candidate["candidate_sort"],
        ),
    )


def resolve_historical_roster_base(
    canonical_name: str,
    source_name: str,
    status: str,
    comment: str,
    base_location_raw: str,
    demand_df: pd.DataFrame,
    demand_prepared: pd.DataFrame,
    node_avg_days: dict[str, float],
    airports_df: pd.DataFrame,
) -> dict:
    """Resolve one archived roster row to an exact or estimated historical base."""
    manual_override = dict(getattr(config, "HISTORICAL_BASE_MANUAL_OVERRIDES", {})).get(
        canonical_name
    )
    if manual_override:
        city = str(manual_override.get("city", "")).strip()
        state = normalize_state(manual_override.get("state"))
        country = str(manual_override.get("country", "")).strip() or country_from_state(state)
        airport_iata = str(
            manual_override.get("airport_iata")
            or choose_airport_for_location(f"{city}, {state}", airports_df)
            or ""
        ).strip().upper()
        lat, lon = airport_location_fields(airports_df, airport_iata)
        return build_historical_base_row(
            canonical_name=canonical_name,
            source_name=source_name,
            status=status,
            comment=comment,
            base_location_raw=base_location_raw,
            base_city=city,
            base_state=state,
            base_country=country,
            base_airport_iata=airport_iata,
            base_lat=lat,
            base_lon=lon,
            base_source="manual_override",
            base_confidence="manual_override",
        )

    city, state_abbr = parse_city_state(base_location_raw)
    is_region_only = base_location_raw in config.HISTORICAL_BASE_REGION_CANDIDATES

    if not is_region_only and city and state_abbr and state_abbr not in {"CANADA"}:
        airport_iata = choose_airport_for_location(base_location_raw, airports_df) or ""
        lat, lon = airport_location_fields(airports_df, airport_iata)
        return build_historical_base_row(
            canonical_name=canonical_name,
            source_name=source_name,
            status=status,
            comment=comment,
            base_location_raw=base_location_raw,
            base_city=city,
            base_state=state_abbr,
            base_country=country_from_state(state_abbr),
            base_airport_iata=airport_iata,
            base_lat=lat,
            base_lon=lon,
            base_source="archived_roster",
            base_confidence="high",
        )

    tech_appts = demand_df[
        demand_df["Service Resource: Name"].apply(canonicalize_tech_name) == canonical_name
    ].copy()
    tech_appts = prepare_demand(tech_appts)
    estimate = choose_historical_estimated_base(
        canonical_name,
        base_location_raw,
        tech_appts,
        demand_prepared,
        node_avg_days,
        airports_df,
    )
    if estimate:
        return build_historical_base_row(
            canonical_name=canonical_name,
            source_name=source_name,
            status=status,
            comment=comment,
            base_location_raw=base_location_raw,
            base_city=estimate["city"],
            base_state=estimate["state"],
            base_country=estimate["country"],
            base_airport_iata=estimate["airport_iata"],
            base_lat=estimate["lat"],
            base_lon=estimate["lon"],
            base_source=(
                "archived_region"
                if base_location_raw in config.HISTORICAL_BASE_REGION_CANDIDATES
                else "estimated_from_history"
            ),
            base_confidence=(
                "estimated_region_to_city"
                if base_location_raw in config.HISTORICAL_BASE_REGION_CANDIDATES
                else "estimated"
            ),
        )

    return build_historical_base_row(
        canonical_name=canonical_name,
        source_name=source_name,
        status=status,
        comment=comment,
        base_location_raw=base_location_raw,
        base_city="",
        base_state="",
        base_country="",
        base_airport_iata="",
        base_lat=np.nan,
        base_lon=np.nan,
        base_source="unknown",
        base_confidence="unknown",
    )


def build_historical_roster(
    roster_path: str,
    airports_df: pd.DataFrame,
    demand_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build archived technician roster used by the historical map view."""
    raw = pd.read_excel(roster_path, sheet_name=config.APPTS_REPORT_RESOURCES_SHEET)
    required_cols = {"Service Resource Name", "Location", "Comment"}
    missing = sorted(required_cols - set(raw.columns))
    if missing:
        raise ValueError(
            f"Historical roster workbook at {roster_path} is missing columns: {missing}"
        )

    demand_prepared, node_avg_days = build_historical_cost_context(demand_df)
    rows_by_name: dict[str, dict] = {}

    for _, row in raw.iterrows():
        source_name = (
            ""
            if pd.isna(row.get("Service Resource Name"))
            else str(row.get("Service Resource Name", "")).strip()
        )
        if not source_name:
            continue
        canonical_name = canonicalize_tech_name(source_name)
        base_location_raw = (
            "" if pd.isna(row.get("Location")) else str(row.get("Location", "")).strip()
        )
        comment = "" if pd.isna(row.get("Comment")) else str(row.get("Comment", "")).strip()
        status = classify_historical_roster_status(comment)
        resolved = resolve_historical_roster_base(
            canonical_name=canonical_name,
            source_name=source_name,
            status=status,
            comment=comment,
            base_location_raw=base_location_raw,
            demand_df=demand_df,
            demand_prepared=demand_prepared,
            node_avg_days=node_avg_days,
            airports_df=airports_df,
        )
        rows_by_name[canonical_name] = resolved

    historical_roster = (
        pd.DataFrame(rows_by_name.values())
        .sort_values(["status", "tech_name"])
        .reset_index(drop=True)
    )
    return historical_roster


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
            config.CLEAN_TECHS_CSV,
            config.SERVICE_APPTS_REPORT,
            os.path.join(config.OPTIMIZATION_DIR, "tech_master.csv"),
        ],
        "technician workbook",
    )

    airports_df = build_airports_df(config.MAJOR_AIRPORTS)
    anchor_allocations = load_anchor_allocations(config.TECHNICIAN_ANCHOR_ALLOCATIONS_CSV)
    tech_master, tech_source_meta = build_tech_master(
        tech_path,
        airports_df,
        args.contractor_assignment_scope,
        anchor_allocations=anchor_allocations,
        appointments_workbook_path=appts_path,
    )
    demand = build_demand_appointments(appts_path, config.GEOCODED_APPTS_CSV, airports_df)
    candidates, candidate_meta = build_candidate_bases(
        airports_df, demand, args.top_demand_cities
    )
    historical_roster = build_historical_roster(config.SERVICE_APPTS_REPORT, airports_df, demand)

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
    historical_roster_out = out_dir / "historical_roster.csv"
    parse_ambiguous_out = out_dir / "skill_parse_ambiguous.csv"
    summary_out = out_dir / "optimization_input_summary.json"

    tech_master.to_csv(tech_path_out, index=False)
    demand.to_csv(demand_path_out, index=False)
    candidates.to_csv(candidate_path_out, index=False)
    historical_roster.to_csv(historical_roster_out, index=False)
    demand[demand["parse_confidence"] != "high"].to_csv(parse_ambiguous_out, index=False)

    summary = {
        "appointments_source": appts_path,
        "tech_source": tech_path,
        "appointments_source_kind": (
            "repo_raw_workbook"
            if os.path.abspath(appts_path) == os.path.abspath(config.SERVICE_APPTS_DISPATCH)
            else "external_workbook_override"
        ),
        "tech_source_kind": tech_source_meta["tech_source_kind"],
        "tech_source_is_cached": bool(tech_source_meta["tech_source_is_cached"]),
        "source_warnings": list(tech_source_meta.get("source_warnings", [])),
        "technician_anchor_allocations_source": (
            config.TECHNICIAN_ANCHOR_ALLOCATIONS_CSV
            if os.path.exists(config.TECHNICIAN_ANCHOR_ALLOCATIONS_CSV)
            else None
        ),
        "historical_roster_source": config.SERVICE_APPTS_REPORT,
        "rows": {
            "tech_master": int(len(tech_master)),
            "demand_appointments": int(len(demand)),
            "candidate_bases": int(len(candidates)),
            "historical_roster": int(len(historical_roster)),
        },
        "skills": {
            "hps_required_appointments": int(demand["required_hps"].sum()),
            "ls_required_appointments": int(demand["required_ls"].sum()),
            "hps_ls_required_appointments": int((demand["skill_class"] == "hps_ls").sum()),
        },
        "candidate_pool": candidate_meta,
        "contractor_assignment_scope": args.contractor_assignment_scope,
        "contractor_policy_defaults": {
            "travel_cost_policy": config.TRAVEL_COST_POLICY_CONTRACTOR,
            "zone_policy": config.ZONE_POLICY_CONTRACTOR_SOFT,
            "contractor_cost_multiplier": config.CONTRACTOR_COST_MULTIPLIER,
            "contractor_cost_cap_usd": config.CONTRACTOR_COST_CAP_USD,
            "contractor_dispatch_surcharge_usd": config.CONTRACTOR_DISPATCH_SURCHARGE_USD,
        },
        "operational_zone_anchor": "airport_based",
        "special_tech_constraints": [
            {
                "tech_name": str(row["tech_name"]),
                "anchor_site_name": str(row["anchor_site_name"]),
                "anchor_reserved_fte": float(row["anchor_reserved_fte"]),
                "external_field_fte": float(row["external_field_fte"]),
                "external_assignment_mode": str(row["assignment_scope_mode"]),
                "external_allowed_states": str(row["assignment_scope_states"]),
            }
            for _, row in tech_master[
                tech_master["anchor_site_name"].fillna("").astype(str).str.strip().ne("")
            ].iterrows()
        ],
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
    print(f"Saved: {historical_roster_out}")
    print(f"Saved: {parse_ambiguous_out}")
    print(f"Saved: {summary_out}")
    print("\nInput summary:")
    print(json.dumps(summary, indent=2))
    print("Step 6 complete.")


if __name__ == "__main__":
    main()
