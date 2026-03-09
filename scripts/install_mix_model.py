"""Patient simulator install history cleaning and forward mix modeling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import math
import re

import pandas as pd

import config


INSTALL_SIGNAL_RE = re.compile(r"\b(?:iso|install|installation|re-?install(?:ation)?)\b", re.I)
LEARNING_SPACE_RE = re.compile(r"\bAVS(?:\s+ISO)?\b|learning\s*space|\bLS\b", re.I)
NON_NET_NEW_RE = re.compile(r"\breturn\b|\bre-?install(?:ation)?\b|\breplac(?:e|ement)\b", re.I)
SUPPORT_ONLY_RE = re.compile(
    r"tech onsite to oversee htx installs|already onsite|multisim router",
    re.I,
)
CASE_ID_RE = re.compile(r"\bcase\s*#?\s*(\d+)\b", re.I)
SERIAL_TOKEN_RE = re.compile(r"\b(?:MFS|APN|APP|ARIA|JUNO|ARES|EVO|LUNA|HPS)[0-9A-Z/-]*\b", re.I)
FAMILY_TOKEN_PATTERNS = [
    ("Apollo Nursing", "Apollo", re.compile(r"\bAPN[0-9A-Z-]*\b|APOLLO\s+NURS(?:ING)?", re.I)),
    (
        "Apollo Pre-Hospital",
        "Apollo",
        re.compile(r"\bAPP[0-9A-Z-]*\b|APOLLO\s+PRE[- ]?HOSPITAL", re.I),
    ),
    ("Apollo", "Apollo", re.compile(r"\bAPOLLO\b", re.I)),
    ("Lucina", "Lucina", re.compile(r"\b(?:MFS|LUCINA)[0-9A-Z-]*\b", re.I)),
    ("Juno", "Juno", re.compile(r"\bJUNO[0-9A-Z-]*\b|\bJUNOS\b", re.I)),
    ("Ares", "Ares", re.compile(r"\bARES[0-9A-Z-]*\b", re.I)),
    ("Aria", "Aria", re.compile(r"\bARIA[0-9A-Z-]*\b", re.I)),
    ("Evo", "Evo", re.compile(r"\bEVO[0-9A-Z-]*\b", re.I)),
    ("Luna", "Luna", re.compile(r"\bLUNA[0-9A-Z-]*\b", re.I)),
    ("HPS", "HPS", re.compile(r"\b(?:HPS|PHPS)[0-9A-Z/-]*\b", re.I)),
]


@dataclass
class PatientSimInstallModel:
    """Install history, historical mix, forward mix, and economics outputs."""

    source_metadata: dict[str, Any]
    history_rows: pd.DataFrame
    historical_mix_events: pd.DataFrame
    historical_mix_units: pd.DataFrame
    forward_mix: pd.DataFrame
    family_economics: pd.DataFrame
    assumptions: dict[str, Any]


def _safe_text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _match_text(description: object, subject: object) -> str:
    desc = _safe_text(description)
    if desc:
        return desc
    return _safe_text(subject)


def _combined_text(service_type: object, description: object, subject: object) -> str:
    parts = [_safe_text(service_type), _safe_text(description), _safe_text(subject)]
    return " | ".join([p for p in parts if p])


def _normalize_description_for_dedupe(text: str) -> str:
    normalized = text.upper()
    normalized = CASE_ID_RE.sub(" ", normalized)
    normalized = re.sub(r"\b\d{5,7}\b", " ", normalized)
    normalized = re.sub(r"[^A-Z0-9]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _extract_case_id(text: str) -> str | None:
    match = CASE_ID_RE.search(text)
    return match.group(1) if match else None


def _extract_serial_tokens(text: str) -> list[str]:
    return SERIAL_TOKEN_RE.findall(text.upper())


def _extract_family_matches(text: str) -> list[dict[str, str]]:
    matches: list[dict[str, str]] = []
    for raw_family, normalized_family, pattern in FAMILY_TOKEN_PATTERNS:
        if pattern.search(text):
            matches.append(
                {
                    "raw_family": raw_family,
                    "normalized_family": normalized_family,
                }
            )

    raw_names = {m["raw_family"] for m in matches}
    if "Apollo" in raw_names and (
        "Apollo Nursing" in raw_names or "Apollo Pre-Hospital" in raw_names
    ):
        matches = [m for m in matches if m["raw_family"] != "Apollo"]

    deduped: list[dict[str, str]] = []
    seen = set()
    for match in matches:
        key = (match["raw_family"], match["normalized_family"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(match)
    return deduped


def _infer_units(
    text: str,
    raw_family: str,
    normalized_family: str,
    allow_implicit_single: bool,
) -> float | None:
    upper = text.upper()

    if normalized_family == "Juno":
        other_match = re.search(r"\bWITH\s+(\d+)\s+OTHER\s+JUNOS?\b", upper)
        if other_match:
            return float(int(other_match.group(1)) + 1)
        one_of_match = re.search(r"\b1\s+OF\s+(\d+)\s+JUNOS?\b", upper)
        if one_of_match:
            return float(int(one_of_match.group(1)))
        with_count_match = re.search(r"\bWITH\s+(\d+)\s+JUNOS?\b", upper)
        if with_count_match:
            return float(int(with_count_match.group(1)))
        direct_qty_match = re.search(r"\b(\d+)\s+JUNOS?\b", upper)
        if direct_qty_match:
            return float(int(direct_qty_match.group(1)))

    qty_pattern_map = {
        "Apollo Nursing": r"\b(\d+)\s+APN\b",
        "Apollo Pre-Hospital": r"\b(\d+)\s+APP\b",
        "Lucina": r"\b(\d+)\s+MFS\b",
        "Aria": r"\b(\d+)\s+ARIA\b",
        "Ares": r"\b(\d+)\s+ARES\b",
        "Evo": r"\b(\d+)\s+EVO\b",
        "Luna": r"\b(\d+)\s+LUNA\b",
        "HPS": r"\b(\d+)\s+HPS\b",
    }
    qty_pattern = qty_pattern_map.get(raw_family)
    if qty_pattern:
        qty_match = re.search(qty_pattern, upper)
        if qty_match:
            return float(int(qty_match.group(1)))

    if allow_implicit_single and normalized_family == "Apollo" and raw_family == "Apollo":
        return 1.0

    if allow_implicit_single and raw_family in {
        "Apollo Nursing",
        "Apollo Pre-Hospital",
        "Lucina",
        "Juno",
        "Ares",
        "Aria",
        "Evo",
        "Luna",
        "HPS",
    }:
        return 1.0

    return None


def parse_install_history_row(
    appointment_number: object,
    account_name: object,
    service_type: object,
    description: object,
    subject: object,
    scheduled_start: object,
    duration_days: object,
    duration_hours: object,
) -> list[dict[str, Any]]:
    """Parse one appointment row into zero or more family component records."""

    stype = _safe_text(service_type).upper()
    desc = _safe_text(description)
    subj = _safe_text(subject)
    text = _combined_text(service_type, description, subject)
    family_text = _match_text(description, subject)

    install_like = (
        ("ISO" in stype)
        or ("AVS" in stype)
        or bool(INSTALL_SIGNAL_RE.search(text))
    )
    if not install_like:
        return []

    base = {
        "appointment_number": _safe_text(appointment_number),
        "account_name": _safe_text(account_name),
        "service_type": _safe_text(service_type),
        "description": desc,
        "subject": subj,
        "scheduled_start": pd.to_datetime(scheduled_start, errors="coerce"),
        "duration_days": pd.to_numeric(duration_days, errors="coerce"),
        "duration_hours": pd.to_numeric(duration_hours, errors="coerce"),
        "source_text_for_matching": family_text,
        "raw_family_matches": [],
        "normalized_family": None,
        "units_inferred": math.nan,
        "event_weight": math.nan,
        "history_exclusion_reason": None,
        "history_resolution_type": None,
        "case_id": _extract_case_id(text),
        "serial_tokens": _extract_serial_tokens(text),
    }

    if stype in {"PM"} or "REPAIR" in stype or "RETURN/FOLLOW-UP" in stype:
        row = dict(base)
        row["history_exclusion_reason"] = "service_type_non_install"
        return [row]
    if LEARNING_SPACE_RE.search(text):
        row = dict(base)
        row["history_exclusion_reason"] = "learning_space_excluded"
        return [row]
    if NON_NET_NEW_RE.search(text):
        row = dict(base)
        row["history_exclusion_reason"] = "non_net_new_install"
        return [row]
    if SUPPORT_ONLY_RE.search(text):
        row = dict(base)
        row["history_exclusion_reason"] = "support_only_install_activity"
        return [row]

    family_matches = _extract_family_matches(family_text)
    if not family_matches:
        row = dict(base)
        row["history_exclusion_reason"] = "ambiguous_family_unresolved"
        return [row]

    resolved_family_count = len({m["normalized_family"] for m in family_matches})
    rows: list[dict[str, Any]] = []
    for match in family_matches:
        units = _infer_units(
            family_text,
            match["raw_family"],
            match["normalized_family"],
            allow_implicit_single=resolved_family_count == 1,
        )
        if units is None:
            row = dict(base)
            row["raw_family_matches"] = [m["raw_family"] for m in family_matches]
            row["history_exclusion_reason"] = "ambiguous_bundle_units"
            row["history_resolution_type"] = "ambiguous_bundle_units"
            return [row]

        row = dict(base)
        row["raw_family_matches"] = [m["raw_family"] for m in family_matches]
        row["normalized_family"] = match["normalized_family"]
        row["units_inferred"] = float(units)
        row["event_weight"] = 1.0 / float(resolved_family_count)
        row["history_resolution_type"] = (
            "single_family"
            if resolved_family_count == 1
            else "explicit_bundle_split"
        )
        rows.append(row)
    return rows


def _history_source_from_workbook(
    workbook_path: str,
    config_module: Any = config,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_excel(workbook_path, sheet_name=config_module.APPTS_RAW_SHEET)
    derived = pd.read_excel(workbook_path, sheet_name=config_module.APPTS_DISPATCH_SHEET)
    keep_raw = [
        "Appointment Number",
        "Account: Account Name",
        "Service Resource: Name",
        "Scheduled Start",
        "Scheduled End",
        "Dispatched Date/Time",
        "Description",
        "Subject",
    ]
    keep_derived = [
        "Appointment Number",
        "Service Type",
        "Duration Days",
        "Duration Hours",
    ]
    raw = raw[[c for c in keep_raw if c in raw.columns]].copy()
    derived = derived[[c for c in keep_derived if c in derived.columns]].copy()
    merged = raw.merge(derived, on="Appointment Number", how="left")
    metadata = {
        "source_kind": "raw_workbook_merge",
        "source_path": workbook_path,
        "raw_sheet": config_module.APPTS_RAW_SHEET,
        "derived_sheet": config_module.APPTS_DISPATCH_SHEET,
    }
    return merged, metadata


def _history_source_from_demand_csv(demand_path: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    demand = pd.read_csv(demand_path)
    required = {
        "Appointment Number",
        "Account: Account Name",
        "scheduled_start",
        "Description",
        "Subject",
        "Service Type",
        "duration_hours",
    }
    missing = sorted(required - set(demand.columns))
    if missing:
        raise ValueError(f"demand_appointments fallback missing required columns: {missing}")

    mapped = demand.rename(
        columns={
            "scheduled_start": "Scheduled Start",
            "duration_hours": "Duration Hours",
        }
    ).copy()
    if "Duration Days" not in mapped.columns:
        mapped["Duration Days"] = pd.to_numeric(mapped["Duration Hours"], errors="coerce") / 24.0
    metadata = {
        "source_kind": "demand_appointments_fallback",
        "source_path": demand_path,
        "raw_sheet": None,
        "derived_sheet": None,
    }
    return mapped, metadata


def select_install_history_source(
    appointments_workbook: str | None = None,
    demand_appointments_csv: str | None = None,
    config_module: Any = config,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return the richest install history source available."""

    workbook = appointments_workbook or config_module.SERVICE_APPTS_DISPATCH
    demand_csv = demand_appointments_csv or str(Path(config_module.OPTIMIZATION_DIR) / "demand_appointments.csv")

    if workbook and Path(workbook).exists():
        return _history_source_from_workbook(workbook, config_module=config_module)
    if demand_csv and Path(demand_csv).exists():
        return _history_source_from_demand_csv(demand_csv)
    raise FileNotFoundError(
        "Could not resolve an install history source from the raw workbook or demand_appointments.csv"
    )


def _build_history_rows(
    source_df: pd.DataFrame,
    config_module: Any = config,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in source_df.iterrows():
        rows.extend(
            parse_install_history_row(
                appointment_number=row.get("Appointment Number"),
                account_name=row.get("Account: Account Name"),
                service_type=row.get("Service Type"),
                description=row.get("Description"),
                subject=row.get("Subject"),
                scheduled_start=row.get("Scheduled Start"),
                duration_days=row.get("Duration Days"),
                duration_hours=row.get("Duration Hours"),
            )
        )

    history = pd.DataFrame(rows)
    if history.empty:
        return pd.DataFrame(
            columns=[
                "appointment_number",
                "account_name",
                "service_type",
                "description",
                "subject",
                "scheduled_start",
                "duration_days",
                "duration_hours",
                "normalized_family",
                "units_inferred",
                "event_weight",
                "history_exclusion_reason",
                "history_resolution_type",
                "case_id",
                "serial_token",
                "raw_family_matches",
                "include_in_history",
                "include_in_future_mix",
                "future_mix_exclusion_reason",
                "duplicate_group_key",
                "duplicate_group_size",
                "duplicate_group_rank",
            ]
        )

    history["serial_token"] = history["serial_tokens"].apply(
        lambda tokens: tokens[0] if tokens else None
    )
    history["raw_family_matches"] = history["raw_family_matches"].apply(
        lambda values: ",".join(values) if values else ""
    )
    history["normalized_description"] = history["description"].apply(_normalize_description_for_dedupe)
    history["include_in_history_pre_dedupe"] = history["history_exclusion_reason"].isna()

    history["duplicate_group_key"] = history.apply(
        lambda r: (
            f"case::{r['case_id']}::{r['normalized_family']}"
            if r["include_in_history_pre_dedupe"] and r["case_id"] and r["normalized_family"]
            else (
                f"desc::{r['account_name'].upper()}::{r['normalized_description']}::{r['normalized_family']}"
                if r["include_in_history_pre_dedupe"] and r["normalized_family"]
                else f"appt::{r['appointment_number']}::{r.name}"
            )
        ),
        axis=1,
    )

    sort_df = history.assign(
        _duration_sort=pd.to_numeric(history["duration_days"], errors="coerce").fillna(-1.0),
        _start_sort=pd.to_datetime(history["scheduled_start"], errors="coerce"),
    ).sort_values(
        ["duplicate_group_key", "_duration_sort", "_start_sort", "appointment_number"],
        ascending=[True, False, True, True],
    )
    sort_df["duplicate_group_rank"] = sort_df.groupby("duplicate_group_key").cumcount() + 1
    sort_df["duplicate_group_size"] = sort_df.groupby("duplicate_group_key")["appointment_number"].transform("size")
    history = sort_df.drop(columns=["_duration_sort", "_start_sort"]).copy()

    history["include_in_history"] = history["include_in_history_pre_dedupe"] & history["duplicate_group_rank"].eq(1)
    history["future_mix_exclusion_reason"] = None
    history.loc[
        history["include_in_history"]
        & history["normalized_family"].eq("HPS")
        & config_module.PATIENT_SIM_EXCLUDE_HPS_FROM_FUTURE_MIX,
        "future_mix_exclusion_reason",
    ] = "hps_excluded_by_config"
    history["include_in_future_mix"] = history["include_in_history"] & history["future_mix_exclusion_reason"].isna()
    return history


def _with_family_order(
    df: pd.DataFrame,
    config_module: Any = config,
) -> pd.DataFrame:
    family_order = list(config_module.PATIENT_SIM_FAMILY_ASSUMPTIONS.keys())
    df["family_order"] = df["family"].apply(
        lambda value: family_order.index(value) if value in family_order else len(family_order)
    )
    return df.sort_values(["family_order", "family"]).drop(columns=["family_order"])


def _summarize_historical_mix(
    history: pd.DataFrame,
    config_module: Any = config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    included = history[history["include_in_history"] & history["normalized_family"].notna()].copy()
    events = (
        included.groupby("normalized_family", as_index=False)
        .agg(
            event_equivalent_count=("event_weight", "sum"),
            deduped_component_rows=("appointment_number", "count"),
        )
        .rename(columns={"normalized_family": "family"})
    )
    units = (
        included.groupby("normalized_family", as_index=False)
        .agg(
            units_inferred=("units_inferred", "sum"),
            deduped_component_rows=("appointment_number", "count"),
        )
        .rename(columns={"normalized_family": "family"})
    )

    if not events.empty:
        total_events = float(events["event_equivalent_count"].sum())
        events["event_share"] = events["event_equivalent_count"] / total_events if total_events else 0.0
        events = _with_family_order(events, config_module=config_module)
    if not units.empty:
        total_units = float(units["units_inferred"].sum())
        units["unit_share"] = units["units_inferred"] / total_units if total_units else 0.0
        units = _with_family_order(units, config_module=config_module)
    return events, units


def _normalize_timestamp(value: object) -> pd.Timestamp | None:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return None
    if getattr(timestamp, "tzinfo", None):
        timestamp = timestamp.tz_localize(None)
    return timestamp.normalize()


def _recency_weight(
    start_value: object,
    reference_date: pd.Timestamp | None,
    config_module: Any = config,
) -> float:
    if not config_module.PATIENT_SIM_RECENCY_WEIGHTING_ENABLED:
        return 1.0
    start = _normalize_timestamp(start_value)
    if start is None or reference_date is None:
        return 1.0
    age_years = max((reference_date - start).days, 0) / 365.25
    halflife = float(config_module.PATIENT_SIM_RECENCY_HALFLIFE_YEARS or 1.0)
    if halflife <= 0:
        return 1.0
    return 0.5 ** (age_years / halflife)


def _build_forward_mix(
    history: pd.DataFrame,
    config_module: Any = config,
) -> pd.DataFrame:
    included = history[history["include_in_history"] & history["normalized_family"].notna()].copy()
    if included.empty:
        return pd.DataFrame(
            columns=[
                "family",
                "historical_units",
                "historical_event_equivalent_count",
                "recency_weighted_units",
                "manual_multiplier",
                "forward_mix_excluded",
                "forward_mix_exclusion_reason",
                "forward_units_after_adjustments",
                "forward_share",
            ]
        )

    reference_date = None
    normalized_starts = included["scheduled_start"].apply(_normalize_timestamp)
    valid_starts = normalized_starts.dropna()
    if not valid_starts.empty:
        reference_date = valid_starts.max()

    included["recency_weight"] = included["scheduled_start"].apply(
        lambda value: _recency_weight(
            value,
            reference_date=reference_date,
            config_module=config_module,
        )
    )
    included["manual_multiplier"] = included["normalized_family"].map(
        config_module.PATIENT_SIM_FORWARD_MIX_MANUAL_MULTIPLIERS
    ).fillna(1.0)
    included["forward_weighted_units"] = (
        included["units_inferred"] * included["recency_weight"] * included["manual_multiplier"]
    )

    summary = (
        included.groupby("normalized_family", as_index=False)
        .agg(
            historical_units=("units_inferred", "sum"),
            recency_weighted_units=("forward_weighted_units", "sum"),
            historical_event_equivalent_count=("event_weight", "sum"),
        )
        .rename(columns={"normalized_family": "family"})
    )
    summary["manual_multiplier"] = summary["family"].map(
        config_module.PATIENT_SIM_FORWARD_MIX_MANUAL_MULTIPLIERS
    ).fillna(1.0)
    summary["forward_mix_excluded"] = False
    summary["forward_mix_exclusion_reason"] = None
    summary.loc[
        summary["family"].eq("HPS") & config_module.PATIENT_SIM_EXCLUDE_HPS_FROM_FUTURE_MIX,
        "forward_mix_excluded",
    ] = True
    summary.loc[
        summary["family"].eq("HPS") & config_module.PATIENT_SIM_EXCLUDE_HPS_FROM_FUTURE_MIX,
        "forward_mix_exclusion_reason",
    ] = "hps_excluded_by_config"
    summary["forward_units_after_adjustments"] = summary["recency_weighted_units"].where(
        ~summary["forward_mix_excluded"],
        0.0,
    )
    total_forward_units = float(summary["forward_units_after_adjustments"].sum())
    summary["forward_share"] = (
        summary["forward_units_after_adjustments"] / total_forward_units if total_forward_units else 0.0
    )
    return _with_family_order(summary, config_module=config_module)


def _build_family_economics(
    history: pd.DataFrame,
    forward_mix: pd.DataFrame,
    config_module: Any = config,
) -> pd.DataFrame:
    included = history[history["include_in_history"] & history["normalized_family"].notna()].copy()
    all_medians = included.groupby("normalized_family")["duration_days"].median()
    single_unit_medians = included[included["units_inferred"].eq(1.0)].groupby("normalized_family")["duration_days"].median()

    rows: list[dict[str, Any]] = []
    for family, assumptions in config_module.PATIENT_SIM_FAMILY_ASSUMPTIONS.items():
        observed_single = single_unit_medians.get(family, math.nan)
        observed_all = all_medians.get(family, math.nan)
        override = assumptions.get("install_calendar_days_override")
        if override is not None:
            used_days = float(override)
            install_days_source = "config_override"
        elif not pd.isna(observed_single):
            used_days = float(observed_single)
            install_days_source = "historical_single_unit_median"
        elif not pd.isna(observed_all):
            used_days = float(observed_all)
            install_days_source = "historical_all_rows_median"
        else:
            used_days = math.nan
            install_days_source = "missing"

        rows.append(
            {
                "family": family,
                "install_revenue_usd": assumptions.get("install_revenue_usd"),
                "install_margin": assumptions.get("install_margin"),
                "install_calendar_days_override": override,
                "historical_single_unit_calendar_days_median": observed_single,
                "historical_all_rows_calendar_days_median": observed_all,
                "install_calendar_days_used": used_days,
                "install_calendar_days_source": install_days_source,
                "revenue_range_low_usd": assumptions.get("revenue_range_low_usd"),
                "revenue_range_high_usd": assumptions.get("revenue_range_high_usd"),
                "revenue_source_note": assumptions.get("revenue_source_note"),
                "forward_mix_excluded": bool(
                    not forward_mix.empty
                    and forward_mix.loc[forward_mix["family"].eq(family), "forward_mix_excluded"].any()
                ),
                "forward_share": float(
                    forward_mix.loc[forward_mix["family"].eq(family), "forward_share"].iloc[0]
                )
                if not forward_mix.empty and family in set(forward_mix["family"])
                else 0.0,
            }
        )

    economics = pd.DataFrame(rows)
    active = economics[economics["forward_share"] > 0].copy()
    missing_revenue = active[active["install_revenue_usd"].isna()]
    if not missing_revenue.empty:
        families = ", ".join(sorted(missing_revenue["family"].tolist()))
        raise ValueError(
            "Install model has active forward-mix families without revenue assumptions: "
            f"{families}"
        )
    missing_days = active[active["install_calendar_days_used"].isna()]
    if not missing_days.empty:
        families = ", ".join(sorted(missing_days["family"].tolist()))
        raise ValueError(
            "Install model has active forward-mix families without calendar-day assumptions: "
            f"{families}"
        )
    return _with_family_order(economics, config_module=config_module)


def build_patient_sim_install_model(
    appointments_workbook: str | None = None,
    demand_appointments_csv: str | None = None,
    config_module: Any = config,
) -> PatientSimInstallModel:
    """Build cleaned install history, historical mix, forward mix, and economics."""

    source_df, source_metadata = select_install_history_source(
        appointments_workbook=appointments_workbook,
        demand_appointments_csv=demand_appointments_csv,
        config_module=config_module,
    )
    history_rows = _build_history_rows(source_df, config_module=config_module)
    historical_mix_events, historical_mix_units = _summarize_historical_mix(
        history_rows,
        config_module=config_module,
    )
    forward_mix = _build_forward_mix(history_rows, config_module=config_module)
    family_economics = _build_family_economics(
        history_rows,
        forward_mix,
        config_module=config_module,
    )

    assumptions = {
        "capacity_model_time_unit": config_module.PATIENT_SIM_CAPACITY_TIME_UNIT,
        "forward_mix_basis": "unit_equivalents",
        "exclude_hps_from_future_mix": bool(config_module.PATIENT_SIM_EXCLUDE_HPS_FROM_FUTURE_MIX),
        "recency_weighting_enabled": bool(config_module.PATIENT_SIM_RECENCY_WEIGHTING_ENABLED),
        "recency_halflife_years": float(config_module.PATIENT_SIM_RECENCY_HALFLIFE_YEARS),
        "forward_mix_manual_multipliers": dict(config_module.PATIENT_SIM_FORWARD_MIX_MANUAL_MULTIPLIERS),
        "freed_capacity_utilization_factor": float(config_module.FREED_CAPACITY_UTILIZATION_FACTOR),
    }

    return PatientSimInstallModel(
        source_metadata=source_metadata,
        history_rows=history_rows,
        historical_mix_events=historical_mix_events,
        historical_mix_units=historical_mix_units,
        forward_mix=forward_mix,
        family_economics=family_economics,
        assumptions=assumptions,
    )
