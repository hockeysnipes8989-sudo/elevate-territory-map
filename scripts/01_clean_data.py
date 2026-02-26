"""Step 1: Load Excel files, clean data, export to CSV."""
import pandas as pd
import os
import sys
import re
sys.path.insert(0, os.path.dirname(__file__))
import config


def clean_city(city):
    """Strip leading dots and whitespace from city names."""
    if pd.isna(city):
        return city
    return str(city).lstrip(". ").strip()


def normalize_state_for_geocode(state):
    """Normalize state/province values for geocoding only."""
    if pd.isna(state):
        return state
    s = str(state).strip()
    # US state name → abbreviation mapping
    state_abbrevs = {
        "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
        "California": "CA", "Colorado": "CO", "Connecticut": "CT",
        "Delaware": "DE", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI",
        "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
        "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME",
        "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI",
        "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO",
        "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
        "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM",
        "New York": "NY", "North Carolina": "NC", "North Dakota": "ND",
        "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
        "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD",
        "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
        "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
        "Wisconsin": "WI", "Wyoming": "WY",
        "District of Columbia": "DC",
    }
    if s in state_abbrevs:
        return state_abbrevs[s]
    return s


def normalize_tech_name(name):
    """Map technician name variants to canonical form."""
    if pd.isna(name):
        return name
    name = str(name).strip()
    return config.TECH_NAME_MAP.get(name, name)


def load_appointments():
    """Load and clean service appointments from Derived Fields sheet."""
    print("Loading service appointments...")
    df = pd.read_excel(
        config.SERVICE_APPTS_DISPATCH,
        sheet_name=config.APPTS_DISPATCH_SHEET,
    )
    print(f"  Raw rows: {len(df)}")

    # Clean city values for output + geocoding consistency
    df["City"] = df["City"].apply(clean_city)

    # Normalize technician names
    df["Service Resource: Name"] = df["Service Resource: Name"].apply(normalize_tech_name)

    # Build geocode key from cleaned city + normalized state, while preserving raw values
    df["geocode_key"] = df.apply(
        lambda r: build_geocode_key(
            clean_city(r["City"]),
            normalize_state_for_geocode(r["State/Province"]),
        ),
        axis=1,
    )

    print(f"  Clean rows: {len(df)}")
    print(f"  Unique cities: {df['City'].nunique()}")
    print(f"  Unique states: {df['State/Province'].nunique()}")
    return df


_CANADA_PROVINCE_ABBR = {"AB", "BC", "MB", "NB", "NL", "NS", "NT", "NU", "ON", "PE", "QC", "SK", "YT"}
_CANADA_PROVINCE_NAMES = {
    "Alberta", "British Columbia", "Manitoba", "New Brunswick",
    "Newfoundland and Labrador", "Nova Scotia", "Northwest Territories",
    "Nunavut", "Ontario", "Prince Edward Island", "Quebec", "Saskatchewan", "Yukon",
}


def build_geocode_key(city, state):
    """Build a geocode-friendly string from city and state."""
    if pd.isna(city) or pd.isna(state):
        return None
    city = str(city).strip()
    state = str(state).strip()
    # Detect Canadian provinces by abbreviation or full name
    if state in _CANADA_PROVINCE_ABBR or state in _CANADA_PROVINCE_NAMES or state == "Canada":
        if state in _CANADA_PROVINCE_ABBR:
            return f"{city}, {state}, Canada"
        return f"{city}, {state}, Canada"
    return f"{city}, {state}, USA"


def load_technicians():
    """Load technician home bases from source-of-truth roster workbook."""
    print("Loading technicians...")

    source_path = getattr(config, "EXTERNAL_TECH_ROSTER_XLSX", None)
    use_source_truth = bool(source_path and os.path.exists(source_path))

    if use_source_truth:
        raw = pd.read_excel(source_path, sheet_name=0, header=None)
        headers = raw.iloc[1].tolist()
        df = raw.iloc[2:].copy()
        df.columns = headers
        df = df.dropna(how="all")
        df = df[["Tech", "Location", "Comments"]].copy()
        df.columns = ["name", "location", "comment"]
        print(f"  Source: {source_path}")
    else:
        # Backward-compatible fallback to legacy Resources sheet.
        df = pd.read_excel(
            config.SERVICE_APPTS_REPORT,
            sheet_name=config.APPTS_REPORT_RESOURCES_SHEET,
        )
        df = df[["Service Resource Name", "Location", "Comment"]].copy()
        df.columns = ["name", "location", "comment"]
        print(f"  Source fallback: {config.SERVICE_APPTS_REPORT}::{config.APPTS_REPORT_RESOURCES_SHEET}")

    df["name"] = df["name"].fillna("").astype(str).str.strip()
    df["location"] = df["location"].fillna("").astype(str).str.strip()
    df["comment"] = df["comment"].fillna("").astype(str).str.strip()
    df = df[df["name"] != ""].copy()

    # Exclude former/inactive technicians from current-state outputs.
    inactive_name_mask = df["name"].isin(getattr(config, "INACTIVE_TECH_NAMES", set()))
    inactive_comment_mask = df["comment"].str.contains(
        "no longer with elevate", case=False, na=False
    )
    inactive_mask = inactive_name_mask | inactive_comment_mask
    inactive_count = int(inactive_mask.sum())
    if inactive_count:
        df = df[~inactive_mask].copy()
        print(f"  Excluded inactive technicians: {inactive_count}")

    # Status for map styling.
    contractor_mask = df["name"].str.contains("contractor", case=False, na=False)
    on_demand_mask = df["comment"].str.contains("use him when required", case=False, na=False)
    df["status"] = "active"
    df.loc[contractor_mask | on_demand_mask, "status"] = "special"

    # Build geocode key from location with US/Canada detection.
    canada_abbr = {"AB", "BC", "MB", "NB", "NL", "NS", "NT", "NU", "ON", "PE", "QC", "SK", "YT"}
    usa_abbr = {
        "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "HI", "IA", "ID", "IL",
        "IN", "KS", "KY", "LA", "MA", "MD", "ME", "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE",
        "NH", "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT",
        "VA", "VT", "WA", "WI", "WV", "WY",
    }

    def tech_geocode_key(loc):
        if pd.isna(loc):
            return None
        loc = str(loc).strip()
        if not loc:
            return None
        if re.search(r"\bCanada\b", loc, flags=re.IGNORECASE):
            return loc
        tail = re.split(r"[,\s]+", loc.strip())[-1].upper()
        if tail in canada_abbr:
            return f"{loc}, Canada"
        if tail in usa_abbr:
            return f"{loc}, USA"
        return f"{loc}, USA"

    df["geocode_key"] = df["location"].apply(tech_geocode_key)
    expected = getattr(config, "EXPECTED_CURRENT_TECH_COUNT", None)
    if expected is not None and int(len(df)) != int(expected):
        print(
            f"  WARNING: Expected {int(expected)} current technicians but loaded {len(df)} from source roster."
        )
    print(f"  Technicians: {len(df)}")
    return df.reset_index(drop=True)


def load_install_base():
    """Load and clean install base data."""
    print("Loading install base...")
    df = pd.read_excel(
        config.INSTALL_BASE,
        sheet_name=config.INSTALL_BASE_SHEET,
    )
    print(f"  Raw rows: {len(df)}")

    # Keep relevant columns
    keep_cols = [
        "Asset Name", "Serial Number", "Status", "Account Name",
        "Product Code", "Product Family", "Product Family Short Name",
        "Active Service", "Territory", "Service Contract Status Simple",
        "Contract_Status_Clean", "Service Start Date", "Service End Date",
    ]
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    # Flag active contracts — use word-boundary match to avoid "inactive" matching "active".
    df["has_active_contract"] = df["Contract_Status_Clean"].str.lower().str.contains(
        r"\bactive\b", regex=True, na=False
    )

    print(f"  Total assets: {len(df)}")
    print(f"  Active contracts: {df['has_active_contract'].sum()}")
    print(f"  Unique accounts: {df['Account Name'].nunique()}")
    print(f"  Territories: {df['Territory'].nunique()}")
    return df


def main():
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)

    appts = load_appointments()
    techs = load_technicians()
    install = load_install_base()

    appts.to_csv(config.CLEAN_APPTS_CSV, index=False)
    techs.to_csv(config.CLEAN_TECHS_CSV, index=False)
    install.to_csv(config.CLEAN_INSTALL_CSV, index=False)

    print(f"\nSaved: {config.CLEAN_APPTS_CSV}")
    print(f"Saved: {config.CLEAN_TECHS_CSV}")
    print(f"Saved: {config.CLEAN_INSTALL_CSV}")
    print("Step 1 complete.")


if __name__ == "__main__":
    main()
