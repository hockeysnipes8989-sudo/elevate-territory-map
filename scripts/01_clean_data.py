"""Step 1: Load Excel files, clean data, export to CSV."""
import pandas as pd
import os
import sys
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


def build_geocode_key(city, state):
    """Build a geocode-friendly string from city and state."""
    if pd.isna(city) or pd.isna(state):
        return None
    city = str(city).strip()
    state = str(state).strip()
    if state in ("Canada", "Ontario"):
        if state == "Ontario":
            return f"{city}, Ontario, Canada"
        return f"{city}, Canada"
    return f"{city}, {state}, USA"


def load_technicians():
    """Load technician home bases from Resources sheet."""
    print("Loading technicians...")
    df = pd.read_excel(
        config.SERVICE_APPTS_REPORT,
        sheet_name=config.APPTS_REPORT_RESOURCES_SHEET,
    )
    df = df[["Service Resource Name", "Location", "Comment"]].copy()
    df.columns = ["name", "location", "comment"]
    df["name"] = df["name"].str.strip()

    # Add status
    df["status"] = df["name"].map(config.TECH_STATUS).fillna("active")

    # Build geocode key from location
    def tech_geocode_key(loc):
        if pd.isna(loc):
            return None
        loc = str(loc).strip()
        if "Canada" in loc:
            return f"{loc}"
        return f"{loc}, USA"

    df["geocode_key"] = df["location"].apply(tech_geocode_key)
    print(f"  Technicians: {len(df)}")
    return df


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

    # Flag active contracts
    df["has_active_contract"] = df["Contract_Status_Clean"].str.lower().str.contains(
        "active", na=False
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
