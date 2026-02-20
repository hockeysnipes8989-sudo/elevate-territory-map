"""Step 3: Match install base accounts to service appointment locations."""
import os
import sys
import pandas as pd
from fuzzywuzzy import fuzz

sys.path.insert(0, os.path.dirname(__file__))
import config


def exact_match(install_accounts, appt_account_coords):
    """Exact match on Account Name (case-insensitive, stripped)."""
    matched = {}
    for acct in install_accounts:
        acct_clean = acct.strip().lower()
        if acct_clean in appt_account_coords:
            matched[acct] = appt_account_coords[acct_clean]
    return matched


def fuzzy_match(unmatched_accounts, appt_accounts_list, threshold=85):
    """Fuzzy match remaining accounts against appointment account names."""
    matched = {}
    for acct in unmatched_accounts:
        best_score = 0
        best_match = None
        for appt_acct in appt_accounts_list:
            score = fuzz.ratio(acct.lower(), appt_acct.lower())
            if score > best_score:
                best_score = score
                best_match = appt_acct
        if best_score >= threshold:
            matched[acct] = best_match
    return matched


def main():
    # Load data
    install = pd.read_csv(config.CLEAN_INSTALL_CSV)
    appts = pd.read_csv(config.GEOCODED_APPTS_CSV)

    # Only keep active-contract assets for the map
    active = install[install["has_active_contract"]].copy()
    print(f"Active contract assets: {len(active)}")
    print(f"Unique active accounts: {active['Account Name'].nunique()}")

    # Build account → average coordinates from appointments
    appts_with_coords = appts.dropna(subset=["lat", "lon"])
    appt_account_coords = {}
    for acct, group in appts_with_coords.groupby("Account: Account Name"):
        acct_clean = str(acct).strip().lower()
        appt_account_coords[acct_clean] = {
            "lat": group["lat"].mean(),
            "lon": group["lon"].mean(),
            "city": group["City"].iloc[0] if "City" in group.columns else None,
            "state": group["State/Province"].iloc[0] if "State/Province" in group.columns else None,
        }

    appt_accounts_list = list(appt_account_coords.keys())

    # Unique install base accounts with active contracts
    install_accounts = active["Account Name"].dropna().unique().tolist()
    print(f"\nMatching {len(install_accounts)} install base accounts to {len(appt_accounts_list)} appointment accounts...")

    # Step 1: Exact match
    exact = exact_match(install_accounts, appt_account_coords)
    print(f"  Exact matches: {len(exact)}")

    # Step 2: Fuzzy match remaining
    unmatched = [a for a in install_accounts if a not in exact]
    fuzzy = fuzzy_match(unmatched, appt_accounts_list, threshold=85)
    print(f"  Fuzzy matches (>=85): {len(fuzzy)}")

    # Combine matches
    all_matches = {}
    for acct, coords in exact.items():
        all_matches[acct] = coords
    for acct, appt_acct in fuzzy.items():
        all_matches[acct] = appt_account_coords[appt_acct.lower()]

    still_unmatched = [a for a in install_accounts if a not in all_matches]
    print(f"  Unmatched: {len(still_unmatched)}")

    # Apply coordinates to install base
    active["lat"] = active["Account Name"].map(lambda a: all_matches.get(a, {}).get("lat") if a in all_matches else None)
    active["lon"] = active["Account Name"].map(lambda a: all_matches.get(a, {}).get("lon") if a in all_matches else None)
    active["matched"] = active["Account Name"].isin(all_matches)

    matched_assets = active["matched"].sum()
    print(f"\nAssets with coordinates: {matched_assets}/{len(active)} ({matched_assets/len(active)*100:.1f}%)")

    active.to_csv(config.INSTALL_MATCHED_CSV, index=False)
    print(f"Saved: {config.INSTALL_MATCHED_CSV}")

    # Build territory summary for choropleth
    territory_summary = active.groupby("Territory").agg(
        total_assets=("Asset Name", "count"),
        matched_assets=("matched", "sum"),
        unique_accounts=("Account Name", "nunique"),
    ).reset_index()
    territory_summary["unmatched_assets"] = territory_summary["total_assets"] - territory_summary["matched_assets"]

    print(f"\nTerritory summary:")
    print(territory_summary.to_string(index=False))

    territory_summary.to_csv(config.TERRITORY_SUMMARY_CSV, index=False)
    print(f"\nSaved: {config.TERRITORY_SUMMARY_CSV}")
    print("Step 3 complete.")


if __name__ == "__main__":
    main()
