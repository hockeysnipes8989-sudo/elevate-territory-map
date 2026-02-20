"""Step 3: Match install base accounts to service appointment locations."""
import os
import sys
import pandas as pd
from fuzzywuzzy import fuzz

sys.path.insert(0, os.path.dirname(__file__))
import config


# High-confidence account aliases (install base account -> appointment account)
MANUAL_ACCOUNT_MAP = {
    "Sarasota Memorial Hospital": "Sarasota Memorial Hospital-Venice Campus",
    "Toronto Metropolitan University": "Toronto Metropolitan University - School of Medicine",
    "University of Alabama at Birmingham": (
        "University of Alabama at Birmingham - Office of Interprofessional Simulation"
    ),
    "University of the Virgin Islands - School of Nursing- St. Thomas Campus": (
        "University of the Virgin Islands"
    ),
    "University of the Virgin Islands School of Nursing- Kingshill Campus": (
        "University of the Virgin Islands"
    ),
    "Cleveland Clinic - Akron General": "Cleveland Clinic",
    "North Georgia Technical College - Blairsville Campus": "North Georgia Technical College",
    "Valencia College School of Public Safety": "Valencia College - West Campus",
}


# Known false-positive fuzzy pairs to reject.
# Store normalized lowercase pairs: (install_account, appointment_account).
FUZZY_REJECT_PAIRS = {
    ("lenoir community college", "owens community college"),
    (
        "pierpont community and technical college",
        "elizabethtown community and technical college",
    ),
    ("clovis community college", "owens community college"),
    ("tri-county high school", "harris county high school"),
    ("jersey college of nursing - jacksonville", "chamberlain college of nursing - jacksonville"),
    ("sonoma state university", "arizona state university"),
    ("durham technical community college", "sowela technical community college"),
    ("gateway community college", "washtenaw community college"),
    (
        "jesse brown veterans affairs medical center",
        "detroit veterans affairs medical center",
    ),
    ("heartland community college", "martin community college"),
}


def norm_account(name):
    """Normalize account names for matching."""
    return str(name).strip().lower()


def exact_match(install_accounts, appt_account_coords):
    """Exact match on account name (case-insensitive)."""
    matched = {}
    for acct in install_accounts:
        acct_norm = norm_account(acct)
        if acct_norm in appt_account_coords:
            matched[acct] = {
                "appt_account_norm": acct_norm,
                "match_method": "exact",
                "score": 100,
            }
    return matched


def manual_match(unmatched_accounts, appt_account_coords):
    """Apply explicit manual account aliases."""
    matched = {}
    for acct in unmatched_accounts:
        mapped = MANUAL_ACCOUNT_MAP.get(acct)
        if not mapped:
            continue
        mapped_norm = norm_account(mapped)
        if mapped_norm in appt_account_coords:
            matched[acct] = {
                "appt_account_norm": mapped_norm,
                "match_method": "manual",
                "score": 100,
            }
    return matched


def fuzzy_match(unmatched_accounts, appt_accounts_list, threshold=85):
    """Fuzzy match remaining accounts against appointment account names."""
    matched = {}
    kept_rows = []
    rejected_rows = []

    for acct in unmatched_accounts:
        acct_norm = norm_account(acct)
        scored = []
        for appt_norm in appt_accounts_list:
            score = fuzz.ratio(acct_norm, appt_norm)
            scored.append((score, appt_norm))

        scored.sort(reverse=True)
        selected = None
        for score, appt_norm in scored:
            if score < threshold:
                break
            pair = (acct_norm, appt_norm)
            if pair in FUZZY_REJECT_PAIRS:
                rejected_rows.append(
                    {
                        "install_account": acct,
                        "appt_account_norm": appt_norm,
                        "score": score,
                    }
                )
                continue
            selected = (score, appt_norm)
            break

        if selected:
            score, appt_norm = selected
            matched[acct] = {
                "appt_account_norm": appt_norm,
                "match_method": "fuzzy",
                "score": score,
            }
            kept_rows.append(
                {
                    "install_account": acct,
                    "appt_account_norm": appt_norm,
                    "score": score,
                }
            )

    return matched, pd.DataFrame(kept_rows), pd.DataFrame(rejected_rows)


def main():
    # Load data
    install = pd.read_csv(config.CLEAN_INSTALL_CSV)
    appts = pd.read_csv(config.GEOCODED_APPTS_CSV)

    # Only keep active-contract assets for the map
    active = install[install["has_active_contract"]].copy()
    print(f"Active contract assets: {len(active)}")
    print(f"Unique active accounts: {active['Account Name'].nunique()}")

    # Build account -> averaged coordinates from appointments with coordinates
    appts_with_coords = appts.dropna(subset=["lat", "lon"]).copy()
    appts_with_coords["acct_norm"] = appts_with_coords["Account: Account Name"].apply(norm_account)

    appt_account_coords = {}
    appt_account_display = {}
    for acct_norm, group in appts_with_coords.groupby("acct_norm"):
        appt_account_coords[acct_norm] = {
            "lat": group["lat"].mean(),
            "lon": group["lon"].mean(),
            "city": group["City"].iloc[0] if "City" in group.columns else None,
            "state": group["State/Province"].iloc[0] if "State/Province" in group.columns else None,
        }
        appt_account_display[acct_norm] = group["Account: Account Name"].iloc[0]

    appt_accounts_list = list(appt_account_coords.keys())

    # Unique install base accounts with active contracts
    install_accounts = active["Account Name"].dropna().unique().tolist()
    print(
        f"\nMatching {len(install_accounts)} install base accounts "
        f"to {len(appt_accounts_list)} appointment accounts..."
    )

    # Step 1: Exact
    exact = exact_match(install_accounts, appt_account_coords)
    print(f"  Exact matches: {len(exact)}")

    # Step 2: Manual aliases
    unmatched = [a for a in install_accounts if a not in exact]
    manual = manual_match(unmatched, appt_account_coords)
    print(f"  Manual matches: {len(manual)}")

    # Step 3: Fuzzy with explicit false-positive rejects
    unmatched = [a for a in install_accounts if a not in exact and a not in manual]
    fuzzy, fuzzy_kept, fuzzy_rejected = fuzzy_match(unmatched, appt_accounts_list, threshold=85)
    print(f"  Fuzzy matches (>=85): {len(fuzzy)}")
    print(f"  Fuzzy rejects (known bad pairs): {len(fuzzy_rejected)}")

    # Combine matches
    all_matches = {}
    all_matches.update(exact)
    all_matches.update(manual)
    all_matches.update(fuzzy)

    still_unmatched = [a for a in install_accounts if a not in all_matches]
    print(f"  Unmatched accounts: {len(still_unmatched)}")

    # Top unmatched accounts (for audit reporting)
    unmatched_top20 = (
        active[active["Account Name"].isin(still_unmatched)]
        .groupby(["Account Name", "Territory"])
        .size()
        .reset_index(name="active_assets")
        .sort_values("active_assets", ascending=False)
        .head(20)
    )
    print("\nTop 20 unmatched accounts (active assets):")
    if unmatched_top20.empty:
        print("  None")
    else:
        print(unmatched_top20.to_string(index=False))

    print("\n10 lowest-scoring fuzzy matches kept:")
    if fuzzy_kept.empty:
        print("  None")
    else:
        low = fuzzy_kept.sort_values("score", ascending=True).head(10).copy()
        low["appt_account"] = low["appt_account_norm"].map(appt_account_display).fillna(
            low["appt_account_norm"]
        )
        print(low[["install_account", "appt_account", "score"]].to_string(index=False))

    if not fuzzy_rejected.empty:
        print("\nRejected fuzzy pairs:")
        rej = fuzzy_rejected.copy()
        rej["appt_account"] = rej["appt_account_norm"].map(appt_account_display).fillna(
            rej["appt_account_norm"]
        )
        print(rej[["install_account", "appt_account", "score"]].to_string(index=False))

    # Apply coordinates/match metadata to install base assets
    def matched_meta(account_name, key):
        match = all_matches.get(account_name)
        if not match:
            return None
        appt_norm = match["appt_account_norm"]
        if key == "lat":
            return appt_account_coords.get(appt_norm, {}).get("lat")
        if key == "lon":
            return appt_account_coords.get(appt_norm, {}).get("lon")
        if key == "appt_account":
            return appt_account_display.get(appt_norm)
        if key == "method":
            return match["match_method"]
        if key == "score":
            return match["score"]
        return None

    active["lat"] = active["Account Name"].map(lambda a: matched_meta(a, "lat"))
    active["lon"] = active["Account Name"].map(lambda a: matched_meta(a, "lon"))
    active["matched_appointment_account"] = active["Account Name"].map(
        lambda a: matched_meta(a, "appt_account")
    )
    active["match_method"] = active["Account Name"].map(lambda a: matched_meta(a, "method"))
    active["match_score"] = active["Account Name"].map(lambda a: matched_meta(a, "score"))
    active["matched"] = active["Account Name"].isin(all_matches)

    matched_assets = int(active["matched"].sum())
    print(
        f"\nAssets with coordinates: {matched_assets}/{len(active)} "
        f"({matched_assets/len(active)*100:.1f}%)"
    )

    active.to_csv(config.INSTALL_MATCHED_CSV, index=False)
    print(f"Saved: {config.INSTALL_MATCHED_CSV}")

    # Build territory summary for choropleth
    territory_summary = (
        active.groupby("Territory")
        .agg(
            total_assets=("Asset Name", "count"),
            matched_assets=("matched", "sum"),
            unique_accounts=("Account Name", "nunique"),
        )
        .reset_index()
    )
    territory_summary["unmatched_assets"] = (
        territory_summary["total_assets"] - territory_summary["matched_assets"]
    )

    print("\nTerritory summary:")
    print(territory_summary.to_string(index=False))

    territory_summary.to_csv(config.TERRITORY_SUMMARY_CSV, index=False)
    print(f"\nSaved: {config.TERRITORY_SUMMARY_CSV}")
    print("Step 3 complete.")


if __name__ == "__main__":
    main()
