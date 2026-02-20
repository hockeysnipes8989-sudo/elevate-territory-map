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
    ("boston medical center", "tucson medical center"),
    ("kellogg community college", "pueblo community college"),
    ("cleveland community college", "craven community college"),
    ("xavier university", "viterbo university"),
    ("manatee technical college", "tooele technical college"),
    ("castle medical center", "wesley medical center"),
    ("wayne state university", "weber state university"),
    ("muskegon community college", "sampson community college"),
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


def fuzzy_match(
    unmatched_accounts,
    appt_accounts_by_territory,
    install_account_territory,
    threshold=92,
):
    """Fuzzy match within the same territory using a high-confidence threshold."""
    matched = {}
    kept_rows = []
    rejected_rows = []

    for acct in unmatched_accounts:
        acct_norm = norm_account(acct)
        territory = install_account_territory.get(acct)
        territory_candidates = sorted(appt_accounts_by_territory.get(territory, set()))
        if not territory_candidates:
            continue

        scored = []
        for appt_norm in territory_candidates:
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
                        "territory": territory,
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
                    "territory": territory,
                }
            )

    return matched, pd.DataFrame(kept_rows), pd.DataFrame(rejected_rows)


def build_matches(install_df, appt_account_coords, appt_accounts_by_territory):
    """Build account-level match dictionary for a given install dataframe."""
    install_accounts = install_df["Account Name"].dropna().unique().tolist()
    install_account_territory = (
        install_df.groupby("Account Name")["Territory"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
        .to_dict()
    )

    exact = exact_match(install_accounts, appt_account_coords)
    unmatched = [a for a in install_accounts if a not in exact]
    manual = manual_match(unmatched, appt_account_coords)
    unmatched = [a for a in install_accounts if a not in exact and a not in manual]
    fuzzy, fuzzy_kept, fuzzy_rejected = fuzzy_match(
        unmatched,
        appt_accounts_by_territory=appt_accounts_by_territory,
        install_account_territory=install_account_territory,
        threshold=92,
    )

    all_matches = {}
    all_matches.update(exact)
    all_matches.update(manual)
    all_matches.update(fuzzy)

    return all_matches, exact, manual, fuzzy, fuzzy_kept, fuzzy_rejected


def apply_matches(install_df, all_matches, appt_account_coords, appt_account_display):
    """Apply account-level matching metadata to install rows."""

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

    out = install_df.copy()
    out["lat"] = out["Account Name"].map(lambda a: matched_meta(a, "lat"))
    out["lon"] = out["Account Name"].map(lambda a: matched_meta(a, "lon"))
    out["matched_appointment_account"] = out["Account Name"].map(
        lambda a: matched_meta(a, "appt_account")
    )
    out["match_method"] = out["Account Name"].map(lambda a: matched_meta(a, "method"))
    out["match_score"] = out["Account Name"].map(lambda a: matched_meta(a, "score"))
    out["matched"] = out["Account Name"].isin(all_matches)
    return out


def print_unmatched_top20(df, title):
    """Print top-20 unmatched accounts by asset count for a subset dataframe."""
    unmatched_top20 = (
        df[~df["matched"]]
        .groupby(["Account Name", "Territory"])
        .size()
        .reset_index(name="asset_count")
        .sort_values("asset_count", ascending=False)
        .head(20)
    )
    print(f"\nTop 20 unmatched accounts ({title}):")
    if unmatched_top20.empty:
        print("  None")
    else:
        print(unmatched_top20.to_string(index=False))


def main():
    # Load data
    install = pd.read_csv(config.CLEAN_INSTALL_CSV)
    appts = pd.read_csv(config.GEOCODED_APPTS_CSV)

    print(f"Install assets (all): {len(install)}")
    print(f"Install unique accounts (all): {install['Account Name'].nunique()}")

    # Build appointment account lookup
    appts_with_coords = appts.dropna(subset=["lat", "lon"]).copy()
    appts_with_coords["acct_norm"] = appts_with_coords["Account: Account Name"].apply(norm_account)

    appt_account_coords = {}
    appt_account_display = {}
    appt_accounts_by_territory = {}
    for acct_norm, group in appts_with_coords.groupby("acct_norm"):
        appt_account_coords[acct_norm] = {
            "lat": group["lat"].mean(),
            "lon": group["lon"].mean(),
            "city": group["City"].iloc[0] if "City" in group.columns else None,
            "state": group["State/Province"].iloc[0] if "State/Province" in group.columns else None,
        }
        appt_account_display[acct_norm] = group["Account: Account Name"].iloc[0]

    for territory, group in appts_with_coords.groupby("Territory"):
        appt_accounts_by_territory[territory] = set(group["acct_norm"])

    # Match all install accounts (active + non-active)
    all_matches, exact, manual, fuzzy, fuzzy_kept, fuzzy_rejected = build_matches(
        install,
        appt_account_coords=appt_account_coords,
        appt_accounts_by_territory=appt_accounts_by_territory,
    )

    install_all_matched = apply_matches(
        install,
        all_matches=all_matches,
        appt_account_coords=appt_account_coords,
        appt_account_display=appt_account_display,
    )

    print(
        f"\nMatching all install accounts to {len(appt_account_coords)} appointment accounts..."
    )
    print(f"  Exact matches: {len(exact)}")
    print(f"  Manual matches: {len(manual)}")
    print(f"  Fuzzy matches (same territory, >=92): {len(fuzzy)}")
    print(f"  Fuzzy rejects (known bad pairs): {len(fuzzy_rejected)}")
    print(
        f"  Matched assets (all): {int(install_all_matched['matched'].sum())}/{len(install_all_matched)} "
        f"({install_all_matched['matched'].mean()*100:.1f}%)"
    )
    print(
        f"  Matched accounts (all): "
        f"{install_all_matched[install_all_matched['matched']]['Account Name'].nunique()}/"
        f"{install_all_matched['Account Name'].nunique()}"
    )

    # Print lowest-scoring fuzzy matches kept
    print("\n10 lowest-scoring fuzzy matches kept:")
    if fuzzy_kept.empty:
        print("  None")
    else:
        low = fuzzy_kept.sort_values("score", ascending=True).head(10).copy()
        low["appt_account"] = low["appt_account_norm"].map(appt_account_display).fillna(
            low["appt_account_norm"]
        )
        print(low[["install_account", "appt_account", "territory", "score"]].to_string(index=False))

    # Save all-assets matched output
    install_all_matched.to_csv(config.INSTALL_ALL_MATCHED_CSV, index=False)
    print(f"\nSaved: {config.INSTALL_ALL_MATCHED_CSV}")

    # Split active vs non-active for downstream map layers
    active = install_all_matched[install_all_matched["has_active_contract"]].copy()
    non_active = install_all_matched[~install_all_matched["has_active_contract"]].copy()

    active.to_csv(config.INSTALL_MATCHED_CSV, index=False)
    non_active.to_csv(config.INSTALL_NONACTIVE_MATCHED_CSV, index=False)
    print(f"Saved: {config.INSTALL_MATCHED_CSV}")
    print(f"Saved: {config.INSTALL_NONACTIVE_MATCHED_CSV}")

    print(
        f"\nActive assets with coordinates: {int(active['matched'].sum())}/{len(active)} "
        f"({active['matched'].mean()*100:.1f}%)"
    )
    print(
        f"Non-active assets with coordinates: {int(non_active['matched'].sum())}/{len(non_active)} "
        f"({non_active['matched'].mean()*100:.1f}%)"
    )

    print_unmatched_top20(active, title="active-contract assets")
    print_unmatched_top20(non_active, title="non-active assets")

    # Build active-contract territory summary for existing choropleth
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

    print("\nTerritory summary (active contracts):")
    print(territory_summary.to_string(index=False))

    territory_summary.to_csv(config.TERRITORY_SUMMARY_CSV, index=False)
    print(f"\nSaved: {config.TERRITORY_SUMMARY_CSV}")
    print("Step 3 complete.")


if __name__ == "__main__":
    main()
