"""Step 2: Geocode unique city/state pairs via Nominatim with caching."""
import json
import os
import sys
import time
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

sys.path.insert(0, os.path.dirname(__file__))
import config


def load_cache():
    """Load geocode cache from JSON file."""
    if os.path.exists(config.GEOCODE_CACHE):
        with open(config.GEOCODE_CACHE, "r") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    """Save geocode cache to JSON file."""
    with open(config.GEOCODE_CACHE, "w") as f:
        json.dump(cache, f, indent=2, sort_keys=True)


def apply_overrides(cache):
    """Apply deterministic geocode overrides from config."""
    applied = 0
    for key, coords in config.GEOCODE_OVERRIDES.items():
        if cache.get(key) != coords:
            cache[key] = coords
            applied += 1
    return applied


def geocode_key(key, geolocator, cache, max_retries=3):
    """Geocode a single key, using cache if available. Retries on transient errors."""
    if key in cache:
        return cache[key]

    for attempt in range(1, max_retries + 1):
        try:
            location = geolocator.geocode(key, timeout=10)
            if location:
                result = {"lat": location.latitude, "lon": location.longitude}
                cache[key] = result
                return result
            else:
                print(f"  NOT FOUND: {key}")
                cache[key] = None
                return None
        except GeocoderTimedOut as e:
            wait = 2 ** attempt
            print(f"  TIMEOUT for {key} (attempt {attempt}/{max_retries}), retrying in {wait}s: {e}")
            if attempt < max_retries:
                time.sleep(wait)
        except GeocoderServiceError as e:
            print(f"  SERVICE ERROR for {key}: {e}")
            return None
    print(f"  FAILED after {max_retries} attempts: {key}")
    return None


def main():
    geolocator = Nominatim(user_agent=config.GEOCODE_USER_AGENT)
    cache = load_cache()
    overrides_applied = apply_overrides(cache)
    if overrides_applied:
        print(f"Applied geocode overrides: {overrides_applied}")

    # Load cleaned data
    appts = pd.read_csv(config.CLEAN_APPTS_CSV)
    techs = pd.read_csv(config.CLEAN_TECHS_CSV)

    # Collect all unique geocode keys
    appt_keys = set(appts["geocode_key"].dropna().unique())
    tech_keys = set(techs["geocode_key"].dropna().unique())
    all_keys = appt_keys | tech_keys

    # Filter out already-cached keys
    to_geocode = [k for k in sorted(all_keys) if k not in cache]

    print(f"Total unique keys: {len(all_keys)}")
    print(f"Already cached: {len(all_keys) - len(to_geocode)}")
    print(f"To geocode: {len(to_geocode)}")

    if to_geocode:
        print(f"Estimated time: {len(to_geocode) * config.GEOCODE_DELAY / 60:.1f} minutes")
        for i, key in enumerate(to_geocode):
            result = geocode_key(key, geolocator, cache)
            status = f"({result['lat']:.4f}, {result['lon']:.4f})" if result else "NOT FOUND"
            print(f"  [{i+1}/{len(to_geocode)}] {key} → {status}")
            if i < len(to_geocode) - 1:
                time.sleep(config.GEOCODE_DELAY)
            # Save cache every 50 lookups
            if (i + 1) % 50 == 0:
                save_cache(cache)

    save_cache(cache)

    # Apply geocoded coordinates to appointments
    appts["lat"] = appts["geocode_key"].map(lambda k: cache.get(k, {}).get("lat") if cache.get(k) else None)
    appts["lon"] = appts["geocode_key"].map(lambda k: cache.get(k, {}).get("lon") if cache.get(k) else None)

    geocoded_count = appts["lat"].notna().sum()
    print(f"\nAppointments geocoded: {geocoded_count}/{len(appts)} ({geocoded_count/len(appts)*100:.1f}%)")

    appts.to_csv(config.GEOCODED_APPTS_CSV, index=False)
    print(f"Saved: {config.GEOCODED_APPTS_CSV}")

    # Apply geocoded coordinates to technicians
    techs["lat"] = techs["geocode_key"].map(lambda k: cache.get(k, {}).get("lat") if cache.get(k) else None)
    techs["lon"] = techs["geocode_key"].map(lambda k: cache.get(k, {}).get("lon") if cache.get(k) else None)

    tech_geocoded = techs["lat"].notna().sum()
    print(f"Technicians geocoded: {tech_geocoded}/{len(techs)}")

    techs.to_csv(config.GEOCODED_TECHS_CSV, index=False)
    print(f"Saved: {config.GEOCODED_TECHS_CSV}")

    # Report any failures
    failed = [k for k in all_keys if cache.get(k) is None]
    if failed:
        print(f"\nFailed geocodes ({len(failed)}):")
        for f in sorted(failed):
            print(f"  - {f}")

    print("Step 2 complete.")


if __name__ == "__main__":
    main()
