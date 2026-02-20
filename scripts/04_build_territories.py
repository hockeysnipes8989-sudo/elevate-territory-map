"""Step 4: Build territory boundary polygons via convex hull from appointment coordinates."""
import json
import os
import sys
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

sys.path.insert(0, os.path.dirname(__file__))
import config


def buffer_hull(points, buffer_deg=0.3):
    """Expand convex hull points outward by buffer_deg from centroid."""
    centroid = np.mean(points, axis=0)
    buffered = []
    for pt in points:
        direction = pt - centroid
        dist = np.linalg.norm(direction)
        if dist > 0:
            unit = direction / dist
            buffered.append(pt + unit * buffer_deg)
        else:
            buffered.append(pt + buffer_deg)
    return np.array(buffered)


def build_polygon(coords, buffer_deg):
    """Build a GeoJSON polygon from a set of lat/lon coordinates."""
    if len(coords) < 3:
        # Not enough points for a hull — create a small box
        center = np.mean(coords, axis=0)
        half = buffer_deg
        return [
            [center[1] - half, center[0] - half],
            [center[1] + half, center[0] - half],
            [center[1] + half, center[0] + half],
            [center[1] - half, center[0] + half],
            [center[1] - half, center[0] - half],
        ]

    points = np.array(coords)

    try:
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
    except Exception:
        # Degenerate case — all points collinear
        hull_points = points

    buffered = buffer_hull(hull_points, buffer_deg)

    # Re-compute hull on buffered points
    try:
        hull2 = ConvexHull(buffered)
        final_points = buffered[hull2.vertices]
    except Exception:
        final_points = buffered

    # Convert to GeoJSON [lon, lat] and close the ring
    ring = [[float(p[1]), float(p[0])] for p in final_points]
    ring.append(ring[0])
    return ring


def main():
    appts = pd.read_csv(config.GEOCODED_APPTS_CSV)
    appts = appts.dropna(subset=["lat", "lon", "Territory"])

    print(f"Appointments with coordinates: {len(appts)}")
    print(f"Territories: {appts['Territory'].nunique()}")

    # Load territory summary for asset counts
    territory_summary = pd.read_csv(config.TERRITORY_SUMMARY_CSV)
    asset_counts = dict(zip(territory_summary["Territory"], territory_summary["total_assets"]))

    features = []
    for territory, group in appts.groupby("Territory"):
        coords = list(zip(group["lat"], group["lon"]))
        ring = build_polygon(coords, config.TERRITORY_BUFFER_DEG)

        color = config.TERRITORY_COLORS.get(territory, "#999999")
        assets = asset_counts.get(territory, 0)

        feature = {
            "type": "Feature",
            "properties": {
                "name": territory,
                "color": color,
                "appointments": len(group),
                "active_assets": assets,
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [ring],
            },
        }
        features.append(feature)
        print(f"  {territory}: {len(coords)} points → {len(ring)-1} hull vertices, {assets} active assets")

    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    with open(config.TERRITORIES_GEOJSON, "w") as f:
        json.dump(geojson, f, indent=2)

    print(f"\nSaved: {config.TERRITORIES_GEOJSON}")
    print(f"Total territories: {len(features)}")
    print("Step 4 complete.")


if __name__ == "__main__":
    main()
