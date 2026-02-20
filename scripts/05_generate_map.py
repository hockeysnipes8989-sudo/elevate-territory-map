"""Step 5: Generate the interactive Folium map with 4 toggleable layers."""
import json
import os
import sys
import pandas as pd
import folium
from folium.plugins import MarkerCluster

sys.path.insert(0, os.path.dirname(__file__))
import config


def classify_service_type(service_type):
    """Map service type to color category."""
    if pd.isna(service_type):
        return "Other"
    st = str(service_type).strip()
    if st == "PM":
        return "PM"
    elif st == "Repair":
        return "Repair"
    elif st in ("AVS ISO", "ISO"):
        return "Install"
    else:
        return "Other"


def get_choropleth_color(value, min_val, max_val, colors):
    """Map a value to a color in the gradient."""
    if max_val == min_val:
        return colors[len(colors) // 2]
    ratio = (value - min_val) / (max_val - min_val)
    idx = int(ratio * (len(colors) - 1))
    idx = min(idx, len(colors) - 1)
    return colors[idx]


def add_territory_boundaries(m, geojson_path):
    """Add territory boundary polygons as a toggleable layer."""
    with open(geojson_path, "r") as f:
        geojson = json.load(f)

    fg = folium.FeatureGroup(name="Territory Boundaries", show=True)

    for feature in geojson["features"]:
        name = feature["properties"]["name"]
        color = feature["properties"]["color"]
        appts = feature["properties"]["appointments"]
        assets = feature["properties"]["active_assets"]

        popup_html = (
            f"<b>{name}</b><br>"
            f"Service appointments: {appts}<br>"
            f"Active contract assets: {assets}"
        )

        folium.GeoJson(
            {"type": "FeatureCollection", "features": [feature]},
            style_function=lambda x, c=color: {
                "fillColor": c,
                "color": c,
                "weight": 3,
                "fillOpacity": 0.15,
            },
            tooltip=name,
            popup=folium.Popup(popup_html, max_width=250),
        ).add_to(fg)

    fg.add_to(m)
    return fg


def add_active_contracts_choropleth(m, geojson_path, territory_summary):
    """Add territory-level choropleth colored by active contract count."""
    with open(geojson_path, "r") as f:
        geojson = json.load(f)

    fg = folium.FeatureGroup(name="Active Contract Simulators", show=True)

    asset_counts = dict(zip(territory_summary["Territory"], territory_summary["total_assets"]))
    account_counts = dict(zip(territory_summary["Territory"], territory_summary["unique_accounts"]))

    min_assets = territory_summary["total_assets"].min()
    max_assets = territory_summary["total_assets"].max()

    for feature in geojson["features"]:
        name = feature["properties"]["name"]
        assets = asset_counts.get(name, 0)
        accounts = account_counts.get(name, 0)
        fill_color = get_choropleth_color(assets, min_assets, max_assets, config.CHOROPLETH_COLORS)

        popup_html = (
            f"<b>{name}</b><br>"
            f"Active assets: <b>{assets}</b><br>"
            f"Unique accounts: {accounts}"
        )

        folium.GeoJson(
            {"type": "FeatureCollection", "features": [feature]},
            style_function=lambda x, fc=fill_color: {
                "fillColor": fc,
                "color": "#333",
                "weight": 1,
                "fillOpacity": 0.55,
            },
            tooltip=f"{name}: {assets} active assets",
            popup=folium.Popup(popup_html, max_width=250),
        ).add_to(fg)

    # Add a legend
    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
         background: white; padding: 10px 14px; border-radius: 6px;
         box-shadow: 0 2px 6px rgba(0,0,0,0.3); font-size: 12px; line-height: 1.6;">
        <b>Active Contract Assets</b><br>
    """
    labels = ["Low", "", "Medium", "", "High"]
    for color, label in zip(config.CHOROPLETH_COLORS, labels):
        legend_html += f'<span style="background:{color};width:20px;height:12px;display:inline-block;margin-right:4px;border:1px solid #999;"></span>{label}<br>'
    legend_html += f"<span style='font-size:10px;color:#666;'>Range: {min_assets} – {max_assets}</span></div>"
    m.get_root().html.add_child(folium.Element(legend_html))

    fg.add_to(m)
    return fg


def add_matched_install_markers(m, install_matched):
    """Add point markers for install base accounts that matched to coordinates."""
    matched = install_matched[install_matched["matched"] & install_matched["lat"].notna()].copy()
    if matched.empty:
        return

    # Aggregate by account for cleaner display
    fg = folium.FeatureGroup(name="Active Contract Simulators", show=True)

    acct_agg = matched.groupby("Account Name").agg(
        lat=("lat", "first"),
        lon=("lon", "first"),
        asset_count=("Asset Name", "count"),
        territory=("Territory", "first"),
        products=("Product Family Short Name", lambda x: ", ".join(sorted(x.dropna().unique()))),
    ).reset_index()

    for _, row in acct_agg.iterrows():
        popup_html = (
            f"<b>{row['Account Name']}</b><br>"
            f"Territory: {row['territory']}<br>"
            f"Active assets: <b>{row['asset_count']}</b><br>"
            f"Products: {row['products']}"
        )
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=max(4, min(row["asset_count"] * 1.5, 15)),
            color="#bd0026",
            fill=True,
            fill_color="#fd8d3c",
            fill_opacity=0.7,
            weight=1.5,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{row['Account Name']}: {row['asset_count']} assets",
        ).add_to(fg)

    fg.add_to(m)


def add_service_appointments(m, appts):
    """Add clustered service appointment markers."""
    fg = folium.FeatureGroup(name="Service Appointments", show=False)
    cluster = MarkerCluster(name="Service Appointments Cluster").add_to(fg)

    appts_with_coords = appts.dropna(subset=["lat", "lon"])

    for _, row in appts_with_coords.iterrows():
        stype = classify_service_type(row.get("Service Type"))
        color = config.SERVICE_TYPE_COLORS.get(stype, "purple")

        popup_parts = [
            f"<b>{row.get('Account: Account Name', 'N/A')}</b>",
            f"Appt: {row.get('Appointment Number', '')}",
            f"Type: <b>{stype}</b>",
            f"Tech: {row.get('Service Resource: Name', 'N/A')}",
            f"Territory: {row.get('Territory', 'N/A')}",
        ]

        scheduled = row.get("Scheduled Start")
        if pd.notna(scheduled):
            popup_parts.append(f"Date: {str(scheduled)[:10]}")

        subject = row.get("Subject")
        if pd.notna(subject):
            popup_parts.append(f"Subject: {str(subject)[:80]}")

        popup_html = "<br>".join(popup_parts)

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            weight=1,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{stype}: {row.get('Account: Account Name', '')}",
        ).add_to(cluster)

    fg.add_to(m)
    return fg


def add_technician_markers(m, techs):
    """Add technician home base markers."""
    fg = folium.FeatureGroup(name="Technician Home Bases", show=True)

    techs_with_coords = techs.dropna(subset=["lat", "lon"])

    for _, row in techs_with_coords.iterrows():
        status = row.get("status", "active")
        color = config.TECH_COLORS.get(status, "blue")

        icon_map = {
            "active": "user",
            "former": "user-times",
            "special": "star",
        }
        icon_name = icon_map.get(status, "user")

        popup_html = (
            f"<b>{row['name']}</b><br>"
            f"Location: {row['location']}<br>"
            f"Status: {status.title()}<br>"
        )
        comment = row.get("comment")
        if pd.notna(comment):
            popup_html += f"Role: {comment}"

        folium.Marker(
            location=[row["lat"], row["lon"]],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{row['name']} ({status})",
            icon=folium.Icon(color=color, icon=icon_name, prefix="fa"),
        ).add_to(fg)

    fg.add_to(m)
    return fg


def add_service_type_legend(m):
    """Add a legend for service appointment colors."""
    legend_html = """
    <div style="position: fixed; bottom: 30px; right: 30px; z-index: 1000;
         background: white; padding: 10px 14px; border-radius: 6px;
         box-shadow: 0 2px 6px rgba(0,0,0,0.3); font-size: 12px; line-height: 1.6;">
        <b>Service Types</b><br>
    """
    for stype, color in config.SERVICE_TYPE_COLORS.items():
        legend_html += f'<span style="background:{color};width:12px;height:12px;display:inline-block;margin-right:4px;border-radius:50%;border:1px solid #999;"></span>{stype}<br>'

    legend_html += """<br><b>Technicians</b><br>"""
    for status, color in config.TECH_COLORS.items():
        legend_html += f'<span style="background:{color};width:12px;height:12px;display:inline-block;margin-right:4px;border-radius:50%;border:1px solid #999;"></span>{status.title()}<br>'
    legend_html += "</div>"
    m.get_root().html.add_child(folium.Element(legend_html))


def main():
    os.makedirs(config.DOCS_DIR, exist_ok=True)

    print("Loading processed data...")
    appts = pd.read_csv(config.GEOCODED_APPTS_CSV)
    techs = pd.read_csv(config.GEOCODED_TECHS_CSV)
    install_matched = pd.read_csv(config.INSTALL_MATCHED_CSV)
    territory_summary = pd.read_csv(config.TERRITORY_SUMMARY_CSV)

    print(f"  Appointments: {len(appts)} ({appts['lat'].notna().sum()} geocoded)")
    print(f"  Technicians: {len(techs)} ({techs['lat'].notna().sum()} geocoded)")
    print(f"  Install base (active): {len(install_matched)} ({install_matched['matched'].sum()} matched)")
    print(f"  Territories: {len(territory_summary)}")

    # Create base map
    m = folium.Map(
        location=config.MAP_CENTER,
        zoom_start=config.MAP_ZOOM,
        tiles=config.MAP_TILES,
        control_scale=True,
    )

    # Add title
    title_html = """
    <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
         z-index: 1000; background: white; padding: 8px 20px; border-radius: 6px;
         box-shadow: 0 2px 6px rgba(0,0,0,0.3); font-size: 16px; font-weight: bold;">
        Elevate Healthcare — Interactive Service Territory Map
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    # Layer 1: Active Contract Simulators (choropleth)
    print("Adding active contract choropleth...")
    add_active_contracts_choropleth(m, config.TERRITORIES_GEOJSON, territory_summary)

    # Add matched install base point markers to the choropleth layer
    print("Adding matched install base markers...")
    add_matched_install_markers(m, install_matched)

    # Layer 2: Service Appointments (clustered)
    print("Adding service appointments...")
    add_service_appointments(m, appts)

    # Layer 3: Technician Home Bases
    print("Adding technician markers...")
    add_technician_markers(m, techs)

    # Layer 4: Territory Boundaries
    print("Adding territory boundaries...")
    add_territory_boundaries(m, config.TERRITORIES_GEOJSON)

    # Legends
    add_service_type_legend(m)

    # Layer control
    folium.LayerControl(collapsed=False).add_to(m)

    # Save
    m.save(config.MAP_OUTPUT)
    file_size = os.path.getsize(config.MAP_OUTPUT) / (1024 * 1024)
    print(f"\nMap saved: {config.MAP_OUTPUT}")
    print(f"File size: {file_size:.1f} MB")
    print("Step 5 complete.")


if __name__ == "__main__":
    main()
