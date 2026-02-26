"""Step 5: Generate the interactive Folium map and simulation UI."""
import json
import os
import sys
import pandas as pd
import folium

sys.path.insert(0, os.path.dirname(__file__))
import config


SIM_SUMMARY_ENHANCED = "scenario_summary_enhanced.csv"
SIM_SUMMARY = "scenario_summary.csv"
SIM_PLACEMENTS = "scenario_placements.csv"
SIM_CANDIDATES = "candidate_bases.csv"


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


def exclude_inactive_technicians(techs):
    """Remove inactive/former technicians from display/output layers."""
    filtered = techs.copy()
    if "status" in filtered.columns:
        filtered = filtered[filtered["status"].astype(str).str.lower() != "former"]
    inactive_names = set(getattr(config, "INACTIVE_TECH_NAMES", set()))
    if inactive_names and "name" in filtered.columns:
        filtered = filtered[~filtered["name"].isin(inactive_names)]
    if "comment" in filtered.columns:
        filtered = filtered[
            ~filtered["comment"].astype(str).str.contains("no longer with elevate", case=False, na=False)
        ]
    return filtered.copy()


def validate_current_tech_headcount(techs):
    """Warn if current technician roster count differs from configured expectation."""
    expected = getattr(config, "EXPECTED_CURRENT_TECH_COUNT", None)
    if expected is None:
        return
    actual = int(len(techs))
    if actual == int(expected):
        return
    names = sorted(techs["name"].dropna().astype(str).tolist()) if "name" in techs.columns else []
    print(
        f"WARNING: Expected {int(expected)} current technicians but found {actual} in technicians dataset."
    )
    if names:
        print("  Technician names loaded:")
        for name in names:
            print(f"    - {name}")


def get_choropleth_color(value, min_val, max_val, colors):
    """Map a value to a color in the gradient."""
    if max_val == min_val:
        return colors[len(colors) // 2]
    ratio = (value - min_val) / (max_val - min_val)
    idx = int(ratio * (len(colors) - 1))
    idx = min(idx, len(colors) - 1)
    return colors[idx]


def add_territory_boundaries(m, geojson_path, layer_name):
    """Add territory boundary polygons as a toggleable layer."""
    with open(geojson_path, "r") as f:
        geojson = json.load(f)

    fg = folium.FeatureGroup(name=layer_name, show=True)

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
                "fillOpacity": 0.10,
            },
            tooltip=name,
            popup=folium.Popup(popup_html, max_width=250),
        ).add_to(fg)

        # Outlier locations (Alaska, USVI, etc.) are rendered as points, not polygon vertices.
        for marker in feature["properties"].get("outlier_markers", []):
            lat = marker.get("lat")
            lon = marker.get("lon")
            label = marker.get("label", f"{name} outlier")
            if lat is None or lon is None:
                continue
            folium.CircleMarker(
                location=[lat, lon],
                radius=4,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.9,
                weight=1,
                tooltip=f"{name}: {label}",
            ).add_to(fg)

    fg.add_to(m)
    return fg


def add_active_contracts_choropleth(m, geojson_path, territory_summary, layer_name):
    """Add territory-level choropleth colored by active contract count."""
    with open(geojson_path, "r") as f:
        geojson = json.load(f)

    fg = folium.FeatureGroup(name=layer_name, show=True)

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
                "fillOpacity": 0.35,
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


def add_matched_install_markers(m, install_matched, fg=None, layer_name="Active Contract Simulators"):
    """Add point markers for install base accounts that matched to coordinates."""
    matched = install_matched[install_matched["matched"] & install_matched["lat"].notna()].copy()
    if matched.empty:
        return fg

    # Aggregate by account for cleaner display
    created_layer = False
    if fg is None:
        fg = folium.FeatureGroup(name=layer_name, show=True)
        created_layer = True

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

    if created_layer:
        fg.add_to(m)
    return fg


def add_nonactive_install_markers(m, install_all, layer_name):
    """Add point markers for non-active-contract assets with matched coordinates."""
    non_active = install_all[
        (~install_all["has_active_contract"]) & install_all["matched"] & install_all["lat"].notna()
    ].copy()
    fg = folium.FeatureGroup(name=layer_name, show=True)
    if non_active.empty:
        fg.add_to(m)
        return fg

    acct_agg = (
        non_active.groupby("Account Name")
        .agg(
            lat=("lat", "first"),
            lon=("lon", "first"),
            asset_count=("Asset Name", "count"),
            territory=("Territory", "first"),
            statuses=(
                "Contract_Status_Clean",
                lambda x: "; ".join(
                    [f"{k}: {v}" for k, v in x.fillna("Unknown").value_counts().to_dict().items()]
                ),
            ),
            products=("Product Family Short Name", lambda x: ", ".join(sorted(x.dropna().unique()))),
        )
        .reset_index()
    )

    for _, row in acct_agg.iterrows():
        popup_html = (
            f"<b>{row['Account Name']}</b><br>"
            f"Territory: {row['territory']}<br>"
            f"Non-active assets: <b>{row['asset_count']}</b><br>"
            f"Contract status mix: {row['statuses']}<br>"
            f"Products: {row['products']}"
        )
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=max(3, min(row["asset_count"] * 1.2, 12)),
            color="#5f6368",
            fill=True,
            fill_color="#9aa0a6",
            fill_opacity=0.65,
            weight=1,
            popup=folium.Popup(popup_html, max_width=330),
            tooltip=f"{row['Account Name']}: {row['asset_count']} non-active assets",
        ).add_to(fg)

    fg.add_to(m)
    return fg


def add_service_appointments(m, appts, layer_name):
    """Add service appointment markers (no clustering)."""
    fg = folium.FeatureGroup(name=layer_name, show=True)

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
            radius=3,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            weight=0.5,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{stype}: {row.get('Account: Account Name', '')}",
        ).add_to(fg)

    print(f"Service appointment markers added: {len(appts_with_coords)}")

    fg.add_to(m)
    return fg


def add_technician_markers(m, techs, layer_name):
    """Add technician home base markers."""
    fg = folium.FeatureGroup(name=layer_name, show=True)

    techs_with_coords = exclude_inactive_technicians(techs).dropna(subset=["lat", "lon"])
    if techs_with_coords.empty:
        fg.add_to(m)
        return fg

    techs_with_coords = techs_with_coords.copy()
    techs_with_coords["status"] = (
        techs_with_coords["status"].fillna("active").astype(str).str.lower()
    )
    techs_with_coords["location"] = techs_with_coords["location"].fillna("").astype(str)
    techs_with_coords["coord_key"] = techs_with_coords.apply(
        lambda r: f"{float(r['lat']):.6f}|{float(r['lon']):.6f}",
        axis=1,
    )

    grouped = techs_with_coords.groupby("coord_key", sort=False)
    for _, group in grouped:
        group = group.sort_values(["status", "name"])
        lat = float(group.iloc[0]["lat"])
        lon = float(group.iloc[0]["lon"])
        location = str(group.iloc[0].get("location", "")).strip() or "Unknown"
        num_techs = int(len(group))

        if num_techs == 1:
            row = group.iloc[0]
            status = str(row.get("status", "active")).lower()
            color = config.TECH_COLORS.get(status, "blue")
            icon_name = "star" if status == "special" else "user"
            tooltip = f"{row['name']} ({status})"
        else:
            status_counts = group["status"].value_counts().to_dict()
            if len(status_counts) == 1:
                only_status = next(iter(status_counts.keys()))
                color = config.TECH_COLORS.get(only_status, "blue")
            else:
                color = "blue"
            icon_name = "users"
            tooltip = f"{location} ({num_techs} techs)"

        roster_lines = []
        for _, r in group.iterrows():
            name = str(r.get("name", "")).strip()
            status = str(r.get("status", "active")).strip().title()
            comment_raw = r.get("comment", "")
            comment = "" if pd.isna(comment_raw) else str(comment_raw).strip()
            if comment.lower() == "nan":
                comment = ""
            extra = f" - {comment}" if comment else ""
            roster_lines.append(f"<li><b>{name}</b> ({status}){extra}</li>")
        roster_html = "".join(roster_lines)
        popup_html = (
            f"<b>{location}</b><br>"
            f"Technicians at this base: <b>{num_techs}</b><br>"
            "<div style='margin-top:6px;'>"
            "<b>Roster</b>"
            f"<ul style='margin:4px 0 0 16px; padding:0;'>{roster_html}</ul>"
            "</div>"
        )

        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=360),
            tooltip=tooltip,
            icon=folium.Icon(color=color, icon=icon_name, prefix="fa"),
        ).add_to(fg)

    fg.add_to(m)
    return fg


def add_airport_layer(m):
    """Add major airport hub markers as a toggleable layer."""
    fg = folium.FeatureGroup(name=f"Major Airport Hubs ({len(config.MAJOR_AIRPORTS)})", show=True)

    for airport in config.MAJOR_AIRPORTS:
        popup_html = (
            f"<b>{airport['code']} — {airport['name']}</b><br>"
            f"{airport['city']}<br>"
            f"<span style='color:#666;font-size:11px;'>Major service hub</span>"
        )
        folium.Marker(
            location=[airport["lat"], airport["lon"]],
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"{airport['code']}: {airport['name']}",
            icon=folium.Icon(color="darkblue", icon="plane", prefix="fa"),
        ).add_to(fg)

    fg.add_to(m)
    return fg


def add_hub_radius_circles(m):
    """Add ~300-mile (~5hr drive) radius circles around key dispatch hubs."""
    RADIUS_METERS = 483_000  # ~300 miles / 5-hour drive threshold

    KEY_HUBS = [
        "ATL", "ORD", "DFW", "DEN", "LAX", "JFK", "SFO", "SEA",
        "MIA", "BOS", "PHX", "IAH", "MSP", "CLT", "SLC",
    ]

    hub_airports = [a for a in config.MAJOR_AIRPORTS if a["code"] in KEY_HUBS]
    missing_hubs = sorted(set(KEY_HUBS) - {a["code"] for a in hub_airports})
    if missing_hubs:
        raise ValueError(f"Missing KEY_HUBS in MAJOR_AIRPORTS: {', '.join(missing_hubs)}")

    fg = folium.FeatureGroup(name="Hub Dispatch Radius (~300mi / 5hr drive)", show=False)

    for airport in hub_airports:
        folium.Circle(
            location=[airport["lat"], airport["lon"]],
            radius=RADIUS_METERS,
            color="#1a6faf",
            weight=1.5,
            fill=True,
            fill_color="#1a6faf",
            fill_opacity=0.04,
            tooltip=f"{airport['code']}: ~300mi / 5hr drive radius",
        ).add_to(fg)

    fg.add_to(m)
    return fg


def add_service_type_legend(m, service_type_counts):
    """Add a legend for service appointment colors."""
    legend_html = """
    <div style="position: fixed; bottom: 30px; right: 30px; z-index: 1000;
         background: white; padding: 10px 14px; border-radius: 6px;
         box-shadow: 0 2px 6px rgba(0,0,0,0.3); font-size: 12px; line-height: 1.6;">
        <b>Service Types</b><br>
    """
    for stype in ["PM", "Repair", "Install", "Other"]:
        color = config.SERVICE_TYPE_COLORS.get(stype, "purple")
        count = int(service_type_counts.get(stype, 0))
        legend_html += (
            f'<span style="background:{color};width:12px;height:12px;display:inline-block;'
            f'margin-right:4px;border-radius:50%;border:1px solid #999;"></span>{stype} ({count})<br>'
        )

    legend_html += """<br><b>Technicians</b><br>"""
    for status, color in config.TECH_COLORS.items():
        if status == "former":
            continue
        legend_html += f'<span style="background:{color};width:12px;height:12px;display:inline-block;margin-right:4px;border-radius:50%;border:1px solid #999;"></span>{status.title()}<br>'
    legend_html += (
        "<div style='margin-top:6px;color:#666;font-size:10px;'>"
        "Tech markers may represent multiple technicians at the same base location."
        "</div>"
    )
    legend_html += "</div>"
    m.get_root().html.add_child(folium.Element(legend_html))


def load_simulation_data():
    """Load scenario summary + placements for map simulation controls."""
    sim_dir = config.OPTIMIZATION_DIR
    summary_enhanced_path = os.path.join(sim_dir, SIM_SUMMARY_ENHANCED)
    summary_path = os.path.join(sim_dir, SIM_SUMMARY)
    placements_path = os.path.join(sim_dir, SIM_PLACEMENTS)
    candidates_path = os.path.join(sim_dir, SIM_CANDIDATES)

    if not os.path.exists(placements_path):
        return None
    if os.path.exists(summary_enhanced_path):
        summary_df = pd.read_csv(summary_enhanced_path)
    elif os.path.exists(summary_path):
        summary_df = pd.read_csv(summary_path)
    else:
        return None
    if not os.path.exists(candidates_path):
        return None

    placements_df = pd.read_csv(placements_path)
    candidates_df = pd.read_csv(candidates_path)

    if summary_df.empty or "scenario_hires" not in summary_df.columns:
        return None
    if placements_df.empty:
        placements_df = pd.DataFrame(
            columns=[
                "scenario_hires",
                "candidate_id",
                "city",
                "state",
                "airport_iata",
                "hires_allocated",
                "assigned_appointments",
                "assigned_hours",
            ]
        )
    if "candidate_id" not in candidates_df.columns:
        return None

    summary_df["scenario_hires"] = pd.to_numeric(summary_df["scenario_hires"], errors="coerce")
    summary_df = summary_df.dropna(subset=["scenario_hires"]).copy()
    summary_df["scenario_hires"] = summary_df["scenario_hires"].astype(int)

    # Build baseline deltas if enhanced columns are missing.
    if "economic_total_with_overhead_usd" not in summary_df.columns:
        if "modeled_total_cost_usd" in summary_df.columns:
            summary_df["economic_total_with_overhead_usd"] = pd.to_numeric(
                summary_df["modeled_total_cost_usd"], errors="coerce"
            ).fillna(0.0)
        else:
            summary_df["economic_total_with_overhead_usd"] = 0.0
    if "savings_vs_n0_usd" not in summary_df.columns:
        base = float(
            summary_df.loc[summary_df["scenario_hires"] == 0, "economic_total_with_overhead_usd"]
            .head(1)
            .squeeze()
        ) if (summary_df["scenario_hires"] == 0).any() else float(
            summary_df["economic_total_with_overhead_usd"].max()
        )
        summary_df["savings_vs_n0_usd"] = base - summary_df["economic_total_with_overhead_usd"]
    if "savings_vs_n0_pct" not in summary_df.columns:
        base = float(
            summary_df.loc[summary_df["scenario_hires"] == 0, "economic_total_with_overhead_usd"]
            .head(1)
            .squeeze()
        ) if (summary_df["scenario_hires"] == 0).any() else float(
            summary_df["economic_total_with_overhead_usd"].max()
        )
        summary_df["savings_vs_n0_pct"] = (
            (summary_df["savings_vs_n0_usd"] / base) * 100.0 if base > 0 else 0.0
        )
    if "marginal_savings_from_prev_usd" not in summary_df.columns:
        summary_df = summary_df.sort_values("scenario_hires")
        summary_df["marginal_savings_from_prev_usd"] = (
            summary_df["economic_total_with_overhead_usd"].shift(1)
            - summary_df["economic_total_with_overhead_usd"]
        ).fillna(0.0)

    if "unmet_appointments" not in summary_df.columns:
        summary_df["unmet_appointments"] = 0.0
    if "hire_cost_usd" not in summary_df.columns:
        summary_df["hire_cost_usd"] = 0.0
    if "mean_existing_utilization" not in summary_df.columns:
        summary_df["mean_existing_utilization"] = 0.0
    if "max_existing_utilization" not in summary_df.columns:
        summary_df["max_existing_utilization"] = 0.0

    placements_df["scenario_hires"] = pd.to_numeric(
        placements_df.get("scenario_hires"), errors="coerce"
    )
    placements_df = placements_df.dropna(subset=["scenario_hires"]).copy()
    placements_df["scenario_hires"] = placements_df["scenario_hires"].astype(int)

    # Attach coordinates from candidate pool.
    placements_df = placements_df.merge(
        candidates_df[["candidate_id", "lat", "lon"]],
        on="candidate_id",
        how="left",
    )
    placements_df = placements_df.dropna(subset=["lat", "lon"]).copy()

    # Keep only configured scenario range.
    min_s = getattr(config, "SIM_SCENARIO_MIN", 0)
    max_s = getattr(config, "SIM_SCENARIO_MAX", 4)
    summary_df = summary_df[
        (summary_df["scenario_hires"] >= min_s) & (summary_df["scenario_hires"] <= max_s)
    ].copy()
    placements_df = placements_df[
        (placements_df["scenario_hires"] >= min_s) & (placements_df["scenario_hires"] <= max_s)
    ].copy()
    if summary_df.empty:
        return None

    # Skip infeasible scenarios (solver_status == -1) so NaN/None cost
    # fields don't break json.dumps or the simulation panel UI.
    if "solver_status" in summary_df.columns:
        infeasible_mask = pd.to_numeric(summary_df["solver_status"], errors="coerce") == -1
        n_infeasible = int(infeasible_mask.sum())
        if n_infeasible:
            print(f"  Skipping {n_infeasible} infeasible scenario(s) from simulation panel.")
            summary_df = summary_df[~infeasible_mask].copy()
    if summary_df.empty:
        return None

    payload = {}
    for _, row in summary_df.sort_values("scenario_hires").iterrows():
        scenario = int(row["scenario_hires"])
        subset = (
            placements_df[placements_df["scenario_hires"] == scenario]
            .copy()
            .sort_values(["hires_allocated", "assigned_hours"], ascending=[False, False])
        )
        placement_records = []
        for _, p in subset.iterrows():
            placement_records.append(
                {
                    "candidate_id": p.get("candidate_id"),
                    "city": p.get("city"),
                    "state": p.get("state"),
                    "airport_iata": p.get("airport_iata"),
                    "hires_allocated": float(p.get("hires_allocated", 0)),
                    "assigned_appointments": float(p.get("assigned_appointments", 0)),
                    "assigned_hours": float(p.get("assigned_hours", 0)),
                    "lat": float(p.get("lat")),
                    "lon": float(p.get("lon")),
                }
            )
        payload[str(scenario)] = {
            "scenario_hires": scenario,
            "kpis": {
                "economic_total_with_overhead_usd": float(
                    row.get("economic_total_with_overhead_usd", 0)
                ),
                "savings_vs_n0_usd": float(row.get("savings_vs_n0_usd", 0)),
                "savings_vs_n0_pct": float(row.get("savings_vs_n0_pct", 0)),
                "marginal_savings_from_prev_usd": float(
                    row.get("marginal_savings_from_prev_usd", 0)
                ),
                "unmet_appointments": float(row.get("unmet_appointments", 0)),
                "hire_cost_usd": float(row.get("hire_cost_usd", 0)),
                "mean_existing_utilization": float(row.get("mean_existing_utilization", 0)),
                "max_existing_utilization": float(row.get("max_existing_utilization", 0)),
            },
            "placements": placement_records,
        }
    return payload


def add_simulation_layers(m, simulation_payload):
    """Add one marker layer per simulation scenario and return layer JS names."""
    if not simulation_payload:
        return {}

    scenario_layers = {}
    scenario_colors = ["#4a4e69", "#2a9d8f", "#f4a261", "#e76f51", "#c1121f"]
    ordered_keys = sorted(simulation_payload.keys(), key=lambda x: int(x))
    default_key = "0" if "0" in simulation_payload else ordered_keys[0]

    for key in ordered_keys:
        scenario = int(key)
        color = scenario_colors[min(scenario, len(scenario_colors) - 1)]
        fg = folium.FeatureGroup(
            name=f"Simulation Scenario N={scenario}",
            show=(key == default_key),
            control=False,
        )

        placements = simulation_payload[key]["placements"]
        for p in placements:
            hires = max(float(p["hires_allocated"]), 1.0)
            diameter = int(max(24, min(44, 20 + 6 * hires)))
            font_size = int(max(13, min(24, 11 + 3 * hires)))
            marker_html = (
                "<div style=\""
                f"width:{diameter}px;height:{diameter}px;border-radius:50%;"
                f"background:{color};border:2px solid #fff;color:#fff;"
                f"display:flex;align-items:center;justify-content:center;"
                f"font-size:{font_size}px;font-weight:700;"
                "box-shadow:0 3px 10px rgba(0,0,0,0.35);"
                "\">&#9733;</div>"
            )
            popup_html = (
                f"<b>Scenario N={scenario}</b><br>"
                f"<b>{p['city']}, {p['state']}</b> ({p['airport_iata']})<br>"
                f"Hires allocated: <b>{int(round(p['hires_allocated']))}</b><br>"
                f"Assigned appointments: {p['assigned_appointments']:.1f}<br>"
                f"Assigned hours: {p['assigned_hours']:.1f}"
            )
            folium.Marker(
                location=[p["lat"], p["lon"]],
                icon=folium.DivIcon(html=marker_html),
                tooltip=f"Scenario N={scenario}: {p['city']}, {p['state']}",
                popup=folium.Popup(popup_html, max_width=280),
            ).add_to(fg)

        fg.add_to(m)
        scenario_layers[key] = fg.get_name()
    return scenario_layers


def add_simulation_panel(m, simulation_payload, scenario_layer_names):
    """Inject scenario controls and KPI cards into the map page."""
    if not simulation_payload or not scenario_layer_names:
        return

    map_var = m.get_name()
    ordered_keys = sorted(simulation_payload.keys(), key=lambda x: int(x))
    default_key = "0" if "0" in simulation_payload else ordered_keys[0]

    # Build JS object with layer variable names and resolve them at runtime.
    layer_entries = []
    for key in ordered_keys:
        layer_entries.append(f'"{key}": "{scenario_layer_names[key]}"')
    layer_js = "{\n" + ",\n".join(layer_entries) + "\n}"
    payload_js = json.dumps(simulation_payload)

    panel_html = """
    <style>
      #sim-panel {
        position: fixed;
        top: 72px;
        left: 12px;
        z-index: 1200;
        width: 320px;
        max-height: calc(100vh - 110px);
        overflow-y: auto;
        background: rgba(255, 255, 255, 0.96);
        border: 1px solid #cfcfcf;
        border-radius: 10px;
        box-shadow: 0 3px 12px rgba(0,0,0,0.22);
        padding: 12px;
        font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
      }
      #sim-panel h3 {
        margin: 0 0 8px 0;
        font-size: 14px;
      }
      #sim-subtitle {
        margin: 0 0 10px 0;
        color: #4d4d4d;
        font-size: 11px;
      }
      #sim-buttons {
        display: flex;
        gap: 6px;
        flex-wrap: wrap;
        margin-bottom: 10px;
      }
      .sim-btn {
        border: 1px solid #9aa0a6;
        background: #fff;
        color: #222;
        padding: 4px 8px;
        border-radius: 14px;
        font-size: 12px;
        cursor: pointer;
      }
      .sim-btn.active {
        background: #163b59;
        border-color: #163b59;
        color: #fff;
      }
      #sim-kpis {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 6px;
        margin-bottom: 10px;
      }
      .sim-kpi {
        background: #f7f8fa;
        border: 1px solid #e2e5e9;
        border-radius: 8px;
        padding: 6px 8px;
      }
      .sim-kpi .label {
        font-size: 10px;
        color: #5f6368;
      }
      .sim-kpi .value {
        margin-top: 3px;
        font-weight: 700;
        font-size: 12px;
      }
      #sim-recs-title {
        margin: 0 0 6px 0;
        font-size: 12px;
        font-weight: 700;
      }
      #sim-recs {
        border: 1px solid #e2e5e9;
        border-radius: 8px;
        background: #fbfbfc;
        padding: 6px 8px;
        font-size: 11px;
      }
      .sim-rec-row {
        margin: 0 0 5px 0;
        padding-bottom: 5px;
        border-bottom: 1px dashed #e1e4e8;
      }
      .sim-rec-row:last-child {
        margin-bottom: 0;
        padding-bottom: 0;
        border-bottom: none;
      }
      #sim-footnote {
        margin-top: 8px;
        font-size: 10px;
        color: #666;
      }
      #sim-panel-toggle {
        display: none;
        position: fixed;
        top: 72px;
        left: 12px;
        z-index: 1250;
        border: 1px solid #9aa0a6;
        border-radius: 14px;
        background: #fff;
        padding: 4px 10px;
        font-size: 12px;
      }
      @media (max-width: 900px) {
        #sim-panel { display: none; width: 280px; }
        #sim-panel.mobile-open { display: block; }
        #sim-panel-toggle { display: block; }
      }
    </style>
    <button id="sim-panel-toggle" title="Show/hide simulation panel">Simulation</button>
    <div id="sim-panel">
      <h3>Simulation Scenarios</h3>
      <p id="sim-subtitle">Cost-first optimization with fixed current tech bases.</p>
      <div id="sim-buttons"></div>
      <div id="sim-kpis">
        <div class="sim-kpi"><div class="label">Total Cost</div><div class="value" id="kpi-total">-</div></div>
        <div class="sim-kpi"><div class="label">Cost Change vs N=0</div><div class="value" id="kpi-savings">-</div></div>
        <div class="sim-kpi"><div class="label">Marginal Cost Change</div><div class="value" id="kpi-marginal">-</div></div>
        <div class="sim-kpi" id="kpi-unmet-card"><div class="label">Unmet Appointments</div><div class="value" id="kpi-unmet">-</div></div>
        <div class="sim-kpi"><div class="label">Annual Hire Payroll</div><div class="value" id="kpi-hire-cost">-</div></div>
        <div class="sim-kpi"><div class="label">Mean Utilization</div><div class="value" id="kpi-mean-util">-</div></div>
        <div class="sim-kpi"><div class="label">Max Utilization</div><div class="value" id="kpi-max-util">-</div></div>
      </div>
      <div id="sim-recs-title">Recommended Bases</div>
      <div id="sim-recs">No recommendations.</div>
      <div id="sim-footnote">Shows N=0..4 scenario outputs from optimization pipeline.</div>
    </div>
    """

    script_js = f"""
    (function() {{
      const mapVarName = "{map_var}";
      const scenarioData = {payload_js};
      const scenarioLayerNames = {layer_js};
      const orderedScenarios = {json.dumps(ordered_keys)};
      const defaultScenario = "{default_key}";
      const showUnmetKpi = orderedScenarios.some((s) =>
        Number(((scenarioData[s] || {{}}).kpis || {{}}).unmet_appointments || 0) > 0
      );
      let mapRef = null;
      let scenarioLayers = {{}};

      function money(v) {{
        const n = Number(v || 0);
        return "$" + n.toLocaleString(undefined, {{maximumFractionDigits: 0}});
      }}
      function pct(v) {{
        const n = Number(v || 0);
        return n.toFixed(2) + "%";
      }}

      function renderButtons() {{
        const container = document.getElementById("sim-buttons");
        container.innerHTML = "";
        orderedScenarios.forEach((s) => {{
          const b = document.createElement("button");
          b.className = "sim-btn";
          b.textContent = "N=" + s;
          b.setAttribute("data-scenario", s);
          b.onclick = () => showScenario(s);
          container.appendChild(b);
        }});
      }}

      function setActiveButton(scenario) {{
        document.querySelectorAll(".sim-btn").forEach((b) => {{
          b.classList.toggle("active", b.getAttribute("data-scenario") === scenario);
        }});
      }}

      function renderKpis(scenario) {{
        const item = scenarioData[scenario];
        if (!item) return;
        const k = item.kpis || {{}};
        document.getElementById("kpi-total").textContent = money(k.economic_total_with_overhead_usd);
        document.getElementById("kpi-savings").textContent = money(k.savings_vs_n0_usd) + " (" + pct(k.savings_vs_n0_pct) + ")";
        document.getElementById("kpi-marginal").textContent = money(k.marginal_savings_from_prev_usd);
        if (showUnmetKpi) {{
          document.getElementById("kpi-unmet").textContent = Number(k.unmet_appointments || 0).toFixed(1);
        }}
        document.getElementById("kpi-hire-cost").textContent = money(k.hire_cost_usd);
        document.getElementById("kpi-mean-util").textContent = Number(k.mean_existing_utilization || 0).toFixed(3);
        document.getElementById("kpi-max-util").textContent = Number(k.max_existing_utilization || 0).toFixed(3);
      }}

      function renderRecommendations(scenario) {{
        const recWrap = document.getElementById("sim-recs");
        const item = scenarioData[scenario];
        const recs = item ? (item.placements || []) : [];
        if (!recs.length) {{
          recWrap.innerHTML = "No new-hire placements in this scenario.";
          return;
        }}
        recWrap.innerHTML = recs.map((r) => {{
          return `
            <div class="sim-rec-row">
              <div><b>${{r.city}}, ${{r.state}}</b> (${{r.airport_iata}})</div>
              <div>Hires: <b>${{Math.round(r.hires_allocated || 0)}}</b></div>
              <div>Hours: ${{Number(r.assigned_hours || 0).toFixed(1)}}</div>
              <div>Appointments: ${{Number(r.assigned_appointments || 0).toFixed(1)}}</div>
            </div>`;
        }}).join("");
      }}

      function showScenario(scenario) {{
        if (!mapRef) return;
        orderedScenarios.forEach((s) => {{
          const layer = scenarioLayers[s];
          if (!layer) return;
          if (mapRef.hasLayer(layer)) {{
            mapRef.removeLayer(layer);
          }}
        }});
        const target = scenarioLayers[scenario];
        if (target && !mapRef.hasLayer(target)) {{
          mapRef.addLayer(target);
        }}
        setActiveButton(scenario);
        renderKpis(scenario);
        renderRecommendations(scenario);
      }}

      function resolveLayers() {{
        mapRef = window[mapVarName] || null;
        if (!mapRef) {{
          return orderedScenarios.length + 1;
        }}
        const resolved = {{}};
        let missing = 0;
        orderedScenarios.forEach((s) => {{
          const layerVar = scenarioLayerNames[s];
          const layerObj = layerVar ? window[layerVar] : null;
          if (layerObj) {{
            resolved[s] = layerObj;
          }} else {{
            missing += 1;
          }}
        }});
        scenarioLayers = resolved;
        return missing;
      }}

      function wireMobileToggle() {{
        const btn = document.getElementById("sim-panel-toggle");
        const panel = document.getElementById("sim-panel");
        if (!btn || !panel) return;
        btn.addEventListener("click", () => {{
          panel.classList.toggle("mobile-open");
        }});
      }}

      function initWhenReady(remainingAttempts) {{
        const missing = resolveLayers();
        if (missing > 0 && remainingAttempts > 0) {{
          window.setTimeout(() => initWhenReady(remainingAttempts - 1), 50);
          return;
        }}
        if (missing > 0) {{
          const recWrap = document.getElementById("sim-recs");
          if (recWrap) {{
            recWrap.innerHTML = "Simulation layers unavailable in this map build.";
          }}
          return;
        }}
        renderButtons();
        const unmetCard = document.getElementById("kpi-unmet-card");
        if (unmetCard && !showUnmetKpi) {{
          unmetCard.style.display = "none";
        }}
        wireMobileToggle();
        showScenario(defaultScenario);
      }}

      initWhenReady(80);
    }})();
    """

    m.get_root().html.add_child(folium.Element(panel_html))
    m.get_root().script.add_child(folium.Element(script_js))


def add_simulation_unavailable_notice(m):
    """Show small notice when simulation outputs are unavailable."""
    notice_html = """
    <div style="position: fixed; top: 72px; left: 12px; z-index: 1200;
         background: rgba(255,255,255,0.95); border: 1px solid #d0d0d0;
         border-radius: 8px; padding: 8px 10px; font-size: 11px; color: #444;
         box-shadow: 0 2px 8px rgba(0,0,0,0.2);">
      Simulation panel unavailable.<br>
      Run scripts 06-09 to generate optimization outputs.
    </div>
    """
    m.get_root().html.add_child(folium.Element(notice_html))


def main():
    os.makedirs(config.DOCS_DIR, exist_ok=True)

    print("Loading processed data...")
    appts = pd.read_csv(config.GEOCODED_APPTS_CSV)
    techs = pd.read_csv(config.GEOCODED_TECHS_CSV)
    raw_tech_count = len(techs)
    techs = exclude_inactive_technicians(techs)
    validate_current_tech_headcount(techs)
    filtered_out = raw_tech_count - len(techs)
    if filtered_out:
        print(f"  Filtered inactive technicians from map layer: {filtered_out}")
    install_matched = pd.read_csv(config.INSTALL_MATCHED_CSV)
    if os.path.exists(config.INSTALL_ALL_MATCHED_CSV):
        install_all_matched = pd.read_csv(config.INSTALL_ALL_MATCHED_CSV)
    else:
        install_all_matched = install_matched.copy()
        install_all_matched["has_active_contract"] = True
    territory_summary = pd.read_csv(config.TERRITORY_SUMMARY_CSV)

    if install_all_matched["has_active_contract"].dtype != bool:
        install_all_matched["has_active_contract"] = (
            install_all_matched["has_active_contract"]
            .astype(str)
            .str.lower()
            .isin({"true", "1", "yes"})
        )
    non_active_assets_count = int((~install_all_matched["has_active_contract"]).sum())

    print(f"  Appointments: {len(appts)} ({appts['lat'].notna().sum()} geocoded)")
    print(f"  Technicians: {len(techs)} ({techs['lat'].notna().sum()} geocoded)")
    print(f"  Install base (active): {len(install_matched)} ({install_matched['matched'].sum()} matched)")
    print(
        f"  Install base (non-active): {non_active_assets_count} "
        f"({int((~install_all_matched['has_active_contract'] & install_all_matched['matched']).sum())} matched)"
    )
    print(f"  Territories: {len(territory_summary)}")

    active_assets_count = int(len(install_matched))
    appointments_count = int(len(appts))
    technicians_count = int(len(techs))
    territories_count = int(len(territory_summary))
    layer_active_name = f"Active Contract Simulators ({active_assets_count:,} assets)"
    layer_appt_name = f"Service Appointments ({appointments_count:,})"
    layer_tech_name = f"Technician Home Bases ({technicians_count:,})"
    layer_territory_name = f"Territory Boundaries ({territories_count:,})"
    layer_nonactive_name = f"Non-Active/No Contract Assets ({non_active_assets_count:,} assets)"
    service_type_counts = appts["Service Type"].apply(classify_service_type).value_counts()

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
    active_layer = add_active_contracts_choropleth(
        m,
        config.TERRITORIES_GEOJSON,
        territory_summary,
        layer_name=layer_active_name,
    )

    # Add matched install base point markers to the choropleth layer
    print("Adding matched install base markers...")
    add_matched_install_markers(
        m,
        install_matched,
        fg=active_layer,
        layer_name=layer_active_name,
    )

    # Layer 2: Non-active install base assets (matched to coordinates)
    print("Adding non-active install markers...")
    add_nonactive_install_markers(
        m,
        install_all=install_all_matched,
        layer_name=layer_nonactive_name,
    )

    # Layer 3: Service Appointments
    print("Adding service appointments...")
    add_service_appointments(m, appts, layer_name=layer_appt_name)

    # Layer 4: Technician Home Bases
    print("Adding technician markers...")
    add_technician_markers(m, techs, layer_name=layer_tech_name)

    # Layer 5: Territory Boundaries
    print("Adding territory boundaries...")
    add_territory_boundaries(m, config.TERRITORIES_GEOJSON, layer_name=layer_territory_name)

    # Layer 6: Airport Hubs
    print("Adding airport hubs...")
    add_airport_layer(m)

    # Layer 7: Hub Dispatch Radius Circles
    print("Adding hub dispatch radius circles (300mi / 5hr drive)...")
    add_hub_radius_circles(m)

    # Legends
    add_service_type_legend(m, service_type_counts=service_type_counts)

    # Simulation scenario UI/layers
    if getattr(config, "ENABLE_SIMULATION_UI", False):
        print("Adding simulation scenario panel...")
        simulation_payload = load_simulation_data()
        if simulation_payload:
            scenario_layer_names = add_simulation_layers(m, simulation_payload)
            add_simulation_panel(m, simulation_payload, scenario_layer_names)
            print(f"  Loaded scenarios: {', '.join(sorted(simulation_payload.keys(), key=int))}")
        else:
            add_simulation_unavailable_notice(m)
            print("  Simulation outputs not found; panel disabled.")

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
