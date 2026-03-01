"""Step 5: Generate the interactive Folium map and simulation UI."""
import json
import os
import sys
import pandas as pd
import folium
from collections import defaultdict

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


def add_territory_boundaries(m, geojson_path, layer_name, show=False):
    """Add territory boundary polygons as a toggleable layer."""
    with open(geojson_path, "r") as f:
        geojson = json.load(f)

    fg = folium.FeatureGroup(name=layer_name, show=show)

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


def add_service_appointments(m, appts, layer_name, show=True):
    """Add service appointment markers (no clustering)."""
    fg = folium.FeatureGroup(name=layer_name, show=show)

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
        def safe_float(r, col, default=0):
            v = r.get(col, default)
            return default if pd.isna(v) else float(v)

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
                "realistic_installations_enabled": safe_float(row, "realistic_installations_enabled"),
                "total_profit_enabled_moderate_usd": safe_float(row, "total_profit_enabled_moderate_usd"),
                "gross_revenue_moderate_usd": safe_float(row, "gross_revenue_moderate_usd"),
                "net_economic_value_moderate_usd": safe_float(row, "net_economic_value_moderate_usd"),
                "break_even_installations_moderate": safe_float(row, "break_even_installations_moderate"),
                "roi_moderate_pct": safe_float(row, "roi_moderate_pct"),
            },
            "placements": placement_records,
        }
    return payload


# ---------------------------------------------------------------------------
# Territory visualization: per-tech assignment dots
# ---------------------------------------------------------------------------


def load_territory_assignment_data():
    """Load assignment CSVs and demand appointments for territory visualization."""
    sim_dir = config.OPTIMIZATION_DIR
    files = {
        "existing": os.path.join(sim_dir, "scenario_assignments_existing.csv"),
        "newhires": os.path.join(sim_dir, "scenario_assignments_newhires.csv"),
        "demand_appts": os.path.join(sim_dir, "demand_appointments.csv"),
        "tech_master": os.path.join(sim_dir, "tech_master.csv"),
        "candidates": os.path.join(sim_dir, "candidate_bases.csv"),
        "utilization": os.path.join(sim_dir, "scenario_tech_utilization.csv"),
    }

    # Critical files: existing assignments + demand appointments + tech master
    for key in ("existing", "demand_appts", "tech_master"):
        if not os.path.exists(files[key]):
            print(f"  Territory viz: missing {os.path.basename(files[key])}, skipping.")
            return None

    data = {}
    data["existing"] = pd.read_csv(files["existing"])
    data["newhires"] = (
        pd.read_csv(files["newhires"]) if os.path.exists(files["newhires"]) else pd.DataFrame()
    )
    data["demand_appts"] = pd.read_csv(files["demand_appts"])
    data["tech_master"] = pd.read_csv(files["tech_master"])
    data["candidates"] = (
        pd.read_csv(files["candidates"]) if os.path.exists(files["candidates"]) else pd.DataFrame()
    )
    data["utilization"] = (
        pd.read_csv(files["utilization"]) if os.path.exists(files["utilization"]) else pd.DataFrame()
    )

    # Determine available scenarios (intersection of assignment data with configured range)
    min_s = getattr(config, "SIM_SCENARIO_MIN", 0)
    max_s = getattr(config, "SIM_SCENARIO_MAX", 4)
    existing_scenarios = set(
        pd.to_numeric(data["existing"]["scenario_hires"], errors="coerce").dropna().astype(int)
    )
    if not data["newhires"].empty and "scenario_hires" in data["newhires"].columns:
        newhire_scenarios = set(
            pd.to_numeric(data["newhires"]["scenario_hires"], errors="coerce").dropna().astype(int)
        )
        existing_scenarios = existing_scenarios | newhire_scenarios
    data["available_scenarios"] = sorted(
        s for s in existing_scenarios if min_s <= s <= max_s
    )

    if not data["available_scenarios"]:
        print("  Territory viz: no scenarios in configured range, skipping.")
        return None

    print(f"  Territory viz: loaded assignment data for scenarios {data['available_scenarios']}")
    return data


def resolve_appointment_assignments(territory_data):
    """Resolve node-level assignments down to individual appointments.

    Returns {scenario_int: {appointment_id_str: assignee_id_str}}.
    """
    existing_df = territory_data["existing"]
    newhires_df = territory_data["newhires"]
    demand_appts = territory_data["demand_appts"]
    tech_master = territory_data["tech_master"]
    candidates = territory_data["candidates"]
    scenarios = territory_data["available_scenarios"]

    # Build assignee base coordinate lookup
    base_coords = {}
    for _, t in tech_master.iterrows():
        tid = str(t["tech_id"])
        lat = t.get("base_lat")
        lon = t.get("base_lon")
        if pd.notna(lat) and pd.notna(lon):
            base_coords[tid] = (float(lat), float(lon))
    if not candidates.empty:
        for _, c in candidates.iterrows():
            cid = str(c["candidate_id"])
            lat = c.get("lat")
            lon = c.get("lon")
            if pd.notna(lat) and pd.notna(lon):
                base_coords[cid] = (float(lat), float(lon))

    # Build demand node key for each individual appointment
    demand_appts = demand_appts.copy()
    demand_appts["node_key"] = (
        demand_appts["state_norm"].astype(str) + "__" + demand_appts["skill_class"].astype(str)
    )

    # Group appointments by node key
    node_appointments = defaultdict(list)
    for _, row in demand_appts.iterrows():
        appt_id = str(row.get("appointment_id", row.get("Appointment Number", "")))
        lat = row.get("lat")
        lon = row.get("lon")
        if pd.isna(lat) or pd.isna(lon):
            continue
        node_appointments[row["node_key"]].append({
            "id": appt_id,
            "lat": float(lat),
            "lon": float(lon),
        })

    result = {}
    for scenario in scenarios:
        assignment_map = {}

        # Gather all assignee quotas per node for this scenario
        node_assignees = defaultdict(list)  # node_key -> [(assignee_id, quota)]

        # Existing tech assignments
        sc_existing = existing_df[
            pd.to_numeric(existing_df["scenario_hires"], errors="coerce") == scenario
        ]
        for _, row in sc_existing.iterrows():
            node_key = str(row["node_id"])
            assignee_id = str(row["tech_id"])
            quota = float(row.get("assigned_appointments", 0))
            if quota > 0:
                node_assignees[node_key].append((assignee_id, quota))

        # New hire assignments
        if not newhires_df.empty and "scenario_hires" in newhires_df.columns:
            sc_newhires = newhires_df[
                pd.to_numeric(newhires_df["scenario_hires"], errors="coerce") == scenario
            ]
            for _, row in sc_newhires.iterrows():
                node_key = str(row["node_id"])
                assignee_id = str(row["candidate_id"])
                quota = float(row.get("assigned_appointments", 0))
                if quota > 0:
                    node_assignees[node_key].append((assignee_id, quota))

        # For each node, assign individual appointments to assignees via nearest-neighbor
        for node_key, assignees in node_assignees.items():
            appts_list = node_appointments.get(node_key, [])
            if not appts_list:
                continue

            # Round fractional quotas to integers
            raw_quotas = [(aid, q) for aid, q in assignees]
            rounded = [(aid, int(round(q))) for aid, q in raw_quotas]

            # Adjust if rounding mismatch
            total_rounded = sum(q for _, q in rounded)
            total_appts = len(appts_list)
            if total_rounded != total_appts and rounded:
                # Adjust the largest quota
                diff = total_appts - total_rounded
                idx_max = max(range(len(rounded)), key=lambda i: rounded[i][1])
                aid, q = rounded[idx_max]
                rounded[idx_max] = (aid, max(0, q + diff))

            # Build remaining quota dict
            remaining = {}
            for aid, q in rounded:
                remaining[aid] = remaining.get(aid, 0) + q

            # Nearest-neighbor greedy assignment
            # Compute distances from each appointment to each assignee base
            assignee_ids = [aid for aid in remaining if remaining[aid] > 0 and aid in base_coords]
            if not assignee_ids:
                continue

            for appt in sorted(appts_list, key=lambda a: a["id"]):
                if not assignee_ids:
                    break
                best_aid = None
                best_dist = float("inf")
                for aid in assignee_ids:
                    if remaining.get(aid, 0) <= 0:
                        continue
                    bcoord = base_coords[aid]
                    dist = (appt["lat"] - bcoord[0]) ** 2 + (appt["lon"] - bcoord[1]) ** 2
                    if dist < best_dist or (dist == best_dist and (best_aid is None or aid < best_aid)):
                        best_dist = dist
                        best_aid = aid
                if best_aid is not None:
                    assignment_map[appt["id"]] = best_aid
                    remaining[best_aid] -= 1
                    if remaining[best_aid] <= 0:
                        assignee_ids = [a for a in assignee_ids if remaining.get(a, 0) > 0]

        result[scenario] = assignment_map

    return result


def build_tech_color_map(territory_data):
    """Assign a unique color from TECH_TERRITORY_PALETTE to each assignee."""
    tech_master = territory_data["tech_master"]
    newhires_df = territory_data["newhires"]
    palette = config.TECH_TERRITORY_PALETTE

    # Existing techs with availability > 0, sorted alphabetically
    active_techs = tech_master[
        pd.to_numeric(tech_master["availability_fte"], errors="coerce").fillna(0) > 0
    ].copy()
    sorted_tech_ids = sorted(active_techs["tech_id"].astype(str).tolist())

    # New hire candidate IDs from assignments
    candidate_ids = set()
    if not newhires_df.empty and "candidate_id" in newhires_df.columns:
        candidate_ids = set(newhires_df["candidate_id"].astype(str).unique())
    sorted_candidate_ids = sorted(candidate_ids)

    color_map = {}
    idx = 0
    for tid in sorted_tech_ids:
        color_map[tid] = palette[idx % len(palette)]
        idx += 1
    for cid in sorted_candidate_ids:
        if cid not in color_map:
            color_map[cid] = palette[idx % len(palette)]
            idx += 1

    return color_map


def add_territory_assignment_layers(m, assignment_map, territory_data, tech_color_map):
    """Add per-scenario territory dot layers to the map.

    Returns {scenario_str: {"dots_layer": js_name, "tech_stats": {...}}}.
    """
    demand_appts = territory_data["demand_appts"]
    tech_master = territory_data["tech_master"]
    candidates = territory_data["candidates"]
    utilization_df = territory_data["utilization"]
    existing_df = territory_data["existing"]
    newhires_df = territory_data["newhires"]
    scenarios = territory_data["available_scenarios"]

    # Build name lookup
    name_lookup = {}
    for _, t in tech_master.iterrows():
        name_lookup[str(t["tech_id"])] = str(t["tech_name"])
    if not candidates.empty:
        for _, c in candidates.iterrows():
            cid = str(c["candidate_id"])
            city = c.get("city", "")
            state = c.get("state", "")
            name_lookup[cid] = f"New Hire ({city}, {state})"

    # Build appointment detail lookup
    appt_details = {}
    for _, row in demand_appts.iterrows():
        appt_id = str(row.get("appointment_id", row.get("Appointment Number", "")))
        appt_details[appt_id] = {
            "lat": float(row["lat"]) if pd.notna(row.get("lat")) else None,
            "lon": float(row["lon"]) if pd.notna(row.get("lon")) else None,
            "account": str(row.get("Account: Account Name", "N/A")),
            "service_type": str(row.get("Service Type", "")),
            "state": str(row.get("state_norm", "")),
            "city": str(row.get("city", "")),
            "skill_class": str(row.get("skill_class", "")),
        }

    # Build base location lookup for popups
    base_location = {}
    for _, t in tech_master.iterrows():
        tid = str(t["tech_id"])
        city = t.get("base_city", "")
        state = t.get("base_state", "")
        base_location[tid] = f"{city}, {state}" if pd.notna(city) else str(state)
    if not candidates.empty:
        for _, c in candidates.iterrows():
            cid = str(c["candidate_id"])
            city = c.get("city", "")
            state = c.get("state", "")
            base_location[cid] = f"{city}, {state}" if pd.notna(city) else str(state)

    result = {}

    for scenario in scenarios:
        scenario_str = str(scenario)
        appt_assignments = assignment_map.get(scenario, {})
        is_default = scenario == scenarios[0]

        # Gather per-tech appointment locations and details
        tech_appts = defaultdict(list)  # assignee_id -> [(lat, lon, appt_id)]
        for appt_id, assignee_id in appt_assignments.items():
            detail = appt_details.get(appt_id)
            if detail and detail["lat"] is not None:
                tech_appts[assignee_id].append((detail["lat"], detail["lon"], appt_id))

        # --- Dots layer ---
        dots_fg = folium.FeatureGroup(
            name=f"Territory Dots N={scenario}",
            show=is_default,
            control=False,
        )
        for assignee_id, appts_list in tech_appts.items():
            color = tech_color_map.get(assignee_id, "#888888")
            tech_name = name_lookup.get(assignee_id, assignee_id)
            for lat, lon, appt_id in appts_list:
                detail = appt_details.get(appt_id, {})
                popup_html = (
                    f"<b>{detail.get('account', 'N/A')}</b><br>"
                    f"Appt: {appt_id}<br>"
                    f"Type: {detail.get('service_type', '')}<br>"
                    f"Location: {detail.get('city', '')}, {detail.get('state', '')}<br>"
                    f"Skill: {detail.get('skill_class', '')}<br>"
                    f"Assigned to: <b>{tech_name}</b>"
                )
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=config.TERRITORY_DOT_RADIUS,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=config.TERRITORY_DOT_OPACITY,
                    weight=1,
                    popup=folium.Popup(popup_html, max_width=300),
                    tooltip=f"{tech_name}: {detail.get('account', '')}",
                ).add_to(dots_fg)
        dots_fg.add_to(m)

        # Build per-tech stats
        tech_stats = {}
        for assignee_id, appts_list in tech_appts.items():
            tech_name = name_lookup.get(assignee_id, assignee_id)
            base_loc = base_location.get(assignee_id, "Unknown")
            states_served = sorted(set(
                appt_details.get(a_id, {}).get("state", "")
                for _, _, a_id in appts_list if appt_details.get(a_id, {}).get("state", "")
            ))

            # Travel cost: sum from assignment CSVs
            travel_cost = 0.0
            sc_existing = existing_df[
                (pd.to_numeric(existing_df["scenario_hires"], errors="coerce") == scenario)
                & (existing_df["tech_id"].astype(str) == assignee_id)
            ]
            if not sc_existing.empty:
                travel_cost += pd.to_numeric(
                    sc_existing["total_travel_cost_usd"], errors="coerce"
                ).fillna(0).sum()
            if not newhires_df.empty and "candidate_id" in newhires_df.columns:
                sc_newhire = newhires_df[
                    (pd.to_numeric(newhires_df["scenario_hires"], errors="coerce") == scenario)
                    & (newhires_df["candidate_id"].astype(str) == assignee_id)
                ]
                if not sc_newhire.empty:
                    travel_cost += pd.to_numeric(
                        sc_newhire["total_travel_cost_usd"], errors="coerce"
                    ).fillna(0).sum()

            # Utilization from utilization CSV
            utilization = 0.0
            if not utilization_df.empty and "tech_id" in utilization_df.columns:
                util_row = utilization_df[
                    (pd.to_numeric(utilization_df["scenario_hires"], errors="coerce") == scenario)
                    & (utilization_df["tech_id"].astype(str) == assignee_id)
                ]
                if not util_row.empty:
                    utilization = float(
                        pd.to_numeric(util_row["utilization"].iloc[0], errors="coerce") or 0
                    )

            tech_stats[assignee_id] = {
                "name": tech_name,
                "base": base_loc,
                "appointments": len(appts_list),
                "travel_cost_usd": round(travel_cost, 2),
                "utilization": round(utilization, 4),
                "states": states_served,
            }

        result[scenario_str] = {
            "dots_layer": dots_fg.get_name(),
            "tech_stats": tech_stats,
        }

    return result


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


def add_simulation_panel(m, simulation_payload, scenario_layer_names,
                         territory_layer_names=None, tech_color_map=None):
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

    # Territory layer JS maps
    territory_dots_entries = []
    tech_colors_js = json.dumps(tech_color_map) if tech_color_map else "{}"
    if territory_layer_names:
        for key in ordered_keys:
            info = territory_layer_names.get(key)
            if info:
                territory_dots_entries.append(f'"{key}": "{info["dots_layer"]}"')
    territory_dots_js = "{\n" + ",\n".join(territory_dots_entries) + "\n}" if territory_dots_entries else "{}"

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
      #sim-tech-legend {
        max-height: 180px;
        overflow-y: auto;
        font-size: 10px;
        margin-top: 8px;
        border: 1px solid #e2e5e9;
        border-radius: 8px;
        background: #fbfbfc;
        padding: 6px 8px;
      }
      #sim-tech-legend-title {
        margin: 8px 0 0 0;
        font-size: 12px;
        font-weight: 700;
      }
      .tech-legend-row {
        display: flex;
        align-items: center;
        gap: 5px;
        padding: 2px 0;
      }
      .tech-legend-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        flex-shrink: 0;
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
      .kpi-clickable { cursor: pointer; transition: background 0.15s, border-color 0.15s; }
      .kpi-clickable:hover { background: #eef1f5; border-color: #163b59; }
      #kpi-modal-overlay {
        display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background: rgba(0,0,0,0.35); z-index: 2000; justify-content: center; align-items: center;
      }
      #kpi-modal-overlay[style*="display: block"] { display: flex !important; }
      #kpi-modal {
        background: #fff; border-radius: 12px; padding: 20px 24px; max-width: 420px; width: 90%;
        box-shadow: 0 8px 32px rgba(0,0,0,0.25); position: relative; font-size: 13px; line-height: 1.5;
      }
      #kpi-modal-title { font-size: 15px; font-weight: 700; margin-bottom: 10px; color: #163b59; }
      #kpi-modal-body { color: #333; }
      #kpi-modal-body p { margin: 6px 0; }
      #kpi-modal-close {
        position: absolute; top: 10px; right: 14px; font-size: 20px; cursor: pointer;
        color: #888; background: none; border: none; line-height: 1;
      }
      #kpi-modal-close:hover { color: #333; }
      @media (max-width: 900px) {
        #sim-panel { display: none; width: 280px; }
        #sim-panel.mobile-open { display: block; }
        #sim-panel-toggle { display: block; }
      }
    </style>
    <button id="sim-panel-toggle" title="Show/hide simulation panel">Simulation</button>
    <div id="sim-panel">
      <h3>Simulation Scenarios</h3>
      <p id="sim-subtitle">Cost-first optimization — all figures annualized.</p>
      <div id="sim-buttons"></div>
      <div id="sim-kpis">
        <div class="sim-kpi kpi-clickable" data-kpi="total-cost"><div class="label">Total Cost</div><div class="value" id="kpi-total">-</div></div>
        <div class="sim-kpi kpi-clickable" data-kpi="cost-change"><div class="label">Cost Change vs N=0</div><div class="value" id="kpi-savings">-</div></div>
        <div class="sim-kpi kpi-clickable" data-kpi="marginal-cost"><div class="label">Marginal Cost Change</div><div class="value" id="kpi-marginal">-</div></div>
        <div class="sim-kpi kpi-clickable" data-kpi="unmet" id="kpi-unmet-card"><div class="label">Unmet Appointments</div><div class="value" id="kpi-unmet">-</div></div>
        <div class="sim-kpi kpi-clickable" data-kpi="hire-payroll"><div class="label">Annual Hire Payroll</div><div class="value" id="kpi-hire-cost">-</div></div>
        <div class="sim-kpi kpi-clickable" data-kpi="mean-util"><div class="label">Mean Utilization</div><div class="value" id="kpi-mean-util">-</div></div>
        <div class="sim-kpi kpi-clickable" data-kpi="max-util"><div class="label">Max Utilization</div><div class="value" id="kpi-max-util">-</div></div>
      </div>
      <div style="border-top: 2px solid #163b59; margin: 10px 0 8px 0;"></div>
      <div style="font-size: 12px; font-weight: 700; margin-bottom: 2px;">Freed Capacity &amp; Profit Potential</div>
      <div style="font-size: 9px; color: #666; margin-bottom: 6px;">Moderate scenario: $120K/install, 25% margin</div>
      <div id="sim-kpis-revenue" style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-bottom: 10px;">
        <div class="sim-kpi kpi-clickable" data-kpi="installs"><div class="label">Realistic Installations</div><div class="value" id="kpi-installs">-</div></div>
        <div class="sim-kpi kpi-clickable" data-kpi="gross-rev"><div class="label">Gross Revenue Enabled</div><div class="value" id="kpi-gross-rev">-</div></div>
        <div class="sim-kpi kpi-clickable" data-kpi="profit"><div class="label">Est. Profit Enabled</div><div class="value" id="kpi-profit">-</div></div>
        <div class="sim-kpi kpi-clickable" data-kpi="net-value"><div class="label">Net Economic Value</div><div class="value" id="kpi-net-value">-</div></div>
        <div class="sim-kpi kpi-clickable" data-kpi="break-even"><div class="label">Break-Even Installations</div><div class="value" id="kpi-break-even">-</div></div>
        <div class="sim-kpi kpi-clickable" data-kpi="roi"><div class="label">ROI</div><div class="value" id="kpi-roi">-</div></div>
      </div>
      <div id="sim-recs-title">Recommended Bases</div>
      <div id="sim-recs">No recommendations.</div>
      <div id="sim-tech-legend-title" style="display:none;">Territory Assignments</div>
      <div id="sim-tech-legend" style="display:none;"></div>
      <div id="sim-footnote">Shows N=0..4 scenario outputs from optimization pipeline. Click any card for details.</div>
    </div>
    <div id="kpi-modal-overlay">
      <div id="kpi-modal">
        <button id="kpi-modal-close">&times;</button>
        <div id="kpi-modal-title"></div>
        <div id="kpi-modal-body"></div>
      </div>
    </div>
    """

    script_js = f"""
    (function() {{
      const mapVarName = "{map_var}";
      const scenarioData = {payload_js};
      const scenarioLayerNames = {layer_js};
      const territoryDotLayerNames = {territory_dots_js};
      const techColors = {tech_colors_js};
      const orderedScenarios = {json.dumps(ordered_keys)};
      const defaultScenario = "{default_key}";
      const showUnmetKpi = orderedScenarios.some((s) =>
        Number(((scenarioData[s] || {{}}).kpis || {{}}).unmet_appointments || 0) > 0
      );
      const hasTerritory = Object.keys(territoryDotLayerNames).length > 0;
      let mapRef = null;
      let scenarioLayers = {{}};
      let territoryDotLayers = {{}};

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

        // Revenue / profit KPIs
        var installs = Number(k.realistic_installations_enabled || 0);
        var hasInstalls = installs > 0;
        document.getElementById("kpi-installs").textContent = hasInstalls ? installs.toFixed(1) : "\u2014";
        document.getElementById("kpi-gross-rev").textContent = hasInstalls ? money(k.gross_revenue_moderate_usd) : "\u2014";
        document.getElementById("kpi-profit").textContent = hasInstalls ? money(k.total_profit_enabled_moderate_usd) : "\u2014";
        var netVal = Number(k.net_economic_value_moderate_usd || 0);
        var netValEl = document.getElementById("kpi-net-value");
        netValEl.textContent = hasInstalls ? money(netVal) : "\u2014";
        netValEl.style.color = hasInstalls ? (netVal >= 0 ? "#1a7f37" : "#d1242f") : "";
        var be = Number(k.break_even_installations_moderate || 0);
        document.getElementById("kpi-break-even").textContent = hasInstalls ? be.toFixed(1) : "\u2014";
        var roi = Number(k.roi_moderate_pct || 0);
        document.getElementById("kpi-roi").textContent = (hasInstalls && roi) ? roi.toFixed(0) + "%" : "\u2014";
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

      function renderTechLegend(scenario) {{
        const titleEl = document.getElementById("sim-tech-legend-title");
        const legendEl = document.getElementById("sim-tech-legend");
        if (!titleEl || !legendEl || !hasTerritory) return;
        const item = scenarioData[scenario];
        const stats = item ? (item.tech_stats || {{}}) : {{}};
        const entries = Object.entries(stats);
        if (!entries.length) {{
          titleEl.style.display = "none";
          legendEl.style.display = "none";
          return;
        }}
        titleEl.style.display = "block";
        legendEl.style.display = "block";
        entries.sort((a, b) => (b[1].appointments || 0) - (a[1].appointments || 0));
        legendEl.innerHTML = entries.map(([tid, s]) => {{
          const color = techColors[tid] || "#888";
          return `<div class="tech-legend-row">` +
            `<span class="tech-legend-dot" style="background:${{color}};"></span>` +
            `<span>${{s.name}} (${{s.appointments}})</span>` +
            `</div>`;
        }}).join("");
      }}

      function showScenario(scenario) {{
        if (!mapRef) return;
        orderedScenarios.forEach((s) => {{
          const layer = scenarioLayers[s];
          if (layer && mapRef.hasLayer(layer)) {{
            mapRef.removeLayer(layer);
          }}
          const dotLayer = territoryDotLayers[s];
          if (dotLayer && mapRef.hasLayer(dotLayer)) {{
            mapRef.removeLayer(dotLayer);
          }}
        }});
        const target = scenarioLayers[scenario];
        if (target && !mapRef.hasLayer(target)) {{
          mapRef.addLayer(target);
        }}
        const dotTarget = territoryDotLayers[scenario];
        if (dotTarget && !mapRef.hasLayer(dotTarget)) {{
          mapRef.addLayer(dotTarget);
        }}
        setActiveButton(scenario);
        renderKpis(scenario);
        renderRecommendations(scenario);
        renderTechLegend(scenario);
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
        // Resolve territory dot layers (no-op if empty)
        orderedScenarios.forEach((s) => {{
          const dotVar = territoryDotLayerNames[s];
          if (dotVar && window[dotVar]) territoryDotLayers[s] = window[dotVar];
        }});
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

      var kpiExplanations = {{
        "total-cost": {{
          title: "Total Cost",
          body: "<p><b>What:</b> The total annual economic cost for this hiring scenario, including travel costs for all technicians, new-hire payroll, and fixed canceled/voided overhead ($35,632).</p><p><b>Formula:</b> Travel Cost + Hire Payroll + Canceled/Voided Overhead</p><p><b>Interpretation:</b> Lower is better from a pure cost perspective. N=0 is the cheapest scenario because adding hires increases payroll faster than it reduces travel costs.</p>"
        }},
        "cost-change": {{
          title: "Cost Change vs N=0",
          body: "<p><b>What:</b> How much more (or less) this scenario costs compared to the N=0 baseline with no new hires.</p><p><b>Formula:</b> Total Cost(N=0) - Total Cost(this scenario)</p><p><b>Interpretation:</b> Negative values mean this scenario costs more than N=0. The optimizer minimizes cost, so all hiring scenarios show increased cost vs baseline.</p>"
        }},
        "marginal-cost": {{
          title: "Marginal Cost Change",
          body: "<p><b>What:</b> The incremental cost change from adding one more hire compared to the previous scenario (N-1).</p><p><b>Formula:</b> Total Cost(N-1) - Total Cost(N)</p><p><b>Interpretation:</b> Shows diminishing returns on travel savings. Each additional hire saves less on travel while adding a fixed $146,640 in payroll.</p>"
        }},
        "unmet": {{
          title: "Unmet Appointments",
          body: "<p><b>What:</b> Number of service appointments that could not be assigned to any technician in the optimal solution.</p><p><b>Formula:</b> Count of demand appointments with no feasible assignment</p><p><b>Interpretation:</b> Zero means all 1,480 appointments are served. Non-zero would indicate capacity or skill constraints preventing full coverage.</p>"
        }},
        "hire-payroll": {{
          title: "Annual Hire Payroll",
          body: "<p><b>What:</b> Total burdened annual cost for all new hires in this scenario.</p><p><b>Formula:</b> N new hires &times; $146,640/hire (burdened company planning cost)</p><p><b>Interpretation:</b> This is the fully-loaded cost including salary, benefits, equipment, and overhead \u2014 not take-home pay. It's the incremental payroll cost of expanding the workforce.</p>"
        }},
        "mean-util": {{
          title: "Mean Utilization (Existing Techs)",
          body: "<p><b>What:</b> Average utilization rate across all existing technicians with non-zero capacity.</p><p><b>Formula:</b> Mean of (assigned hours / capacity hours) for each existing tech</p><p><b>Interpretation:</b> Target is 0.850 (85%). At N=0 it's ~0.857 \u2014 the workforce is tightly loaded. Adding hires reduces existing-tech utilization by offloading appointments, freeing capacity for installations.</p>"
        }},
        "max-util": {{
          title: "Max Utilization (Existing Techs)",
          body: "<p><b>What:</b> The highest utilization rate among all existing technicians.</p><p><b>Formula:</b> Max of (assigned hours / capacity hours) across existing techs</p><p><b>Interpretation:</b> Values near 1.000 indicate at least one tech is at full capacity with no scheduling buffer. At N=0, max util is 0.9999 \u2014 one tech is essentially maxed out.</p>"
        }},
        "installs": {{
          title: "Realistic Installations Enabled",
          body: "<p><b>What:</b> Estimated number of new installations that freed existing-tech time could support.</p><p><b>Formula:</b> (Freed duration days &times; 75% utilization factor) &divide; (3.25 avg install days + 1.0 travel day)</p><p><b>Interpretation:</b> This accounts for scheduling gaps, PTO, and travel overhead. It's what freed capacity <i>could enable</i>, not guaranteed bookings. N=0 shows 0 because no capacity is freed without hiring.</p>"
        }},
        "gross-rev": {{
          title: "Gross Revenue Enabled",
          body: "<p><b>What:</b> Total MSRP gross revenue if all realistic installations were completed (moderate scenario: $120K/install).</p><p><b>Formula:</b> Realistic installations &times; $120,000 per install</p><p><b>Interpretation:</b> This is top-line revenue before any costs or margins. Shown for reference only \u2014 the profit figure below is more meaningful for business decisions.</p>"
        }},
        "profit": {{
          title: "Estimated Profit Enabled",
          body: "<p><b>What:</b> Estimated profit from installations and service contracts, with industry-typical margins applied.</p><p><b>Formula:</b> (Installs &times; $120K &times; 25% install margin) + (Installs &times; $7K &times; 70% service margin)</p><p><b>Interpretation:</b> Uses moderate scenario assumptions. Installation margin (25%) covers COGS, shipping, commissions, and warranty. Service contract margin (70%) reflects higher-margin recurring revenue.</p>"
        }},
        "net-value": {{
          title: "Net Economic Value",
          body: "<p><b>What:</b> Profit enabled minus the net cost increase of hiring. The bottom-line value of this scenario.</p><p><b>Formula:</b> Total Profit Enabled - Net Cost Increase vs N=0</p><p><b>Interpretation:</b> <span style='color:#1a7f37'>Green = positive</span> means the profit from freed capacity exceeds the cost of hiring. This is the key metric for justifying new hires from a revenue perspective.</p>"
        }},
        "break-even": {{
          title: "Break-Even Installations",
          body: "<p><b>What:</b> Minimum installations needed to recoup the net cost increase of hiring.</p><p><b>Formula:</b> Net Cost Increase &divide; ($120K &times; 25% + $7K &times; 70% = $34,900 profit per install)</p><p><b>Interpretation:</b> A low break-even (e.g., 1.7) means only ~2 installations are needed to justify the hire. Compare to realistic installations enabled (~56.7 for N=1) to assess feasibility.</p>"
        }},
        "roi": {{
          title: "ROI (Return on Investment)",
          body: "<p><b>What:</b> Net economic value as a percentage of net cost increase.</p><p><b>Formula:</b> (Net Economic Value &divide; Net Cost Increase) &times; 100%</p><p><b>Interpretation:</b> A 3,000%+ ROI means each dollar of incremental hiring cost could generate ~$30 in profit from freed capacity. High ROI reflects the low marginal cost of hiring relative to the high value of installation capacity.</p>"
        }}
      }};

      document.addEventListener("click", function(e) {{
        var card = e.target.closest(".kpi-clickable");
        if (card) {{
          var kpiKey = card.getAttribute("data-kpi");
          var info = kpiExplanations[kpiKey];
          if (info) {{
            document.getElementById("kpi-modal-title").textContent = info.title;
            document.getElementById("kpi-modal-body").innerHTML = info.body;
            document.getElementById("kpi-modal-overlay").style.display = "block";
          }}
          return;
        }}
        var overlay = document.getElementById("kpi-modal-overlay");
        if (e.target === overlay || e.target.id === "kpi-modal-close") {{
          overlay.style.display = "none";
        }}
      }});

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

    # Pre-check: load territory assignment data (needed to decide show param for Layer 3)
    territory_data = None
    territory_layer_info = None
    if getattr(config, "ENABLE_SIMULATION_UI", False):
        print("  Pre-loading territory assignment data...")
        territory_data = load_territory_assignment_data()

    # Layer 3: Service Appointments
    # Hidden by default when territory viz is active (user can toggle on via LayerControl)
    show_static_appts = territory_data is None
    print("Adding service appointments...")
    add_service_appointments(m, appts, layer_name=layer_appt_name, show=show_static_appts)

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
            # Territory visualization
            if territory_data:
                print("  Resolving appointment-to-tech assignments...")
                assignment_map = resolve_appointment_assignments(territory_data)
                tech_color_map = build_tech_color_map(territory_data)
                total_assigned = sum(len(v) for v in assignment_map.values())
                print(f"  Resolved {total_assigned} total appointment assignments across {len(assignment_map)} scenarios")
                print("  Generating territory dot layers...")
                territory_layer_info = add_territory_assignment_layers(
                    m, assignment_map, territory_data, tech_color_map
                )
                # Inject tech_stats into per-scenario payload entries for JS
                for key, info in territory_layer_info.items():
                    if key in simulation_payload:
                        simulation_payload[key]["tech_stats"] = info["tech_stats"]
                print(f"  Territory layers added for scenarios: {list(territory_layer_info.keys())}")
            else:
                print("  Territory assignment data not available; territory layers skipped.")

            scenario_layer_names = add_simulation_layers(m, simulation_payload)
            add_simulation_panel(
                m, simulation_payload, scenario_layer_names,
                territory_layer_names=territory_layer_info,
                tech_color_map=tech_color_map if territory_data else None,
            )
            print(f"  Loaded scenarios: {', '.join(sorted(simulation_payload.keys(), key=lambda x: int(x) if x.isdigit() else 999))}")
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
