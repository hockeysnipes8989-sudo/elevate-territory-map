"""Step 5: Generate the interactive Folium map and simulation UI."""
import json
import os
import sys
import pandas as pd
import numpy as np
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


def get_map_ui_preset():
    """Return the active stakeholder/debug UI preset."""
    mode = str(getattr(config, "MAP_UI_MODE", "stakeholder")).strip().lower()
    stakeholder = mode == "stakeholder"
    return {
        "mode": mode,
        "include_airports": True,
        "show_active_assets": not stakeholder,
        "show_nonactive_assets": not stakeholder,
        "show_service_appointments": not stakeholder,
        "show_territory_boundaries": not stakeholder,
        "show_hub_radius": not stakeholder,
        "airport_default_visible": (
            getattr(config, "MAP_STAKEHOLDER_HUBS_DEFAULT_VISIBLE", False)
            if stakeholder
            else True
        ),
        "show_hub_toggle": bool(
            getattr(config, "MAP_STAKEHOLDER_SHOW_HUB_TOGGLE", True)
        ) and stakeholder,
        "hub_toggle_label": getattr(config, "MAP_HUB_TOGGLE_LABEL", "Flight hubs"),
        "show_layer_control": not stakeholder,
        "show_legends": not stakeholder,
        "panel_width_px": getattr(config, "MAP_PANEL_WIDTH_PX", 360),
        "panel_radius_px": getattr(config, "MAP_PANEL_RADIUS_PX", 18),
        "panel_shadow": getattr(
            config, "MAP_PANEL_SHADOW", "0 18px 40px rgba(15, 23, 42, 0.16)"
        ),
        "card_radius_px": getattr(config, "MAP_CARD_RADIUS_PX", 14),
        "font_family": getattr(
            config,
            "MAP_PANEL_FONT_FAMILY",
            '"Public Sans", "Segoe UI", "Helvetica Neue", Arial, sans-serif',
        ),
        "coverage_default_count": getattr(
            config, "MAP_COVERAGE_ASSIGNMENTS_DEFAULT_COUNT", 6
        ),
        "tech_marker_colors": getattr(config, "MAP_TECH_MARKER_COLORS", {}),
        "new_hire_marker_color": getattr(config, "MAP_NEW_HIRE_MARKER_COLOR", "#C5662D"),
        "airport_marker_color": getattr(config, "MAP_AIRPORT_MARKER_COLOR", "#6B7280"),
        "airport_marker_fill": getattr(config, "MAP_AIRPORT_MARKER_FILL", "#F8FAFC"),
        "airport_marker_border": getattr(config, "MAP_AIRPORT_MARKER_BORDER", "#CBD5E1"),
        "assignment_dot_radius": getattr(
            config,
            "MAP_ASSIGNMENT_DOT_STAKEHOLDER_RADIUS"
            if stakeholder
            else "MAP_ASSIGNMENT_DOT_DEBUG_RADIUS",
            getattr(config, "TERRITORY_DOT_RADIUS", 6),
        ),
        "assignment_dot_opacity": getattr(
            config,
            "MAP_ASSIGNMENT_DOT_STAKEHOLDER_OPACITY"
            if stakeholder
            else "MAP_ASSIGNMENT_DOT_DEBUG_OPACITY",
            getattr(config, "TERRITORY_DOT_OPACITY", 0.85),
        ),
        "assignment_dot_light_stroke": getattr(
            config, "MAP_ASSIGNMENT_DOT_LIGHT_STROKE", "#FFFFFF"
        ),
        "assignment_dot_dark_stroke": getattr(
            config, "MAP_ASSIGNMENT_DOT_DARK_STROKE", "#1F2937"
        ),
        "assignment_dot_stroke_width": getattr(config, "MAP_ASSIGNMENT_DOT_STROKE_WIDTH", 1.1),
        "territory_palette": list(
            getattr(
                config,
                "STAKEHOLDER_TERRITORY_PALETTE" if stakeholder else "TECH_TERRITORY_PALETTE",
                getattr(config, "TECH_ASSIGNMENT_FALLBACK_PALETTE", []),
            )
        ),
    }


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


def load_anchor_tech_metadata():
    """Load anchored-tech metadata from optimization tech_master if available."""
    tech_master_path = os.path.join(config.OPTIMIZATION_DIR, "tech_master.csv")
    if not os.path.exists(tech_master_path):
        return {"by_name": {}, "anchors": []}

    tech_master = pd.read_csv(tech_master_path)
    required_anchor_cols = {
        "tech_name",
        "anchor_site_name",
        "anchor_site_lat",
        "anchor_site_lon",
        "anchor_reserved_fte",
        "external_field_fte",
        "anchor_show_on_map",
        "assignment_scope_states",
        "anchor_notes",
    }
    if not required_anchor_cols.issubset(set(tech_master.columns)):
        return {"by_name": {}, "anchors": []}

    by_name = {}
    anchor_rows = []
    anchored = tech_master[
        tech_master["anchor_site_name"].fillna("").astype(str).str.strip().ne("")
    ].copy()

    for _, row in anchored.iterrows():
        tech_name = str(row.get("tech_name", "")).strip()
        if not tech_name:
            continue
        metadata = {
            "anchor_site_name": str(row.get("anchor_site_name", "")).strip(),
            "anchor_site_location_raw": str(row.get("anchor_site_location_raw", "")).strip(),
            "anchor_reserved_fte": pd.to_numeric(row.get("anchor_reserved_fte"), errors="coerce"),
            "external_field_fte": pd.to_numeric(row.get("external_field_fte"), errors="coerce"),
            "assignment_scope_states": str(row.get("assignment_scope_states", "")).strip(),
            "anchor_notes": str(row.get("anchor_notes", "")).strip(),
            "anchor_show_on_map": bool(
                pd.to_numeric(row.get("anchor_show_on_map"), errors="coerce")
                if pd.notna(pd.to_numeric(row.get("anchor_show_on_map"), errors="coerce"))
                else 0
            ),
            "anchor_site_lat": pd.to_numeric(row.get("anchor_site_lat"), errors="coerce"),
            "anchor_site_lon": pd.to_numeric(row.get("anchor_site_lon"), errors="coerce"),
        }
        by_name[tech_name] = metadata
        if (
            metadata["anchor_show_on_map"]
            and pd.notna(metadata["anchor_site_lat"])
            and pd.notna(metadata["anchor_site_lon"])
        ):
            anchor_rows.append(
                {
                    "tech_name": tech_name,
                    **metadata,
                }
            )
    return {"by_name": by_name, "anchors": anchor_rows}


def format_state_scope(states_value):
    """Render semicolon-delimited state scope as slash-delimited copy."""
    if states_value is None or pd.isna(states_value):
        return ""
    parts = [part.strip() for part in str(states_value).split(";") if part.strip()]
    return " / ".join(parts)


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
        omitted_split_states = feature["properties"].get("omitted_split_states", [])
        split_note = ""
        if omitted_split_states:
            split_list = ", ".join(omitted_split_states)
            split_note = (
                "<br><span style='color:#666;font-size:11px;'>"
                f"Split-state coverage shown as reference markers only: {split_list}"
                "</span>"
            )

        popup_html = (
            f"<b>{name}</b><br>"
            f"Service appointments: {appts}<br>"
            f"Active contract assets: {assets}"
            f"{split_note}"
        )

        if feature.get("geometry", {}).get("coordinates"):
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

        for marker in feature["properties"].get("reference_markers", []):
            lat = marker.get("lat")
            lon = marker.get("lon")
            label = marker.get("label", f"{name} reference")
            if lat is None or lon is None:
                continue
            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.95,
                weight=2,
                tooltip=f"{name}: {label}",
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


def add_technician_markers(m, techs, layer_name, ui_preset, anchor_metadata_by_name=None):
    """Add technician home base markers."""
    fg = folium.FeatureGroup(name=layer_name, show=True)
    anchor_metadata_by_name = anchor_metadata_by_name or {}

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
            color = ui_preset["tech_marker_colors"].get(
                status, ui_preset["tech_marker_colors"].get("active", "#2E6F5E")
            )
            tooltip = f"{row['name']} ({status})"
        else:
            status_counts = group["status"].value_counts().to_dict()
            if len(status_counts) == 1:
                only_status = next(iter(status_counts.keys()))
                color = ui_preset["tech_marker_colors"].get(
                    only_status, ui_preset["tech_marker_colors"].get("mixed", "#51606E")
                )
            else:
                color = ui_preset["tech_marker_colors"].get("mixed", "#51606E")
            tooltip = f"{location} ({num_techs} techs)"

        roster_lines = []
        for _, r in group.iterrows():
            name = str(r.get("name", "")).strip()
            status = str(r.get("status", "active")).strip().title()
            comment_raw = r.get("comment", "")
            comment = "" if pd.isna(comment_raw) else str(comment_raw).strip()
            if comment.lower() == "nan":
                comment = ""
            anchor_meta = anchor_metadata_by_name.get(name, {})
            extra = f" - {comment}" if comment else ""
            detail_lines = []
            anchor_site = str(anchor_meta.get("anchor_site_name", "")).strip()
            if anchor_site:
                detail_lines.append(f"Anchor site: {anchor_site}")
            reserved = anchor_meta.get("anchor_reserved_fte")
            external = anchor_meta.get("external_field_fte")
            allowed_states = format_state_scope(anchor_meta.get("assignment_scope_states", ""))
            if pd.notna(reserved):
                detail_lines.append(f"Reserved duty: {float(reserved):.0%}")
            if pd.notna(external):
                detail_lines.append(f"External field capacity: {float(external):.0%}")
            if allowed_states:
                detail_lines.append(f"External assignment region: {allowed_states}")
            if anchor_meta.get("anchor_notes"):
                detail_lines.append(str(anchor_meta["anchor_notes"]))
            extra_html = (
                "<br><span style='display:block;margin-top:3px;font-size:11px;color:#64748b;'>"
                + "<br>".join(detail_lines)
                + "</span>"
                if detail_lines
                else ""
            )
            roster_lines.append(f"<li><b>{name}</b> ({status}){extra}{extra_html}</li>")
        roster_html = "".join(roster_lines)
        popup_html = (
            f"<b>{location}</b><br>"
            f"Technicians at this base: <b>{num_techs}</b><br>"
            "<div style='margin-top:6px;'>"
            "<b>Roster</b>"
            f"<ul style='margin:4px 0 0 16px; padding:0;'>{roster_html}</ul>"
            "</div>"
        )

        diameter = 24 if num_techs == 1 else 30
        label = str(num_techs)
        marker_html = (
            "<div style=\""
            f"width:{diameter}px;height:{diameter}px;border-radius:999px;"
            f"background:{color};border:2px solid #ffffff;color:#ffffff;"
            "display:flex;align-items:center;justify-content:center;"
            f"font:700 {11 if num_techs == 1 else 12}px {ui_preset['font_family']};"
            "box-shadow:0 6px 14px rgba(15,23,42,0.22);"
            "\">"
            f"{label}"
            "</div>"
        )

        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=360),
            tooltip=tooltip,
            icon=folium.DivIcon(
                html=marker_html,
                icon_size=(diameter, diameter),
                icon_anchor=(diameter // 2, diameter // 2),
                class_name="elevate-tech-marker",
            ),
        ).add_to(fg)

    fg.add_to(m)
    return fg


def add_anchor_site_markers(m, anchor_rows, ui_preset):
    """Add subtle anchor-site markers for anchored technicians."""
    fg = folium.FeatureGroup(name="Anchored Technician Sites", show=True, control=False)
    if not anchor_rows:
        fg.add_to(m)
        return fg

    seen_keys = set()
    for row in anchor_rows:
        key = (
            row.get("tech_name"),
            row.get("anchor_site_name"),
            float(row.get("anchor_site_lat")),
            float(row.get("anchor_site_lon")),
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        reserved = row.get("anchor_reserved_fte")
        external = row.get("external_field_fte")
        state_scope = format_state_scope(row.get("assignment_scope_states", ""))
        popup_parts = [
            f"<b>{row.get('anchor_site_name', 'Anchor site')}</b>",
            f"Anchored technician: <b>{row.get('tech_name', 'Unknown')}</b>",
        ]
        if pd.notna(reserved):
            popup_parts.append(f"Reserved duty: {float(reserved):.0%}")
        if pd.notna(external):
            popup_parts.append(f"External field capacity: {float(external):.0%}")
        if state_scope:
            popup_parts.append(f"External assignment region: {state_scope}")
        if row.get("anchor_notes"):
            popup_parts.append(str(row["anchor_notes"]))
        popup_html = "<br>".join(popup_parts)
        marker_html = (
            "<div style=\""
            "width:18px;height:18px;border-radius:999px;"
            f"background:{config.MAP_ANCHOR_MARKER_FILL};"
            f"border:1.5px solid {config.MAP_ANCHOR_MARKER_BORDER};"
            "box-shadow:0 4px 10px rgba(15,23,42,0.14);"
            "display:flex;align-items:center;justify-content:center;"
            f"color:{config.MAP_ANCHOR_MARKER_COLOR};"
            "font-size:10px;font-weight:700;"
            "\">"
            "M"
            "</div>"
        )
        folium.Marker(
            location=[float(row["anchor_site_lat"]), float(row["anchor_site_lon"])],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{row.get('anchor_site_name', 'Anchor site')}: {row.get('tech_name', '')}",
            icon=folium.DivIcon(
                html=marker_html,
                icon_size=(18, 18),
                icon_anchor=(9, 9),
                class_name="elevate-anchor-marker",
            ),
        ).add_to(fg)
    fg.add_to(m)
    return fg


def add_airport_layer(m, ui_preset, show=True):
    """Add major airport hub markers as a toggleable layer."""
    fg = folium.FeatureGroup(
        name=f"Major Airport Hubs ({len(config.MAJOR_AIRPORTS)})",
        show=show,
    )

    for airport in config.MAJOR_AIRPORTS:
        popup_html = (
            f"<b>{airport['code']} — {airport['name']}</b><br>"
            f"{airport['city']}<br>"
            f"<span style='color:#666;font-size:11px;'>Major service hub</span>"
        )
        marker_html = (
            "<div style=\""
            "width:16px;height:16px;border-radius:999px;"
            f"background:{ui_preset['airport_marker_fill']};"
            f"border:1.5px solid {ui_preset['airport_marker_border']};"
            f"box-shadow:0 3px 8px rgba(15,23,42,0.12);"
            "display:flex;align-items:center;justify-content:center;"
            f"color:{ui_preset['airport_marker_color']};"
            "font-size:9px;font-weight:700;opacity:0.92;"
            "\">"
            "A"
            "</div>"
        )
        folium.Marker(
            location=[airport["lat"], airport["lon"]],
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"{airport['code']}: {airport['name']}",
            icon=folium.DivIcon(
                html=marker_html,
                icon_size=(16, 16),
                icon_anchor=(8, 8),
                class_name="elevate-airport-marker",
            ),
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

    summary_df = summary_df.sort_values("scenario_hires").reset_index(drop=True)
    if (summary_df["scenario_hires"] == 0).any():
        baseline_cost = float(
            summary_df.loc[
                summary_df["scenario_hires"] == 0, "economic_total_with_overhead_usd"
            ].iloc[0]
        )
    else:
        baseline_cost = float(summary_df["economic_total_with_overhead_usd"].iloc[0])
    summary_df["annual_cost_delta_vs_n0_usd"] = (
        summary_df["economic_total_with_overhead_usd"] - baseline_cost
    )
    summary_df["annual_cost_delta_vs_n0_pct"] = np.where(
        baseline_cost != 0,
        (summary_df["annual_cost_delta_vs_n0_usd"] / baseline_cost) * 100.0,
        0.0,
    )
    summary_df["incremental_cost_vs_prior_usd"] = (
        summary_df["economic_total_with_overhead_usd"]
        - summary_df["economic_total_with_overhead_usd"].shift(1)
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
                "annual_cost_delta_vs_n0_usd": float(
                    row.get("annual_cost_delta_vs_n0_usd", 0)
                ),
                "annual_cost_delta_vs_n0_pct": float(
                    row.get("annual_cost_delta_vs_n0_pct", 0)
                ),
                "incremental_cost_vs_prior_usd": float(
                    row.get("incremental_cost_vs_prior_usd", 0)
                ),
                "unmet_appointments": float(row.get("unmet_appointments", 0)),
                "hire_cost_usd": float(row.get("hire_cost_usd", 0)),
                "mean_existing_utilization": float(row.get("mean_existing_utilization", 0)),
                "max_existing_utilization": float(row.get("max_existing_utilization", 0)),
                "install_units_enabled": safe_float(row, "install_units_enabled"),
                "install_revenue_enabled_usd": safe_float(row, "install_revenue_enabled_usd"),
                "install_profit_enabled_usd": safe_float(row, "install_profit_enabled_usd"),
                "net_economic_value_install_usd": safe_float(row, "net_economic_value_install_usd"),
                "break_even_install_units": safe_float(row, "break_even_install_units"),
                "roi_install_pct": safe_float(row, "roi_install_pct"),
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


def build_tech_color_map(territory_data, palette):
    """Assign a stable, high-contrast color to each assignee."""
    tech_master = territory_data["tech_master"]
    newhires_df = territory_data["newhires"]
    explicit_color_map = dict(getattr(config, "TECH_ASSIGNMENT_COLOR_MAP", {}))
    fallback_palette = palette or getattr(
        config, "TECH_ASSIGNMENT_FALLBACK_PALETTE", getattr(config, "TECH_TERRITORY_PALETTE", [])
    )

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
        if tid in explicit_color_map:
            color_map[tid] = explicit_color_map[tid]
            continue
        color_map[tid] = fallback_palette[idx % len(fallback_palette)]
        idx += 1
    for cid in sorted_candidate_ids:
        if cid not in color_map:
            if cid in explicit_color_map:
                color_map[cid] = explicit_color_map[cid]
                continue
            color_map[cid] = fallback_palette[idx % len(fallback_palette)]
            idx += 1

    return color_map


def get_assignment_dot_stroke(fill_hex, ui_preset):
    """Choose a dark stroke for light fills and a white stroke for dark fills."""
    value = str(fill_hex or "").strip().lstrip("#")
    if len(value) != 6:
        return ui_preset["assignment_dot_light_stroke"]
    try:
        red = int(value[0:2], 16)
        green = int(value[2:4], 16)
        blue = int(value[4:6], 16)
    except ValueError:
        return ui_preset["assignment_dot_light_stroke"]

    # Simple perceived brightness check keeps yellow/lime dots readable.
    brightness = (red * 299 + green * 587 + blue * 114) / 1000
    if brightness >= 165:
        return ui_preset["assignment_dot_dark_stroke"]
    return ui_preset["assignment_dot_light_stroke"]


def add_territory_assignment_layers(m, assignment_map, territory_data, tech_color_map, ui_preset):
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
            stroke_color = get_assignment_dot_stroke(color, ui_preset)
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
                    radius=ui_preset["assignment_dot_radius"],
                    color=stroke_color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=ui_preset["assignment_dot_opacity"],
                    weight=ui_preset["assignment_dot_stroke_width"],
                    opacity=0.95,
                    stroke=True,
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


def add_simulation_layers(m, simulation_payload, ui_preset):
    """Add one marker layer per simulation scenario and return layer JS names."""
    if not simulation_payload:
        return {}

    scenario_layers = {}
    ordered_keys = sorted(simulation_payload.keys(), key=lambda x: int(x))
    default_key = "0" if "0" in simulation_payload else ordered_keys[0]

    for key in ordered_keys:
        scenario = int(key)
        color = ui_preset["new_hire_marker_color"]
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
                "box-shadow:0 10px 18px rgba(15,23,42,0.28);"
                "\">&#10010;</div>"
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


def build_simulation_kpi_explanations():
    """Return the current KPI explanation copy for the map UI."""
    return {
        "total-cost": {
            "title": "Total Cost",
            "body": (
                "<p><b>What:</b> The model's annual cost for this scenario.</p>"
                "<p><b>Formula:</b> Travel Cost + Time-Zone Penalties + Hire Payroll + "
                "Configured Overhead + Any Unmet-Appointment Penalties</p>"
                "<p><b>Interpretation:</b> Lower is better in the core cost-first optimization. "
                "This is the number the solver is minimizing before the install-upside view is applied.</p>"
            ),
        },
        "cost-change": {
            "title": "Annual Cost Delta vs N=0",
            "body": (
                "<p><b>What:</b> The difference between this scenario's total cost and the N=0 baseline.</p>"
                "<p><b>Formula:</b> Total Cost(this scenario) - Total Cost(N=0)</p>"
                "<p><b>Interpretation:</b> Negative means this scenario is cheaper than N=0. "
                "Positive means it is more expensive than the no-hire baseline.</p>"
            ),
        },
        "marginal-cost": {
            "title": "Incremental Cost vs Prior Scenario",
            "body": (
                "<p><b>What:</b> The cost change from moving from the previous hire level to this one.</p>"
                "<p><b>Formula:</b> Total Cost(this scenario) - Total Cost(previous scenario)</p>"
                "<p><b>Interpretation:</b> Negative means this step is cheaper than the previous scenario. "
                "Positive means the added hire increases total cost.</p>"
            ),
        },
        "unmet": {
            "title": "Unmet Appointments",
            "body": (
                "<p><b>What:</b> Service appointments that the model could not feasibly assign.</p>"
                "<p><b>Interpretation:</b> Zero means the scenario covers the full modeled demand. "
                "A positive value means coverage, skill, or feasibility constraints left some work unserved.</p>"
            ),
        },
        "hire-payroll": {
            "title": "Incremental Hire Payroll",
            "body": (
                "<p><b>What:</b> The burdened annual payroll cost for the new hires in this scenario.</p>"
                "<p><b>Interpretation:</b> This is the company planning cost of the added headcount, "
                "not employee take-home pay.</p>"
            ),
        },
        "mean-util": {
            "title": "Mean Load Ratio",
            "body": (
                "<p><b>What:</b> Average modeled load ratio across existing technicians.</p>"
                "<p><b>Formula:</b> Mean of assigned hours / capacity hours for existing techs</p>"
                "<p><b>Interpretation:</b> This is a calendar-based capacity proxy, not literal "
                "timesheet or weekday payroll utilization.</p>"
            ),
        },
        "max-util": {
            "title": "Peak Load Ratio",
            "body": (
                "<p><b>What:</b> The highest modeled load ratio among existing technicians.</p>"
                "<p><b>Interpretation:</b> Values near 1.000 mean at least one tech is near the top "
                "of the model's normalized capacity range.</p>"
            ),
        },
        "installs": {
            "title": "Install Units Enabled",
            "body": (
                "<p><b>What:</b> Estimated patient-sim install units that freed capacity could support.</p>"
                "<p><b>Formula:</b> Freed calendar days available / weighted-average install calendar days</p>"
                "<p><b>Interpretation:</b> This is a family-weighted patient-sim install-only capacity view, "
                "not guaranteed bookings.</p>"
            ),
        },
        "install-revenue": {
            "title": "Install Revenue Enabled",
            "body": (
                "<p><b>What:</b> Estimated top-line install revenue if all enabled patient-sim installs were completed.</p>"
                "<p><b>Formula:</b> Install units enabled × weighted-average family install revenue</p>"
                "<p><b>Interpretation:</b> This uses the current forward-looking patient-sim family mix, "
                "not the old generic bucket model.</p>"
            ),
        },
        "install-profit": {
            "title": "Install Profit Enabled",
            "body": (
                "<p><b>What:</b> Estimated install gross profit from those enabled patient-sim installs.</p>"
                "<p><b>Formula:</b> Install units enabled × weighted-average family install profit</p>"
                "<p><b>Interpretation:</b> This intentionally stays install-only and does not add recurring "
                "service-contract profit to the headline map KPIs.</p>"
            ),
        },
        "net-value": {
            "title": "Net Economic Value",
            "body": (
                "<p><b>What:</b> Install profit enabled minus the net cost increase of hiring.</p>"
                "<p><b>Formula:</b> Install Profit Enabled - Net Cost Increase vs N=0</p>"
                "<p><b>Interpretation:</b> Positive means the install-upside view outweighs the added cost. "
                "Negative means it does not.</p>"
            ),
        },
        "break-even": {
            "title": "Break-Even Units",
            "body": (
                "<p><b>What:</b> The install units needed to recover the net cost increase of hiring.</p>"
                "<p><b>Formula:</b> Net Cost Increase / weighted-average install profit per unit</p>"
                "<p><b>Interpretation:</b> Lower is better because fewer installs are needed to justify the hire cost.</p>"
            ),
        },
        "roi": {
            "title": "ROI",
            "body": (
                "<p><b>What:</b> Net economic value as a percent of the scenario's net cost increase.</p>"
                "<p><b>Formula:</b> Net Economic Value / Net Cost Increase × 100%</p>"
                "<p><b>Interpretation:</b> Higher means the install-upside view produces more value per dollar "
                "of added hiring cost.</p>"
            ),
        },
    }


def build_simulation_panel_css(ui_preset):
    """Return the panel CSS for the simulation UI."""
    return f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600;700&display=swap');
      #sim-panel {{
        position: fixed;
        top: 16px;
        left: 16px;
        z-index: 1200;
        width: {int(ui_preset["panel_width_px"])}px;
        max-height: calc(100vh - 32px);
        overflow-y: auto;
        background: rgba(255, 255, 255, 0.94);
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: {int(ui_preset["panel_radius_px"])}px;
        box-shadow: {ui_preset["panel_shadow"]};
        backdrop-filter: blur(8px);
        padding: 16px;
        font-family: {ui_preset["font_family"]};
        color: #0f172a;
      }}
      #sim-panel-toggle {{
        display: none;
        position: fixed;
        top: 14px;
        left: 14px;
        z-index: 1250;
        border: 1px solid rgba(15, 23, 42, 0.12);
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.96);
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.12);
        color: #0f172a;
        padding: 8px 14px;
        font: 600 12px {ui_preset["font_family"]};
        cursor: pointer;
      }}
      .sim-header {{
        margin-bottom: 18px;
      }}
      .sim-header-top {{
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 12px;
      }}
      .sim-header-copy {{
        min-width: 0;
      }}
      .sim-eyebrow {{
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #8b5e34;
        margin-bottom: 6px;
      }}
      .sim-title {{
        margin: 0;
        font-size: 26px;
        line-height: 1.1;
        font-weight: 700;
        color: #112033;
      }}
      #sim-subtitle {{
        margin: 8px 0 0 0;
        font-size: 12px;
        color: #475569;
      }}
      .sim-chip {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        border: 1px solid rgba(15, 23, 42, 0.10);
        border-radius: 999px;
        background: #f8fafc;
        color: #475569;
        padding: 7px 10px;
        font: 600 11px {ui_preset["font_family"]};
        cursor: pointer;
        transition: background 0.15s, color 0.15s, border-color 0.15s, transform 0.15s;
        flex-shrink: 0;
      }}
      .sim-chip:hover {{
        background: #f1f5f9;
        border-color: rgba(15, 23, 42, 0.22);
        color: #102235;
      }}
      .sim-chip.active {{
        background: #e8eff4;
        border-color: #183b58;
        color: #183b58;
      }}
      .sim-chip-indicator {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #94a3b8;
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.7);
      }}
      .sim-chip.active .sim-chip-indicator {{
        background: #183b58;
      }}
      .sim-section {{
        margin-top: 16px;
      }}
      .sim-section:first-of-type {{
        margin-top: 0;
      }}
      .sim-section-label {{
        margin-bottom: 8px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #64748b;
      }}
      .sim-section-heading {{
        margin: 0;
        font-size: 14px;
        font-weight: 700;
        color: #122236;
      }}
      .sim-section-caption {{
        margin: 4px 0 0 0;
        font-size: 11px;
        color: #64748b;
      }}
      #sim-buttons {{
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 6px;
      }}
      .sim-btn {{
        border: 1px solid rgba(15, 23, 42, 0.12);
        background: #f8fafc;
        color: #102235;
        padding: 8px 0;
        border-radius: 999px;
        font: 600 12px {ui_preset["font_family"]};
        cursor: pointer;
        transition: background 0.15s, color 0.15s, border-color 0.15s, transform 0.15s;
      }}
      .sim-btn:hover {{
        border-color: rgba(17, 32, 51, 0.32);
        background: #f1f5f9;
      }}
      .sim-btn.active {{
        background: #183b58;
        border-color: #183b58;
        color: #fff;
      }}
      .sim-card-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
        margin-top: 10px;
      }}
      .sim-kpi {{
        background: #f8fafc;
        border: 1px solid #e7edf3;
        border-radius: {int(ui_preset["card_radius_px"])}px;
        padding: 10px 11px;
      }}
      .kpi-clickable {{
        cursor: pointer;
        transition: background 0.15s, border-color 0.15s, transform 0.15s;
      }}
      .kpi-clickable:hover {{
        background: #f1f5f9;
        border-color: #cbd5e1;
        transform: translateY(-1px);
      }}
      .sim-kpi .label {{
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: #64748b;
      }}
      .sim-kpi .value {{
        margin-top: 6px;
        font-size: 18px;
        font-weight: 700;
        line-height: 1.15;
        color: #102235;
      }}
      .sim-list-box {{
        margin-top: 10px;
        background: #fbfdff;
        border: 1px solid #e7edf3;
        border-radius: {int(ui_preset["card_radius_px"])}px;
        padding: 10px;
      }}
      .sim-empty {{
        font-size: 12px;
        color: #64748b;
      }}
      .sim-rec-row {{
        padding: 10px 0;
        border-top: 1px solid #edf2f7;
      }}
      .sim-rec-row:first-child {{
        padding-top: 0;
        border-top: none;
      }}
      .sim-rec-row:last-child {{
        padding-bottom: 0;
      }}
      .sim-rec-top {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 8px;
        margin-bottom: 6px;
      }}
      .sim-rec-city {{
        font-size: 13px;
        font-weight: 700;
        color: #0f172a;
      }}
      .sim-airport-chip {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 42px;
        padding: 3px 8px;
        border-radius: 999px;
        background: #eef2f7;
        color: #475569;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.04em;
      }}
      .sim-rec-stats {{
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 8px;
        font-size: 11px;
        color: #475569;
      }}
      .sim-stat-label {{
        display: block;
        font-size: 10px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.03em;
      }}
      .sim-stat-value {{
        display: block;
        margin-top: 2px;
        font-weight: 600;
        color: #102235;
      }}
      .coverage-row {{
        display: grid;
        grid-template-columns: 12px 1fr;
        gap: 10px;
        align-items: start;
        padding: 8px 0;
        border-top: 1px solid #edf2f7;
      }}
      .coverage-row:first-child {{
        padding-top: 0;
        border-top: none;
      }}
      .coverage-row:last-child {{
        padding-bottom: 0;
      }}
      .coverage-dot {{
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-top: 2px;
        border: 1px solid rgba(255, 255, 255, 0.9);
        box-shadow: 0 1px 4px rgba(15, 23, 42, 0.18);
      }}
      .coverage-name {{
        font-size: 12px;
        font-weight: 700;
        color: #102235;
      }}
      .coverage-meta {{
        margin-top: 2px;
        font-size: 11px;
        color: #64748b;
      }}
      .sim-link-btn {{
        margin-top: 8px;
        border: none;
        background: transparent;
        padding: 0;
        color: #183b58;
        font: 600 12px {ui_preset["font_family"]};
        cursor: pointer;
      }}
      .sim-link-btn:hover {{
        color: #102235;
      }}
      #sim-footnote {{
        margin-top: 16px;
        padding-top: 12px;
        border-top: 1px solid #edf2f7;
        font-size: 11px;
        line-height: 1.45;
        color: #64748b;
      }}
      #kpi-modal-overlay {{
        display: none;
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(15, 23, 42, 0.34);
        z-index: 2000;
        justify-content: center;
        align-items: center;
        padding: 20px;
      }}
      #kpi-modal-overlay[style*="display: block"] {{
        display: flex !important;
      }}
      #kpi-modal {{
        width: min(520px, 100%);
        background: #ffffff;
        border-radius: 18px;
        border: 1px solid rgba(15, 23, 42, 0.08);
        box-shadow: 0 24px 60px rgba(15, 23, 42, 0.24);
        padding: 22px 22px 18px 22px;
        position: relative;
        font: 400 13px/1.55 {ui_preset["font_family"]};
        color: #334155;
      }}
      #kpi-modal-title {{
        margin-bottom: 10px;
        padding-right: 24px;
        font-size: 17px;
        font-weight: 700;
        color: #102235;
      }}
      #kpi-modal-body p {{
        margin: 0 0 10px 0;
      }}
      #kpi-modal-close {{
        position: absolute;
        top: 12px;
        right: 14px;
        border: none;
        background: transparent;
        color: #64748b;
        font-size: 24px;
        line-height: 1;
        cursor: pointer;
      }}
      #kpi-modal-close:hover {{
        color: #0f172a;
      }}
      @media (max-width: 900px) {{
        #sim-panel {{
          display: none;
          width: min(calc(100vw - 24px), 340px);
          top: 58px;
          left: 12px;
          max-height: calc(100vh - 70px);
        }}
        #sim-panel.mobile-open {{
          display: block;
        }}
        #sim-panel-toggle {{
          display: inline-flex;
          align-items: center;
          justify-content: center;
        }}
      }}
    </style>
    """


def build_simulation_panel_markup():
    """Return the panel HTML shell."""
    return """
    <button id="sim-panel-toggle" title="Show or hide scenarios">Scenarios</button>
    <div id="sim-panel">
      <div class="sim-header">
        <div class="sim-header-top">
          <div class="sim-header-copy">
            <div class="sim-eyebrow">Elevate Healthcare</div>
            <h2 class="sim-title">Service territory scenarios</h2>
            <p id="sim-subtitle">Cost-first optimization &middot; annualized results</p>
          </div>
          <button id="sim-hub-toggle" class="sim-chip" type="button" style="display:none;">
            <span class="sim-chip-indicator" aria-hidden="true"></span>
            <span id="sim-hub-toggle-label">Flight hubs</span>
          </button>
        </div>
      </div>

      <div class="sim-section">
        <div class="sim-section-label">Scenarios</div>
        <div id="sim-buttons"></div>
      </div>

      <div class="sim-section">
        <h3 class="sim-section-heading">Cost &amp; Load</h3>
        <div id="sim-kpis" class="sim-card-grid">
          <div class="sim-kpi kpi-clickable" data-kpi="total-cost">
            <div class="label">Total Cost</div>
            <div class="value" id="kpi-total">&mdash;</div>
          </div>
          <div class="sim-kpi kpi-clickable" data-kpi="cost-change">
            <div class="label">Annual Cost Delta vs N=0</div>
            <div class="value" id="kpi-annual-delta">&mdash;</div>
          </div>
          <div class="sim-kpi kpi-clickable" data-kpi="marginal-cost">
            <div class="label">Incremental Cost vs Prior Scenario</div>
            <div class="value" id="kpi-incremental">&mdash;</div>
          </div>
          <div class="sim-kpi kpi-clickable" data-kpi="hire-payroll">
            <div class="label">Incremental Hire Payroll</div>
            <div class="value" id="kpi-hire-cost">&mdash;</div>
          </div>
          <div class="sim-kpi kpi-clickable" data-kpi="mean-util">
            <div class="label">Mean Load Ratio</div>
            <div class="value" id="kpi-mean-load">&mdash;</div>
          </div>
          <div class="sim-kpi kpi-clickable" data-kpi="max-util">
            <div class="label">Peak Load Ratio</div>
            <div class="value" id="kpi-peak-load">&mdash;</div>
          </div>
          <div class="sim-kpi kpi-clickable" data-kpi="unmet" id="kpi-unmet-card">
            <div class="label">Unmet Appointments</div>
            <div class="value" id="kpi-unmet">&mdash;</div>
          </div>
        </div>
      </div>

      <div class="sim-section">
        <h3 class="sim-section-heading">Install Upside</h3>
        <p class="sim-section-caption">Family-weighted patient-sim install-only view</p>
        <div id="sim-kpis-revenue" class="sim-card-grid">
          <div class="sim-kpi kpi-clickable" data-kpi="installs">
            <div class="label">Install Units Enabled</div>
            <div class="value" id="kpi-install-units">&mdash;</div>
          </div>
          <div class="sim-kpi kpi-clickable" data-kpi="install-revenue">
            <div class="label">Install Revenue Enabled</div>
            <div class="value" id="kpi-install-revenue">&mdash;</div>
          </div>
          <div class="sim-kpi kpi-clickable" data-kpi="install-profit">
            <div class="label">Install Profit Enabled</div>
            <div class="value" id="kpi-install-profit">&mdash;</div>
          </div>
          <div class="sim-kpi kpi-clickable" data-kpi="net-value">
            <div class="label">Net Economic Value</div>
            <div class="value" id="kpi-net-value">&mdash;</div>
          </div>
          <div class="sim-kpi kpi-clickable" data-kpi="break-even">
            <div class="label">Break-Even Units</div>
            <div class="value" id="kpi-break-even">&mdash;</div>
          </div>
          <div class="sim-kpi kpi-clickable" data-kpi="roi">
            <div class="label">ROI</div>
            <div class="value" id="kpi-roi">&mdash;</div>
          </div>
        </div>
      </div>

      <div class="sim-section">
        <h3 class="sim-section-heading">Recommended Bases</h3>
        <div id="sim-recs" class="sim-list-box">
          <div class="sim-empty">No new-hire placements in this scenario.</div>
        </div>
      </div>

      <div class="sim-section" id="sim-coverage-section" style="display:none;">
        <h3 class="sim-section-heading">Coverage Assignments</h3>
        <p class="sim-section-caption">Top assignees stay visible so the dot colors remain readable.</p>
        <div id="sim-tech-legend" class="sim-list-box"></div>
        <button id="sim-tech-toggle" class="sim-link-btn" style="display:none;">Show all</button>
      </div>

      <div id="sim-footnote">
        Selected scenario results from the cost-first optimization. Load ratios are modeled
        capacity proxies. Install cards show family-weighted patient-sim install-only upside.
      </div>
    </div>

    <div id="kpi-modal-overlay">
      <div id="kpi-modal">
        <button id="kpi-modal-close" aria-label="Close explanation">&times;</button>
        <div id="kpi-modal-title"></div>
        <div id="kpi-modal-body"></div>
      </div>
    </div>
    """


def build_simulation_panel_script(
    map_var,
    payload_js,
    layer_js,
    territory_dots_js,
    tech_colors_js,
    airport_layer_name,
    ordered_keys,
    default_key,
    ui_preset,
):
    """Return the panel JavaScript."""
    explanations_js = json.dumps(build_simulation_kpi_explanations())
    return f"""
    <script>
    (function() {{
      const mapVarName = "{map_var}";
      const scenarioData = {payload_js};
      const scenarioLayerNames = {layer_js};
      const territoryDotLayerNames = {territory_dots_js};
      const techColors = {tech_colors_js};
      const airportLayerName = {json.dumps(airport_layer_name)};
      const orderedScenarios = {json.dumps(ordered_keys)};
      const defaultScenario = "{default_key}";
      const coverageDefaultCount = {int(ui_preset["coverage_default_count"])};
      const lightDotStroke = {json.dumps(ui_preset["assignment_dot_light_stroke"])};
      const darkDotStroke = {json.dumps(ui_preset["assignment_dot_dark_stroke"])};
      const showHubToggle = {json.dumps(bool(ui_preset["show_hub_toggle"]))};
      const hubToggleLabel = {json.dumps(ui_preset["hub_toggle_label"])};
      const airportDefaultVisible = {json.dumps(bool(ui_preset["airport_default_visible"]))};
      const kpiExplanations = {explanations_js};
      const showUnmetKpi = orderedScenarios.some((s) =>
        Number(((scenarioData[s] || {{}}).kpis || {{}}).unmet_appointments || 0) > 0
      );
      const hasTerritory = Object.keys(territoryDotLayerNames).length > 0;
      const coverageExpandedByScenario = {{}};
      let mapRef = null;
      let scenarioLayers = {{}};
      let territoryDotLayers = {{}};
      let airportLayer = null;
      let airportVisible = airportDefaultVisible;

      function money(v) {{
        const n = Math.abs(Number(v || 0));
        return "$" + n.toLocaleString(undefined, {{ maximumFractionDigits: 0 }});
      }}

      function signedMoney(v) {{
        const n = Number(v || 0);
        if (n === 0) return "$0";
        return (n > 0 ? "+" : "-") + money(n);
      }}

      function pct(v, digits) {{
        const n = Number(v || 0);
        return n.toFixed(digits === undefined ? 2 : digits) + "%";
      }}

      function signedPct(v, digits) {{
        const n = Number(v || 0);
        if (n === 0) return "0.00%";
        return (n > 0 ? "+" : "-") + Math.abs(n).toFixed(digits === undefined ? 2 : digits) + "%";
      }}

      function getDotStroke(fillHex) {{
        const raw = String(fillHex || "").replace("#", "");
        if (raw.length !== 6) return lightDotStroke;
        const red = parseInt(raw.slice(0, 2), 16);
        const green = parseInt(raw.slice(2, 4), 16);
        const blue = parseInt(raw.slice(4, 6), 16);
        if ([red, green, blue].some(Number.isNaN)) return lightDotStroke;
        const brightness = (red * 299 + green * 587 + blue * 114) / 1000;
        return brightness >= 165 ? darkDotStroke : lightDotStroke;
      }}

      function renderButtons() {{
        const container = document.getElementById("sim-buttons");
        if (!container) return;
        container.innerHTML = "";
        orderedScenarios.forEach((s) => {{
          const button = document.createElement("button");
          button.className = "sim-btn";
          button.textContent = "N=" + s;
          button.setAttribute("data-scenario", s);
          button.addEventListener("click", () => showScenario(s));
          container.appendChild(button);
        }});
      }}

      function setActiveButton(scenario) {{
        document.querySelectorAll(".sim-btn").forEach((button) => {{
          button.classList.toggle("active", button.getAttribute("data-scenario") === scenario);
        }});
      }}

      function updateHubToggleUi() {{
        const toggle = document.getElementById("sim-hub-toggle");
        const label = document.getElementById("sim-hub-toggle-label");
        if (!toggle || !label) return;
        if (!showHubToggle || !airportLayer) {{
          toggle.style.display = "none";
          return;
        }}
        toggle.style.display = "inline-flex";
        label.textContent = hubToggleLabel;
        toggle.classList.toggle("active", !!airportVisible);
        toggle.setAttribute("aria-pressed", airportVisible ? "true" : "false");
      }}

      function setSignedValueColor(element, value) {{
        if (!element) return;
        if (Number(value || 0) > 0) {{
          element.style.color = "#b42318";
        }} else if (Number(value || 0) < 0) {{
          element.style.color = "#1f7a40";
        }} else {{
          element.style.color = "#102235";
        }}
      }}

      function renderKpis(scenario) {{
        const item = scenarioData[scenario];
        if (!item) return;
        const k = item.kpis || {{}};

        const totalEl = document.getElementById("kpi-total");
        const annualDeltaEl = document.getElementById("kpi-annual-delta");
        const incrementalEl = document.getElementById("kpi-incremental");
        const hireEl = document.getElementById("kpi-hire-cost");
        const meanLoadEl = document.getElementById("kpi-mean-load");
        const peakLoadEl = document.getElementById("kpi-peak-load");
        const unmetEl = document.getElementById("kpi-unmet");
        const installUnitsEl = document.getElementById("kpi-install-units");
        const installRevenueEl = document.getElementById("kpi-install-revenue");
        const installProfitEl = document.getElementById("kpi-install-profit");
        const netValueEl = document.getElementById("kpi-net-value");
        const breakEvenEl = document.getElementById("kpi-break-even");
        const roiEl = document.getElementById("kpi-roi");

        totalEl.textContent = money(k.economic_total_with_overhead_usd);
        annualDeltaEl.textContent = signedMoney(k.annual_cost_delta_vs_n0_usd) + " (" + signedPct(k.annual_cost_delta_vs_n0_pct, 2) + ")";
        incrementalEl.textContent = signedMoney(k.incremental_cost_vs_prior_usd);
        hireEl.textContent = money(k.hire_cost_usd);
        meanLoadEl.textContent = Number(k.mean_existing_utilization || 0).toFixed(3);
        peakLoadEl.textContent = Number(k.max_existing_utilization || 0).toFixed(3);

        setSignedValueColor(annualDeltaEl, k.annual_cost_delta_vs_n0_usd);
        setSignedValueColor(incrementalEl, k.incremental_cost_vs_prior_usd);

        if (showUnmetKpi && unmetEl) {{
          unmetEl.textContent = Number(k.unmet_appointments || 0).toFixed(1);
        }}

        const installUnits = Number(k.install_units_enabled || 0);
        const hasInstallUpside = installUnits > 0;
        installUnitsEl.textContent = hasInstallUpside ? installUnits.toFixed(1) : "—";
        installRevenueEl.textContent = hasInstallUpside ? money(k.install_revenue_enabled_usd) : "—";
        installProfitEl.textContent = hasInstallUpside ? money(k.install_profit_enabled_usd) : "—";
        netValueEl.textContent = hasInstallUpside ? signedMoney(k.net_economic_value_install_usd) : "—";
        breakEvenEl.textContent = hasInstallUpside ? Number(k.break_even_install_units || 0).toFixed(1) : "—";
        roiEl.textContent = hasInstallUpside ? pct(k.roi_install_pct, 0) : "—";
        setSignedValueColor(netValueEl, k.net_economic_value_install_usd);
      }}

      function renderRecommendations(scenario) {{
        const recWrap = document.getElementById("sim-recs");
        if (!recWrap) return;
        const item = scenarioData[scenario];
        const recs = item ? (item.placements || []) : [];
        if (!recs.length) {{
          recWrap.innerHTML = '<div class="sim-empty">No new-hire placements in this scenario.</div>';
          return;
        }}
        recWrap.innerHTML = recs.map((r) => {{
          return `
            <div class="sim-rec-row">
              <div class="sim-rec-top">
                <div class="sim-rec-city">${{r.city}}, ${{r.state}}</div>
                <span class="sim-airport-chip">${{r.airport_iata}}</span>
              </div>
              <div class="sim-rec-stats">
                <div>
                  <span class="sim-stat-label">Hires</span>
                  <span class="sim-stat-value">${{Math.round(r.hires_allocated || 0)}}</span>
                </div>
                <div>
                  <span class="sim-stat-label">Hours</span>
                  <span class="sim-stat-value">${{Number(r.assigned_hours || 0).toFixed(1)}}</span>
                </div>
                <div>
                  <span class="sim-stat-label">Appointments</span>
                  <span class="sim-stat-value">${{Number(r.assigned_appointments || 0).toFixed(1)}}</span>
                </div>
              </div>
            </div>`;
        }}).join("");
      }}

      function renderCoverageAssignments(scenario) {{
        const sectionEl = document.getElementById("sim-coverage-section");
        const listEl = document.getElementById("sim-tech-legend");
        const toggleEl = document.getElementById("sim-tech-toggle");
        if (!sectionEl || !listEl || !toggleEl) return;
        if (!hasTerritory) {{
          sectionEl.style.display = "none";
          return;
        }}

        const item = scenarioData[scenario];
        const stats = item ? (item.tech_stats || {{}}) : {{}};
        const entries = Object.entries(stats).sort(
          (a, b) => Number(b[1].appointments || 0) - Number(a[1].appointments || 0)
        );

        if (!entries.length) {{
          sectionEl.style.display = "none";
          return;
        }}

        sectionEl.style.display = "block";
        const expanded = !!coverageExpandedByScenario[scenario];
        const visibleEntries = expanded ? entries : entries.slice(0, coverageDefaultCount);

        listEl.innerHTML = visibleEntries.map(([techId, stat]) => {{
          const color = techColors[techId] || "#64748b";
          const stroke = getDotStroke(color);
          const base = stat.base ? String(stat.base) : "Base unavailable";
          return `
            <div class="coverage-row">
              <span class="coverage-dot" style="background:${{color}}; border-color:${{stroke}};"></span>
              <div>
                <div class="coverage-name">${{stat.name}}</div>
                <div class="coverage-meta">${{Number(stat.appointments || 0).toFixed(0)}} appointments &middot; ${{base}}</div>
              </div>
            </div>`;
        }}).join("");

        if (entries.length > coverageDefaultCount) {{
          toggleEl.style.display = "inline-block";
          toggleEl.textContent = expanded
            ? "Show fewer"
            : `Show all ${{entries.length}} assignees`;
        }} else {{
          toggleEl.style.display = "none";
        }}
      }}

      function syncAirportLayerVisibility() {{
        if (!mapRef || !airportLayer) return;
        if (airportVisible) {{
          if (!mapRef.hasLayer(airportLayer)) {{
            mapRef.addLayer(airportLayer);
          }}
        }} else if (mapRef.hasLayer(airportLayer)) {{
          mapRef.removeLayer(airportLayer);
        }}
        updateHubToggleUi();
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

        const targetLayer = scenarioLayers[scenario];
        if (targetLayer && !mapRef.hasLayer(targetLayer)) {{
          mapRef.addLayer(targetLayer);
        }}
        const dotTarget = territoryDotLayers[scenario];
        if (dotTarget && !mapRef.hasLayer(dotTarget)) {{
          mapRef.addLayer(dotTarget);
        }}

        setActiveButton(scenario);
        renderKpis(scenario);
        renderRecommendations(scenario);
        renderCoverageAssignments(scenario);
      }}

      function resolveLayers() {{
        mapRef = window[mapVarName] || null;
        if (!mapRef) {{
          return orderedScenarios.length + 1;
        }}

        const resolved = {{}};
        let missing = 0;
        orderedScenarios.forEach((scenario) => {{
          const layerVar = scenarioLayerNames[scenario];
          const layerObj = layerVar ? window[layerVar] : null;
          if (layerObj) {{
            resolved[scenario] = layerObj;
          }} else {{
            missing += 1;
          }}
        }});
        scenarioLayers = resolved;

        orderedScenarios.forEach((scenario) => {{
          const dotVar = territoryDotLayerNames[scenario];
          if (dotVar && window[dotVar]) {{
            territoryDotLayers[scenario] = window[dotVar];
          }}
        }});

        airportLayer = airportLayerName ? window[airportLayerName] || null : null;
        airportVisible = airportLayer ? mapRef.hasLayer(airportLayer) : false;

        return missing;
      }}

      function wireMobileToggle() {{
        const button = document.getElementById("sim-panel-toggle");
        const panel = document.getElementById("sim-panel");
        if (!button || !panel) return;
        button.addEventListener("click", () => {{
          panel.classList.toggle("mobile-open");
        }});
      }}

      function wireCoverageToggle() {{
        const button = document.getElementById("sim-tech-toggle");
        if (!button) return;
        button.addEventListener("click", () => {{
          const active = document.querySelector(".sim-btn.active");
          if (!active) return;
          const scenario = active.getAttribute("data-scenario");
          coverageExpandedByScenario[scenario] = !coverageExpandedByScenario[scenario];
          renderCoverageAssignments(scenario);
        }});
      }}

      function wireHubToggle() {{
        const toggle = document.getElementById("sim-hub-toggle");
        if (!toggle) return;
        updateHubToggleUi();
        toggle.addEventListener("click", () => {{
          if (!airportLayer) return;
          airportVisible = !airportVisible;
          syncAirportLayerVisibility();
        }});
      }}

      function wireKpiModal() {{
        document.addEventListener("click", (event) => {{
          const card = event.target.closest(".kpi-clickable");
          if (card) {{
            const kpiKey = card.getAttribute("data-kpi");
            const info = kpiExplanations[kpiKey];
            if (info) {{
              document.getElementById("kpi-modal-title").textContent = info.title;
              document.getElementById("kpi-modal-body").innerHTML = info.body;
              document.getElementById("kpi-modal-overlay").style.display = "block";
            }}
            return;
          }}
          const overlay = document.getElementById("kpi-modal-overlay");
          if (event.target === overlay || event.target.id === "kpi-modal-close") {{
            overlay.style.display = "none";
          }}
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
            recWrap.innerHTML = '<div class="sim-empty">Scenario layers unavailable in this map build.</div>';
          }}
          return;
        }}

        renderButtons();
        const unmetCard = document.getElementById("kpi-unmet-card");
        if (unmetCard && !showUnmetKpi) {{
          unmetCard.style.display = "none";
        }}
        wireMobileToggle();
        wireCoverageToggle();
        wireHubToggle();
        wireKpiModal();
        showScenario(defaultScenario);
        syncAirportLayerVisibility();
      }}

      initWhenReady(80);
    }})();
    </script>
    """


def add_simulation_panel(
    m,
    simulation_payload,
    scenario_layer_names,
    territory_layer_names=None,
    tech_color_map=None,
    airport_layer_name=None,
    ui_preset=None,
):
    """Inject scenario controls and KPI cards into the map page."""
    if not simulation_payload or not scenario_layer_names:
        return

    ui_preset = ui_preset or get_map_ui_preset()
    map_var = m.get_name()
    ordered_keys = sorted(simulation_payload.keys(), key=lambda x: int(x))
    default_key = "0" if "0" in simulation_payload else ordered_keys[0]

    layer_entries = [f'"{key}": "{scenario_layer_names[key]}"' for key in ordered_keys]
    layer_js = "{\n" + ",\n".join(layer_entries) + "\n}"
    payload_js = json.dumps(simulation_payload)

    territory_dots_entries = []
    tech_colors_js = json.dumps(tech_color_map) if tech_color_map else "{}"
    if territory_layer_names:
        for key in ordered_keys:
            info = territory_layer_names.get(key)
            if info:
                territory_dots_entries.append(f'"{key}": "{info["dots_layer"]}"')
    territory_dots_js = (
        "{\n" + ",\n".join(territory_dots_entries) + "\n}"
        if territory_dots_entries
        else "{}"
    )

    panel_html = (
        build_simulation_panel_css(ui_preset)
        + build_simulation_panel_markup()
        + build_simulation_panel_script(
            map_var,
            payload_js,
            layer_js,
            territory_dots_js,
            tech_colors_js,
            airport_layer_name,
            ordered_keys,
            default_key,
            ui_preset,
        )
    )
    m.get_root().html.add_child(folium.Element(panel_html))


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
    ui_preset = get_map_ui_preset()

    print("Loading processed data...")
    appts = pd.read_csv(config.GEOCODED_APPTS_CSV)
    techs = pd.read_csv(config.GEOCODED_TECHS_CSV)
    anchor_metadata = load_anchor_tech_metadata()
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
    if anchor_metadata["anchors"]:
        print(f"  Anchored technician sites: {len(anchor_metadata['anchors'])}")

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
    airport_layer = None

    if ui_preset["show_active_assets"]:
        print("Adding active contract choropleth...")
        active_layer = add_active_contracts_choropleth(
            m,
            config.TERRITORIES_GEOJSON,
            territory_summary,
            layer_name=layer_active_name,
        )

        print("Adding matched install base markers...")
        add_matched_install_markers(
            m,
            install_matched,
            fg=active_layer,
            layer_name=layer_active_name,
        )

    if ui_preset["show_nonactive_assets"]:
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
    # Stakeholder mode removes this from the visible UI, but debug mode still supports it.
    if ui_preset["show_service_appointments"]:
        show_static_appts = territory_data is None
        print("Adding service appointments...")
        add_service_appointments(m, appts, layer_name=layer_appt_name, show=show_static_appts)

    # Layer 4: Technician Home Bases
    print("Adding technician markers...")
    add_technician_markers(
        m,
        techs,
        layer_name=layer_tech_name,
        ui_preset=ui_preset,
        anchor_metadata_by_name=anchor_metadata["by_name"],
    )

    if anchor_metadata["anchors"]:
        print("Adding anchored technician sites...")
        add_anchor_site_markers(m, anchor_metadata["anchors"], ui_preset=ui_preset)

    # Layer 5: Territory Boundaries
    if ui_preset["show_territory_boundaries"]:
        print("Adding territory boundaries...")
        add_territory_boundaries(m, config.TERRITORIES_GEOJSON, layer_name=layer_territory_name)

    # Layer 6: Airport Hubs
    if ui_preset["include_airports"]:
        print("Adding airport hubs...")
        airport_layer = add_airport_layer(
            m,
            ui_preset=ui_preset,
            show=ui_preset["airport_default_visible"],
        )

    # Layer 7: Hub Dispatch Radius Circles
    if ui_preset["show_hub_radius"]:
        print("Adding hub dispatch radius circles (300mi / 5hr drive)...")
        add_hub_radius_circles(m)

    # Legends
    if ui_preset["show_legends"]:
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
                tech_color_map = build_tech_color_map(
                    territory_data, ui_preset["territory_palette"]
                )
                total_assigned = sum(len(v) for v in assignment_map.values())
                print(f"  Resolved {total_assigned} total appointment assignments across {len(assignment_map)} scenarios")
                print("  Generating territory dot layers...")
                territory_layer_info = add_territory_assignment_layers(
                    m, assignment_map, territory_data, tech_color_map, ui_preset
                )
                # Inject tech_stats into per-scenario payload entries for JS
                for key, info in territory_layer_info.items():
                    if key in simulation_payload:
                        simulation_payload[key]["tech_stats"] = info["tech_stats"]
                print(f"  Territory layers added for scenarios: {list(territory_layer_info.keys())}")
            else:
                print("  Territory assignment data not available; territory layers skipped.")

            scenario_layer_names = add_simulation_layers(m, simulation_payload, ui_preset)
            add_simulation_panel(
                m, simulation_payload, scenario_layer_names,
                territory_layer_names=territory_layer_info,
                tech_color_map=tech_color_map if territory_data else None,
                airport_layer_name=airport_layer.get_name() if airport_layer else None,
                ui_preset=ui_preset,
            )
            print(f"  Loaded scenarios: {', '.join(sorted(simulation_payload.keys(), key=lambda x: int(x) if x.isdigit() else 999))}")
        else:
            add_simulation_unavailable_notice(m)
            print("  Simulation outputs not found; panel disabled.")

    # Layer control
    if ui_preset["show_layer_control"]:
        folium.LayerControl(collapsed=False).add_to(m)

    # Save
    m.save(config.MAP_OUTPUT)
    file_size = os.path.getsize(config.MAP_OUTPUT) / (1024 * 1024)
    print(f"\nMap saved: {config.MAP_OUTPUT}")
    print(f"File size: {file_size:.1f} MB")
    print("Step 5 complete.")


if __name__ == "__main__":
    main()
