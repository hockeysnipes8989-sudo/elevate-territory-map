import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import folium
import pandas as pd


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import config


def load_module(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


step05 = load_module("step05_map_heat", "05_generate_map.py")


class HeatMapViewTests(unittest.TestCase):
    def test_load_heat_map_data_uses_optimization_scope_rows_with_valid_coords(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pd.DataFrame(
                [
                    {
                        "appointment_id": "A1",
                        "country": "USA",
                        "lat": 32.77,
                        "lon": -96.79,
                        "duration_hours": 8.0,
                    },
                    {
                        "appointment_id": "A2",
                        "country": "Canada",
                        "lat": 43.65,
                        "lon": -79.38,
                        "duration_hours": 6.0,
                    },
                    {
                        "appointment_id": "A3",
                        "country": "USA",
                        "lat": None,
                        "lon": -84.39,
                        "duration_hours": 5.0,
                    },
                    {
                        "appointment_id": "A4",
                        "country": "USA",
                        "lat": 33.75,
                        "lon": -84.39,
                        "duration_hours": 12.0,
                    },
                ]
            ).to_csv(root / "demand_appointments.csv", index=False)

            old_opt_dir = config.OPTIMIZATION_DIR
            try:
                config.OPTIMIZATION_DIR = str(root)
                heat_df = step05.load_heat_map_data()
            finally:
                config.OPTIMIZATION_DIR = old_opt_dir

            self.assertIsNotNone(heat_df)
            self.assertEqual(len(heat_df), 2)
            self.assertTrue((heat_df["country"] == "USA").all())
            self.assertTrue(heat_df["lat"].notna().all())
            self.assertTrue(heat_df["lon"].notna().all())
            self.assertEqual(
                heat_df.sort_values("appointment_id")["appointment_id"].tolist(),
                ["A1", "A4"],
            )

    def test_add_heat_map_layers_returns_total_appointments_from_loaded_rows(self):
        heat_df = pd.DataFrame(
            [
                {"appointment_id": "A1", "lat": 32.77, "lon": -96.79, "duration_hours": 8.0},
                {"appointment_id": "A2", "lat": 33.75, "lon": -84.39, "duration_hours": 12.0},
                {"appointment_id": "A3", "lat": 41.88, "lon": -87.63, "duration_hours": 6.0},
            ]
        )
        m = folium.Map(location=[39.5, -98.35], zoom_start=4)

        layer_meta = step05.add_heat_map_layers(m, heat_df, step05.get_map_ui_preset())

        self.assertIsNotNone(layer_meta)
        self.assertEqual(layer_meta["total_appointments"], len(heat_df))
        self.assertIn("count_layer", layer_meta)
        self.assertIn("dots_layer", layer_meta)

    def test_load_operational_zone_data_uses_saved_demand_fields_and_code_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pd.DataFrame(
                [
                    {
                        "appointment_id": "A1",
                        "country": "USA",
                        "lat": 34.05,
                        "lon": -118.24,
                        "nearest_hub_airport": "LAX",
                        "nearest_hub_operational_zone_label": "",
                        "nearest_hub_operational_zone_rank": "",
                    },
                    {
                        "appointment_id": "A2",
                        "country": "USA",
                        "lat": 39.29,
                        "lon": -76.61,
                        "nearest_hub_airport": "BWI",
                        "nearest_hub_operational_zone_label": "Eastern",
                        "nearest_hub_operational_zone_rank": 0,
                    },
                    {
                        "appointment_id": "A3",
                        "country": "Canada",
                        "lat": 43.65,
                        "lon": -79.38,
                        "nearest_hub_airport": "YYZ",
                    },
                ]
            ).to_csv(root / "demand_appointments.csv", index=False)

            old_opt_dir = config.OPTIMIZATION_DIR
            try:
                config.OPTIMIZATION_DIR = str(root)
                zone_data = step05.load_operational_zone_data()
            finally:
                config.OPTIMIZATION_DIR = old_opt_dir

            self.assertIsNotNone(zone_data)
            self.assertEqual(zone_data["panel"]["plotted_appointments"], 2)
            appointments = zone_data["appointments"].sort_values("appointment_id")
            self.assertEqual(
                appointments["nearest_hub_operational_zone_label"].tolist(),
                ["Pacific", "Eastern"],
            )
            self.assertEqual(
                appointments["nearest_hub_operational_zone_rank"].astype(int).tolist(),
                [3, 0],
            )
            self.assertEqual(
                [zone["label"] for zone in zone_data["panel"]["zones"]],
                list(config.OPERATIONAL_ZONE_DEFINITIONS.keys()),
            )

    def test_simulation_panel_markup_and_script_include_heat_and_operational_zone_tabs(self):
        markup = step05.build_simulation_panel_markup()
        optimized_idx = markup.index('data-view="optimized"')
        blank_idx = markup.index('data-view="blank"')
        historical_idx = markup.index('data-view="historical"')
        zones_idx = markup.index('data-view="zones"')
        heat_idx = markup.index('data-view="heat"')

        self.assertLess(optimized_idx, blank_idx)
        self.assertLess(blank_idx, historical_idx)
        self.assertLess(historical_idx, zones_idx)
        self.assertLess(zones_idx, heat_idx)
        self.assertLess(historical_idx, heat_idx)
        self.assertIn('id="heat-plotted-count"', markup)
        self.assertIn('id="zones-legend"', markup)
        self.assertIn('data-view="zones"', markup)

        script = step05.build_simulation_panel_script(
            "map_ref",
            '{"0":{"placements":[],"kpis":{}}}',
            '{"0":"scenario_layer"}',
            "{}",
            "{}",
            None,
            ["0"],
            "0",
            step05.get_map_ui_preset(),
            heat_map_layer_names_js='{"dots_layer":"heat_dots","count_layer":"heat_count","total_appointments":1466}',
            operational_zones_payload_js='{"zones":[{"label":"Eastern","rank":0,"color":"#2563EB","airport_count":10,"demand_count":5}],"plotted_airports":65,"plotted_appointments":2,"employee_two_zone_penalty_usd":450.0,"contractor_two_zone_penalty_usd":250.0,"contractor_three_plus_zone_penalty_usd":1500.0}',
            operational_zones_layer_names_js='{"airports_layer":"zones_airports","appointments_layer":"zones_appts","plotted_airports":65,"plotted_appointments":2}',
        )
        self.assertIn('const heatLayerNames = {"dots_layer":"heat_dots","count_layer":"heat_count","total_appointments":1466};', script)
        self.assertIn('const operationalZonesData = {"zones":[{"label":"Eastern","rank":0,"color":"#2563EB","airport_count":10,"demand_count":5}],"plotted_airports":65,"plotted_appointments":2,"employee_two_zone_penalty_usd":450.0,"contractor_two_zone_penalty_usd":250.0,"contractor_three_plus_zone_penalty_usd":1500.0};', script)
        self.assertIn('const operationalZoneLayerNames = {"airports_layer":"zones_airports","appointments_layer":"zones_appts","plotted_airports":65,"plotted_appointments":2};', script)
        self.assertIn('renderHeatSummary();', script)
        self.assertIn('renderOperationalZonesSummary();', script)
        self.assertIn('Pacific to Central is 2 jumps because Pacific = 3 and Central = 1.', script)
        self.assertIn('data-view="heat"', markup)


if __name__ == "__main__":
    unittest.main()
