import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

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


step05 = load_module("step05_map_territories", "05_generate_map.py")
step12 = load_module("step12_optimize_territories", "12_optimize_territories.py")


class OptimizeTerritoriesViewTests(unittest.TestCase):
    def test_build_optimize_territories_color_map_uses_dedicated_palette(self):
        territory_data = {
            "tech_master": pd.DataFrame(
                [
                    {
                        "tech_id": "ben_walker",
                        "tech_name": "Ben Walker",
                        "availability_fte": 1.0,
                    },
                    {
                        "tech_id": "scott_fogo",
                        "tech_name": "Scott Fogo",
                        "availability_fte": 1.0,
                    },
                    {
                        "tech_id": "curt_corder",
                        "tech_name": "Curt Corder",
                        "availability_fte": 1.0,
                    },
                ]
            )
        }

        color_map = step05.build_optimize_territories_color_map(territory_data)

        self.assertEqual(color_map["ben_walker"], config.OPTIMIZE_TERRITORIES_COLOR_MAP["ben_walker"])
        self.assertEqual(color_map["scott_fogo"], config.OPTIMIZE_TERRITORIES_COLOR_MAP["scott_fogo"])
        self.assertEqual(color_map["curt_corder"], config.OPTIMIZE_TERRITORIES_COLOR_MAP["curt_corder"])

    def test_filter_optimize_territories_marker_techs_hides_only_requested_tab_markers(self):
        techs = pd.DataFrame(
            [
                {"name": "Ben Walker", "lat": 1.0, "lon": 1.0},
                {"name": "Scott Fogo", "lat": 1.0, "lon": 1.0},
                {"name": "Curt Corder", "lat": 1.0, "lon": 1.0},
                {"name": "Damion Lyn", "lat": 1.0, "lon": 1.0},
                {"name": "Elier Martin", "lat": 1.0, "lon": 1.0},
                {"name": "Hakim Mouazer", "lat": 1.0, "lon": 1.0},
                {"name": "HTX Contractor Alex", "lat": 1.0, "lon": 1.0},
                {"name": "HTX Contractor Robert", "lat": 1.0, "lon": 1.0},
                {"name": "James Sanchez", "lat": 1.0, "lon": 1.0},
            ]
        )

        filtered = step05.filter_optimize_territories_marker_techs(techs)

        self.assertEqual(
            filtered["name"].tolist(),
            ["Ben Walker", "Scott Fogo", "James Sanchez"],
        )

    def test_filter_roster_excludes_requested_people_and_keeps_ben_scott_and_curt(self):
        tech_df = pd.DataFrame(
            [
                {
                    "tech_id": "ben_walker",
                    "tech_name": "Ben Walker",
                    "employment_type": "fte",
                    "base_country": "USA",
                    "availability_fte": 1.0,
                },
                {
                    "tech_id": "scott_fogo",
                    "tech_name": "Scott Fogo",
                    "employment_type": "fte",
                    "base_country": "USA",
                    "availability_fte": 1.0,
                },
                {
                    "tech_id": "curt_corder",
                    "tech_name": "Curt Corder",
                    "employment_type": "fte",
                    "base_country": "USA",
                    "availability_fte": 1.0,
                },
                {
                    "tech_id": "james_sanchez",
                    "tech_name": "James Sanchez",
                    "employment_type": "fte",
                    "base_country": "USA",
                    "availability_fte": 1.0,
                },
                {
                    "tech_id": "elier_martin",
                    "tech_name": "Elier Martin",
                    "employment_type": "fte",
                    "base_country": "USA",
                    "availability_fte": 0.1,
                },
                {
                    "tech_id": "damion_lyn",
                    "tech_name": "Damion Lyn",
                    "employment_type": "fte",
                    "base_country": "USA",
                    "availability_fte": 0.2,
                },
                {
                    "tech_id": "hakim_mouazer",
                    "tech_name": "Hakim Mouazer",
                    "employment_type": "fte",
                    "base_country": "Canada",
                    "availability_fte": 1.0,
                },
                {
                    "tech_id": "alex_contractor",
                    "tech_name": "Alex Contractor",
                    "employment_type": "contractor",
                    "base_country": "USA",
                    "availability_fte": 1.0,
                },
            ]
        )

        kept, excluded = step12.filter_roster(tech_df)

        self.assertEqual(
            kept.sort_values("tech_name")["tech_name"].tolist(),
            ["Ben Walker", "Curt Corder", "Scott Fogo"],
        )
        self.assertIn("James Sanchez", excluded)
        self.assertIn("Elier Martin", excluded)
        self.assertIn("Damion Lyn", excluded)
        self.assertIn("Hakim Mouazer", excluded)
        self.assertIn("Alex Contractor", excluded)

    def test_filter_2025_demand_removes_florida_and_splits_skill_views(self):
        demand_df = pd.DataFrame(
            [
                {
                    "appointment_id": "A1",
                    "scheduled_start": "2025-01-15",
                    "country": "USA",
                    "state_norm": "GA",
                    "skill_class": "regular",
                    "required_ls": 0,
                    "required_hps": 0,
                },
                {
                    "appointment_id": "A2",
                    "scheduled_start": "2025-02-20",
                    "country": "USA",
                    "state_norm": "FL",
                    "skill_class": "regular",
                    "required_ls": 0,
                    "required_hps": 0,
                },
                {
                    "appointment_id": "A3",
                    "scheduled_start": "2025-03-05",
                    "country": "USA",
                    "state_norm": "NC",
                    "skill_class": "ls",
                    "required_ls": 1,
                    "required_hps": 0,
                },
                {
                    "appointment_id": "A4",
                    "scheduled_start": "2025-04-10",
                    "country": "USA",
                    "state_norm": "TX",
                    "skill_class": "hps",
                    "required_ls": 0,
                    "required_hps": 1,
                },
                {
                    "appointment_id": "A5",
                    "scheduled_start": "2024-12-10",
                    "country": "USA",
                    "state_norm": "GA",
                    "skill_class": "regular",
                    "required_ls": 0,
                    "required_hps": 0,
                },
                {
                    "appointment_id": "A6",
                    "scheduled_start": "2025-02-20",
                    "country": "Canada",
                    "state_norm": "ON",
                    "skill_class": "regular",
                    "required_ls": 0,
                    "required_hps": 0,
                },
            ]
        )

        patient_df, patient_meta = step12.filter_2025_demand(demand_df, "patient_sim")
        ls_df, _ = step12.filter_2025_demand(demand_df, "learning_space")
        hps_df, _ = step12.filter_2025_demand(demand_df, "hps")
        combined_df, combined_meta = step12.filter_2025_demand(demand_df, "combined")

        self.assertEqual(patient_df["appointment_id"].tolist(), ["A1"])
        self.assertEqual(ls_df["appointment_id"].tolist(), ["A3"])
        self.assertEqual(hps_df["appointment_id"].tolist(), ["A4"])
        self.assertEqual(
            combined_df.sort_values("appointment_id")["appointment_id"].tolist(),
            ["A1", "A3", "A4"],
        )
        self.assertEqual(patient_meta["appointments_removed_excluded_states"], 1)
        self.assertEqual(combined_meta["states_in_mode"], 3)

    def test_load_optimize_territories_data_and_panel_include_new_tab(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            territory_dir = root / config.OPTIMIZE_TERRITORIES_SUBDIR
            territory_dir.mkdir(parents=True, exist_ok=True)

            summary_payload = {
                "available_modes": ["patient_sim", "learning_space", "combined"],
                "default_mode": "patient_sim",
                "mode_panels": {
                    "patient_sim": {
                        "key": "patient_sim",
                        "label": "Patient Sim",
                        "appointments_plotted": 12,
                        "active_owner_tech_count": 3,
                        "raw_travel_cost_usd": 1200.0,
                        "coverage_gap_count": 1,
                        "training_gap_count": 0,
                    },
                    "learning_space": {
                        "key": "learning_space",
                        "label": "LearningSpace",
                        "appointments_plotted": 4,
                        "active_owner_tech_count": 2,
                        "raw_travel_cost_usd": 600.0,
                        "coverage_gap_count": 1,
                        "training_gap_count": 1,
                    },
                    "combined": {
                        "key": "combined",
                        "label": "Combined",
                        "appointments_plotted": 16,
                        "active_owner_tech_count": 3,
                        "raw_travel_cost_usd": 1800.0,
                        "coverage_gap_count": 1,
                        "training_gap_count": 1,
                    },
                },
                "assumptions": {"demand_year": 2025},
            }
            (territory_dir / "territory_summary.json").write_text(
                json.dumps(summary_payload)
            )
            pd.DataFrame(
                [
                    {
                        "territory_mode": "patient_sim",
                        "appointment_id": "A1",
                        "tech_id": "ben_walker",
                        "tech_name": "Ben Walker",
                        "lat": 33.75,
                        "lon": -84.39,
                        "account_name": "Site 1",
                        "state_norm": "GA",
                    }
                ]
            ).to_csv(territory_dir / "territory_appointment_assignments.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "territory_mode": "patient_sim",
                        "tech_id": "ben_walker",
                        "tech_name": "Ben Walker",
                        "assigned_appointments": 12,
                        "primary_states": "GA;SC",
                        "covered_skill_labels": "Patient Sim",
                        "raw_travel_cost_usd": 1200.0,
                        "share_two_zone_plus": 0.0,
                    },
                    {
                        "territory_mode": "combined",
                        "tech_id": "curt_corder",
                        "tech_name": "Curt Corder",
                        "assigned_appointments": 0,
                        "primary_states": None,
                        "owned_states": None,
                        "covered_skill_labels": "Patient Sim;HPS",
                        "raw_travel_cost_usd": 0.0,
                        "share_two_zone_plus": 0.0,
                    },
                ]
            ).to_csv(territory_dir / "territory_tech_summary.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "territory_mode": "learning_space",
                        "state_norm": "NC",
                        "state_name": "North Carolina",
                        "skill_class": "ls",
                        "appointment_count": 3,
                        "coverage_gap_flag": True,
                        "training_gap_flag": True,
                        "gap_reason": "no same-zone qualified owner",
                    }
                ]
            ).to_csv(territory_dir / "territory_gap_summary.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "territory_mode": "patient_sim",
                        "state_norm": "GA",
                        "state_name": "Georgia",
                        "dominant_owner_tech_name": "Ben Walker",
                        "dominant_owner_color_key": "ben_walker",
                    }
                ]
            ).to_csv(territory_dir / "territory_state_summary.csv", index=False)

            old_opt_dir = config.OPTIMIZATION_DIR
            try:
                config.OPTIMIZATION_DIR = str(root)
                territory_data = step05.load_optimize_territories_data()
            finally:
                config.OPTIMIZATION_DIR = old_opt_dir

            self.assertIsNotNone(territory_data)
            self.assertEqual(territory_data["default_mode"], "patient_sim")
            self.assertEqual(
                territory_data["modes"]["patient_sim"]["label"],
                "Patient Sim",
            )
            self.assertEqual(
                territory_data["modes"]["learning_space"]["gap_rows"][0]["skill_class"],
                "ls",
            )
            combined_rows = territory_data["modes"].get("combined", {}).get("tech_summaries", [])
            if combined_rows:
                curt_row = [row for row in combined_rows if row["tech_id"] == "curt_corder"][0]
                self.assertEqual(curt_row["primary_states"], "FL")

        markup = step05.build_simulation_panel_markup()
        optimized_idx = markup.index('data-view="optimized"')
        territories_idx = markup.index('data-view="territories"')
        blank_idx = markup.index('data-view="blank"')
        self.assertLess(optimized_idx, territories_idx)
        self.assertLess(territories_idx, blank_idx)
        self.assertIn('id="territory-tech-list"', markup)

        script = step05.build_simulation_panel_script(
            "map_ref",
            '{"0":{"placements":[],"kpis":{}}}',
            '{"0":"scenario_layer"}',
            "{}",
            '{"ben_walker":"#123456"}',
            None,
            ["0"],
            "0",
            step05.get_map_ui_preset(),
            territory_tech_colors_js='{"ben_walker":"#2E7D32"}',
            territory_opt_payload_js='{"available_modes":["patient_sim"],"default_mode":"patient_sim","modes":{"patient_sim":{"label":"Patient Sim","tech_summaries":[],"gap_rows":[]}}}',
            territory_opt_layer_names_js='{"patient_sim":{"dots_layer":"territory_dots","states_layer":"territory_states"}}',
        )
        self.assertIn('const territoryTechColors = {"ben_walker":"#2E7D32"};', script)
        self.assertIn('const territoryOptimizeData = {"available_modes":["patient_sim"],"default_mode":"patient_sim","modes":{"patient_sim":{"label":"Patient Sim","tech_summaries":[],"gap_rows":[]}}};', script)
        self.assertIn('const territoryOptimizeLayerNames = {"patient_sim":{"dots_layer":"territory_dots","states_layer":"territory_states"}};', script)
        self.assertIn("const territoryTechMarkersLayerName = null;", script)
        self.assertIn('renderTerritoryModeButtons();', script)
        self.assertIn('data-view="territories"', markup)


if __name__ == "__main__":
    unittest.main()
