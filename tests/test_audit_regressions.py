import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import config
from optimization_utils import (
    build_airports_df,
    choose_airport_for_location,
    nearest_airport_code,
)


def load_module(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


step05 = load_module("step05_map_audit", "05_generate_map.py")
step06 = load_module("step06_inputs_audit", "06_build_optimization_inputs.py")


class AuditRegressionTests(unittest.TestCase):
    def test_nearest_airport_prefers_actual_nearest_within_country(self):
        airports_df = build_airports_df(config.MAJOR_AIRPORTS)

        cases = [
            ("Fort Smith, AR", 35.3859, -94.3985, "AR", "TUL"),
            ("Fayetteville, AR", 36.0822, -94.1719, "AR", "TUL"),
            ("Mobile, AL", 30.6954, -88.0399, "AL", "MSY"),
            ("Stratford, NJ", 39.8268, -75.0154, "NJ", "PHL"),
            ("Vineland, NJ", 39.4862, -75.0257, "NJ", "PHL"),
            ("Tyler, TX", 32.3513, -95.3011, "TX", "SHV"),
        ]

        for label, lat, lon, state_hint, expected in cases:
            with self.subTest(label=label):
                code, distance_km = nearest_airport_code(
                    lat, lon, airports_df, state_hint=state_hint
                )
                self.assertEqual(code, expected)
                self.assertIsNotNone(distance_km)
                self.assertGreater(distance_km, 0.0)

    def test_choose_airport_for_unknown_city_does_not_guess_from_state(self):
        airports_df = build_airports_df(config.MAJOR_AIRPORTS)

        self.assertIsNone(choose_airport_for_location("Randomville, TX", airports_df))
        self.assertIsNone(choose_airport_for_location("Unknown, AR", airports_df))

    def test_build_candidate_bases_re_ranks_after_major_airport_overlap(self):
        airports_df = pd.DataFrame(
            [
                {
                    "airport_code": "AAA",
                    "airport_name": "Alpha Airport",
                    "city_name": "Alpha",
                    "city_norm": "alpha",
                    "state_abbr": "TX",
                    "country": "USA",
                    "lat": 32.0,
                    "lon": -97.0,
                    "hub_tier": config.HUB_TIER_MEDIUM,
                    "operational_zone_label": "Central",
                    "operational_zone_rank": 1,
                    "utc_offset_standard": -6,
                },
                {
                    "airport_code": "BBB",
                    "airport_name": "Bravo Airport",
                    "city_name": "Bravo",
                    "city_norm": "bravo",
                    "state_abbr": "TX",
                    "country": "USA",
                    "lat": 31.0,
                    "lon": -96.0,
                    "hub_tier": config.HUB_TIER_MEDIUM,
                    "operational_zone_label": "Central",
                    "operational_zone_rank": 1,
                    "utc_offset_standard": -6,
                },
            ]
        )
        demand_df = pd.DataFrame(
            [
                {
                    "appointment_id": "1",
                    "city": "Alpha",
                    "state_norm": "TX",
                    "country": "USA",
                    "duration_hours": 10.0,
                    "lat": 32.01,
                    "lon": -97.01,
                },
                {
                    "appointment_id": "2",
                    "city": "Alpha",
                    "state_norm": "TX",
                    "country": "USA",
                    "duration_hours": 8.0,
                    "lat": 32.02,
                    "lon": -97.02,
                },
                {
                    "appointment_id": "3",
                    "city": "Charlie",
                    "state_norm": "TX",
                    "country": "USA",
                    "duration_hours": 7.0,
                    "lat": 30.0,
                    "lon": -95.0,
                },
                {
                    "appointment_id": "4",
                    "city": "Delta",
                    "state_norm": "TX",
                    "country": "USA",
                    "duration_hours": 6.0,
                    "lat": 29.5,
                    "lon": -94.5,
                },
            ]
        )

        candidates, meta = step06.build_candidate_bases(
            airports_df=airports_df,
            demand_df=demand_df,
            top_demand_cities=3,
        )
        demand_candidates = candidates[candidates["candidate_type"] == "demand_city"].copy()

        self.assertEqual(meta["requested_top_demand_cities"], 3)
        self.assertEqual(meta["selected_pre_dedupe_demand_cities"], 3)
        self.assertEqual(meta["dropped_due_major_airport_overlap"], 1)
        self.assertEqual(meta["final_demand_city_candidates"], 2)
        self.assertEqual(demand_candidates["demand_rank"].tolist(), [1, 2])
        self.assertEqual(demand_candidates["city"].tolist(), ["Charlie", "Delta"])

    def test_build_tech_master_backfills_missing_skills_from_cached_master(self):
        airports_df = build_airports_df(config.MAJOR_AIRPORTS)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workbook_path = root / "tech_roster.xlsx"
            cached_path = root / "tech_master.csv"

            pd.DataFrame(
                [
                    {
                        "Service Resource Name": "Robert Cohen",
                        "Location": "St Louis, MO",
                        "Comment": "",
                    }
                ]
            ).to_excel(
                workbook_path,
                sheet_name=config.APPTS_REPORT_RESOURCES_SHEET,
                index=False,
            )
            pd.DataFrame(
                [
                    {
                        "tech_id": "robert_cohen",
                        "tech_name": "Robert Cohen",
                        "source_name": "Robert Cohen",
                        "employment_type": "fte",
                        "base_location_raw": "St Louis, MO",
                        "base_city": "St Louis",
                        "base_state": "MO",
                        "base_country": "USA",
                        "base_airport_iata": "STL",
                        "base_lat": 38.7487,
                        "base_lon": -90.37,
                        "skill_hps": 1,
                        "skill_ls": 0,
                        "skill_patient": 1,
                        "availability_fte": 1.0,
                        "fixed_base": 1,
                        "can_relocate": 0,
                        "contractor_assignment_scope": "",
                        "constraint_florida_only": 0,
                        "notes": "",
                    }
                ]
            ).to_csv(cached_path, index=False)

            old_opt_dir = config.OPTIMIZATION_DIR
            try:
                config.OPTIMIZATION_DIR = str(root)
                tech_master, source_meta = step06.build_tech_master(
                    str(workbook_path),
                    airports_df,
                    contractor_scope="anywhere",
                )
            finally:
                config.OPTIMIZATION_DIR = old_opt_dir

            row = tech_master.set_index("tech_name").loc["Robert Cohen"]
            self.assertEqual(int(row["skill_hps"]), 1)
            self.assertEqual(int(row["skill_ls"]), 0)
            self.assertEqual(int(row["skill_patient"]), 1)
            self.assertTrue(
                any("Backfilled" in warning for warning in source_meta["source_warnings"])
            )

    def test_load_simulation_data_marks_stale_enhanced_outputs_and_preserves_provenance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            pd.DataFrame(
                [
                    {
                        "scenario_hires": 0,
                        "solver_status": 1,
                        "solver_proven_optimal": 0,
                        "solver_mip_gap": 0.125,
                        "solver_message": "Time limit",
                        "economic_total_with_overhead_usd": 111.0,
                        "hire_cost_usd": 0.0,
                        "mean_existing_utilization": 0.5,
                        "max_existing_utilization": 0.8,
                    }
                ]
            ).to_csv(root / "scenario_summary.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "scenario_hires": 0,
                        "solver_status": 0,
                        "solver_proven_optimal": 1,
                        "solver_mip_gap": 0.0,
                        "solver_message": "Optimal",
                        "economic_total_with_overhead_usd": 999.0,
                    }
                ]
            ).to_csv(root / "scenario_summary_enhanced.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "scenario_hires": 0,
                        "candidate_id": "airport_dfw",
                        "city": "Dallas",
                        "state": "TX",
                        "airport_iata": "DFW",
                        "hires_allocated": 0.0,
                        "assigned_appointments": 0.0,
                        "assigned_hours": 0.0,
                    }
                ]
            ).to_csv(root / "scenario_placements.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "candidate_id": "airport_dfw",
                        "lat": 32.8998,
                        "lon": -97.0403,
                    }
                ]
            ).to_csv(root / "candidate_bases.csv", index=False)
            pd.DataFrame(columns=["scenario_hires"]).to_csv(
                root / "scenario_tech_utilization.csv",
                index=False,
            )
            with open(root / "model_assumptions.json", "w") as f:
                json.dump({"data_span_years": 1.0}, f)
            with open(root / "optimization_input_summary.json", "w") as f:
                json.dump(
                    {
                        "tech_source_kind": "cached_tech_master_csv",
                        "tech_source_is_cached": True,
                        "source_warnings": ["cached fallback used"],
                    },
                    f,
                )

            stale_time = 1_700_000_000
            fresh_time = stale_time + 10
            os.utime(root / "scenario_summary_enhanced.csv", (stale_time, stale_time))
            for path in [
                root / "scenario_summary.csv",
                root / "scenario_placements.csv",
                root / "scenario_tech_utilization.csv",
                root / "model_assumptions.json",
            ]:
                os.utime(path, (fresh_time, fresh_time))

            old_opt_dir = config.OPTIMIZATION_DIR
            old_min = config.SIM_SCENARIO_MIN
            old_max = config.SIM_SCENARIO_MAX
            try:
                config.OPTIMIZATION_DIR = str(root)
                config.SIM_SCENARIO_MIN = 0
                config.SIM_SCENARIO_MAX = 0
                payload = step05.load_simulation_data()
            finally:
                config.OPTIMIZATION_DIR = old_opt_dir
                config.SIM_SCENARIO_MIN = old_min
                config.SIM_SCENARIO_MAX = old_max

            self.assertIsNotNone(payload)
            scenario = payload["0"]
            self.assertTrue(scenario["analysis_is_stale"])
            self.assertIn("older than raw scenario outputs", scenario["analysis_staleness_reason"])
            self.assertEqual(
                scenario["kpis"]["economic_total_with_overhead_usd"],
                111.0,
            )
            self.assertFalse(scenario["solver_proven_optimal"])
            self.assertEqual(scenario["solver_status"], 1)
            self.assertTrue(scenario["source_provenance"]["tech_source_is_cached"])
            self.assertEqual(
                scenario["source_provenance"]["source_warnings"],
                ["cached fallback used"],
            )


if __name__ == "__main__":
    unittest.main()
