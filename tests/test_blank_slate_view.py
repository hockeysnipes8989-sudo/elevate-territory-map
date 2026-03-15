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


step05 = load_module("step05_map_blank", "05_generate_map.py")
step08 = load_module("step08_optimize_blank", "08_optimize_locations.py")


class BlankSlateTests(unittest.TestCase):
    def test_blank_slate_solver_uses_hire_count_capacity_and_allows_hps(self):
        tech = pd.DataFrame(
            [
                {
                    "tech_id": "inactive_tech",
                    "tech_name": "Inactive Tech",
                    "employment_type": "fte",
                    "availability_fte": 0.0,
                    "base_state": "TX",
                    "base_airport_iata": "DFW",
                    "skill_hps": 1,
                    "skill_ls": 1,
                    "skill_patient": 1,
                    "constraint_florida_only": 0,
                    "assignment_scope_mode": config.ASSIGNMENT_SCOPE_MODE_NATIONAL,
                    "assignment_scope_state": "",
                    "zone_policy": config.ZONE_POLICY_STANDARD,
                    "base_operational_zone_rank": 1,
                }
            ]
        )
        nodes = pd.DataFrame(
            [
                {
                    "node_id": "TX__hps",
                    "state_norm": "TX",
                    "skill_class": "hps",
                    "required_hps": 1,
                    "required_ls": 0,
                    "appointment_count": 5.0,
                    "demand_hours": 10.0,
                    "avg_hours_per_appointment": 2.0,
                    "node_operational_zone_label": "Central",
                    "node_operational_zone_rank": 1,
                }
            ]
        )
        candidates = pd.DataFrame(
            [
                {
                    "candidate_id": "airport_dfw",
                    "candidate_type": "major_airport",
                    "city": "Dallas",
                    "state": "TX",
                    "airport_iata": "DFW",
                    "operational_zone_label": "Central",
                    "operational_zone_rank": 1,
                }
            ]
        )
        full_cost_lookup = {
            ("airport_dfw", "TX__hps"): {
                "unit_cost_usd": 100.0,
                "travel_cost_policy": config.TRAVEL_COST_POLICY_EMPLOYEE,
            }
        }

        result = step08.solve_scenario(
            hire_count=1,
            tech=tech,
            nodes=nodes,
            candidates=candidates,
            full_cost_lookup=full_cost_lookup,
            contractor_scope="anywhere",
            target_utilization=0.5,
            out_of_region_penalty=0.0,
            unmet_penalty=5000.0,
            annual_hire_cost_usd=1000.0,
            max_hires_per_base=1,
            time_limit_sec=5,
            blank_slate=True,
        )

        self.assertAlmostEqual(result["summary"]["hours_per_capacity_unit"], 20.0)
        self.assertEqual(result["summary"]["unmet_appointments"], 0.0)
        self.assertEqual(
            float(result["new_assignments"]["assigned_appointments"].sum()),
            5.0,
        )
        self.assertTrue(result["existing_assignments"].empty)

    def test_load_blank_slate_data_annualizes_costs_and_merges_coordinates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            blank_dir = root / config.BLANK_SLATE_SUBDIR
            blank_dir.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(
                [
                    {
                        "candidate_id": "airport_jfk",
                        "city": "New York",
                        "state": "NY",
                        "lat": 40.6413,
                        "lon": -73.7781,
                    }
                ]
            ).to_csv(root / "candidate_bases.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "scenario_hires": 16,
                        "solver_status": 0,
                        "solver_message": "Optimal",
                        "total_appointments": 120.0,
                        "travel_cost_usd": 400000.0,
                        "hire_cost_usd": 200000.0,
                        "modeled_total_cost_usd": 600000.0,
                        "economic_total_with_overhead_usd": 600000.0,
                    }
                ]
            ).to_csv(blank_dir / "scenario_summary.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "scenario_hires": 16,
                        "candidate_id": "airport_jfk",
                        "city": "New York",
                        "state": "NY",
                        "airport_iata": "JFK",
                        "hires_allocated": 1,
                        "assigned_appointments": 120.0,
                        "assigned_hours": 480.0,
                    }
                ]
            ).to_csv(blank_dir / "scenario_placements.csv", index=False)
            with open(blank_dir / "model_assumptions.json", "w") as f:
                json.dump({"data_span_years": 2.0, "blank_slate": True}, f)

            old_opt_dir = config.OPTIMIZATION_DIR
            try:
                config.OPTIMIZATION_DIR = str(root)
                payload = step05.load_blank_slate_data()
            finally:
                config.OPTIMIZATION_DIR = old_opt_dir

            self.assertIsNotNone(payload)
            self.assertEqual(payload["scenario_hires"], 16)
            self.assertEqual(payload["total_placements"], 1)
            self.assertEqual(payload["annualized_total_cost_usd"], 300000.0)
            self.assertEqual(payload["annualized_travel_cost_usd"], 200000.0)
            self.assertEqual(payload["placements"][0]["lat"], 40.6413)
            self.assertEqual(payload["placements"][0]["lon"], -73.7781)

    def test_load_blank_slate_assignment_data_keeps_scenario_16(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            blank_dir = root / config.BLANK_SLATE_SUBDIR
            blank_dir.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(
                [
                    {
                        "appointment_id": "A1",
                        "Appointment Number": "A1",
                        "lat": 32.0,
                        "lon": -97.0,
                        "state_norm": "TX",
                        "skill_class": "regular",
                    }
                ]
            ).to_csv(root / "demand_appointments.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "tech_id": "tech_1",
                        "tech_name": "Tech One",
                        "availability_fte": 1.0,
                        "base_lat": 32.0,
                        "base_lon": -97.0,
                    }
                ]
            ).to_csv(root / "tech_master.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "candidate_id": "airport_dfw",
                        "city": "Dallas",
                        "state": "TX",
                        "lat": 32.8998,
                        "lon": -97.0403,
                    }
                ]
            ).to_csv(root / "candidate_bases.csv", index=False)
            pd.DataFrame(columns=step08.EXISTING_ASSIGNMENT_COLUMNS).to_csv(
                blank_dir / "scenario_assignments_existing.csv",
                index=False,
            )
            pd.DataFrame(
                [
                    {
                        "scenario_hires": 16,
                        "candidate_id": "airport_dfw",
                        "candidate_type": "major_airport",
                        "candidate_city": "Dallas",
                        "candidate_state": "TX",
                        "airport_iata": "DFW",
                        "node_id": "TX__regular",
                        "state_norm": "TX",
                        "skill_class": "regular",
                        "assigned_appointments": 1.0,
                        "assigned_hours": 8.0,
                        "total_travel_cost_usd": 100.0,
                    }
                ]
            ).to_csv(blank_dir / "scenario_assignments_newhires.csv", index=False)
            pd.DataFrame(columns=step08.UTILIZATION_COLUMNS).to_csv(
                blank_dir / "scenario_tech_utilization.csv",
                index=False,
            )

            old_opt_dir = config.OPTIMIZATION_DIR
            old_max = config.SIM_SCENARIO_MAX
            try:
                config.OPTIMIZATION_DIR = str(root)
                config.SIM_SCENARIO_MAX = 4
                data = step05.load_blank_slate_assignment_data()
            finally:
                config.OPTIMIZATION_DIR = old_opt_dir
                config.SIM_SCENARIO_MAX = old_max

            self.assertIsNotNone(data)
            self.assertEqual(data["available_scenarios"], [16])


if __name__ == "__main__":
    unittest.main()
