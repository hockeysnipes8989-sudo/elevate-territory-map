import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import config
from optimization_utils import build_airports_df


def load_module(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


step06 = load_module("step06_inputs_hub", "06_build_optimization_inputs.py")
step08 = load_module("step08_optimize_hub", "08_optimize_locations.py")


class HubPenaltyTests(unittest.TestCase):
    def test_build_airports_df_preserves_and_defaults_hub_tier(self):
        airports_df = build_airports_df(
            [
                {
                    "code": "AAA",
                    "name": "Airport A",
                    "city": "Austin, TX",
                    "lat": 1.0,
                    "lon": 2.0,
                    "hub_tier": config.HUB_TIER_SMALL,
                },
                {
                    "code": "BBB",
                    "name": "Airport B",
                    "city": "Boston, MA",
                    "lat": 3.0,
                    "lon": 4.0,
                },
            ]
        ).set_index("airport_code")

        self.assertEqual(airports_df.loc["AAA", "hub_tier"], config.HUB_TIER_SMALL)
        self.assertEqual(airports_df.loc["BBB", "hub_tier"], config.HUB_TIER_MEDIUM)

    def test_enrich_fields_attach_hub_tier_metadata(self):
        airports_df = build_airports_df(config.MAJOR_AIRPORTS)

        tech_master = pd.DataFrame(
            [
                {
                    "tech_id": "tech_1",
                    "tech_name": "Tech One",
                    "employment_type": "fte",
                    "base_airport_iata": "LIT",
                }
            ]
        )
        candidates = pd.DataFrame(
            [
                {
                    "candidate_id": "airport_ord",
                    "airport_iata": "ORD",
                }
            ]
        )

        tech_enriched = step06.enrich_tech_policy_fields(
            tech_master,
            airports_df,
            contractor_scope="anywhere",
        )
        candidate_enriched = step06.enrich_candidate_zone_fields(candidates, airports_df)

        self.assertEqual(
            tech_enriched.iloc[0]["base_hub_tier"],
            config.HUB_TIER_SMALL,
        )
        self.assertEqual(
            candidate_enriched.iloc[0]["hub_tier"],
            config.HUB_TIER_LARGE,
        )

    def test_evaluate_hub_penalty_is_flight_only(self):
        self.assertEqual(
            step08.evaluate_hub_penalty(config.HUB_TIER_SMALL, "fly"),
            config.HUB_PENALTY_SMALL,
        )
        self.assertEqual(
            step08.evaluate_hub_penalty(config.HUB_TIER_SMALL, "drive_day"),
            0.0,
        )
        self.assertEqual(
            step08.evaluate_hub_penalty(config.HUB_TIER_UNKNOWN, "fly"),
            config.HUB_PENALTY_UNKNOWN,
        )

    def test_solve_scenario_adds_hub_penalty_to_fly_assignments(self):
        tech = pd.DataFrame(
            columns=[
                "tech_id",
                "tech_name",
                "employment_type",
                "availability_fte",
                "base_state",
                "base_airport_iata",
                "base_hub_tier",
                "skill_hps",
                "skill_ls",
                "skill_patient",
                "constraint_florida_only",
                "assignment_scope_mode",
                "assignment_scope_state",
                "zone_policy",
                "base_operational_zone_rank",
            ]
        )
        nodes = pd.DataFrame(
            [
                {
                    "node_id": "TX__regular",
                    "state_norm": "TX",
                    "skill_class": "regular",
                    "required_hps": 0,
                    "required_ls": 0,
                    "appointment_count": 1.0,
                    "demand_hours": 1.0,
                    "avg_hours_per_appointment": 1.0,
                    "node_operational_zone_rank": 1,
                }
            ]
        )
        candidates = pd.DataFrame(
            [
                {
                    "candidate_id": "airport_lit",
                    "candidate_type": "major_airport",
                    "city": "Little Rock",
                    "state": "AR",
                    "airport_iata": "LIT",
                    "hub_tier": config.HUB_TIER_SMALL,
                    "operational_zone_rank": 1,
                    "operational_zone_label": "Central",
                }
            ]
        )
        full_cost_lookup = {
            ("airport_lit", "TX__regular"): {
                "unit_cost_usd": 100.0,
                "trip_mode": "fly",
            }
        }

        result = step08.solve_scenario(
            hire_count=1,
            tech=tech,
            nodes=nodes,
            candidates=candidates,
            full_cost_lookup=full_cost_lookup,
            contractor_scope="anywhere",
            target_utilization=1.0,
            out_of_region_penalty=0.0,
            unmet_penalty=1000.0,
            annual_hire_cost_usd=0.0,
            max_hires_per_base=1,
            time_limit_sec=5,
            blank_slate=True,
        )

        summary = result["summary"]
        new_assignments = result["new_assignments"]

        self.assertEqual(summary["travel_cost_usd"], 100.0)
        self.assertEqual(summary["hub_penalty_usd"], config.HUB_PENALTY_SMALL)
        self.assertEqual(summary["modeled_total_cost_usd"], 250.0)
        self.assertEqual(new_assignments.iloc[0]["unit_hub_penalty_usd"], config.HUB_PENALTY_SMALL)
        self.assertEqual(new_assignments.iloc[0]["total_hub_penalty_usd"], config.HUB_PENALTY_SMALL)


if __name__ == "__main__":
    unittest.main()
