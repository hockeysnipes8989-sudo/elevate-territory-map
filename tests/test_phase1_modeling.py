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


step08 = load_module("step08_optimize", "08_optimize_locations.py")
step11 = load_module("step11_costs", "11_build_full_cost_table.py")


class PhaseOneModelingTests(unittest.TestCase):
    def test_airport_operational_zone_mapping_handles_arizona(self):
        airports_df = build_airports_df(config.MAJOR_AIRPORTS)
        phx = airports_df.loc[airports_df["airport_code"] == "PHX"].iloc[0]
        bwi = airports_df.loc[airports_df["airport_code"] == "BWI"].iloc[0]
        self.assertEqual(phx["operational_zone_label"], "Mountain")
        self.assertEqual(int(phx["operational_zone_rank"]), 2)
        self.assertEqual(bwi["operational_zone_label"], "Eastern")

    def test_build_demand_nodes_uses_plurality_operational_zone(self):
        demand = pd.DataFrame(
            [
                {
                    "appointment_id": "1",
                    "state_norm": "TX",
                    "skill_class": "regular",
                    "required_hps": 0,
                    "required_ls": 0,
                    "duration_hours": 10,
                    "territory": "A",
                    "nearest_hub_operational_zone_label": "Central",
                    "nearest_hub_operational_zone_rank": 1,
                },
                {
                    "appointment_id": "2",
                    "state_norm": "TX",
                    "skill_class": "regular",
                    "required_hps": 0,
                    "required_ls": 0,
                    "duration_hours": 12,
                    "territory": "A",
                    "nearest_hub_operational_zone_label": "Central",
                    "nearest_hub_operational_zone_rank": 1,
                },
                {
                    "appointment_id": "3",
                    "state_norm": "TX",
                    "skill_class": "regular",
                    "required_hps": 0,
                    "required_ls": 0,
                    "duration_hours": 8,
                    "territory": "B",
                    "nearest_hub_operational_zone_label": "Mountain",
                    "nearest_hub_operational_zone_rank": 2,
                },
            ]
        )
        nodes = step08.build_demand_nodes(demand)
        row = nodes.iloc[0]
        self.assertEqual(row["node_operational_zone_label"], "Central")
        self.assertEqual(int(row["node_operational_zone_rank"]), 1)
        self.assertAlmostEqual(float(row["node_operational_zone_share"]), 2 / 3)

    def test_zone_policy_blocks_standard_three_plus_but_softens_contractors(self):
        feasible, penalty = step08.evaluate_zone_policy(3, config.ZONE_POLICY_STANDARD)
        self.assertFalse(feasible)
        self.assertEqual(penalty, 0.0)

        feasible, penalty = step08.evaluate_zone_policy(2, config.ZONE_POLICY_STANDARD)
        self.assertTrue(feasible)
        self.assertEqual(penalty, config.EMPLOYEE_TWO_ZONE_JUMP_PENALTY_USD)

        feasible, penalty = step08.evaluate_zone_policy(3, config.ZONE_POLICY_CONTRACTOR_SOFT)
        self.assertTrue(feasible)
        self.assertEqual(penalty, config.CONTRACTOR_THREE_PLUS_ZONE_JUMP_PENALTY_USD)

    def test_ground_transport_threshold_uses_one_way_median_distance(self):
        under = step11.compute_ground_transport(124.0, 3)
        over = step11.compute_ground_transport(125.0, 3)

        self.assertEqual(under["ground_transport_mode"], "personal_vehicle")
        self.assertAlmostEqual(under["mileage_cost_usd"], 124.0 * 2.0 * config.IRS_MILEAGE_RATE_USD_PER_MI)
        self.assertEqual(under["rental_cost_usd"], 0.0)

        self.assertEqual(over["ground_transport_mode"], "rental_car")
        self.assertEqual(over["mileage_cost_usd"], 0.0)
        self.assertAlmostEqual(over["rental_cost_usd"], 3 * config.RENTAL_CAR_DAILY_RATE_USD)

    def test_trip_span_days_uses_duration_proxy(self):
        self.assertEqual(step11.compute_trip_span_days(2.01), 3)
        self.assertEqual(step11.compute_trip_span_days(0.2), 1)

    def test_contractor_cost_policy_is_compressed_and_capped(self):
        effective = step11.apply_travel_cost_policy(
            employee_style_unit_cost=2000.0,
            travel_cost_policy=config.TRAVEL_COST_POLICY_CONTRACTOR,
            contractor_cost_multiplier=0.65,
            contractor_cost_cap_usd=900.0,
            contractor_dispatch_surcharge_usd=125.0,
        )
        self.assertEqual(effective, 900.0)

    def test_tech_eligibility_uses_scope_and_zone_policy(self):
        standard_tech = pd.Series(
            {
                "skill_hps": 0,
                "skill_ls": 0,
                "skill_patient": 1,
                "constraint_florida_only": 0,
                "employment_type": "fte",
                "base_airport_iata": "LAX",
                "zone_policy": config.ZONE_POLICY_STANDARD,
                "base_operational_zone_rank": 3,
                "assignment_scope_mode": config.ASSIGNMENT_SCOPE_MODE_NATIONAL,
                "assignment_scope_state": "",
            }
        )
        contractor = pd.Series(
            {
                "skill_hps": 0,
                "skill_ls": 0,
                "skill_patient": 1,
                "constraint_florida_only": 0,
                "employment_type": "contractor",
                "base_airport_iata": "IAH",
                "zone_policy": config.ZONE_POLICY_CONTRACTOR_SOFT,
                "base_operational_zone_rank": 1,
                "assignment_scope_mode": config.ASSIGNMENT_SCOPE_MODE_NATIONAL,
                "assignment_scope_state": "",
            }
        )
        node = pd.Series(
            {
                "required_hps": 0,
                "required_ls": 0,
                "state_norm": "PA",
                "node_operational_zone_rank": 0,
            }
        )
        far_node = pd.Series(
            {
                "required_hps": 0,
                "required_ls": 0,
                "state_norm": "NY",
                "node_operational_zone_rank": 0,
            }
        )
        self.assertFalse(step08.tech_eligible_for_node(standard_tech, far_node, "anywhere"))
        self.assertTrue(step08.tech_eligible_for_node(contractor, node, "anywhere"))


if __name__ == "__main__":
    unittest.main()
