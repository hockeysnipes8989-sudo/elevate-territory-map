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
step06 = load_module("step06_inputs", "06_build_optimization_inputs.py")
step02 = load_module("step02_geocode", "02_geocode.py")


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

    def test_synthetic_new_hire_eligibility_blocks_ls_and_hps_outside_blank_slate(self):
        regular_node = pd.Series({"required_hps": 0, "required_ls": 0})
        ls_node = pd.Series({"required_hps": 0, "required_ls": 1})
        hps_node = pd.Series({"required_hps": 1, "required_ls": 0})

        self.assertTrue(
            step08.synthetic_new_hire_eligible_for_node(regular_node, blank_slate=False)
        )
        self.assertFalse(
            step08.synthetic_new_hire_eligible_for_node(ls_node, blank_slate=False)
        )
        self.assertFalse(
            step08.synthetic_new_hire_eligible_for_node(hps_node, blank_slate=False)
        )
        self.assertTrue(
            step08.synthetic_new_hire_eligible_for_node(ls_node, blank_slate=True)
        )
        self.assertTrue(
            step08.synthetic_new_hire_eligible_for_node(hps_node, blank_slate=True)
        )

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

    def test_apply_anchor_allocations_sets_tameka_scope_and_capacity(self):
        tech_master = pd.DataFrame(
            [
                {
                    "tech_name": "Tameka Gongs",
                    "availability_fte": 1.0,
                    "assignment_scope_mode": config.ASSIGNMENT_SCOPE_MODE_NATIONAL,
                    "assignment_scope_state": "",
                    "notes": "25% travel, rest at customer site",
                }
            ]
        )
        anchors = pd.DataFrame(
            [
                {
                    "tech_name": "Tameka Gongs",
                    "anchor_site_name": "Morgan State University",
                    "anchor_site_location_raw": "Baltimore, MD",
                    "anchor_site_lat": 39.2908816,
                    "anchor_site_lon": -76.610759,
                    "anchor_reserved_fte": 0.75,
                    "external_field_fte": 0.25,
                    "external_assignment_mode": config.ASSIGNMENT_SCOPE_MODE_STATE_SET_LIMITED,
                    "external_allowed_states": "MD;VA;WV;DC;DE",
                    "anchor_show_on_map": 1,
                    "anchor_notes": "Reserved duty",
                }
            ]
        )

        enriched = step06.apply_anchor_allocations(tech_master, anchors)
        row = enriched.iloc[0]
        self.assertAlmostEqual(float(row["availability_fte"]), 0.25)
        self.assertEqual(
            row["assignment_scope_mode"], config.ASSIGNMENT_SCOPE_MODE_STATE_SET_LIMITED
        )
        self.assertEqual(row["assignment_scope_states"], "MD;VA;WV;DC;DE")
        self.assertEqual(row["anchor_site_name"], "Morgan State University")

    def test_account_coordinate_override_pins_morgan_state(self):
        appts = pd.DataFrame(
            [
                {
                    "Account: Account Name": "Morgan State University",
                    "lat": 39.2908816,
                    "lon": -76.610759,
                },
                {
                    "Account: Account Name": "Other Account",
                    "lat": 10.0,
                    "lon": 20.0,
                },
            ]
        )
        updated, count = step02.apply_account_coordinate_overrides(appts)
        self.assertEqual(count, 1)
        row = updated.loc[
            updated["Account: Account Name"] == "Morgan State University"
        ].iloc[0]
        self.assertAlmostEqual(float(row["lat"]), 39.3438)
        self.assertAlmostEqual(float(row["lon"]), -76.5844)

    def test_state_set_limited_scope_blocks_out_of_region_assignments(self):
        anchored_tech = pd.Series(
            {
                "skill_hps": 0,
                "skill_ls": 1,
                "skill_patient": 1,
                "constraint_florida_only": 0,
                "employment_type": "fte",
                "base_airport_iata": "BWI",
                "zone_policy": config.ZONE_POLICY_STANDARD,
                "base_operational_zone_rank": 0,
                "assignment_scope_mode": config.ASSIGNMENT_SCOPE_MODE_STATE_SET_LIMITED,
                "assignment_scope_state": "",
                "assignment_scope_states": "MD;VA;WV;DC;DE",
            }
        )
        md_node = pd.Series(
            {
                "required_hps": 0,
                "required_ls": 1,
                "state_norm": "MD",
                "node_operational_zone_rank": 0,
            }
        )
        ny_node = pd.Series(
            {
                "required_hps": 0,
                "required_ls": 1,
                "state_norm": "NY",
                "node_operational_zone_rank": 0,
            }
        )
        self.assertTrue(step08.tech_eligible_for_node(anchored_tech, md_node, "anywhere"))
        self.assertFalse(step08.tech_eligible_for_node(anchored_tech, ny_node, "anywhere"))

    def test_normal_solver_keeps_ls_with_incumbent_and_regular_with_new_hire(self):
        tech = pd.DataFrame(
            [
                {
                    "tech_id": "tech_ls",
                    "tech_name": "LS Tech",
                    "employment_type": "fte",
                    "availability_fte": 1.0,
                    "base_state": "TX",
                    "base_airport_iata": "DFW",
                    "base_hub_tier": config.HUB_TIER_LARGE,
                    "skill_hps": 0,
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
                    "node_id": "TX__ls",
                    "state_norm": "TX",
                    "skill_class": "ls",
                    "required_hps": 0,
                    "required_ls": 1,
                    "appointment_count": 1.0,
                    "demand_hours": 1.0,
                    "avg_hours_per_appointment": 1.0,
                    "node_operational_zone_label": "Central",
                    "node_operational_zone_rank": 1,
                },
                {
                    "node_id": "TX__regular",
                    "state_norm": "TX",
                    "skill_class": "regular",
                    "required_hps": 0,
                    "required_ls": 0,
                    "appointment_count": 1.0,
                    "demand_hours": 100.0,
                    "avg_hours_per_appointment": 100.0,
                    "node_operational_zone_label": "Central",
                    "node_operational_zone_rank": 1,
                },
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
                    "hub_tier": config.HUB_TIER_LARGE,
                    "operational_zone_label": "Central",
                    "operational_zone_rank": 1,
                }
            ]
        )
        full_cost_lookup = {
            ("tech_ls", "TX__ls"): {"unit_cost_usd": 10.0, "trip_mode": "drive_day"},
            ("tech_ls", "TX__regular"): {"unit_cost_usd": 1000.0, "trip_mode": "drive_day"},
            ("airport_dfw", "TX__ls"): {"unit_cost_usd": 1.0, "trip_mode": "drive_day"},
            ("airport_dfw", "TX__regular"): {"unit_cost_usd": 1.0, "trip_mode": "drive_day"},
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
            unmet_penalty=5000.0,
            annual_hire_cost_usd=0.0,
            max_hires_per_base=1,
            time_limit_sec=5,
            blank_slate=False,
        )

        new_assignments = result["new_assignments"].set_index("node_id")
        existing_assignments = result["existing_assignments"].set_index("node_id")

        self.assertNotIn("TX__ls", new_assignments.index)
        self.assertIn("TX__regular", new_assignments.index)
        self.assertEqual(
            float(new_assignments.loc["TX__regular", "assigned_appointments"]),
            1.0,
        )
        self.assertIn("TX__ls", existing_assignments.index)
        self.assertEqual(
            float(existing_assignments.loc["TX__ls", "assigned_appointments"]),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
