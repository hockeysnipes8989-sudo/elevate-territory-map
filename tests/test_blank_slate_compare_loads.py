import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def load_module(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


compare = load_module("step10_blank_slate_compare", "10_compare_blank_slate_loads.py")


class BlankSlateCompareTests(unittest.TestCase):
    def test_build_detail_and_summary_tables_use_annualized_underload_cutoff(self):
        placements_13 = pd.DataFrame(
            [
                {
                    "candidate_id": "airport_slc",
                    "city": "Salt Lake City",
                    "state": "UT",
                    "airport_iata": "SLC",
                    "assigned_appointments": 41.0,
                    "assigned_hours": 500.0,
                },
                {
                    "candidate_id": "airport_cle",
                    "city": "Cleveland",
                    "state": "OH",
                    "airport_iata": "CLE",
                    "assigned_appointments": 180.0,
                    "assigned_hours": 6000.0,
                },
            ]
        )
        placements_12 = pd.DataFrame(
            [
                {
                    "candidate_id": "airport_den",
                    "city": "Denver",
                    "state": "CO",
                    "airport_iata": "DEN",
                    "assigned_appointments": 90.0,
                    "assigned_hours": 900.0,
                },
                {
                    "candidate_id": "airport_bna",
                    "city": "Nashville",
                    "state": "TN",
                    "airport_iata": "BNA",
                    "assigned_appointments": 160.0,
                    "assigned_hours": 6200.0,
                },
            ]
        )

        details_13 = compare.build_detail_table(
            placements_13,
            scenario_hires=13,
            data_span_years=2.078,
            annual_underload_threshold=75.0,
        )
        details_12 = compare.build_detail_table(
            placements_12,
            scenario_hires=12,
            data_span_years=2.078,
            annual_underload_threshold=75.0,
        )
        summary = compare.build_summary_table(pd.concat([details_13, details_12], ignore_index=True))

        slc_row = details_13.loc[details_13["candidate_id"] == "airport_slc"].iloc[0]
        den_row = details_12.loc[details_12["candidate_id"] == "airport_den"].iloc[0]
        self.assertLess(slc_row["annualized_assigned_appointments"], 75.0)
        self.assertTrue(bool(slc_row["underloaded_flag"]))
        self.assertLess(den_row["annualized_assigned_appointments"], 75.0)
        self.assertTrue(bool(den_row["underloaded_flag"]))

        summary_13 = summary.loc[summary["scenario_hires"] == 13].iloc[0]
        summary_12 = summary.loc[summary["scenario_hires"] == 12].iloc[0]
        self.assertEqual(int(summary_13["placement_count"]), 2)
        self.assertEqual(int(summary_13["underloaded_placements_count"]), 1)
        self.assertEqual(float(summary_13["min_assigned_appointments"]), 41.0)
        self.assertEqual(float(summary_12["max_assigned_appointments"]), 160.0)


if __name__ == "__main__":
    unittest.main()
