import importlib.util
import json
import re
import sys
import unittest
from pathlib import Path

import pandas as pd


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import config
from optimization_utils import build_airports_df, canonicalize_tech_name


def load_module(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


step05 = load_module("step05_map", "05_generate_map.py")
step06 = load_module("step06_inputs", "06_build_optimization_inputs.py")


class HistoricalViewTests(unittest.TestCase):
    def test_canonicalize_tech_name_collapses_historical_variants(self):
        self.assertEqual(canonicalize_tech_name("[S] Alex Rondero"), "Alex Rondero")
        self.assertEqual(canonicalize_tech_name("Clarence Bonner, Jr"), "Clarence Bonner")
        self.assertEqual(canonicalize_tech_name("Elier Alvarez Martin"), "Elier Martin")

    def test_build_historical_roster_uses_archived_and_estimated_bases(self):
        airports_df = build_airports_df(config.MAJOR_AIRPORTS)
        demand_df = pd.read_csv(Path(config.OPTIMIZATION_DIR) / "demand_appointments.csv")

        roster = step06.build_historical_roster(
            config.SERVICE_APPTS_REPORT, airports_df, demand_df
        ).set_index("tech_name")

        david = roster.loc["David Bazany"]
        self.assertEqual(david["status"], "former")
        self.assertEqual(david["base_label"], "San Antonio, TX")
        self.assertEqual(david["base_source"], "archived_roster")

        john = roster.loc["John Aleksa"]
        self.assertEqual(john["base_label"], "New York, NY")
        self.assertEqual(john["base_airport_iata"], "JFK")

        trent = roster.loc["Trent Osborne"]
        self.assertEqual(trent["base_label"], "Calgary, AB")
        self.assertEqual(trent["base_airport_iata"], "YYC")
        self.assertEqual(trent["base_source"], "archived_region")
        self.assertEqual(trent["base_confidence"], "estimated_region_to_city")

        alex = roster.loc["Alex Rondero"]
        self.assertEqual(alex["status"], "special")
        self.assertEqual(alex["base_label"], "Sarasota, FL")

    def test_load_historical_data_uses_resolved_bases_and_canonical_names(self):
        territory_data = step05.load_territory_assignment_data()
        self.assertIsNotNone(territory_data)

        historical = step05.load_historical_data(territory_data, config.OPTIMIZATION_DIR)
        self.assertIsNotNone(historical)

        stats = historical["tech_stats"]
        self.assertIn("Trent Osborne", stats)
        self.assertEqual(stats["Trent Osborne"]["base"], "Calgary, AB")
        self.assertEqual(stats["Trent Osborne"]["base_confidence"], "estimated_region_to_city")
        self.assertEqual(stats["David Bazany"]["base"], "San Antonio, TX")
        self.assertEqual(stats["John Aleksa"]["base"], "New York, NY")
        self.assertEqual(stats["Alex Rondero"]["status"], "special")

        self.assertNotIn("[S] Alex Rondero", stats)
        self.assertNotIn("Clarence Bonner, Jr", stats)
        self.assertNotIn("Elier Alvarez Martin", stats)

    def test_generated_map_payload_embeds_updated_historical_data(self):
        html = Path(config.MAP_OUTPUT).read_text()
        prefix = "const historicalData = "
        suffix = ";\n      const historicalDotLayerName"
        self.assertIn(prefix, html)
        self.assertIn(suffix, html)
        start = html.index(prefix) + len(prefix)
        end = html.index(suffix, start)
        historical = json.loads(html[start:end])
        stats = historical["tech_stats"]

        self.assertEqual(stats["David Bazany"]["base"], "San Antonio, TX")
        self.assertEqual(stats["John Aleksa"]["base"], "New York, NY")
        self.assertEqual(stats["Trent Osborne"]["base"], "Calgary, AB")
        self.assertEqual(stats["Alex Rondero"]["status"], "special")
        self.assertNotIn("[S] Alex Rondero", stats)


if __name__ == "__main__":
    unittest.main()
