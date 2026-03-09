import sys
import tempfile
import types
import unittest
from pathlib import Path

import pandas as pd


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import config
from install_mix_model import (
    build_patient_sim_install_model,
    parse_install_history_row,
    select_install_history_source,
)


def make_config(**overrides):
    values = {
        "APPTS_RAW_SHEET": config.APPTS_RAW_SHEET,
        "APPTS_DISPATCH_SHEET": config.APPTS_DISPATCH_SHEET,
        "OPTIMIZATION_DIR": config.OPTIMIZATION_DIR,
        "SERVICE_APPTS_DISPATCH": config.SERVICE_APPTS_DISPATCH,
        "PATIENT_SIM_CAPACITY_TIME_UNIT": config.PATIENT_SIM_CAPACITY_TIME_UNIT,
        "PATIENT_SIM_EXCLUDE_HPS_FROM_FUTURE_MIX": config.PATIENT_SIM_EXCLUDE_HPS_FROM_FUTURE_MIX,
        "PATIENT_SIM_RECENCY_WEIGHTING_ENABLED": config.PATIENT_SIM_RECENCY_WEIGHTING_ENABLED,
        "PATIENT_SIM_RECENCY_HALFLIFE_YEARS": config.PATIENT_SIM_RECENCY_HALFLIFE_YEARS,
        "PATIENT_SIM_FORWARD_MIX_MANUAL_MULTIPLIERS": dict(
            config.PATIENT_SIM_FORWARD_MIX_MANUAL_MULTIPLIERS
        ),
        "PATIENT_SIM_FAMILY_ASSUMPTIONS": {
            family: assumptions.copy()
            for family, assumptions in config.PATIENT_SIM_FAMILY_ASSUMPTIONS.items()
        },
        "FREED_CAPACITY_UTILIZATION_FACTOR": config.FREED_CAPACITY_UTILIZATION_FACTOR,
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


def write_workbook(path: Path, rows: list[dict]) -> None:
    raw_rows = []
    derived_rows = []
    for row in rows:
        raw_rows.append(
            {
                "Appointment Number": row["appointment_number"],
                "Account: Account Name": row.get("account_name", "Test Account"),
                "Service Resource: Name": row.get("service_resource", "Test Tech"),
                "Scheduled Start": row.get("scheduled_start"),
                "Scheduled End": row.get("scheduled_end"),
                "Dispatched Date/Time": row.get("dispatched_time"),
                "Description": row.get("description"),
                "Subject": row.get("subject"),
            }
        )
        derived_rows.append(
            {
                "Appointment Number": row["appointment_number"],
                "Service Type": row.get("service_type", "ISO"),
                "Duration Days": row.get("duration_days"),
                "Duration Hours": row.get("duration_hours"),
            }
        )

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(raw_rows).to_excel(
            writer,
            sheet_name=config.APPTS_RAW_SHEET,
            index=False,
        )
        pd.DataFrame(derived_rows).to_excel(
            writer,
            sheet_name=config.APPTS_DISPATCH_SHEET,
            index=False,
        )


class InstallMixModelTests(unittest.TestCase):
    def test_parse_excludes_learning_space_and_non_net_new(self):
        avs_rows = parse_install_history_row(
            appointment_number="1",
            account_name="Acct",
            service_type="AVS ISO",
            description="AVS ISO install",
            subject="LS room",
            scheduled_start="2025-01-01",
            duration_days=1,
            duration_hours=24,
        )
        self.assertEqual(avs_rows[0]["history_exclusion_reason"], "learning_space_excluded")

        replacement_rows = parse_install_history_row(
            appointment_number="2",
            account_name="Acct",
            service_type="ISO",
            description="Apollo replacement install",
            subject="",
            scheduled_start="2025-01-01",
            duration_days=1,
            duration_hours=24,
        )
        self.assertEqual(replacement_rows[0]["history_exclusion_reason"], "non_net_new_install")

    def test_select_install_history_source_prefers_workbook(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            workbook = tmp / "appointments.xlsx"
            write_workbook(
                workbook,
                [
                    {
                        "appointment_number": "1001",
                        "description": "Install 1 APN",
                        "subject": "Apollo install",
                        "scheduled_start": "2025-01-01",
                        "duration_days": 2,
                        "duration_hours": 48,
                    }
                ],
            )

            demand_csv = tmp / "demand_appointments.csv"
            pd.DataFrame(
                [
                    {
                        "Appointment Number": "1001",
                        "Account: Account Name": "Fallback Account",
                        "scheduled_start": "2025-01-01",
                        "Description": "Install 1 Juno",
                        "Subject": "Juno install",
                        "Service Type": "ISO",
                        "duration_hours": 24,
                    }
                ]
            ).to_csv(demand_csv, index=False)

            _, metadata = select_install_history_source(
                appointments_workbook=str(workbook),
                demand_appointments_csv=str(demand_csv),
            )

            self.assertEqual(metadata["source_kind"], "raw_workbook_merge")

    def test_select_install_history_source_falls_back_to_demand_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            demand_csv = tmp / "demand_appointments.csv"
            pd.DataFrame(
                [
                    {
                        "Appointment Number": "1002",
                        "Account: Account Name": "Fallback Account",
                        "scheduled_start": "2025-01-01",
                        "Description": "Install 1 Juno",
                        "Subject": "Juno install",
                        "Service Type": "ISO",
                        "duration_hours": 24,
                    }
                ]
            ).to_csv(demand_csv, index=False)

            _, metadata = select_install_history_source(
                appointments_workbook=str(tmp / "missing.xlsx"),
                demand_appointments_csv=str(demand_csv),
            )

            self.assertEqual(metadata["source_kind"], "demand_appointments_fallback")

    def test_build_model_separates_history_and_forward_mix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workbook = Path(tmpdir) / "appointments.xlsx"
            write_workbook(
                workbook,
                [
                    {
                        "appointment_number": "2001",
                        "description": "Install 1 MFS & 1 APN CASE 2001",
                        "subject": "Lucina and Apollo",
                        "scheduled_start": "2025-01-10",
                        "duration_days": 2,
                        "duration_hours": 48,
                    },
                    {
                        "appointment_number": "2002",
                        "description": "Install 1 of 8 Junos CASE 2002",
                        "subject": "Juno lab",
                        "scheduled_start": "2025-02-01",
                        "duration_days": 1,
                        "duration_hours": 24,
                    },
                    {
                        "appointment_number": "2003",
                        "description": "Install HPS CASE 2003",
                        "subject": "Legacy HPS install",
                        "scheduled_start": "2025-03-01",
                        "duration_days": 3,
                        "duration_hours": 72,
                    },
                    {
                        "appointment_number": "2004",
                        "description": "AVS ISO install",
                        "subject": "LearningSpace room",
                        "service_type": "AVS ISO",
                        "scheduled_start": "2025-04-01",
                        "duration_days": 1,
                        "duration_hours": 24,
                    },
                    {
                        "appointment_number": "2005",
                        "description": "Apollo replacement install",
                        "subject": "Replacement",
                        "scheduled_start": "2025-04-02",
                        "duration_days": 1,
                        "duration_hours": 24,
                    },
                ],
            )

            model = build_patient_sim_install_model(
                appointments_workbook=str(workbook),
            )

            historical_units = model.historical_mix_units.set_index("family")["units_inferred"].to_dict()
            self.assertEqual(historical_units["Apollo"], 1.0)
            self.assertEqual(historical_units["Lucina"], 1.0)
            self.assertEqual(historical_units["Juno"], 8.0)
            self.assertEqual(historical_units["HPS"], 1.0)

            historical_events = model.historical_mix_events.set_index("family")["event_equivalent_count"].to_dict()
            self.assertEqual(historical_events["Apollo"], 0.5)
            self.assertEqual(historical_events["Lucina"], 0.5)

            forward_mix = model.forward_mix.set_index("family")
            self.assertTrue(bool(forward_mix.loc["HPS", "forward_mix_excluded"]))
            self.assertEqual(forward_mix.loc["HPS", "forward_share"], 0.0)
            self.assertAlmostEqual(float(model.forward_mix["forward_share"].sum()), 1.0, places=6)

    def test_duplicate_collapse_keeps_one_primary_row(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workbook = Path(tmpdir) / "appointments.xlsx"
            write_workbook(
                workbook,
                [
                    {
                        "appointment_number": "3001",
                        "description": "Install 1 APN CASE 77",
                        "subject": "Apollo install",
                        "scheduled_start": "2025-01-01",
                        "duration_days": 1,
                        "duration_hours": 24,
                    },
                    {
                        "appointment_number": "3002",
                        "description": "Install 1 APN CASE 77",
                        "subject": "Apollo install follow-up duplicate row",
                        "scheduled_start": "2025-01-02",
                        "duration_days": 3,
                        "duration_hours": 72,
                    },
                ],
            )

            model = build_patient_sim_install_model(
                appointments_workbook=str(workbook),
            )
            history = model.history_rows
            apollo_rows = history[history["normalized_family"].eq("Apollo")].copy()

            self.assertEqual(len(apollo_rows), 2)
            self.assertEqual(int(apollo_rows["duplicate_group_size"].max()), 2)
            self.assertEqual(int(apollo_rows["include_in_history"].sum()), 1)
            primary = apollo_rows[apollo_rows["include_in_history"]].iloc[0]
            self.assertEqual(primary["appointment_number"], "3002")

    def test_ambiguous_bundle_stays_out_of_unit_mix(self):
        rows = parse_install_history_row(
            appointment_number="4001",
            account_name="Acct",
            service_type="ISO",
            description="Apollo and Juno install",
            subject="Multiple sims",
            scheduled_start="2025-01-01",
            duration_days=2,
            duration_hours=48,
        )
        self.assertEqual(rows[0]["history_exclusion_reason"], "ambiguous_bundle_units")

    def test_recency_weighting_can_shift_forward_mix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workbook = Path(tmpdir) / "appointments.xlsx"
            write_workbook(
                workbook,
                [
                    {
                        "appointment_number": "5001",
                        "description": "Install 1 APN",
                        "subject": "Apollo install",
                        "scheduled_start": "2021-01-01",
                        "duration_days": 2,
                        "duration_hours": 48,
                    },
                    {
                        "appointment_number": "5002",
                        "description": "Install 1 Juno",
                        "subject": "Juno install",
                        "scheduled_start": "2025-01-01",
                        "duration_days": 1,
                        "duration_hours": 24,
                    },
                ],
            )

            model = build_patient_sim_install_model(
                appointments_workbook=str(workbook),
                config_module=make_config(
                    PATIENT_SIM_RECENCY_WEIGHTING_ENABLED=True,
                    PATIENT_SIM_RECENCY_HALFLIFE_YEARS=1.0,
                ),
            )

            forward_mix = model.forward_mix.set_index("family")
            self.assertGreater(
                float(forward_mix.loc["Juno", "forward_share"]),
                float(forward_mix.loc["Apollo", "forward_share"]),
            )


if __name__ == "__main__":
    unittest.main()
