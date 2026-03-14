"""Tests for the diminishing-returns function in Step 09."""

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


step09 = load_module("step09_analyze", "09_analyze_scenarios.py")
apply_diminishing_returns = step09.apply_diminishing_returns


class DiminishingReturnsTests(unittest.TestCase):
    """Verify the two-stage diminishing-returns model."""

    def test_alpha_1_is_identity(self):
        """alpha=1.0 with no ceiling disables both stages."""
        linear = pd.Series([0.0, 25.0, 50.0, 75.0])
        result = apply_diminishing_returns(linear, alpha=1.0, ceiling=None)
        pd.testing.assert_series_equal(result, linear)

    def test_ceiling_only(self):
        """Ceiling without power-law clips at ceiling."""
        linear = pd.Series([0.0, 25.0, 50.0, 75.0])
        result = apply_diminishing_returns(linear, alpha=1.0, ceiling=40.0)
        expected = pd.Series([0.0, 25.0, 40.0, 40.0])
        pd.testing.assert_series_equal(result, expected)

    def test_power_law_preserves_reference(self):
        """The smallest positive value (N=1 baseline) is unchanged."""
        linear = pd.Series([0.0, 25.0, 50.0, 75.0, 98.0])
        result = apply_diminishing_returns(linear, alpha=0.7, ceiling=None)
        self.assertAlmostEqual(result.iloc[1], 25.0, places=4)
        # Higher values should be diminished
        self.assertLess(result.iloc[2], 50.0)
        self.assertLess(result.iloc[3], 75.0)
        self.assertLess(result.iloc[4], 98.0)

    def test_power_law_monotonic(self):
        """Output should still be monotonically increasing."""
        linear = pd.Series([0.0, 25.0, 50.0, 75.0, 98.0])
        result = apply_diminishing_returns(linear, alpha=0.7, ceiling=None)
        for i in range(1, len(result)):
            self.assertGreaterEqual(result.iloc[i], result.iloc[i - 1])

    def test_combined_stages(self):
        """Power-law + ceiling both apply."""
        linear = pd.Series([0.0, 25.0, 50.0, 75.0, 98.0])
        result = apply_diminishing_returns(linear, alpha=0.7, ceiling=55.0)
        # N=1 preserved
        self.assertAlmostEqual(result.iloc[1], 25.0, places=4)
        # Last value capped
        self.assertLessEqual(result.iloc[4], 55.0)

    def test_zero_ceiling_disables(self):
        """Ceiling of 0 disables the ceiling stage."""
        linear = pd.Series([0.0, 25.0, 50.0])
        result = apply_diminishing_returns(linear, alpha=0.7, ceiling=0)
        # Should still apply power-law, not clip
        self.assertAlmostEqual(result.iloc[1], 25.0, places=4)
        self.assertLess(result.iloc[2], 50.0)
        self.assertGreater(result.iloc[2], 0.0)

    def test_explicit_reference_linear(self):
        """Explicit reference_linear overrides auto-detection."""
        linear = pd.Series([0.0, 25.0, 50.0])
        result = apply_diminishing_returns(
            linear, alpha=0.7, ceiling=None, reference_linear=25.0
        )
        self.assertAlmostEqual(result.iloc[1], 25.0, places=4)

    def test_zeros_unchanged(self):
        """Zero values stay zero regardless of parameters."""
        linear = pd.Series([0.0, 0.0, 25.0])
        result = apply_diminishing_returns(linear, alpha=0.7, ceiling=55.0)
        self.assertEqual(result.iloc[0], 0.0)
        self.assertEqual(result.iloc[1], 0.0)


if __name__ == "__main__":
    unittest.main()
