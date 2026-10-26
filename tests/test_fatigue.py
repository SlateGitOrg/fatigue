"""Recovery of the generator's planted half-lives by the frequency model,
and failure of the calendar model on the same data.

Thresholds are derived, not tuned:
  * test_fitting measured the curvature SE at ~0.75 of the true sampling
    spread, so a correct estimator should land within 1.96 / 0.75 ~= 2.6
    reported SEs; we allow 3.0.  Across seeds 20260912, 1, 2, 3 the worst
    |z| measured was 2.6.
  * Coverage of the nominal 95% CI is therefore really ~90%.  With 7 fitted
    creatives, P(at least 5 covered | p=0.9) ~= 0.97, so ">= 5 of 7" is the
    honest assertion, not "all 7".
"""

import math
import unittest

from src.fatigue import (MIN_OBSERVED_DAYS, fit_portfolio,
                         pause_recovery_signal, remaining_useful_exposures,
                         report_portfolio)
from src.fitting import ExpFit
from src.generator import default_specs, generate_panel

PANEL = generate_panel()
SPECS = {s.creative_id: s for s in default_specs()}
FREQ = fit_portfolio(PANEL, "frequency")
CAL = fit_portfolio(PANEL, "calendar")


class TestFrequencyModelRecoversTruth(unittest.TestCase):

    def test_halflife_within_three_standard_errors_for_every_creative(self):
        for cid, fit in FREQ.fits.items():
            z = (fit.halflife - SPECS[cid].true_halflife_freq) / fit.halflife_se
            self.assertLess(abs(z), 3.0, "%s z=%.2f" % (cid, z))

    def test_nominal_ci_covers_truth_for_at_least_five_of_seven(self):
        covered = 0
        for cid, fit in FREQ.fits.items():
            lo, hi = fit.ci95()
            covered += lo <= SPECS[cid].true_halflife_freq <= hi
        self.assertEqual(len(FREQ.fits), 7)
        self.assertGreaterEqual(covered, 5)

    def test_relative_error_under_five_percent(self):
        # Measured worst case is 3.6% (C6).  5% keeps a margin while still
        # failing the undeconfounded fit, which is off by up to 32%.
        for cid, fit in FREQ.fits.items():
            rel = abs(fit.halflife / SPECS[cid].true_halflife_freq - 1.0)
            self.assertLess(rel, 0.05, "%s rel err %.3f" % (cid, rel))

    def test_recovery_holds_on_a_different_seed(self):
        pf = fit_portfolio(generate_panel(seed=3), "frequency")
        for cid, fit in pf.fits.items():
            z = (fit.halflife - SPECS[cid].true_halflife_freq) / fit.halflife_se
            self.assertLess(abs(z), 3.0, "%s z=%.2f" % (cid, z))

    def test_fitted_seasonal_curve_tracks_the_planted_season(self):
        # The intercept is absorbed into amplitudes, so compare shapes: the
        # log-ratio of the fitted season between peak (day 0) and trough
        # (day 55) against the truth log(1.3 / 0.7) = 0.619.
        got = math.log(FREQ.season_at(0) / FREQ.season_at(55))
        self.assertAlmostEqual(got, math.log(1.3 / 0.7), delta=0.1)


class TestGenericApproachesFail(unittest.TestCase):

    def test_without_deconfounding_frequency_fit_is_biased(self):
        naive = fit_portfolio(PANEL, "frequency", deconfound=False)
        badly_off = [cid for cid, f in naive.fits.items()
                     if abs(f.halflife / SPECS[cid].true_halflife_freq - 1) > 0.10]
        # Measured: 5 of 7 creatives off by 15-32%.
        self.assertGreaterEqual(len(badly_off), 4, badly_off)

    def test_budget_pair_calendar_halves_frequency_holds(self):
        # C1 and C2 share the planted half-life; C2 delivers 2x.  Any
        # calendar-time model must report C2 as fatiguing ~2x faster.
        f = FREQ.fits["C2_double_rate"].halflife / FREQ.fits["C1_base_rate"].halflife
        c = CAL.fits["C2_double_rate"].halflife / CAL.fits["C1_base_rate"].halflife
        self.assertAlmostEqual(f, 1.0, delta=0.05)
        self.assertAlmostEqual(c, 0.5, delta=0.05)

    def test_pause_creates_spurious_recovery_only_on_calendar_axis(self):
        d = pause_recovery_signal(PANEL, "C3_paused")
        # Frequency: noise CV 3.5% over 7 days -> SE of the mean ~1.3%; allow
        # ~4 SE.  Calendar: 14 dark days at a ~40-day half-life predicts a
        # ~22% decline that never happens -> signal well above 0.3.
        self.assertLess(abs(d["frequency"].recovery_signal), 0.05)
        self.assertGreater(d["calendar"].recovery_signal, 0.30)

    def test_pause_diagnostic_requires_a_gap(self):
        with self.assertRaises(ValueError):
            pause_recovery_signal(PANEL, "C1_base_rate")


class TestRefreshPlanAndSafety(unittest.TestCase):

    def test_thin_creative_reports_insufficient_data(self):
        self.assertIn("C8_thin", FREQ.skipped)
        self.assertIn("insufficient data", FREQ.skipped["C8_thin"])
        rep = {r.creative_id: r for r in report_portfolio(PANEL, FREQ)}
        self.assertFalse(rep["C8_thin"].fitted)
        self.assertTrue(math.isnan(rep["C8_thin"].refresh_day))
        self.assertLess(len(PANEL.for_creative("C8_thin")), MIN_OBSERVED_DAYS)

    def test_refresh_day_matches_the_one_implied_by_the_true_halflife(self):
        # Walk each creative's real delivery path with the PLANTED half-life
        # and check the modelled refresh day lands within 3 days of it.
        for r in report_portfolio(PANEL, FREQ, threshold_fraction=0.6):
            if not r.fitted:
                continue
            f_true = SPECS[r.creative_id].true_halflife_freq * math.log2(1 / 0.6)
            true_day = None
            for row in PANEL.for_creative(r.creative_id):
                if row.cum_frequency + row.freq_today >= f_true:
                    true_day = row.days_since_launch
                    break
            self.assertIsNotNone(true_day, r.creative_id)
            self.assertLessEqual(abs(r.refresh_day - true_day), 3, r.creative_id)

    def test_refresh_days_are_not_uniform(self):
        days = [r.refresh_day for r in report_portfolio(PANEL, FREQ) if r.fitted]
        self.assertGreater(max(days) - min(days), 42)
        self.assertTrue(any(d < 42 for d in days) and any(d > 42 for d in days))

    def test_remaining_exposures_arithmetic(self):
        fit = ExpFit(amplitude=1.0, halflife=6.0, halflife_se=0.1, sse=0.0, n=10)
        self.assertAlmostEqual(remaining_useful_exposures(fit, 0.0, 0.5), 6.0)
        self.assertAlmostEqual(remaining_useful_exposures(fit, 8.0, 0.5), -2.0)
        with self.assertRaises(ValueError):
            remaining_useful_exposures(fit, 0.0, 1.0)


if __name__ == "__main__":
    unittest.main()
