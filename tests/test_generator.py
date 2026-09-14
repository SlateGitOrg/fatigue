"""The generator's ground truth must actually be what the README claims."""

import unittest

from src.generator import (CreativeSpec, DeliverySegment, default_specs,
                           generate_panel, season_index, true_fatigue)


class TestGeneratorTruth(unittest.TestCase):

    def setUp(self):
        self.panel = generate_panel()
        self.specs = {s.creative_id: s for s in default_specs()}

    def test_fatigue_depends_on_frequency_not_on_days(self):
        """Two creatives, same planted half-life, one delivering 5x faster.

        At equal CUMULATIVE FREQUENCY their noiseless fatigue multipliers must
        be identical even though they got there on wildly different days.  If
        the generator had any calendar-time decay term in it, this would fail
        and every downstream conclusion would be circular.
        """
        slow = CreativeSpec("slow", 0, 0.04, 6.0, (DeliverySegment(100, 0.04),))
        fast = CreativeSpec("fast", 0, 0.04, 6.0, (DeliverySegment(100, 0.20),))
        p = generate_panel([slow, fast], seed=5, season_amplitude=0.0,
                           noise_cv=0.0)
        by_freq = {}
        for r in p.rows:
            key = round(r.cum_frequency, 6)
            by_freq.setdefault(key, []).append(r)
        # Frequency 0 is shared on day 0 by construction and says nothing.
        shared = [v for k, v in by_freq.items() if len(v) == 2 and k > 0]
        self.assertGreater(len(shared), 5, "no shared frequency points to compare")
        for rows in shared:
            self.assertNotEqual(rows[0].days_since_launch,
                                rows[1].days_since_launch)
            self.assertAlmostEqual(rows[0].response_rate, rows[1].response_rate,
                                   places=12)

    def test_dark_days_do_not_advance_cumulative_frequency(self):
        rows = self.panel.for_creative("C3_paused")
        before = [r for r in rows if r.days_since_launch == 33][0]
        after = [r for r in rows if r.days_since_launch == 48][0]
        self.assertEqual(after.days_since_launch - before.days_since_launch, 15)
        self.assertAlmostEqual(after.cum_frequency,
                               before.cum_frequency + before.freq_today,
                               places=9)

    def test_paused_creative_emits_no_rows_while_dark(self):
        days = {r.days_since_launch for r in self.panel.for_creative("C3_paused")}
        for d in range(34, 48):
            self.assertNotIn(d, days)

    def test_seasonality_is_shared_and_large_enough_to_confound(self):
        # The confound must be a real one: a swing comparable to a half-life's
        # worth of decay, otherwise the calendar model would "fail" only
        # marginally and the demonstration would be weak.
        vals = [season_index(d, 110, 0.30) for d in range(110)]
        self.assertAlmostEqual(max(vals), 1.30, places=2)
        self.assertAlmostEqual(min(vals), 0.70, places=2)
        self.assertGreater(max(vals) / min(vals), 1.8)

    def test_noise_is_multiplicative_and_unbiased(self):
        # A lognormal draw without the -sigma^2/2 correction would inflate the
        # mean by ~0.06% per unit variance and quietly bias every amplitude.
        spec = CreativeSpec("x", 0, 0.05, 1e9, (DeliverySegment(4000, 0.01),))
        p = generate_panel([spec], seed=11, season_amplitude=0.0, noise_cv=0.05)
        mean = sum(r.response_rate for r in p.rows) / len(p.rows)
        # 4000 draws at cv 5% -> standard error of the mean is 0.05/sqrt(4000)
        # = 0.079%; allow 3 of those.
        self.assertAlmostEqual(mean / 0.05, 1.0, delta=3 * 0.05 / 4000 ** 0.5)

    def test_true_fatigue_is_an_exact_halving(self):
        self.assertAlmostEqual(true_fatigue(6.0, 6.0), 0.5, places=12)
        self.assertAlmostEqual(true_fatigue(12.0, 6.0), 0.25, places=12)
        self.assertAlmostEqual(true_fatigue(0.0, 6.0), 1.0, places=12)

    def test_budget_pair_shares_a_half_life_and_differs_only_in_rate(self):
        c1 = self.specs["C1_base_rate"]
        c2 = self.specs["C2_double_rate"]
        self.assertEqual(c1.true_halflife_freq, c2.true_halflife_freq)
        self.assertEqual(c1.launch_day, c2.launch_day)
        self.assertEqual(len(c1.schedule), len(c2.schedule))
        for a, b in zip(c1.schedule, c2.schedule):
            self.assertEqual(a.days, b.days)
            self.assertAlmostEqual(b.freq_per_day / a.freq_per_day, 2.0, places=9)

    def test_panel_is_reproducible(self):
        a = generate_panel(seed=99)
        b = generate_panel(seed=99)
        self.assertEqual([r.response_rate for r in a.rows],
                         [r.response_rate for r in b.rows])
        c = generate_panel(seed=100)
        self.assertNotEqual([r.response_rate for r in a.rows],
                            [r.response_rate for r in c.rows])


if __name__ == "__main__":
    unittest.main()
