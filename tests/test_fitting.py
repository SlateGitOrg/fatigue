"""The hand-rolled estimator itself, checked against cases with known answers."""

import math
import random
import statistics
import unittest

from src.fitting import (fit_exponential, harmonic_basis,
                         solve_normal_equations)


class TestExponentialFit(unittest.TestCase):

    def setUp(self):
        self.rng = random.Random(7)

    def _draw(self, n, halflife=6.0, noise=0.02):
        xs = [i * 20.0 / n for i in range(n)]
        ys = [2.0 ** (-x / halflife) * (1.0 + self.rng.gauss(0.0, noise))
              for x in xs]
        fit = fit_exponential(xs, ys)
        return fit.halflife_se, fit.halflife

    def test_recovers_exact_curve_with_no_noise(self):
        # Noiseless data: the minimiser must land on the generating parameters
        # to within the golden-section tolerance, not merely somewhere nearby.
        xs = [i * 0.25 for i in range(60)]
        ys = [3.5 * 2.0 ** (-x / 4.75) for x in xs]
        fit = fit_exponential(xs, ys)
        self.assertAlmostEqual(fit.halflife, 4.75, places=4)
        self.assertAlmostEqual(fit.amplitude, 3.5, places=4)
        self.assertLess(fit.sse, 1e-12)

    def test_flat_data_reports_a_very_long_halflife(self):
        # No decay present -> the half-life must run off towards the no-decay
        # limit rather than latching onto a small value from the grid.
        xs = [float(i) for i in range(40)]
        ys = [2.0] * 40
        fit = fit_exponential(xs, ys)
        self.assertGreater(fit.halflife, 1000.0)

    def test_weights_dominate_the_fit(self):
        # One heavily weighted subset of points should pull the fit onto its own
        # curve; if weights were ignored the two answers would coincide.
        xs = [float(i) for i in range(20)]
        clean = [1.0 * 2.0 ** (-x / 5.0) for x in xs]
        polluted = list(clean)
        for i in range(10, 20):
            polluted[i] *= 0.5
        heavy = [1000.0] * 10 + [1.0] * 10
        light = [1.0] * 20
        near = fit_exponential(xs, polluted, heavy).halflife
        far = fit_exponential(xs, polluted, light).halflife
        self.assertAlmostEqual(near, 5.0, places=1)
        self.assertLess(far, near - 0.3)

    def test_standard_error_scales_like_one_over_sqrt_n(self):
        # Asymptotic theory: quadrupling the sample halves the standard error.
        # A curvature-based SE that got the factor of 2 wrong, or dropped the
        # weights, would not obey this.  Averaged over replicates because a
        # single draw's SE is itself noisy enough to swing the ratio by 30%.
        small = statistics.mean(self._draw(60)[0] for _ in range(120))
        big = statistics.mean(self._draw(240)[0] for _ in range(120))
        self.assertAlmostEqual(small / big, 2.0, delta=0.1)

    def test_standard_error_is_slightly_optimistic_but_calibrated(self):
        """Honest calibration check on the reported CI.

        The curvature SE conditions on the fitted amplitude, so it is expected
        to sit a little below the true sampling spread.  Measured over 200
        replicates it lands around 0.75 of the empirical standard deviation,
        i.e. the 95% CI is really closer to a 90% CI.  The bounds below are the
        measured behaviour, not a target: the point is that the SE is neither
        wildly optimistic nor meaningless.
        """
        draws = [self._draw(120) for _ in range(200)]
        mean_se = statistics.mean(d[0] for d in draws)
        empirical_sd = statistics.pstdev(d[1] for d in draws)
        self.assertGreater(mean_se / empirical_sd, 0.7)
        self.assertLess(mean_se / empirical_sd, 1.0)

    def test_too_few_points_is_rejected(self):
        with self.assertRaises(ValueError):
            fit_exponential([0.0, 1.0], [1.0, 0.5])


class TestNormalEquations(unittest.TestCase):

    def test_solves_a_known_regression(self):
        rows = [[1.0, x, x * x] for x in range(-5, 6)]
        ys = [2.0 - 3.0 * r[1] + 0.5 * r[2] for r in rows]
        beta = solve_normal_equations(rows, ys, [1.0] * len(rows))
        for got, want in zip(beta, (2.0, -3.0, 0.5)):
            self.assertAlmostEqual(got, want, places=8)

    def test_singular_design_raises(self):
        rows = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
        with self.assertRaises(ValueError):
            solve_normal_equations(rows, [1.0, 2.0, 3.0], [1.0, 1.0, 1.0])


class TestHarmonicBasis(unittest.TestCase):

    def test_basis_is_periodic(self):
        basis = harmonic_basis(110.0, 2)
        for a, b in zip(basis(17.0), basis(17.0 + 110.0)):
            self.assertAlmostEqual(a, b, places=9)

    def test_basis_cannot_represent_a_monotone_drift(self):
        """This is the property that buys back identification.

        A free per-calendar-bin seasonal index CAN represent exp(k*t), which is
        exactly the direction that is degenerate with fatigue decay.  A two-
        harmonic basis cannot, so the worst least-squares approximation error
        to a real drift must stay large.
        """
        basis = harmonic_basis(110.0, 2)
        days = [float(d) for d in range(110)]
        drift = [0.02 * d for d in days]  # log-space linear drift
        beta = solve_normal_equations([basis(d) for d in days], drift,
                                      [1.0] * len(days))
        resid = [t - sum(b * c for b, c in zip(basis(d), beta))
                 for d, t in zip(days, drift)]
        rms = math.sqrt(sum(r * r for r in resid) / len(resid))
        spread = max(drift) - min(drift)
        # A basis that could soak up the drift would leave rms ~ 0.  Harmonics
        # leave a substantial fraction of the drift's own range unexplained.
        self.assertGreater(rms / spread, 0.05)


if __name__ == "__main__":
    unittest.main()
