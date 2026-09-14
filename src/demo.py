"""60-second artefact: calendar-time fatigue vs frequency-domain fatigue.

Run with:  python -m src.demo
"""

from __future__ import annotations

import math

from .fatigue import (SEASON_PERIOD_DAYS, fit_portfolio, pause_recovery_signal,
                      report_portfolio)
from .generator import default_specs, generate_panel

RULE = "=" * 78
THRESHOLD = 0.60          # refresh when response falls to 60% of launch level
FIXED_POLICY_DAYS = 42    # the six-week calendar policy this project argues with


def _bar(value, scale=40.0, width=28):
    n = max(0, min(width, int(round(value / scale * width))))
    return "#" * n


def main() -> None:
    specs = {s.creative_id: s for s in default_specs()}
    panel = generate_panel()

    print(RULE)
    print("CREATIVE FATIGUE: cumulative frequency vs calendar time")
    print(RULE)
    print("Synthetic panel: %d creatives, %d creative-days, seasonal cycle of %d days."
          % (len(specs), len(panel.rows), int(SEASON_PERIOD_DAYS)))
    print("True fatigue is a function of impressions-per-user ONLY. Calendar time is")
    print("confounded with the seasonal cycle and with flighted/paused delivery.")
    print("")

    freq = fit_portfolio(panel, "frequency")
    cal = fit_portfolio(panel, "calendar")

    print("-- 1. RECOVERING THE PLANTED HALF-LIFE " + "-" * 39)
    print("")
    print("%-16s %10s %10s %16s %12s" % (
        "creative", "true HL", "freq HL", "freq 95% CI", "cal HL"))
    print("%-16s %10s %10s %16s %12s" % (
        "", "(exposures)", "(exposures)", "(exposures)", "(days)"))
    for cid, spec in specs.items():
        if cid in freq.skipped:
            print("%-16s %10.2f %10s %16s %12s" % (
                cid, spec.true_halflife_freq, "-", "insufficient data", "-"))
            continue
        f = freq.fits[cid]
        lo, hi = f.ci95()
        print("%-16s %10.2f %10.2f %16s %12.1f" % (
            cid, spec.true_halflife_freq, f.halflife,
            "[%.2f, %.2f]" % (lo, hi), cal.fits[cid].halflife))
    print("")
    print("The calendar column is not merely imprecise, it is on the wrong axis:")
    print("it reports a half-life in DAYS that depends on how fast you spent.")
    print("")

    print("-- 2. THE BUDGET PAIR " + "-" * 56)
    print("")
    print("C1 and C2 carry the SAME planted half-life (6.00 exposures). C2 simply")
    print("delivers twice as fast.")
    f1, f2 = freq.fits["C1_base_rate"], freq.fits["C2_double_rate"]
    c1, c2 = cal.fits["C1_base_rate"], cal.fits["C2_double_rate"]
    print("")
    print("  frequency model : %.2f vs %.2f exposures   ratio %.3f  (truth 1.000)"
          % (f1.halflife, f2.halflife, f2.halflife / f1.halflife))
    print("  calendar  model : %.1f vs %.1f days        ratio %.3f  (truth 1.000)"
          % (c1.halflife, c2.halflife, c2.halflife / c1.halflife))
    print("")
    print("  The calendar model says C2 fatigues twice as fast. It does not; it is")
    print("  simply shown twice as often. Double the budget, halve the calendar")
    print("  half-life, identical frequency curve.")
    print("")

    print("-- 3. THE PAUSED CREATIVE (the clearest proof) " + "-" * 31)
    print("")
    print("C3 ran for 34 days, went dark for 14 days, then resumed. Both models are")
    print("fitted on PRE-pause data only and asked to predict the first week back.")
    print("")
    diag = pause_recovery_signal(panel, "C3_paused")
    print("  %-10s %12s %12s %18s" % ("model", "predicted", "observed", "recovery signal"))
    for axis in ("calendar", "frequency"):
        d = diag[axis]
        print("  %-10s %12.5f %12.5f %17.1f%%" % (
            axis, d.predicted, d.observed, 100.0 * d.recovery_signal))
    print("")
    print("  Calendar time says 14 more days elapsed, so response should have decayed")
    print("  further; it did not, and the model can only read that as the creative")
    print("  bouncing back. Frequency says the audience saw it zero extra times, so")
    print("  nothing should have changed -- and nothing did.")
    print("")

    print("-- 4. REFRESH PLAN " + "-" * 58)
    print("")
    print("Refresh when modelled response falls to %d%% of its launch-day level,"
          % int(THRESHOLD * 100))
    print("i.e. after %.3f x half-life further exposures per reachable user."
          % math.log2(1.0 / THRESHOLD))
    print("")
    print("%-16s %10s %9s %9s %8s  %s" % (
        "creative", "exposures", "refresh", "vs D%d" % FIXED_POLICY_DAYS,
        "left", "when to refresh (day since launch)"))
    print("%-16s %10s %9s %9s %8s" % (
        "", "at limit", "on day", "(days)", "(exp.)"))
    reports = report_portfolio(panel, freq, threshold_fraction=THRESHOLD)
    fitted = [r for r in reports if r.fitted]
    for r in sorted(fitted, key=lambda r: r.refresh_day):
        print("%-16s %10.2f %9.0f %+9.0f %8.2f  %s" % (
            r.creative_id, r.freq_at_threshold, r.refresh_day,
            r.refresh_day - FIXED_POLICY_DAYS, r.remaining_exposures,
            _bar(r.refresh_day, scale=60.0)))
    for r in reports:
        if not r.fitted:
            print("%-16s %10s %9s %9s %8s  %s" % (
                r.creative_id, "-", "-", "-", "-", r.reason))
    print("")
    span = max(r.refresh_day for r in fitted) - min(r.refresh_day for r in fitted)
    print("Modelled refresh points span day %.0f to day %.0f (%.0f days of spread)"
          % (min(r.refresh_day for r in fitted),
             max(r.refresh_day for r in fitted), span))
    print("against a uniform %d-day policy. Negative 'vs D%d' = the policy keeps"
          % (FIXED_POLICY_DAYS, FIXED_POLICY_DAYS))
    print("spending on a creative already past its limit; positive = it retires a")
    print("creative that is still working. 'left' is remaining useful exposures at")
    print("the end of the panel; negative means already overdue.")
    print("")
    print(RULE)


if __name__ == "__main__":
    main()
