"""Portfolio fatigue analysis on two competing x-axes.

The frequency model regresses response on cumulative impressions-per-user.
The calendar model regresses the same response on days since launch.  Both go
through the same estimator and the same seasonal deconfounder, so the only
difference between them is what "how worn out is this creative" is measured
against.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .fitting import (ExpFit, fit_exponential, fit_with_seasonal_index,
                      harmonic_basis)
from .generator import Panel

# A creative needs enough observed days AND enough travelled distance along the
# fitted axis before a two-parameter decay fit means anything.  With much less
# than one extra exposure per reachable user, the amplitude and the half-life
# trade off almost perfectly: the fit is a straight line with an arbitrary
# curvature attached, and the reported refresh date is noise.
MIN_OBSERVED_DAYS = 12
MIN_FREQUENCY_SPAN = 1.0

# Length of the demand cycle the planner believes they are in, in days.  This
# is configuration, not something estimated from the response data: a media
# planner knows whether they are looking at a weekly, quarterly or annual
# cycle.  Both models receive the same value, so it cannot favour either.
SEASON_PERIOD_DAYS = 110.0
SEASON_HARMONICS = 2


@dataclass(frozen=True)
class CreativeReport:
    creative_id: str
    fitted: bool
    reason: str = ""
    halflife: float = float("nan")
    halflife_ci: tuple = (float("nan"), float("nan"))
    amplitude: float = float("nan")
    n_days: int = 0
    current_frequency: float = float("nan")
    recent_freq_per_day: float = float("nan")
    remaining_exposures: float = float("nan")
    remaining_days: float = float("nan")
    refresh_day: float = float("nan")
    freq_at_threshold: float = float("nan")


@dataclass
class PortfolioFit:
    axis: str  # "frequency" or "calendar"
    fits: dict
    season_fn: object
    season_coeffs: list
    skipped: dict

    def season_at(self, calendar_day) -> float:
        return self.season_fn(calendar_day)


def _eligible(rows):
    if len(rows) < MIN_OBSERVED_DAYS:
        return False, "insufficient data: %d observed days < %d" % (
            len(rows), MIN_OBSERVED_DAYS)
    span = max(r.cum_frequency for r in rows) - min(r.cum_frequency for r in rows)
    if span < MIN_FREQUENCY_SPAN:
        return False, "insufficient data: frequency span %.2f < %.2f" % (
            span, MIN_FREQUENCY_SPAN)
    return True, ""


def build_series(panel: Panel, axis: str, rows_by_creative=None):
    """key -> [(x, calendar_day, y, weight)] with x chosen by axis."""
    series = {}
    skipped = {}
    for cid in panel.creative_ids():
        rows = (rows_by_creative or {}).get(cid) or panel.for_creative(cid)
        ok, why = _eligible(rows)
        if not ok:
            skipped[cid] = why
            continue
        series[cid] = [
            (r.cum_frequency if axis == "frequency" else float(r.days_since_launch),
             r.calendar_day, r.response_rate, r.impressions)
            for r in rows
        ]
    return series, skipped


def fit_portfolio(panel: Panel, axis: str, rows_by_creative=None,
                  deconfound: bool = True,
                  season_period: float = SEASON_PERIOD_DAYS) -> PortfolioFit:
    if axis not in ("frequency", "calendar"):
        raise ValueError("axis must be 'frequency' or 'calendar'")
    series, skipped = build_series(panel, axis, rows_by_creative)
    basis = harmonic_basis(season_period, SEASON_HARMONICS)
    if deconfound:
        fits, season_fn, coeffs = fit_with_seasonal_index(series, basis=basis)
    else:
        fits = {k: fit_exponential([r[0] for r in v], [r[2] for r in v],
                                   [r[3] for r in v])
                for k, v in series.items()}
        season_fn, coeffs = (lambda cal: 1.0), None
    return PortfolioFit(axis=axis, fits=fits, season_fn=season_fn,
                        season_coeffs=coeffs, skipped=skipped)


def remaining_useful_exposures(fit: ExpFit, current_frequency: float,
                               threshold_fraction: float) -> float:
    """Extra impressions-per-user available before response falls to
    `threshold_fraction` of its launch-day level.  Negative means past due."""
    if not 0.0 < threshold_fraction < 1.0:
        raise ValueError("threshold_fraction must be in (0, 1)")
    freq_at_threshold = fit.halflife * math.log2(1.0 / threshold_fraction)
    return freq_at_threshold - current_frequency


def report_portfolio(panel: Panel, pf: PortfolioFit,
                     threshold_fraction: float = 0.60,
                     recent_window: int = 7):
    out = []
    for cid in panel.creative_ids():
        if cid in pf.skipped:
            out.append(CreativeReport(cid, False, pf.skipped[cid]))
            continue
        rows = panel.for_creative(cid)
        fit = pf.fits[cid]
        last = rows[-1]
        current_freq = last.cum_frequency + last.freq_today
        tail = rows[-recent_window:]
        # Rate over the *observed* tail: a paused creative reports no rows while
        # dark, so its remaining-days estimate is driven by how fast it delivers
        # when it is actually running.
        span_days = max(1, tail[-1].days_since_launch - tail[0].days_since_launch + 1)
        rate = sum(r.freq_today for r in tail) / span_days
        rem = remaining_useful_exposures(fit, current_freq, threshold_fraction)
        f_thr = current_freq + rem
        # Walk the ACTUAL delivery path to turn the frequency threshold into a
        # date.  Dividing the threshold by an average rate would silently skip
        # over the dark fortnight in the middle of C3's flight; the point of
        # the whole project is that dark days do not age a creative.
        refresh_day = float("inf")
        for r in rows:
            if r.cum_frequency + r.freq_today >= f_thr:
                refresh_day = float(r.days_since_launch)
                break
        if refresh_day == float("inf") and rate > 0:
            refresh_day = rows[-1].days_since_launch + (f_thr - current_freq) / rate
        out.append(CreativeReport(
            refresh_day=refresh_day, freq_at_threshold=f_thr,
            creative_id=cid, fitted=True, halflife=fit.halflife,
            halflife_ci=fit.ci95(), amplitude=fit.amplitude, n_days=len(rows),
            current_frequency=current_freq, recent_freq_per_day=rate,
            remaining_exposures=rem,
            remaining_days=(rem / rate) if rate > 0 else float("inf"),
        ))
    return out


# --- the pause diagnostic -------------------------------------------------

@dataclass(frozen=True)
class PauseDiagnostic:
    axis: str
    predicted: float
    observed: float
    recovery_signal: float  # observed / predicted - 1, seasonality removed


def pause_recovery_signal(panel: Panel, creative_id: str,
                          gap_min_days: int = 7,
                          post_days: int = 7):
    """Fit each model on PRE-pause data only, then ask what it expects on the
    first days after the creative comes back.

    In calendar time two idle weeks are two more weeks of decay, so the
    calendar model expects a materially lower response than where it left off
    -- and the creative comes back exactly where it was.  That positive
    residual is 'spurious recovery': the calendar model can only explain the
    audience having forgotten nothing as the creative bouncing back.  In
    frequency terms the audience saw the creative zero extra times, so the
    frequency model expects no change and finds none.
    """
    rows = panel.for_creative(creative_id)
    gap_at = None
    for prev, cur in zip(rows, rows[1:]):
        if cur.days_since_launch - prev.days_since_launch >= gap_min_days:
            gap_at = prev.days_since_launch
            break
    if gap_at is None:
        raise ValueError("%s has no delivery gap of >= %d days"
                         % (creative_id, gap_min_days))

    pre = [r for r in rows if r.days_since_launch <= gap_at]
    post = [r for r in rows if r.days_since_launch > gap_at][:post_days]

    held_out = {cid: panel.for_creative(cid) for cid in panel.creative_ids()}
    held_out[creative_id] = pre  # the post-pause window is never fitted

    result = {}
    for axis in ("frequency", "calendar"):
        pf = fit_portfolio(panel, axis, rows_by_creative=held_out)
        fit = pf.fits[creative_id]
        pred = obs = wsum = 0.0
        for r in post:
            x = r.cum_frequency if axis == "frequency" else float(r.days_since_launch)
            w = r.impressions
            pred += w * fit.predict(x) * pf.season_at(r.calendar_day)
            obs += w * r.response_rate
            wsum += w
        result[axis] = PauseDiagnostic(axis, pred / wsum, obs / wsum,
                                       (obs / pred) - 1.0)
    return result
