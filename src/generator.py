"""Synthetic ad-delivery generator with a PLANTED fatigue half-life.

The whole point of this generator is that the ground truth lives in the
*frequency* domain: the true response multiplier for a creative depends only
on how many times the average reachable user has already seen it, never on
how many days have elapsed.  Calendar time is then deliberately polluted with
two things that a calendar-time fatigue model cannot tell apart from decay:

  (a) a seasonal demand cycle shared by every creative, and
  (b) non-constant delivery (flighting, budget changes, a two-week pause).

A creative launched into the downslope of the season looks fatigued when it is
not; a creative whose budget was cut looks recovered when its audience simply
stopped seeing it.  Those two failure modes are what the test suite measures.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

# Response is modelled as  base * 2 ** (-F / halflife)  with F in
# impressions-per-reachable-user.  Expressing decay in half-lives (rather than
# an exponential rate) keeps the planted truth readable: "this creative is
# half as effective by the 6th time you see it".
LN2 = math.log(2.0)


@dataclass(frozen=True)
class DeliverySegment:
    """A stretch of days delivering `freq_per_day` impressions per reach unit."""

    days: int
    freq_per_day: float


@dataclass(frozen=True)
class CreativeSpec:
    creative_id: str
    launch_day: int
    base_response: float
    true_halflife_freq: float  # PLANTED ground truth, in frequency units
    schedule: tuple[DeliverySegment, ...]
    reach_users: int = 200_000
    note: str = ""

    def total_days(self) -> int:
        return sum(seg.days for seg in self.schedule)


@dataclass(frozen=True)
class DayRow:
    creative_id: str
    calendar_day: int
    days_since_launch: int
    freq_today: float
    cum_frequency: float  # cumulative impressions per reachable user, START of day
    impressions: float
    response_rate: float  # observed (noisy) response rate
    season_index: float  # generator truth, never handed to the fitters


@dataclass
class Panel:
    rows: list[DayRow] = field(default_factory=list)
    specs: dict[str, CreativeSpec] = field(default_factory=dict)
    season_period: int = 0
    season_amplitude: float = 0.0

    def for_creative(self, creative_id: str) -> list[DayRow]:
        return [r for r in self.rows if r.creative_id == creative_id]

    def creative_ids(self) -> list[str]:
        return list(self.specs.keys())


def season_index(calendar_day: int, period: int, amplitude: float) -> float:
    """Shared multiplicative demand cycle.

    A cosine peaking at day 0 means every creative that launches early rides a
    downslope -- exactly the confound that makes a calendar-time decay fit look
    convincing while being wrong.
    """
    return 1.0 + amplitude * math.cos(2.0 * math.pi * calendar_day / period)


def true_fatigue(cum_frequency: float, halflife: float) -> float:
    return 2.0 ** (-cum_frequency / halflife)


def default_specs():
    """Eight creatives spanning the failure modes we want to expose.

    C1/C2 are the budget pair: identical delivery SHAPE, one running at double
    the rate, same planted half-life.  C3 is the paused creative.

    Every creative delivers on a stepped (flighted) schedule, and that is a
    modelling requirement, not decoration.  If every creative delivered at a
    constant rate, cumulative frequency would be a straight line in calendar
    day for all of them, and a spurious exponential calendar drift exp(k*t)
    would be *exactly* absorbable into each creative's frequency half-life --
    the seasonal index and the fatigue curves would not be separately
    identified.  Step changes at different days make frequency a nonlinear,
    creative-specific function of calendar time, which is what pins the shared
    seasonal index down.
    """
    return [
        CreativeSpec("C1_base_rate", 0, 0.0420, 6.0,
                     (DeliverySegment(28, 0.06), DeliverySegment(28, 0.14),
                      DeliverySegment(28, 0.05), DeliverySegment(28, 0.16)),
                     note="budget pair: 1x delivery"),
        CreativeSpec("C2_double_rate", 0, 0.0415, 6.0,
                     (DeliverySegment(28, 0.12), DeliverySegment(28, 0.28),
                      DeliverySegment(28, 0.10), DeliverySegment(28, 0.32)),
                     note="budget pair: same shape at 2x delivery"),
        CreativeSpec("C3_paused", 6, 0.0380, 5.0,
                     (DeliverySegment(34, 0.16),
                      DeliverySegment(14, 0.0),      # the two-week dark period
                      DeliverySegment(50, 0.16)),
                     note="paused for 14 days from day 40"),
        CreativeSpec("C4_fast_burn", 3, 0.0505, 2.4,
                     (DeliverySegment(20, 0.05), DeliverySegment(30, 0.18),
                      DeliverySegment(25, 0.08), DeliverySegment(25, 0.20)),
                     note="burns out fast"),
        CreativeSpec("C5_slow_burn", 10, 0.0310, 14.0,
                     (DeliverySegment(30, 0.20), DeliverySegment(30, 0.06),
                      DeliverySegment(40, 0.22)),
                     note="long-lived evergreen"),
        CreativeSpec("C6_flighted", 14, 0.0450, 7.5,
                     (DeliverySegment(20, 0.22), DeliverySegment(20, 0.04),
                      DeliverySegment(20, 0.26), DeliverySegment(30, 0.06)),
                     note="heavy flighting"),
        CreativeSpec("C7_late_launch", 40, 0.0395, 4.0,
                     (DeliverySegment(25, 0.24), DeliverySegment(35, 0.07)),
                     note="launched into the seasonal trough"),
        CreativeSpec("C8_thin", 70, 0.0400, 5.0,
                     (DeliverySegment(6, 0.05),),
                     reach_users=4_000,
                     note="deliberately too little data to fit"),
    ]


def generate_panel(
    specs: list[CreativeSpec] | None = None,
    *,
    seed: int = 20260912,
    season_period: int = 110,
    season_amplitude: float = 0.30,
    noise_cv: float = 0.035,
) -> Panel:
    """Daily aggregate panel.

    noise_cv is the coefficient of variation of the daily observed response
    rate.  3.5% is the order of day-to-day wobble you see on a social ad set
    with six figures of daily impressions once measurement noise, audience mix
    and auction dynamics are folded together; it is small enough that the
    planted half-life is recoverable and large enough that a fit which merely
    interpolates noise does not pass.
    """
    if specs is None:
        specs = default_specs()
    rng = random.Random(seed)
    panel = Panel(specs={s.creative_id: s for s in specs},
                  season_period=season_period,
                  season_amplitude=season_amplitude)

    for spec in specs:
        cum = 0.0
        day_offset = 0
        for seg in spec.schedule:
            for _ in range(seg.days):
                cal = spec.launch_day + day_offset
                s = season_index(cal, season_period, season_amplitude)
                # Fatigue is evaluated at the cumulative frequency the audience
                # carried INTO the day, so a zero-delivery day cannot move it.
                clean = spec.base_response * true_fatigue(cum, spec.true_halflife_freq) * s
                impressions = seg.freq_per_day * spec.reach_users
                if impressions > 0.0:
                    observed = clean * math.exp(rng.gauss(0.0, noise_cv) - 0.5 * noise_cv ** 2)
                    panel.rows.append(DayRow(
                        creative_id=spec.creative_id,
                        calendar_day=cal,
                        days_since_launch=day_offset,
                        freq_today=seg.freq_per_day,
                        cum_frequency=cum,
                        impressions=impressions,
                        response_rate=observed,
                        season_index=s,
                    ))
                cum += seg.freq_per_day
                day_offset += 1
    return panel
