# fatigue

> Creative fatigue modelled against cumulative frequency rather than calendar time, so seasonality stops masquerading as burnout.

> **Implementation note.** The runnable reference is pure Python standard library: a hand-rolled weighted nonlinear least-squares fit of `response = a * 2^(-frequency / half-life)` (one per creative), with a shared smooth seasonal curve estimated by alternating least squares. It uses daily aggregate cumulative frequency per reachable user rather than user-level sequences, and a parametric decay rather than a lifelines/statsmodels hazard model. Output is a plain-text demo, not DuckDB/Plotly/Quarto. Those remain the target stack.



`COMPACT` · **Marketing Analyst** · Intermediate · ~5 days · Retail - social advertising

**Primary language:** Python
**Tags:** `survival-analysis`, `hazard-models`, `marketing`, `deconfounding`, `reporting`

---

## The problem

Creative performance decays with repeated exposure, but teams refresh creative on a fixed calendar - every six weeks, regardless. Some creatives are burned out in nine days and some are still performing at week twelve, so the calendar wastes money in both directions simultaneously.

## ⭐ The differentiator

Models decay as a **function of cumulative per-user frequency rather than calendar time**, fitted as a hazard model. A creative shown twelve times in a week and one shown twelve times over two months sit at the same point on the fatigue curve; a calendar-time model conflates fatigue with every seasonal effect in the period and cannot tell them apart. The output is a per-creative predicted refresh date with a confidence band.

This is the sentence to lead with when someone asks you to walk through the
project. Everything else in this repo exists to make it true and to prove it.

## Data

A documented synthetic generator producing user-level exposure sequences with **known true fatigue parameters** per creative plus confounding seasonality - enabling both recovery testing and a direct demonstration that the calendar-time model fails.

> No paid API key is required to run or demo this project. Where a paid
> service would add value it is wired as an optional enhancement behind an
> interface with an offline mock as the default implementation.

## Stack

- Python: lifelines / statsmodels
- DuckDB, Plotly
- Quarto report
- pytest

## Core capabilities

- Frequency-domain fatigue curve fitting with per-creative parameters
- Seasonality deconfounding separating calendar effects from exposure effects
- Predicted optimal refresh point with uncertainty, per creative
- Portfolio view ranking creatives by remaining useful life
- Counterfactual: modelled spend saved against the current fixed-calendar policy

## Repository layout

```
src/fatigue/
src/deconfound/
generator/
test/recovery/
report/
```

## Build plan

1. Generator with true fatigue parameters and a confounding seasonal effect.
2. Fit the calendar-time model first and watch it fail. Keep that result - it is half the argument.
3. Fit the frequency-domain model and recover the true parameters.
4. Portfolio view and the spend counterfactual last.

## Testing strategy

Assert fitted fatigue parameters **recover the generator's true values within their confidence intervals**. Then assert the calendar-time model **demonstrably fails on the same data** - proving the differentiator empirically rather than asserting it in the README.

Tests assert **correctness**, not merely that the code runs. A green suite on
this repo is a claim about behaviour under adversarial conditions; treat any
test that would pass against a deliberately broken implementation as a bug in
the test.

## Quality & safety layer

Per-creative estimates report their sample size; creatives below a minimum exposure count return 'insufficient data' rather than an unstable refresh date.

## Measurable outcome

> Fatigue curves for eight creatives (seven fitted, one refused as insufficient data) showing refresh points ranging from day 12 to day 70 - against the uniform 42-day policy that was wasting money in both directions.

State it in these terms — business units, not technical ones — in your CV
bullet and in the first thirty seconds of describing the project.

## Measured results

From `python -m src.demo` (seed 20260912, 8 creatives, 664 creative-days, 110-day season at +/-30%):

| creative | true half-life (exposures) | fitted | 95% CI | calendar model (days) |
|---|---|---|---|---|
| C1_base_rate | 6.00 | 5.98 | [5.86, 6.10] | 63.5 |
| C2_double_rate | 6.00 | 6.02 | [5.94, 6.10] | 31.4 |
| C3_paused | 5.00 | 4.97 | [4.88, 5.06] | 39.9 |
| C4_fast_burn | 2.40 | 2.38 | [2.35, 2.42] | 19.4 |
| C5_slow_burn | 14.00 | 13.73 | [13.33, 14.14] | 110.2 |
| C6_flighted | 7.50 | 7.77 | [7.57, 7.97] | 60.1 |
| C7_late_launch | 4.00 | 3.97 | [3.89, 4.05] | 19.8 |
| C8_thin | 5.00 | insufficient data | - | - |

- **Budget pair** (same planted half-life, C2 delivers 2x): the frequency ratio is 1.006; the calendar ratio is 0.495, meaning the calendar model says C2 burns out twice as fast when it doesn't.
- **Paused creative** (14 dark days): fitted on pre-pause data only, the calendar model shows a spurious 85.5% "recovery" after the pause; the frequency model shows 0.8%.
- **Deseasonalising matters:** without the shared seasonal curve, the same frequency fit is biased by 15-32% on 5 of 7 creatives. With it, every creative is within 3.6% of the planted value.
- **Refresh plan** (refresh at 60% of launch response): refresh days run from day 12 (C7) to day 70 (C5) against the uniform 42-day policy. Four creatives are due 11-30 days before day 42, one is due on it, and two can run 5 and 28 days longer.
- Suite: 31 tests, about 30 s.

### Limitations

- The "95%" CI is really about 90%. The curvature SE comes out at ~0.75 of the measured sampling spread (tests/test_fitting.py). C6's CI misses its planted value on the demo seed (z = 2.6), and about 1 creative in 7 misses on each seed tried.
- The model is a parametric exponential decay on daily aggregates, not a semi-parametric hazard model on user-level exposure sequences. Recovery is only proven when the generator's decay matches that form.
- The season period (110 days) is supplied as configuration, not estimated. Identification also relies on creatives having stepped, non-constant delivery. With perfectly constant delivery, frequency is linear in time and fatigue cannot be separated from a seasonal drift.
- The spend counterfactual in money terms is not implemented. The demo reports days early or late versus the 42-day policy, not currency saved.

## Interview questions this project answers

- **Why model against frequency instead of time?**
- **What is a hazard model and why use one here?**
- **How do you separate fatigue from seasonality?**

## What this deliberately is *not*

- Not a creative-management platform. It answers one question the platform does not.


## Run it now

```bash
python -m unittest discover -s tests -v   # the suite
python -m src.demo                        # the 60-second artefact
```

Requires Python 3.11+. The runnable core uses **only the standard
library** (including `sqlite3`), so there is nothing to install.

## Getting started

```bash
git clone <your-fork-url> fatigue
cd fatigue
python -m unittest discover -s tests -v
python -m src.demo
```

Docker is supported but optional — every path above works on a plain
Windows/macOS/Linux laptop without a cloud account.

## Definition of done

- [ ] The differentiator above is implemented, and a test proves it
- [ ] The measurable outcome is produced by a command anyone can run
- [ ] `README` explains the one decision a generic version gets wrong
- [ ] CI runs the full suite on every push and is green on `main`
- [ ] A recruiter can see the headline artefact in under 60 seconds

## Licence

MIT — see [LICENSE](LICENSE).
