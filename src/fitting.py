"""Hand-rolled weighted nonlinear least squares for y = a * 2^(-x / H).

No scipy.  Two observations make this easy enough to do by hand:

  1. `a` is *conditionally linear*.  Fix H and the optimal amplitude has a
     closed form, so the two-parameter fit collapses to a one-dimensional
     profile over H.
  2. The profile SSE(H) is smooth and unimodal on a log-H grid, so a coarse
     geometric grid to bracket the minimum plus golden-section refinement gets
     machine-precision H without derivatives.

Gauss-Newton on both parameters would also work; profiling was chosen because
it cannot diverge and needs no step-size babysitting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

GOLDEN_INV = (math.sqrt(5.0) - 1.0) / 2.0


@dataclass(frozen=True)
class ExpFit:
    amplitude: float
    halflife: float
    halflife_se: float
    sse: float
    n: int

    def predict(self, x: float) -> float:
        return self.amplitude * 2.0 ** (-x / self.halflife)

    def ci95(self) -> tuple[float, float]:
        # Normal approximation; n is in the hundreds here so the t correction
        # is under 1% and would only dress the number up.
        return (self.halflife - 1.96 * self.halflife_se,
                self.halflife + 1.96 * self.halflife_se)


def _profile(xs, ys, ws, halflife):
    """Optimal amplitude and SSE for a fixed half-life (weighted, linear in a)."""
    num = 0.0
    den = 0.0
    for x, y, w in zip(xs, ys, ws):
        d = 2.0 ** (-x / halflife)
        num += w * y * d
        den += w * d * d
    if den <= 0.0:
        return 0.0, float("inf")
    a = num / den
    sse = 0.0
    for x, y, w in zip(xs, ys, ws):
        r = y - a * 2.0 ** (-x / halflife)
        sse += w * r * r
    return a, sse


def _golden(f, lo, hi, tol=1e-6, max_iter=200):
    """Golden-section minimisation on log(H); returns the minimising H."""
    llo, lhi = math.log(lo), math.log(hi)
    c = lhi - GOLDEN_INV * (lhi - llo)
    d = llo + GOLDEN_INV * (lhi - llo)
    fc, fd = f(math.exp(c)), f(math.exp(d))
    for _ in range(max_iter):
        if lhi - llo < tol:
            break
        if fc < fd:
            lhi, d, fd = d, c, fc
            c = lhi - GOLDEN_INV * (lhi - llo)
            fc = f(math.exp(c))
        else:
            llo, c, fc = c, d, fd
            d = llo + GOLDEN_INV * (lhi - llo)
            fd = f(math.exp(d))
    return math.exp(0.5 * (llo + lhi))


def fit_exponential(xs, ys, ws=None, *, lo=0.05, hi=5000.0, grid=60) -> ExpFit:
    xs = list(xs)
    ys = list(ys)
    ws = [1.0] * len(xs) if ws is None else list(ws)
    n = len(xs)
    if n < 3:
        raise ValueError("need at least 3 points to fit two parameters")

    obj = lambda h: _profile(xs, ys, ws, h)[1]

    # Geometric grid first: the profile is unimodal but very flat towards large
    # H (no-decay limit), so starting golden-section from the raw bracket can
    # walk into the flat tail.  Bracketing on a coarse grid avoids that.
    step = (hi / lo) ** (1.0 / (grid - 1))
    best_i, best_v = 0, float("inf")
    vals = []
    for i in range(grid):
        h = lo * step ** i
        v = obj(h)
        vals.append((h, v))
        if v < best_v:
            best_i, best_v = i, v
    blo = vals[max(0, best_i - 1)][0]
    bhi = vals[min(grid - 1, best_i + 1)][0]
    h_hat = _golden(obj, blo, bhi)
    a_hat, sse = _profile(xs, ys, ws, h_hat)

    # Asymptotic SE from the curvature of the profile SSE.  For weighted NLS,
    # Var(H) ~ 2 * sigma2_w / (d2 SSE / dH2), with sigma2_w = SSE / (n - 2).
    dof = max(1, n - 2)
    sigma2 = sse / dof
    eps = max(1e-4, h_hat * 1e-3)
    curv = (obj(h_hat + eps) - 2.0 * sse + obj(h_hat - eps)) / (eps * eps)
    se = math.sqrt(2.0 * sigma2 / curv) if curv > 0 else float("inf")
    return ExpFit(amplitude=a_hat, halflife=h_hat, halflife_se=se, sse=sse, n=n)


def solve_normal_equations(rows, ys, ws):
    """Weighted OLS by Gaussian elimination with partial pivoting.

    Small (<= 6 columns here), so the normal equations are numerically fine
    and a QR would be ceremony.
    """
    k = len(rows[0])
    a = [[0.0] * (k + 1) for _ in range(k)]
    for r, y, w in zip(rows, ys, ws):
        for i in range(k):
            for j in range(k):
                a[i][j] += w * r[i] * r[j]
            a[i][k] += w * r[i] * y
    for c in range(k):
        piv = max(range(c, k), key=lambda i: abs(a[i][c]))
        if abs(a[piv][c]) < 1e-14:
            raise ValueError("singular design matrix in seasonal fit")
        a[c], a[piv] = a[piv], a[c]
        for i in range(k):
            if i == c:
                continue
            f = a[i][c] / a[c][c]
            for j in range(c, k + 1):
                a[i][j] -= f * a[c][j]
    return [a[i][k] / a[i][i] for i in range(k)]


def harmonic_basis(period: float, harmonics: int = 2):
    """Smooth seasonal basis: intercept plus `harmonics` cos/sin pairs.

    A free multiplier per calendar bin was tried first and is a trap.  With
    constant-ish delivery, cumulative frequency is near-linear in calendar day,
    so a spurious exponential calendar drift exp(k*t) is almost perfectly
    absorbable into the fatigue half-lives -- and an unconstrained bin index
    happily supplies exactly that drift.  The measured cost was a ~30% downward
    bias in every recovered half-life while the total SSE went DOWN, i.e. the
    problem is weak identification, not a bad optimiser.  A low-dimensional
    smooth basis cannot represent a monotone drift, which restores
    identification.  Demand seasonality really is smooth, so this is a
    restriction the domain supports rather than one invented to pass a test.
    """
    def basis(calendar_day: float):
        out = [1.0]
        for j in range(1, harmonics + 1):
            w = 2.0 * math.pi * j * calendar_day / period
            out.append(math.cos(w))
            out.append(math.sin(w))
        return out
    return basis


def fit_with_seasonal_index(
    series,
    *,
    basis,
    iterations: int = 40,
):
    """Alternating least squares: per-series decay + one shared seasonal curve.

    `series` maps key -> list of (x, calendar_day, y, weight).  Both the
    frequency model and the calendar model go through this identical routine,
    so any difference in the answers is attributable to the x-axis alone and
    not to one model getting a better seasonality treatment than the other.

    Returns (fits, season_fn) where season_fn(calendar_day) -> multiplier.
    """
    keys = list(series.keys())
    coeffs = None
    season = lambda cal: 1.0
    fits = {}

    for _ in range(iterations):
        for k in keys:
            rows = series[k]
            xs = [r[0] for r in rows]
            ys = [r[2] / season(r[1]) for r in rows]
            ws = [r[3] for r in rows]
            prev_fit = fits.get(k)
            if prev_fit is None:
                fits[k] = fit_exponential(xs, ys, ws)
            else:
                # Warm start: after the first alternation the seasonal curve
                # moves by very little, so the half-life stays inside a narrow
                # bracket.  Re-running the full geometric sweep every iteration
                # made the suite ~8x slower for identical answers.
                h = prev_fit.halflife
                fits[k] = fit_exponential(xs, ys, ws, lo=h / 2.0, hi=h * 2.0,
                                          grid=9)

        # Seasonal step in LOG space: the model is multiplicative, so the
        # seasonal curve is linear in the harmonic coefficients once logged,
        # which makes this half of the alternation a closed-form OLS.
        design, targets, weights = [], [], []
        for k in keys:
            f = fits[k]
            for x, cal, y, w in series[k]:
                pred = f.predict(x)
                if pred <= 0.0 or y <= 0.0:
                    continue
                design.append(basis(cal))
                targets.append(math.log(y / pred))
                weights.append(w)
        new = solve_normal_equations(design, targets, weights)
        # The intercept is not identified separately from the amplitudes; drop
        # it into the amplitudes by zeroing it out here.
        new[0] = 0.0
        prev = coeffs
        coeffs = new
        season = (lambda c: (lambda cal: math.exp(sum(bi * ci for bi, ci in zip(basis(cal), c)))))(coeffs)
        if prev is not None and max(abs(a - b) for a, b in zip(prev, coeffs)) < 1e-10:
            break

    return fits, season, coeffs
