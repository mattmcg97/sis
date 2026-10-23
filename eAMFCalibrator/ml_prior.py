"""The pre-match moneyline model: each match's P(home wins), walk-forward off the match history alone, for
the moneyline nudge (v21-ml) to meet at the kick-off, with its expected total and each side's points beside it,
so one table is a whole prior for v2.1. Every match of a span is priced, the scheduled ones too.

A pre-match moneyline study (Jun-Oct 2026, walk-forward) found the drift-4 form rating's margins too spread out
and the shrink they need drifting (0.58 in June, 0.45 by mid-October); a slower margin with a fitted link beat
prod's pre-match moneyline by about 0.010 log loss on Aug-Oct, level with the walkthrough avg table's own.

    python -m eAMFCalibrator ml-prior                    # every match in out/match_history.csv -> out/ml_prior/
    python -m eAMFCalibrator ml-prior --since 2026-10-01 --table eAMFCalibrator/out/walkthrough/priors/avg.csv
    python -m eAMFCalibrator bets run --candidate v21,v21-ml --v21-ml eAMFCalibrator/out/ml_prior/ml_prior.csv \\
        --v21-prior-model eAMFCalibrator/out/ml_prior/ml_prior.csv

  the margin   the home side's expected margin: a home edge per stream plus the two gamers' strengths and the
               two teams', a ridge on every result in by the kick-off (RESULT_LAG after its own kick-off), each
               weighted 2^(-age / HALF_LIFE days), the penalties in matches of full weight. It is solved again
               at each kick-off a new result has come in before, so a match is priced off every result in by
               then: re-solving once a day, at midnight, cost the study 0.003-0.004 log loss
  the link     P(home wins) = logistic(a + b x margin), fitted each UTC day on the ridge's own walk-forward
               margins of the last LINK_DAYS days' results in by that day's midnight (ties left out)
  the stack    (--table) where a prior table (MATCH_CODE, exp_margin: the walkthrough's avg.csv) prices the
               match, logistic(a + b1 x its margin + b2 x the ridge's), fitted the same way on the matches both
               priced; with fewer than MIN_ROWS of those behind a day, that day's matches get the link. The
               study found the two weights about equal, and the stack 0.001-0.002 better than the link alone
  the total    an online Kalman filter on each result's total: the league's mean drifting, each gamer's and
               team's effect a random walk from when first met, a stream's against the first, the covariance in
               full, each result in RESULT_LAG after its kick-off (TOTAL_FILTER: fitted on Jun-Jul). The study's
               best single expected total: Aug-Oct RMSE 12.47 against a daily ridge's 12.54 (prod's line 12.84)
  the points   Pred_P1_Points / Pred_P2_Points = (exp_total +- exp_margin) / 2, the ridge's margin, for the points
               nudge (--v21-prior-model); the moneyline nudge then moves the margin to meet p_home. sd_total =
               12.3 x (exp_total / 35)^0.75 is the study's fitted spread, informational: the sim makes its own

Writes (to --out) ml_prior.csv, every match of the span: its kick-off, gamers, teams, stream and result, the
ridge's exp_margin, exp_total, sd_total and the points, the day's link (link_a, link_b) and stack, and p_home:
the same file serves --v21-prior-model (the points) and --v21-ml (p_home). Also ml_prior.txt: the run, the link
month by month and, where results are in, the log loss, Brier and calibration slope of p_home and the expected
total's RMSE, beside prod-free baselines.
"""

import contextlib
import csv
import datetime as dt
import math
import os
import time
from collections import namedtuple
from statistics import NormalDist

import numpy as np

from .money_layer import _num, _time, point_columns, read_table

try:
    from threadpoolctl import threadpool_limits
except ImportError:                     # without it BLAS keeps its own threads: fine on a quiet machine
    threadpool_limits = None

RESULT_LAG = dt.timedelta(minutes=36)   # a result is in 36 minutes after its kick-off (eAMFModel's stream, xgb.features)
HALF_LIFE = 30.0           # days: a result counts half as much 30 days on
GAMER_PENALTY = 2.0        # the ridge's penalties, in matches of full weight: gamers play hundreds of matches,
TEAM_PENALTY = 50.0        # so these barely matter; the half-life does
STREAM_PENALTY = 50.0      # a stream's home edge about the common one
HOME_PENALTY = 1e-3        # the common home edge: all but free
LINK_DAYS = 60             # the link's results: those of the last 60 days in by the day's midnight
MIN_ROWS = 300             # the fewest results a day's link, or stack, is fitted on
BURN_IN_DAYS = 36          # the history's first days, while the ridge learns the gamers, kept out of the links
MARGIN_SD = 9.35           # rematch_layer's normal about the margin: the baseline reading the ridge's margin as one
EPS = 1e-6                 # a chance graded is kept this far from 0 and 1
# the total's filter, its variances as ratios to a match's noise (only the ratios move the mean; fitted on Jun-Jul)
TOTAL_FILTER = dict(V_g=0.04589067038085432,      # a gamer's effect when first met,
                    q_g=0.0005070349553707072,    # and its drift a day
                    V_t=10.502638142456593,       # a team's
                    q_t=1.588023246525882e-05,
                    q_mu=0.00022692080486846303,  # the league's mean total: its drift a day, from a diffuse start
                    V_s=5.476205434189408e-07)    # a stream's against the first: static
MU_PRIOR = 1e3
TOTAL_SD = (12.3, 35.0, 0.75)    # the total's sd, 12.3 at an expected 35, x (expected / 35)^0.75: informational

COLUMNS = ["MATCH_CODE", "start", "stream", "home", "away", "home_team", "away_team", "home_score", "away_score",
           "exp_margin", "exp_total", "sd_total", "Pred_P1_Points", "Pred_P2_Points", "link_a", "link_b", "p_ridge",
           "table_margin", "stack_a", "stack_b_table", "stack_b_ridge", "p_home", "priced_by"]

Match = namedtuple("Match", "start code stream home away home_team away_team home_score away_score")


def read_matches(path):
    """[Match] off a match_history.csv in kick-off order (then code); the scores are None until both are in."""
    out = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("SPORT_CODE", "AF") not in ("AF", "", None):
                continue
            t = _time(r.get("SCHEDULED_START_TIME_UTC"))
            code = (r.get("MATCH_CODE") or "").strip()
            home = (r.get("PLAYER_1_HANDLE") or "").strip().upper()
            away = (r.get("PLAYER_2_HANDLE") or "").strip().upper()
            if t is None or not code or not home or not away:
                continue
            s = _num(r.get("STREAM_NUMBER"))
            hs, as_ = _num(r.get("PLAYER_1_FINAL_SCORE")), _num(r.get("PLAYER_2_FINAL_SCORE"))
            if hs is None or as_ is None:
                hs = as_ = None
            out.append(Match(t, code, "" if s is None else f"{s:g}", home, away, (r.get("PLAYER_1_TEAM") or "").strip(),
                             (r.get("PLAYER_2_TEAM") or "").strip(), hs, as_))
    out.sort(key=lambda m: (m.start, m.code))
    return out


def one_thread():
    """BLAS held to one thread while the ridge, the total's filter and the links are fitted (threadpoolctl, where
    it's installed). Their solves are 100 wide or less, quickest on one thread; on a busy machine a thread pool's
    threads wait on each other, and a solve that takes a quarter of a millisecond on one thread took over a second."""
    return threadpool_limits(1, user_api="blas") if threadpool_limits else contextlib.nullcontext()


def day_of(t):
    """Days since 1970 for a naive UTC time (read as UTC, not in the machine's local time)."""
    return t.replace(tzinfo=dt.timezone.utc).timestamp() / 86400.0


def stream_order(streams):
    """The streams in order, by number where they are numbers."""
    return sorted(streams, key=lambda s: (_num(s) if _num(s) is not None else math.inf, s))


# ---------------------------------------------------------------- the margin

class Ridge:
    """The margin model kept as its decayed normal equations: a match's margin = home + home[stream] +
    gamer[home] - gamer[away] + team[home's] - team[away's]. A result is added once it is in, the sums decaying
    by 2^(-days / half_life) as time goes on, and the strengths are solved for at a kick-off. A team left blank
    counts as an average one."""

    def __init__(self, matches, half_life=HALF_LIFE, gamer_penalty=GAMER_PENALTY, team_penalty=TEAM_PENALTY,
                 stream_penalty=STREAM_PENALTY):
        gamers = sorted({m.home for m in matches} | {m.away for m in matches})
        teams = sorted(({m.home_team for m in matches} | {m.away_team for m in matches}) - {""})
        streams = stream_order({m.stream for m in matches} - {""})
        self.gamers = {g: 1 + k for k, g in enumerate(gamers)}
        self.teams = {t: 1 + len(gamers) + k for k, t in enumerate(teams)}
        self.streams = {s: 1 + len(gamers) + len(teams) + k for k, s in enumerate(streams)}
        self.penalty = np.concatenate([[HOME_PENALTY], np.full(len(gamers), float(gamer_penalty)),
                                       np.full(len(teams), float(team_penalty)),
                                       np.full(len(streams), float(stream_penalty))])
        width = len(self.penalty)
        self.S = np.zeros((width, width))      # the weighted X'X and X'y, as of day self.t
        self.r = np.zeros(width)
        self.rate = math.log(2.0) / half_life if half_life else 0.0
        self.t = None
        self.beta = np.zeros(width)
        self.results = 0
        self.solved_at = None

    def row(self, m):
        """(columns, values) of a match's margin."""
        x = {0: 1.0}
        if m.stream in self.streams:
            x[self.streams[m.stream]] = 1.0
        for name, sign, index in ((m.home, 1.0, self.gamers), (m.away, -1.0, self.gamers),
                                  (m.home_team, 1.0, self.teams), (m.away_team, -1.0, self.teams)):
            if name in index:
                x[index[name]] = x.get(index[name], 0.0) + sign
        cols = np.array(sorted(k for k, v in x.items() if v), dtype=int)
        return cols, np.array([x[k] for k in cols])

    def add(self, m, day_in):
        """A result in at `day_in` (days since 1970), weighted by its age since its kick-off."""
        if self.t is None:
            self.t = day_in
        if day_in > self.t:
            f = math.exp(-self.rate * (day_in - self.t))
            self.S *= f
            self.r *= f
            self.t = day_in
        cols, val = self.row(m)
        w = math.exp(-self.rate * (day_in - day_of(m.start)))
        self.S[np.ix_(cols, cols)] += w * np.outer(val, val)
        self.r[cols] += w * val * (m.home_score - m.away_score)
        self.results += 1

    def solve(self, day):
        """The strengths as of `day`: the sums decayed to it, the penalties added."""
        if self.t is None:
            return self.beta
        f = math.exp(-self.rate * max(day - self.t, 0.0))
        self.beta = np.linalg.solve(f * self.S + np.diag(self.penalty), f * self.r)
        self.solved_at = day
        return self.beta

    def margin(self, m):
        cols, val = self.row(m)
        return float(val @ self.beta[cols])


def margins(matches, half_life=HALF_LIFE, gamer_penalty=GAMER_PENALTY, team_penalty=TEAM_PENALTY,
            stream_penalty=STREAM_PENALTY, lag=RESULT_LAG):
    """([each match's expected home margin at its kick-off], the ridge as of the last solve), `matches` in
    kick-off order. A result is in `lag` after its own kick-off; the ridge is solved again at each kick-off a
    new result has come in before, and a match is priced off every result in by its kick-off."""
    ridge = Ridge(matches, half_life, gamer_penalty, team_penalty, stream_penalty)
    settled = sorted((m.start + lag, k) for k, m in enumerate(matches) if m.home_score is not None)
    out, j = [], 0
    with one_thread():
        for m in matches:
            new = False
            while j < len(settled) and settled[j][0] <= m.start:
                t_in, k = settled[j]
                ridge.add(matches[k], day_of(t_in))
                j += 1
                new = True
            if new:
                ridge.solve(day_of(m.start))
            out.append(ridge.margin(m))
    return out, ridge


# ---------------------------------------------------------------- the total

class TotalFilter:
    """The expected total as an online Kalman filter: a match's total = mu + stream + gamer[home] + gamer[away] +
    team[home's] + team[away's] + noise. mu drifts q_mu a day from a diffuse start; a gamer's effect starts
    N(0, V_g) when he is first met and drifts q_g a day from then, a team's N(0, V_t) and q_t; a stream's, against
    the first, N(0, V_s) and stays. The variances are ratios to the noise's. The covariance is kept in full, so a
    result moves every effect it is correlated with (the opponents', the teams')."""

    def __init__(self, matches, settings=TOTAL_FILTER):
        gamers = sorted({m.home for m in matches} | {m.away for m in matches})
        teams = sorted(({m.home_team for m in matches} | {m.away_team for m in matches}) - {""})
        streams = stream_order({m.stream for m in matches} - {""})[1:]
        self.streams = {s: 1 + k for k, s in enumerate(streams)}
        self.gamers = {g: 1 + len(streams) + k for k, g in enumerate(gamers)}
        self.teams = {t: 1 + len(streams) + len(gamers) + k for k, t in enumerate(teams)}
        width = 1 + len(streams) + len(gamers) + len(teams)
        self.s = settings
        self.x = np.zeros(width)
        self.P = np.zeros((width, width))
        self.P[0, 0] = MU_PRIOR
        for k in self.streams.values():
            self.P[k, k] = settings["V_s"]
        self.q = np.zeros(width)
        self.q[0] = settings["q_mu"]
        self.active = np.zeros(width, bool)       # a gamer or team drifts once met
        self.active[0] = True
        self.t = None
        self.updates = 0

    def drift(self, day):
        if self.t is None:
            self.t = day
        elif day > self.t:
            self.P[np.diag_indices_from(self.P)] += self.q * self.active * (day - self.t)
            self.t = day

    def row(self, m):
        """(columns, values) of a match's total; its gamers and teams met."""
        x = {0: 1.0}
        if m.stream in self.streams:
            x[self.streams[m.stream]] = 1.0
        for name, index, v, q in ((m.home, self.gamers, "V_g", "q_g"), (m.away, self.gamers, "V_g", "q_g"),
                                  (m.home_team, self.teams, "V_t", "q_t"), (m.away_team, self.teams, "V_t", "q_t")):
            if name in index:
                k = index[name]
                if not self.active[k]:
                    self.active[k] = True
                    self.P[k, k] = self.s[v]
                    self.q[k] = self.s[q]
                x[k] = x.get(k, 0.0) + 1.0
        cols = np.array(sorted(x), dtype=int)
        return cols, np.array([x[k] for k in cols])

    def update(self, m, day_in):
        """A result in at `day_in` (days since 1970)."""
        self.drift(day_in)
        cols, val = self.row(m)
        Ph = self.P[:, cols] @ val
        gain = Ph / (val @ Ph[cols] + 1.0)
        self.x += gain * (m.home_score + m.away_score - val @ self.x[cols])
        self.P -= np.outer(gain, Ph)
        self.updates += 1

    def total(self, m, day):
        """The expected total of a match kicking off at `day`."""
        self.drift(day)
        cols, val = self.row(m)
        return float(val @ self.x[cols])


def totals(matches, settings=TOTAL_FILTER, lag=RESULT_LAG):
    """[each match's expected total at its kick-off, off every result in by then (NaN before the first)],
    `matches` in kick-off order. A result is in `lag` after its own kick-off."""
    f = TotalFilter(matches, settings)
    settled = sorted((m.start + lag, k) for k, m in enumerate(matches) if m.home_score is not None)
    out, j = [], 0
    with one_thread():
        for m in matches:
            while j < len(settled) and settled[j][0] <= m.start:
                t_in, k = settled[j]
                f.update(matches[k], day_of(t_in))
                j += 1
            total = f.total(m, day_of(m.start))           # met at its kick-off: its gamers drift from then
            out.append(total if f.updates else math.nan)
    return out


def total_sd(total):
    """The total's sd at an expected total (variance growing as the total^1.5): informational, the sims make
    their own."""
    sd, at, power = TOTAL_SD
    return sd * (total / at) ** power if total > 0 else math.nan


# ---------------------------------------------------------------- the link

def logistic_fit(X, y, iters=50, tol=1e-9):
    """The maximum-likelihood logistic regression of y (1 or 0) on X's columns (an intercept column among
    them), by Newton steps."""
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(X @ b)))
        hess = (X * (p * (1.0 - p))[:, None]).T @ X
        step = np.linalg.solve(hess + 1e-10 * np.eye(X.shape[1]), X.T @ (p - y))
        b -= step
        if np.max(np.abs(step)) < tol:
            break
    return b


def daily_links(days, starts, y, X, ok, link_days=LINK_DAYS, min_rows=MIN_ROWS, lag=RESULT_LAG):
    """{day: the link's coefficients, or None with fewer than min_rows results} for each day (a date), fitted at
    its midnight on the results in by then (`lag` after their kick-off; y NaN for a tie or none) whose kick-off
    is within the last link_days days, of the rows with `ok`. starts: the kick-offs (numpy datetime64)."""
    have = ok & np.isfinite(y)
    lag = np.timedelta64(int(lag.total_seconds()), "s")
    out = {}
    for d in days:
        cut = np.datetime64(d.isoformat(), "s")
        use = have & (starts + lag <= cut) & (starts >= cut - np.timedelta64(int(link_days * 86400), "s"))
        out[d] = logistic_fit(X[use], y[use]) if use.sum() >= min_rows else None
    return out


def table_margins(path):
    """{match: (expected home margin, the table's own p_home, its expected total; None where it has none)} off a
    prior table: MATCH_CODE and exp_margin, or the two sides' points."""
    rows, cols = read_table(path)
    pts = point_columns(cols)
    out = {}
    for r in rows:
        code = (r.get("MATCH_CODE") or "").strip()
        m, total = _num(r.get("exp_margin")), _num(r.get("exp_total"))
        h, a = (_num(r.get(pts[0])), _num(r.get(pts[1]))) if pts else (None, None)
        if m is None and h is not None and a is not None:
            m = h - a
        if total is None and h is not None and a is not None:
            total = h + a
        if code and m is not None:
            p = _num(r.get("p_home"))
            out[code] = (m, p if p is not None and 0.0 < p < 1.0 else None, total)
    if not out:
        raise SystemExit(f"--table {path}: no expected margins (exp_margin, or the two sides' points)")
    return out


def price(matches, exp, table=None, days=None, link_days=LINK_DAYS, min_rows=MIN_ROWS, burn_in=BURN_IN_DAYS,
          lag=RESULT_LAG, total=None):
    """Each match of `days` (dates; default every match's) priced: a dict with its link (link_a, link_b,
    p_ridge), the stack's (table_margin, stack_a, stack_b_table, stack_b_ridge, where the table prices) and
    p_home (NaN before the link has min_rows results), and with `total` (the filter's expected totals) its
    exp_total, sd_total and each side's points, in `matches`' order. exp: the ridge's margins; table:
    {match: (margin, p_home, total)} (table_margins). The history's first burn_in days are kept out of the links."""
    n = len(matches)
    total = np.full(n, np.nan) if total is None else np.asarray(total, float)
    starts = np.array([np.datetime64(m.start, "s") for m in matches])
    y = np.array([np.nan if m.home_score is None or m.home_score == m.away_score else float(m.home_score > m.away_score)
                  for m in matches])
    ridge = np.asarray(exp, float)
    first = np.datetime64(matches[0].start.date().isoformat(), "s") + np.timedelta64(int(burn_in * 86400), "s")
    ok = starts >= first
    day = [m.start.date() for m in matches]
    days = sorted(set(day) if days is None else set(days))
    ones = np.ones(n)
    tm = np.array([table[m.code][0] if table and m.code in table else np.nan for m in matches])
    stacks = {}
    with one_thread():
        links = daily_links(days, starts, y, np.column_stack([ones, ridge]), ok, link_days, min_rows, lag)
        if table:
            stacks = daily_links(days, starts, y, np.column_stack([ones, np.nan_to_num(tm), ridge]),
                                 ok & np.isfinite(tm), link_days, min_rows, lag)
    want = set(days)
    out = []
    for k, m in enumerate(matches):
        if day[k] not in want:
            continue
        r = dict(match=m, exp_margin=ridge[k], exp_total=total[k], sd_total=total_sd(total[k]),
                 home_pts=(total[k] + ridge[k]) / 2, away_pts=(total[k] - ridge[k]) / 2, link_a=math.nan,
                 link_b=math.nan, p_ridge=math.nan, table_margin=tm[k], stack_a=math.nan, stack_b_table=math.nan,
                 stack_b_ridge=math.nan, p_home=math.nan, priced_by="")
        b = links.get(day[k])
        if b is not None:
            r.update(link_a=b[0], link_b=b[1], p_ridge=sigmoid(b[0] + b[1] * ridge[k]))
            r.update(p_home=r["p_ridge"], priced_by="ridge")
        s = stacks.get(day[k])
        if s is not None and math.isfinite(tm[k]):
            r.update(stack_a=s[0], stack_b_table=s[1], stack_b_ridge=s[2], priced_by="stack",
                     p_home=sigmoid(s[0] + s[1] * tm[k] + s[2] * ridge[k]))
        out.append(r)
    return out


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))


# ---------------------------------------------------------------- the summary

def grade(y, p):
    """(matches, log loss, Brier, calibration slope: the logistic of y on logit p's, 1 calibrated, below 1 the
    chances too spread out) of chances p against results y (1 or 0)."""
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), EPS, 1 - EPS)
    if not len(y):
        return 0, math.nan, math.nan, math.nan
    ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    x = np.log(p / (1 - p))
    slope = float(logistic_fit(np.column_stack([np.ones(len(x)), x]), y)[1]) if np.ptp(x) > 1e-9 else math.nan
    return len(y), ll, float(np.mean((p - y) ** 2)), slope


def grade_lines(priced, table_name=None, table=None):
    """The model and prod-free baselines graded on the results in (ties left out), on the matches every
    column prices; with a table, on the matches the stack priced too."""
    norm = NormalDist()
    got = [r for r in priced if r["match"].home_score is not None and r["match"].home_score != r["match"].away_score
           and math.isfinite(r["p_home"])]
    if not got:
        return ["  no results in the span yet: nothing graded"]
    y = [float(r["match"].home_score > r["match"].away_score) for r in got]
    rows = [("p_home (the model)", [r["p_home"] for r in got]),
            ("the ridge's margin as a normal, sd 9.35", [norm.cdf(r["exp_margin"] / MARGIN_SD) for r in got]),
            ("evens", [0.5] * len(got))]
    if table:
        rows.insert(1, ("the link on the ridge alone", [r["p_ridge"] for r in got]))
    first, last = got[0]["match"].start, got[-1]["match"].start
    L = [f"  {first:%Y-%m-%d} to {last:%Y-%m-%d}: {len(got):,} matches with a result (ties left out)",
         f"  {'':44s} {'logloss':>8s} {'brier':>7s} {'slope':>6s}"]
    for name, p in rows:
        _, ll, br, sl = grade(y, p)
        L.append(f"  {name:44s} {ll:8.4f} {br:7.4f} " + (f"{sl:6.2f}" if math.isfinite(sl) else f"{'-':>6s}"))
    both = [(yy, r) for yy, r in zip(y, got) if r["priced_by"] == "stack" and table.get(r["match"].code, (0, None))[1]
            is not None] if table else []
    if both:
        yb = [yy for yy, _ in both]
        L.append(f"  where the stack prices and {table_name} has its own p_home: {len(both):,} matches")
        for name, p in (("p_home (the stack)", [r["p_home"] for _, r in both]),
                        ("the link on the ridge alone", [r["p_ridge"] for _, r in both]),
                        (f"{table_name}'s own p_home", [table[r["match"].code][1] for _, r in both])):
            _, ll, br, sl = grade(yb, p)
            L.append(f"  {name:44s} {ll:8.4f} {br:7.4f} {sl:6.2f}")
    return L


def total_lines(priced, table_name=None, table=None):
    """The expected total against the results in: RMSE, bias (expected less real) and the slope of the real on
    the expected (1 calibrated), beside the span's own mean total (in hindsight) and the table's exp_total."""
    got = [r for r in priced if r["match"].home_score is not None and math.isfinite(r["exp_total"])]
    if not got:
        return ["  no results in the span yet: nothing graded"]
    real = np.array([r["match"].home_score + r["match"].away_score for r in got])

    def line(name, exp, real):
        exp = np.asarray(exp, float)
        slope = float(np.polyfit(exp, real, 1)[0]) if np.ptp(exp) > 1e-9 else math.nan
        return (f"  {name:44s} {np.sqrt(np.mean((real - exp) ** 2)):8.3f} {np.mean(exp - real):+7.2f} "
                + (f"{slope:6.2f}" if math.isfinite(slope) else f"{'-':>6s}"))

    exp = np.array([r["exp_total"] for r in got])
    L = [f"  {len(got):,} matches with a result; the residuals' sd {np.std(real - exp):.2f} against sd_total's mean "
         f"{np.mean([r['sd_total'] for r in got]):.2f}",
         f"  {'':44s} {'RMSE':>8s} {'bias':>7s} {'slope':>6s}",
         line("exp_total (the filter)", exp, real),
         line("the span's mean total, in hindsight", np.full(len(got), real.mean()), real)]
    both = [k for k, r in enumerate(got) if table and (table.get(r["match"].code) or (0, 0, None))[2] is not None]
    if both:
        L.append(f"  where {table_name} has an exp_total: {len(both):,} matches")
        L.append(line("exp_total (the filter)", exp[both], real[both]))
        L.append(line(f"{table_name}'s exp_total", [table[got[k]["match"].code][2] for k in both], real[both]))
    return L


def _f(x, spec, width):
    """x formatted to spec, right-aligned in width; blank when it is NaN."""
    return f"{format(x, spec) if x is not None and math.isfinite(x) else '':>{width}s}"


def month_lines(priced, stacked):
    """The link's coefficients month by month (the means over its days' fits) beside the model's log loss."""
    by = {}
    for r in priced:
        by.setdefault(r["match"].start.strftime("%Y-%m"), []).append(r)
    head = f"  {'month':7s} {'matches':>8s} {'graded':>7s} {'logloss':>8s} {'a':>7s} {'b':>6s} {'sd':>5s}"
    if stacked:
        head += f"   {'stacked':>7s} {'a':>7s} {'b_table':>7s} {'b_ridge':>7s}"
    L = [head]
    for month in sorted(by):
        rs = by[month]
        days = {}                                   # a day's link is the same for all its matches
        for r in rs:
            days.setdefault(r["match"].start.date(), r)
        lk = [(r["link_a"], r["link_b"]) for r in days.values() if math.isfinite(r["link_b"])]
        g = [(float(r["match"].home_score > r["match"].away_score), r["p_home"]) for r in rs
             if r["match"].home_score is not None and r["match"].home_score != r["match"].away_score
             and math.isfinite(r["p_home"])]
        ll = grade(*zip(*g))[1] if g else math.nan
        a, b = (float(np.mean(x)) for x in zip(*lk)) if lk else (math.nan, math.nan)
        s = (f"  {month:7s} {len(rs):8,} {len(g):7,} {_f(ll, '.4f', 8)} {_f(a, '+.3f', 7)} {_f(b, '.3f', 6)} "
             f"{_f(1.702 / b if b > 0 else math.nan, '.2f', 5)}")
        if stacked:
            sk = {}
            for r in rs:
                if r["priced_by"] == "stack":
                    sk.setdefault(r["match"].start.date(), (r["stack_a"], r["stack_b_table"], r["stack_b_ridge"]))
            s += f"   {sum(1 for r in rs if r['priced_by'] == 'stack'):7,}"
            if sk:
                sa, st, sr = (float(np.mean(x)) for x in zip(*sk.values()))
                s += f" {sa:+7.3f} {st:7.3f} {sr:7.3f}"
        L.append(s.rstrip())
    return L


def ridge_line(ridge):
    """The ridge's home edges, gamers and teams as of its last solve."""
    if ridge.solved_at is None:
        return "  the ridge: no result in yet"
    b = ridge.beta
    when = dt.datetime(1970, 1, 1) + dt.timedelta(days=ridge.solved_at)
    edges = ", ".join(f"{s} {b[0] + b[k]:+.2f}" for s, k in ridge.streams.items())
    g = b[list(ridge.gamers.values())]
    t = b[list(ridge.teams.values())] if ridge.teams else np.zeros(1)
    return (f"  the ridge as of {when:%Y-%m-%d %H:%M}: the home edge by stream {edges or f'{b[0]:+.2f}'} points; "
            f"{len(ridge.gamers)} gamers (sd {g.std():.2f} points), {len(ridge.teams)} teams "
            f"({t.min():+.2f} to {t.max():+.2f})")


# ---------------------------------------------------------------- the run

def write_csv(path, priced):
    def num(x, nd):
        return "" if x is None or not math.isfinite(x) else round(float(x), nd)

    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        for r in priced:
            m = r["match"]
            w.writerow([m.code, f"{m.start:%Y-%m-%d %H:%M:%S}", m.stream, m.home, m.away, m.home_team, m.away_team,
                        "" if m.home_score is None else f"{m.home_score:g}",
                        "" if m.away_score is None else f"{m.away_score:g}",
                        num(r["exp_margin"], 4), num(r["exp_total"], 4), num(r["sd_total"], 4), num(r["home_pts"], 4),
                        num(r["away_pts"], 4), num(r["link_a"], 6), num(r["link_b"], 6), num(r["p_ridge"], 6),
                        num(r["table_margin"], 4), num(r["stack_a"], 6), num(r["stack_b_table"], 6),
                        num(r["stack_b_ridge"], 6), num(r["p_home"], 6), r["priced_by"]])


def run(out_dir, history, since=None, until=None, table=None, half_life=HALF_LIFE, gamer_penalty=GAMER_PENALTY,
        team_penalty=TEAM_PENALTY, stream_penalty=STREAM_PENALTY, link_days=LINK_DAYS, min_rows=MIN_ROWS):
    """Every match of [since, until) (default: the whole history, the scheduled matches too) priced off every
    result before it; writes ml_prior.csv and ml_prior.txt to out_dir. Returns ml_prior.csv's path."""
    t0 = time.time()
    matches = read_matches(history)
    if not matches:
        raise SystemExit(f"{history}: no matches")
    first = _time(since) if since else dt.datetime.combine(matches[0].start.date(), dt.time())
    last = _time(until) if until else None
    if first is None or (until and last is None):
        raise SystemExit(f"--since {since!r} / --until {until!r}: give each as YYYY-MM-DD (or 'YYYY-MM-DD HH:MM:SS')")
    span = [m for m in matches if m.start >= first and (last is None or m.start < last)]
    if not span:
        raise SystemExit(f"{history}: no matches from {first:%Y-%m-%d %H:%M}" + (f" to {last:%Y-%m-%d %H:%M}" if last else ""))
    view = table_margins(table) if table else None
    exp, ridge = margins(matches, half_life, gamer_penalty, team_penalty, stream_penalty)
    tot = totals(matches)
    keep = {m.code for m in span}
    priced = [r for r in price(matches, exp, view, {m.start.date() for m in span}, link_days, min_rows, total=tot)
              if r["match"].code in keep]
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "ml_prior.csv")
    write_csv(path, priced)
    settled = sum(1 for m in matches if m.home_score is not None)
    open_ = sum(1 for r in priced if r["match"].home_score is None)
    unpriced = sum(1 for r in priced if not math.isfinite(r["p_home"]))
    name = os.path.basename(table) if table else None
    L = [f"\nThe pre-match moneyline and total: {len(matches):,} matches in {history} ({matches[0].start:%Y-%m-%d} to "
         f"{matches[-1].start:%Y-%m-%d}, {settled:,} with a result); {len(priced):,} written, {span[0].start:%Y-%m-%d} "
         f"to {span[-1].start:%Y-%m-%d}, {open_:,} of them without a result yet",
         f"  the margin: a ridge on every result in by the kick-off ({RESULT_LAG.total_seconds() / 60:g} minutes after "
         f"its own), each weighted 2^(-age/{half_life:g} days); penalties in matches of full weight: gamers "
         f"{gamer_penalty:g}, teams {team_penalty:g}, a stream's home edge {stream_penalty:g}",
         f"  the link: logistic(a + b x margin), fitted each day on the last {link_days:g} days' results in by its "
         f"midnight (at least {min_rows:,}; the history's first {BURN_IN_DAYS} days left out)"]
    if view:
        both = sum(1 for r in priced if math.isfinite(r["table_margin"]))
        stacked = sum(1 for r in priced if r["priced_by"] == "stack")
        L.append(f"  the stack: {name}'s exp_margin beside the ridge's, logistic(a + b_table x its + b_ridge x the "
                 f"ridge's), fitted the same way on the matches both priced: {name} has {both:,} of the matches "
                 f"written, {stacked:,} priced by the stack and the rest by the link")
    L.append(ridge_line(ridge))
    L.append(f"  {len(priced) - unpriced:,} matches priced" + (f", {unpriced:,} not: fewer than {min_rows:,} results "
                                                                 "behind their day's link" if unpriced else ""))
    L.append("  the total: an online Kalman filter on each result's total (the league's mean, gamers' and teams' "
             "effects drifting, a stream's), each result in as the ridge's; Pred_P1_Points / Pred_P2_Points = "
             f"(exp_total +- exp_margin) / 2; sd_total = {TOTAL_SD[0]:g} x (exp_total / {TOTAL_SD[1]:g})^"
             f"{TOTAL_SD[2]:g}, informational")
    with one_thread():
        L.append("\nThe link month by month (the means of its days' fits; sd = 1.702 / b, the normal sd the link "
                 "reads the margin at, against the 9.35 the sims assume):")
        L += month_lines(priced, bool(view))
        L.append("\nGraded on the results, on the matches every column prices:")
        L += grade_lines(priced, name, view)
        L.append("\nThe expected total on the results:")
        L += total_lines(priced, name, view)
    L.append(f"\n  done in {time.time() - t0:.1f}s")
    txt = os.path.join(out_dir, "ml_prior.txt")
    with open(txt, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n  -> {path}: a whole prior for v2.1, its points for --v21-prior-model and its p_home for --v21-ml")
    return path
