"""The post-game form layer: a +/- on each side's expected points from how the two gamers have done
against the pre-match model's own prices today and this session. It is built for a pre-match model
refitted every day (rolling_prior.py, `eAMFModel prior-daily`), whose ratings start each day from
every result before it and whose form features follow the results; the layer catches what is left.

Each gamer carries a day form D and a session form S, for the margin and for the total, updated
after every match they play by a Kalman filter on the match's error against the model's price:
  D   sd TAU_DAY when fresh; carried to the next day (00:00 UTC, when a daily prior refits) times
      RHO a day, so RHO 0 starts every day fresh
  S   sd TAU_SESSION when fresh, at each session's start: a gap of more than SESSION_GAP since
      the gamer's last start
A match's expected margin moves by the home gamer's D + S less the away gamer's, its expected total
by both gamers' D + S. A match's error is shared between its two gamers by how unsure each one's
form is, so a gamer whose form has already settled moves less than a fresh one.

TRIGGER > 0 moves the margin only once a trend shows, as a punter would see it: a gamer's form
counts once they have lost (won) at least TRIGGER in a row tonight (kick-offs no more than RUN_GAP
apart), and only the way the run goes. The total is not gated.

Only matches that started before a match count, as stream.known_states needs: a quote priced as of
an earlier time reads the results known then. A settled match's error is taken against the price
the model made out of sample, so the layer reads results from the model's first daily fit (a
single fit: its cut-off) on.

Off unless switched on: for one build by `python -m eAMFModel form-layer DIR --on` (its prior
file's "form_layer"), for every build by ON. It sits on top of whatever the build prices with,
the follow layer included (follow.py; not used on a daily prior).
"""

import datetime as dt
import importlib
import json
import math
import os
from statistics import NormalDist

from . import nb2_prior

ON = False                              # for every build; a build's prior file can say otherwise
NOISE = (9.37, 12.0)                    # sd of a match's margin and total about the price
TAU_DAY = (1.0, 1.0)                    # sd of a gamer's fresh day form: margin, total
RHO = 0.0                               # the share of a day's form carried to the next day
TAU_SESSION = (1.0, 1.0)                # sd of a gamer's fresh session form: margin, total
SESSION_GAP = dt.timedelta(hours=2)     # a longer gap since the gamer's last start: a new session
TRIGGER = 0                             # the margin moves only after a run this long (0: always)
RUN_GAP = dt.timedelta(hours=6)         # a run is a night's: kick-offs no more than this apart
LOOKBACK_DAYS = 14                      # days of results read before the first match, when RHO > 0
EVENING = dt.timedelta(hours=12)        # and the hours before its day, for a session past midnight
MARGIN_SD = 9.35                        # for the trace's win chance off the expected margin
SETTING = "form_layer"                  # the key in a build's prior file: true, false or its own
OWN = ("tau_day", "rho", "tau_session", "trigger")  # settings, e.g. {"tau_day": [1.5, 1.5]}


def _gamers(row):
    return [str(row.get(k) or "").strip().upper() for k in ("PLAYER_1_HANDLE", "PLAYER_2_HANDLE")]


def _finals(row):
    try:
        return float(row["PLAYER_1_FINAL_SCORE"]), float(row["PLAYER_2_FINAL_SCORE"])
    except (KeyError, TypeError, ValueError):
        return None


class Filter:
    """Every gamer's day and session form, for the margin and the total, as results come in.

    A gamer's form for one quantity is (d, s, pdd, pds, pss): the two means and their 2x2
    covariance. Matches must be fed in start order; `form` may be asked at any time at or after
    the gamer's last update."""

    def __init__(self, noise=NOISE, tau_day=TAU_DAY, rho=RHO, tau_session=TAU_SESSION,
                 session_gap=SESSION_GAP, trigger=TRIGGER, run_gap=RUN_GAP):
        self.noise2 = tuple(x * x for x in noise)
        self.day2 = tuple(x * x for x in tau_day)
        self.session2 = tuple(x * x for x in tau_session)
        self.rho, self.gap, self.trigger, self.run_gap = rho, session_gap, trigger, run_gap
        self.gamers = {}                # gamer -> (last start, [margin form, total form])
        self.runs = {}                  # gamer -> tonight's run: +k won the last k, -k lost them

    def _state(self, gamer, t):
        """The gamer's forms as of `t`: a new day's D carried by RHO a day, a new session's S fresh."""
        st = self.gamers.get(gamer)
        if st is None:
            return [(0.0, 0.0, d2, 0.0, s2) for d2, s2 in zip(self.day2, self.session2)]
        last, forms = st
        days = (t.date() - last.date()).days
        out = []
        for q, (d, s, pdd, pds, pss) in enumerate(forms):
            if days > 0:
                r = self.rho ** days
                d, pdd, pds = r * d, r * r * pdd + (1 - r * r) * self.day2[q], r * pds
            if t - last > self.gap:
                s, pds, pss = 0.0, 0.0, self.session2[q]
            out.append((d, s, pdd, pds, pss))
        return out

    def form(self, gamer, t):
        """((day, session) margin form, (day, session) total form) of a gamer as of `t`."""
        return tuple((f[0], f[1]) for f in self._state(gamer, t))

    def run(self, gamer, t):
        """The gamer's run tonight as of `t`: +k won the last k, -k lost the last k."""
        st = self.gamers.get(gamer)
        return 0 if st is None or t - st[0] > self.run_gap else self.runs.get(gamer, 0)

    def margin(self, gamer, t):
        """The gamer's margin form as of `t` that counts: all of it, or with a trigger, only once
        their run tonight is that long, and only the way it goes."""
        m = sum(self.form(gamer, t)[0])
        if not self.trigger:
            return m
        run = self.run(gamer, t)
        return m if abs(run) >= self.trigger and run * m > 0 else 0.0

    def adjustment(self, home, away, t):
        """(margin, total) to add to a match's expected points as of `t`."""
        ht, at = self.form(home, t)[1], self.form(away, t)[1]
        return self.margin(home, t) - self.margin(away, t), sum(ht) + sum(at)

    def update(self, home, away, t, margin_error, total_error, result=None):
        """Take in a match that started at `t`: its margin and total errors against the price, and
        its result (home points less away points) for the runs."""
        if result is not None:
            for g, r in ((home, result), (away, -result)):
                was = self.run(g, t)
                self.runs[g] = max(was, 0) + 1 if r > 0 else min(was, 0) - 1 if r < 0 else 0
        h, a = self._state(home, t), self._state(away, t)
        for q, (y, sign) in enumerate(((margin_error, -1.0), (total_error, 1.0))):
            (d1, s1, a1, b1, c1), (d2, s2, a2, b2, c2) = h[q], a[q]
            r = y - (d1 + s1) - sign * (d2 + s2)
            var = (a1 + 2 * b1 + c1) + (a2 + 2 * b2 + c2) + self.noise2[q]
            k1d, k1s, k2d, k2s = (a1 + b1) / var, (b1 + c1) / var, (a2 + b2) / var, (b2 + c2) / var
            h[q] = (d1 + k1d * r, s1 + k1s * r,
                    a1 - k1d * (a1 + b1), b1 - k1d * (b1 + c1), c1 - k1s * (b1 + c1))
            a[q] = (d2 + sign * k2d * r, s2 + sign * k2s * r,
                    a2 - k2d * (a2 + b2), b2 - k2d * (b2 + c2), c2 - k2s * (b2 + c2))
        self.gamers[home], self.gamers[away] = (t, h), (t, a)


def out_of_sample_since(pre):
    """From when a pre-match model's prices are out of sample: a daily prior's first day, else the
    cut-off of its fit (or of the follow layer's); None when it records neither."""
    days = getattr(pre, "days", None)
    if days:
        return dt.datetime.combine(min(days), dt.time())
    since = getattr(pre, "since", None)
    if isinstance(since, dt.datetime):
        return since
    before = (getattr(pre, "meta", None) or {}).get("before")
    return dt.datetime.fromisoformat(before[:19]) if before else None


class FormLayer:
    """A pre-match model plus each gamer's day and session form against its prices."""

    FOLLOWS_RESULTS = True               # means() reads the results known when pricing

    def __init__(self, pre, since, noise=NOISE, tau_day=TAU_DAY, rho=RHO, tau_session=TAU_SESSION,
                 session_gap=SESSION_GAP, trigger=TRIGGER, lookback_days=LOOKBACK_DAYS):
        self.pre, self.since = pre, since
        self.settings = dict(noise=noise, tau_day=tau_day, rho=rho, tau_session=tau_session,
                             session_gap=session_gap, trigger=trigger)
        self.lookback_days = lookback_days if rho > 0 else 0

    def __getattr__(self, name):
        return getattr(self.pre, name)

    @property
    def AS_OF(self):
        return getattr(self.pre, "AS_OF", True)

    def describe(self):
        base = self.pre.describe() if hasattr(self.pre, "describe") else type(self.pre).__name__
        s = self.settings
        carry = f"carried x{s['rho']:g} a day" if s["rho"] else "fresh each day"
        trigger = (f"; the margin moves once a run of {s['trigger']} shows" if s["trigger"] else "")
        return (f"{base}; post-game form layer from {self.since:%Y-%m-%d} (day form sd "
                f"{s['tau_day'][0]:g} margin, {s['tau_day'][1]:g} total, {carry}; session form sd "
                f"{s['tau_session'][0]:g} margin, {s['tau_session'][1]:g} total{trigger})")

    def run(self, schedule, results=None, **kw):
        """(the model's own expected points for the schedule and the matches the layer read,
        {match: (margin, total) adjustment}, {match: (home, away) (form as filter.form, run)})."""
        starts = {r["MATCH_CODE"]: nb2_prior._start(r) for r in schedule}
        known = [t for t in starts.values() if t is not None]
        if not known or not results:
            return self.pre.means(schedule, results=results, **kw), {}, {}
        # from the first match's day (less the lookback), and the evening before for a session
        # that runs past midnight
        first = max(self.since, dt.datetime.combine(min(known).date(), dt.time())
                    - dt.timedelta(days=self.lookback_days) - EVENING)
        last = max(known)
        settled = [r for r in results if _finals(r) is not None and nb2_prior._start(r) is not None
                   and first <= nb2_prior._start(r) < last]
        extra = [r for r in settled if r["MATCH_CODE"] not in starts]
        got = self.pre.means(list(schedule) + extra, results=results, **kw)
        # in start order; a match asked at the same time as one settles doesn't see it
        events = [(starts[r["MATCH_CODE"]], 0, k, r) for k, r in enumerate(schedule)
                  if starts[r["MATCH_CODE"]] is not None]
        events += [(nb2_prior._start(r), 1, k, r) for k, r in enumerate(settled)]
        events.sort(key=lambda e: e[:3])
        filt = Filter(**self.settings)
        adjust, forms = {}, {}
        for t, settles, _, r in events:
            home, away = _gamers(r)
            if not settles:
                adjust[r["MATCH_CODE"]] = filt.adjustment(home, away, t)
                forms[r["MATCH_CODE"]] = tuple((filt.form(g, t), filt.run(g, t)) for g in (home, away))
                continue
            p, f = got.get(r["MATCH_CODE"]), _finals(r)
            if p is not None:
                filt.update(home, away, t, (f[0] - f[1]) - (p[0] - p[1]), (f[0] + f[1]) - (p[0] + p[1]),
                            f[0] - f[1])
        return got, adjust, forms

    def means(self, schedule, results=None, **kw):
        """Each match's expected (home, away) points: the model's own, the margin and the total moved
        by the two gamers' day and session forms as of its start."""
        got, adjust, _ = self.run(schedule, results=results, **kw)
        out = {}
        for r in schedule:
            code = r["MATCH_CODE"]
            if code not in got:
                continue
            h, a = got[code]
            dm, dt_ = adjust.get(code, (0.0, 0.0))
            out[code] = (max(0.1, h + (dt_ + dm) / 2), max(0.1, a + (dt_ - dm) / 2))
        return out


def switched_on(meta):
    """Whether a build prices with the layer: its prior file's setting (its own settings: on),
    else ON."""
    value = (meta or {}).get(SETTING)
    return ON if value is None else True if isinstance(value, dict) else bool(value)


def own_settings(meta):
    """The settings a build keeps for itself (FormLayer's keywords); the rest are the defaults."""
    value = (meta or {}).get(SETTING)
    return {k: tuple(v) if isinstance(v, list) else v
            for k, v in (value.items() if isinstance(value, dict) else ()) if k in OWN}


def wrap(pre, meta):
    """The pre-match model with the layer on top when the build has it switched on and the model
    records from when its prices are out of sample; else as it is."""
    if pre is None or not switched_on(meta):
        return pre
    since = out_of_sample_since(pre)
    return FormLayer(pre, since, **own_settings(meta)) if since is not None else pre


def peel(pre):
    """A pre-match model without the layer."""
    while isinstance(pre, FormLayer):
        pre = pre.pre
    return pre


def win_chance(margin, sd=MARGIN_SD):
    """The chance a side wins with this expected margin, off a normal margin."""
    return 1.0 - NormalDist(margin, sd).cdf(0.0)


def trace(layer, history, gamer, since, until):
    """One gamer's settled matches from `since` to `until` (dates), each as the model priced it and
    as the layer moves it, from the gamer's side: a list of dicts in start order."""
    gamer = gamer.strip().upper()
    rows = sorted((r for r in history if gamer in _gamers(r) and _finals(r) is not None
                   and nb2_prior._start(r) is not None and since <= nb2_prior._start(r).date() <= until),
                  key=nb2_prior._start)
    if not rows:
        return []
    got, adjust, forms = layer.run(rows, results=history)
    out = []
    for r in rows:
        code = r["MATCH_CODE"]
        if code not in got:
            continue
        home, away = _gamers(r)
        sign = 1.0 if home == gamer else -1.0
        (p1, p2), (s1, s2) = got[code], _finals(r)
        dm, dt_ = adjust.get(code, (0.0, 0.0))
        fresh = (((0.0, 0.0), (0.0, 0.0)), 0)
        (mine, run), (theirs, their_run) = forms.get(code, (fresh, fresh))
        if sign < 0:
            (mine, run), (theirs, their_run) = (theirs, their_run), (mine, run)
        base = sign * (p1 - p2)
        out.append(dict(code=code, start=nb2_prior._start(r), opponent=away if sign > 0 else home,
                        side="home" if sign > 0 else "away",
                        scored=s1 if sign > 0 else s2, conceded=s2 if sign > 0 else s1,
                        base_points=(p1, p2) if sign > 0 else (p2, p1), base_margin=base,
                        run=run, opponent_run=their_run,
                        day=mine[0][0], session=mine[0][1], opponent_form=sum(theirs[0]),
                        adjust=sign * dm, margin=base + sign * dm, total_adjust=dt_))
    return out


def _prior_file(model_dir):
    names = [f for f in os.listdir(model_dir) if f.startswith("v") and f.endswith("prior.json")]
    if len(names) != 1:
        raise SystemExit(f"{model_dir}: not a version's build (no single v*prior.json)")
    return os.path.join(model_dir, names[0])


def switch(model_dir, value, **own):
    """Switch the layer on (True) or off (False) for one build, or back to ON (None); `own`:
    settings the build keeps for itself (OWN), with the layer on. Returns the build's prior file
    as it now is."""
    if own and value is not True:
        raise ValueError("settings of its own come with the layer switched on")
    path = _prior_file(model_dir)
    with open(path, encoding="utf-8") as fh:
        meta = json.load(fh)
    if value is None:
        meta.pop(SETTING, None)
    elif own:
        unknown = set(own) - set(OWN)
        if unknown:
            raise ValueError(f"no layer setting {', '.join(sorted(unknown))}")
        meta[SETTING] = {k: list(v) if isinstance(v, tuple) else v for k, v in own.items()}
    else:
        meta[SETTING] = bool(value)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    return meta


def base_settings(model_dir):
    """The layer settings a build keeps for itself."""
    with open(_prior_file(model_dir), encoding="utf-8") as fh:
        return own_settings(json.load(fh))


def base_model(model_dir):
    """A build's pre-match model as its version prices with it, without this layer."""
    version = os.path.basename(_prior_file(model_dir))[:-len("prior.json")]
    pre = importlib.import_module(f"{__package__}.{version}").prematch_model(model_dir)
    if pre is None:
        raise SystemExit(f"{model_dir}: no pre-match model saved")
    return peel(pre)


def _run_label(run):
    return f"W{run}" if run > 0 else f"L{-run}" if run < 0 else "-"


def report(rows, gamer, layer):
    """Lines for trace's rows: each match as the model priced it and as the layer moves it."""
    if not rows:
        return [f"  no settled matches for {gamer.upper()} in that window"]
    lines = [f"  {gamer.upper()}: {len(rows)} matches, {rows[0]['start']:%Y-%m-%d %H:%M} to "
             f"{rows[-1]['start']:%Y-%m-%d %H:%M}", f"  model: {layer.describe()}", "",
             "  margins and win chances from his side; win chance off a normal margin, sd "
             f"{MARGIN_SD:g}; runs tonight before the match (L3: lost the last 3), his and the "
             "opponent's; +/- = his day + session form less the opponent's",
             "",
             f"  {'kick-off':<16} {'side':<4} {'opponent':<16} {'score':>7} {'runs':>7}   {'model':>6} {'win':>4}"
             f"   {'day':>5} {'sess':>5} {'opp':>5} {'+/-':>5}   {'layer':>6} {'win':>4}"]
    sums = dict(n=0, e0=0.0, e1=0.0, l0=0.0, l1=0.0, k=0)
    day = None
    for r in rows:
        if day is not None and r["start"].date() != day:
            lines.append("")
        day = r["start"].date()
        w0, w1 = win_chance(r["base_margin"]), win_chance(r["margin"])
        runs = f"{_run_label(r['run'])} {_run_label(r['opponent_run'])}"
        lines.append(f"  {r['start']:%Y-%m-%d %H:%M} {r['side']:<4} {r['opponent'][:16]:<16} "
                     f"{r['scored']:>3.0f}-{r['conceded']:<3.0f} {runs:>7}   {r['base_margin']:>+6.1f} {w0:>4.0%}"
                     f"   {r['day']:>+5.2f} {r['session']:>+5.2f} {r['opponent_form']:>+5.2f} "
                     f"{r['adjust']:>+5.2f}   {r['margin']:>+6.1f} {w1:>4.0%}")
        result = r["scored"] - r["conceded"]
        sums["n"] += 1
        sums["e0"] += result - r["base_margin"]
        sums["e1"] += result - r["margin"]
        if result:
            won = result > 0
            sums["k"] += 1
            sums["l0"] -= math.log(max(1e-9, w0 if won else 1 - w0))
            sums["l1"] -= math.log(max(1e-9, w1 if won else 1 - w1))
    n, k = sums["n"], max(1, sums["k"])
    lines += ["", f"  his result against the price, a match: model {sums['e0'] / n:+.2f}, with the layer "
                  f"{sums['e1'] / n:+.2f}; moneyline log loss {sums['l0'] / k:.3f} -> {sums['l1'] / k:.3f}"]
    return lines
