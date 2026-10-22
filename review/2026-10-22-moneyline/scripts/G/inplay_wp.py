"""In-play win probability off the scouting export: P(home wins) at every PLAY_OVER, from the game state and a
pre-match probability, by gradient boosting with monotone constraints, walk-forward by day. The NFL-style
benchmark for prod's in-play moneyline (and v2.1's, given its prices).

    python inplay_wp.py --export eAMFCalibrator/out_scouting_xgb/scouting_playover.csv \
        --prematch preds/A_best.csv --history eAMFCalibrator/out/match_history.csv \
        --since 2026-08-01 --until 2026-10-19 --tune-until 2026-09-01 --out out_inplay_wp \
        [--compare v21=v21_states.csv] [--refit-days 7] [--train-days 120] [--train-every 1] [--tune-every 2]

    Self-contained (pandas, numpy, scipy, xgboost); runs from anywhere. Tests: test_inplay_wp.py beside it.

The export (eAMFCalibrator/scouting.py's EXPORT_FIELDS) has one row per PLAY_OVER in message order: the kind of
play that just ended and the state for the next snap (offense TEAM_A/TEAM_B, down, distance, field position in
yards from the offense's own goal line), the period and the game clock (240 s a quarter), the scoreboard (home
= PLAYER_1 = score_p1), each side's timeouts called in the half (timeouts_used_a/b; overtime per period),
which side TEAM_A is (team_a_side), the opening offense, the final score, and prod's quotes on the same message
(prob_50 = prod's P(home) where it quoted, live_50 = tradeable). The state at a row, as
eAMFModel/playover.state_for reads it:
  touchdown     the scorer's six on the board (score_p*_at_start + 6 where the board lags), the conversion to
                come; the side to receive the kick-off after it has the "ball"
  conversion    its points on the board, the other side to receive
  field goal    good: 3 on the board, the other side to receive; missed: the other side's snap where the row
                has it, else its ball with the spot unknown
  safety        2 on the board; the side that scored receives the free kick
  anything else the row's own snap state: offense, down, distance, field position
A kick to come is set at 1st and 10 from the receiving side's 25 (v21's kick-off start) and flagged.

Models, each refitted every --refit-days on the states of the matches settled by that day's midnight UTC (a
match is settled at its last PLAY_OVER's file_time, else 36 minutes after kick-off), the last --train-days of
them:
  wp       XGBoost binary:logistic from the side with the ball's view (nflfastR's way; P(home) is p or
           1 - p): its lead, its lead / sqrt(share of regulation left + K), its pre-match logit and that x
           sqrt(share left), seconds left, seconds left in the half, period, overtime, down, distance, yards
           to the goal, both sides' timeouts left, the opponent's conversion to come, who receives the
           second-half kick-off, whether it is the home side. Monotone: + lead, lead / sqrt, pre-match terms,
           own timeouts; - down, distance, yards to go, the opponent's timeouts and conversion; the rest free.
  wp_lite  the same from home's view with only what the bet moments carry (lead, time, period, down,
           pre-match): what possession, field position and timeouts add.
  stern    logit P(home) = a + b * lead / sqrt(s + K) + c * pre * sqrt(s) + d * pre (s = share of regulation
           left): the final margin as a Brownian motion drifting at the pre-match edge (Stern 1994).
  pre      the pre-match probability alone.
With --tune-until, the XGBoost settings are chosen on the test days before it (a small grid, best log loss)
and frozen for the days after, which the report leads with.

Scored on every state, and against prod's live in-play price on the PLAY_OVERs it quoted (and any --compare
table: match_code, message, p_home, e.g. v2.1's prices at the same states), each on the states every column
prices: log loss, Brier, AUC, the calibration slope (logit of y on logit p; 1 is calibrated), the log loss
difference with a 95% interval resampling whole days (and whole matches), by quarter, a calibration table, a
monotonicity check and the trees' gains. Ties are left out (no winner).

Written to --out: states.csv (every state, its features and each model's P(home)), report.txt, settings.json.
"""

import argparse
import datetime as dt
import itertools
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb

QUARTER = 240.0
REG = 4 * QUARTER
TIMEOUTS, OT_TIMEOUTS = 3, 2
KICK_YARDS = 25                 # a kick to come: 1st and 10 from the receiving side's own 25 (v21.kickoff_start)
RESULT_LAG = dt.timedelta(minutes=36)
LOST_START_JUMP = 60.0          # eAMFModel/feed.lost_quarter: the clock back up this far within a period
K = 0.02                        # lead / sqrt(share left + K)
EPS = 1e-6
NTHREAD = 2
PARAMS = dict(objective="binary:logistic", eval_metric="logloss", eta=0.05, max_depth=4, min_child_weight=50,
              subsample=0.8, colsample_bytree=1.0, reg_lambda=5.0, tree_method="hist", verbosity=0)
ROUNDS = 300
GRID = dict(max_depth=(3, 4, 5), min_child_weight=(50, 200), rounds=(150, 300, 600))
WANT = ("match_code", "message", "file_time", "period", "clock_seconds", "play_kind", "team_a_side", "offense",
        "down", "distance", "field_position", "score_p1", "score_p2", "score_p1_at_start", "score_p2_at_start",
        "play_messages", "final_p1", "final_p2", "opening_offense", "timeouts_used_a", "timeouts_used_b",
        "prob_50", "live_50", "prematch_prob_50")
KINDS = ("snap", "kick", "conv", "change")

WP_FEATURES = ("lead", "lead_z", "pre", "pre_s", "secs_left", "half_left", "period", "ot", "down", "distance",
               "ytg", "own_to", "opp_to", "opp_conv", "gets_2h_kick", "is_home", "kind")
WP_MONO = dict(lead=1, lead_z=1, pre=1, pre_s=1, own_to=1, down=-1, distance=-1, ytg=-1, opp_to=-1, opp_conv=-1)
LITE_FEATURES = ("lead", "lead_z", "pre", "pre_s", "secs_left", "period", "down")
LITE_MONO = dict(lead=1, lead_z=1, pre=1, pre_s=1)


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, float)))


# ---------------------------------------------------------------- reading

def read_export(path, codes=None):
    """The export's columns this needs (those it has), in match and message order."""
    head = pd.read_csv(path, nrows=0).columns
    missing = [c for c in ("match_code", "message", "period", "clock_seconds", "play_kind", "offense",
                           "score_p1", "score_p2", "final_p1", "final_p2") if c not in head]
    if missing:
        raise SystemExit(f"{path} lacks {', '.join(missing)}: not a scouting PLAY_OVER export")
    use = [c for c in WANT if c in head]
    text = {"match_code", "file_time", "play_kind", "team_a_side", "offense", "play_messages", "opening_offense"}
    ex = pd.read_csv(path, usecols=use, dtype={c: str for c in use if c in text}, low_memory=False)
    if codes is not None:
        ex = ex[ex.match_code.isin(codes)]
    for c in WANT:
        if c not in ex:
            ex[c] = np.nan
    ex["message"] = pd.to_numeric(ex.message, errors="coerce")
    ex = ex[ex.message.notna()].sort_values(["match_code", "message"], kind="stable")
    return ex.drop_duplicates(["match_code", "message"], keep="last").reset_index(drop=True)


def read_prematch(path):
    """{match: p_home} off a table with MATCH_CODE (or code / match_code) and p_home."""
    t = pd.read_csv(path)
    key = next((c for c in ("MATCH_CODE", "code", "match_code") if c in t), None)
    if key is None or "p_home" not in t:
        raise SystemExit(f"{path}: needs MATCH_CODE (or code) and p_home")
    t = t[[key, "p_home"]].dropna().drop_duplicates(key, keep="last")
    return dict(zip(t[key].astype(str), t.p_home.astype(float)))


def read_starts(path):
    """{match: kick-off} off a match history (MATCH_CODE, SCHEDULED_START_TIME_UTC), or {}."""
    if not path or not os.path.exists(path):
        return {}
    h = pd.read_csv(path, usecols=["MATCH_CODE", "SCHEDULED_START_TIME_UTC"])
    h["t"] = pd.to_datetime(h.SCHEDULED_START_TIME_UTC, errors="coerce", format="mixed")
    return dict(zip(h.MATCH_CODE.astype(str), h.t))


def _code_date(code):
    """A match code's date: AF + 3 digits + DDMMYY (AF001011026 is 2026-10-01), or NaT."""
    m = re.match(r"^[A-Z]+\d{3}(\d{2})(\d{2})(\d{2})$", str(code))
    if not m:
        return pd.NaT
    try:
        return pd.Timestamp(2000 + int(m.group(3)), int(m.group(2)), int(m.group(1)))
    except ValueError:
        return pd.NaT


# ---------------------------------------------------------------- the states

def _num(s):
    return pd.to_numeric(s, errors="coerce")


def _last_team(msgs, pattern):
    """The team (TEAM_A / TEAM_B) of the last message matching `pattern` in each row's play_messages."""
    return msgs.str.extract(r".*(?:" + pattern + r")_?(TEAM_[AB])", expand=False)


def build_states(ex, prematch, starts, pre_fallback="prod"):
    """One row per PLAY_OVER: the state from home's view and from the side with the ball's, the label, the
    pre-match probability and prod's live price there. Returns (states, notes)."""
    notes = []
    ex = ex.copy()
    code = ex.match_code
    side = ex.team_a_side.where(ex.team_a_side.isin(["home", "away"]))
    side = side.groupby(code).transform(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)
    notes.append(f"matches whose TEAM_A side is unresolved (read as home, as v21 does): "
                 f"{code[side.isna()].nunique():,} of {code.nunique():,}")
    a_home = np.where(side.fillna("home") == "home", 1, -1)

    def home_of(team, rows=None):          # TEAM_A / TEAM_B -> +1 home, -1 away, 0 unknown
        t = team.fillna("")
        return np.where(t == "TEAM_A", 1, np.where(t == "TEAM_B", -1, 0)) * (a_home if rows is None else a_home[rows])

    kind_raw = ex.play_kind.fillna("")
    msgs = ex.play_messages.fillna("")
    row_side = home_of(ex.offense)
    period = _num(ex.period)
    clock = _num(ex.clock_seconds)
    down, dist, field = _num(ex.down), _num(ex.distance), _num(ex.field_position)
    h, a = _num(ex.score_p1).fillna(0), _num(ex.score_p2).fillna(0)
    h0, a0 = _num(ex.score_p1_at_start), _num(ex.score_p2_at_start)

    scorer = np.zeros(len(ex), int)
    pts = np.zeros(len(ex))
    for kinds, pattern, value in (
            (("TOUCHDOWN",), "TOUCHDOWN", None),
            (("CONVERSION",), "EXTRA_POINT_GOOD|EXTRA_POINT_MISSED|TWO_POINT_CONVERSION_SUCCESSFUL|"
                              "TWO_POINT_CONVERSION_FAILED|EXTRA_POINT|TWO_POINT", None),
            (("FIELD_GOAL",), "FIELD_GOAL_GOOD|FIELD_GOAL_MISSED", None),
            (("SAFETY",), "SAFETY_AWARDED", None)):
        k = kind_raw.isin(kinds).to_numpy()
        if not k.any():
            continue
        team = _last_team(msgs[k], pattern)
        s = home_of(team, k)
        scorer[k] = np.where(s != 0, s, row_side[k])
    td = (kind_raw == "TOUCHDOWN").to_numpy()
    conv = (kind_raw == "CONVERSION").to_numpy()
    fg = (kind_raw == "FIELD_GOAL").to_numpy()
    sf = (kind_raw == "SAFETY").to_numpy()
    fg_good = fg & msgs.str.contains("FIELD_GOAL_GOOD").to_numpy()
    pts[td] = 6
    pts[conv] = np.where(msgs[conv].str.contains("TWO_POINT_CONVERSION_SUCCESSFUL"), 2,
                         np.where(msgs[conv].str.contains("EXTRA_POINT_GOOD"), 1, 0))
    pts[fg_good] = 3
    pts[sf] = 2
    # the board can lag the play: playover._on_the_board
    lag_h = (scorer == 1) & (pts > 0) & h0.notna().to_numpy() & ((h - h0).to_numpy() < pts)
    lag_a = (scorer == -1) & (pts > 0) & a0.notna().to_numpy() & ((a - a0).to_numpy() < pts)
    h = np.where(lag_h, h0 + pts, h)
    a = np.where(lag_a, a0 + pts, a)

    kind = np.full(len(ex), "snap", dtype=object)
    pos = row_side.copy()
    kick = conv | fg_good | sf
    pos = np.where(td | conv | fg_good, -scorer, pos)
    pos = np.where(sf, scorer, pos)
    kind[td] = "conv"
    kind[kick] = "kick"
    fg_miss = fg & ~fg_good
    miss_snap = fg_miss & (row_side == -scorer)
    kind[fg_miss & ~miss_snap] = "change"
    pos = np.where(fg_miss, -scorer, pos)
    not_snap = kind != "snap"
    down = np.where(not_snap, 1, down)
    dist = np.where(not_snap, 10, dist)
    ytg = np.where(not_snap, 100 - KICK_YARDS, 100 - field)
    ytg = np.where(kind == "change", np.nan, ytg)

    p = period.to_numpy(float)
    c = clock.to_numpy(float)
    reg = p <= 4
    secs_left = np.where(reg, (4 - p) * QUARTER + c, 0.0)
    half_left = np.where(np.isin(p, (1, 3)), c + QUARTER, c)
    share = np.where(reg, secs_left, c) / REG

    have = np.where(reg, TIMEOUTS, OT_TIMEOUTS)
    used_a, used_b = _num(ex.timeouts_used_a).to_numpy(float), _num(ex.timeouts_used_b).to_numpy(float)
    to_a, to_b = np.maximum(0, have - used_a), np.maximum(0, have - used_b)
    to_home = np.where(a_home == 1, to_a, to_b)
    to_away = np.where(a_home == 1, to_b, to_a)

    opening = pd.Series(home_of(ex.opening_offense), index=ex.index)
    opening = opening.where(opening != 0).groupby(code).transform("first").to_numpy(float)
    gets_2h_home = np.where(np.isnan(opening), np.nan, (opening == -1).astype(float))   # home receives the 2nd half

    st = pd.DataFrame({
        "match_code": code.to_numpy(), "message": ex.message.to_numpy(), "file_time": ex.file_time.to_numpy(),
        "play_kind": kind_raw.to_numpy(), "kind": kind, "period": p, "clock": c, "secs_left": secs_left,
        "half_left": half_left, "share": share, "home_score": h, "away_score": a, "lead_home": h - a,
        "pos_home": pos, "down": down, "distance": dist, "ytg": ytg, "to_home": to_home, "to_away": to_away,
        "gets_2h_home": gets_2h_home, "final_p1": _num(ex.final_p1).to_numpy(), "final_p2": _num(ex.final_p2).to_numpy(),
        "prod_p_home": np.where(_num(ex.live_50).fillna(0).to_numpy() == 1, _num(ex.prob_50).to_numpy(), np.nan),
        "prod_pre": _num(ex.prematch_prob_50).to_numpy()})
    st.loc[(st.prod_p_home <= 0) | (st.prod_p_home >= 1), "prod_p_home"] = np.nan
    st["y"] = np.where(st.final_p1 > st.final_p2, 1.0, np.where(st.final_p1 < st.final_p2, 0.0, np.nan))

    # a feed that lost a quarter's start mislabels every later period and clock (feed.lost_quarter)
    prev_p = st.groupby("match_code").period.shift()
    prev_c = st.groupby("match_code").clock.shift()
    jump = (st.period == prev_p) & (st.clock > prev_c + LOST_START_JUMP)
    lost = set(st.match_code[jump])
    bad = st.period.isna() | st.clock.isna() | st.match_code.isin(lost)
    notes.append(f"matches dropped for a lost quarter start (the clock back up {LOST_START_JUMP:.0f}s+ in a "
                 f"period): {len(lost):,}; rows without a period or clock: {int((st.period.isna() | st.clock.isna()).sum()):,}")
    st = st[~bad].copy()

    # when: kick-off from the history, else the first row's file_time, else the code's date
    ft = pd.to_datetime(st.file_time, errors="coerce", format="mixed")
    first_ft = ft.groupby(st.match_code).transform("min")
    last_ft = ft.groupby(st.match_code).transform("max")
    start = st.match_code.map(starts) if starts else pd.Series(pd.NaT, index=st.index)
    start = pd.to_datetime(start).fillna(first_ft).fillna(st.match_code.map(_code_date))
    st["start"] = start
    st["settled"] = last_ft.fillna(start + RESULT_LAG)
    st["settled"] = np.maximum(st.settled, start + RESULT_LAG)
    st["day"] = st.start.dt.strftime("%Y-%m-%d")
    notes.append(f"kick-off from the history for {st.match_code[st.match_code.isin(starts)].nunique():,} matches; "
                 "the rest from their first file_time (or the code's date)")

    pre = st.match_code.map(prematch) if prematch else pd.Series(np.nan, index=st.index)
    st["pre_source"] = np.where(pre.notna(), "table", "")
    if pre_fallback == "prod":
        fb = st.prod_pre.where((st.prod_pre > 0) & (st.prod_pre < 1))
        st.loc[pre.isna() & fb.notna(), "pre_source"] = "prod"
        pre = pre.fillna(fb)
    st["pre_p_home"] = pre
    n_m = st.groupby("match_code").pre_source.first().value_counts()
    notes.append("pre-match probability by source (matches): "
                 + ", ".join(f"{k or 'none (dropped)'} {v:,}" for k, v in n_m.items()))
    st = st[st.pre_p_home.notna()].copy()
    return st.reset_index(drop=True), notes


def wp_frame(st, k=K):
    """The side with the ball's features (wp) and its sign: +1 home has it (or nobody does), -1 away."""
    sgn = np.where(st.pos_home.to_numpy() == -1, -1.0, 1.0)
    pre = logit(st.pre_p_home) * sgn
    root = np.sqrt(st.share.to_numpy())
    lead = st.lead_home.to_numpy(float) * sgn
    home = sgn == 1
    X = pd.DataFrame({
        "lead": lead, "lead_z": lead / np.sqrt(st.share.to_numpy() + k), "pre": pre, "pre_s": pre * root,
        "secs_left": st.secs_left.to_numpy(), "half_left": st.half_left.to_numpy(),
        "period": np.minimum(st.period.to_numpy(), 5.0), "ot": (st.period.to_numpy() >= 5).astype(float),
        "down": st.down.to_numpy(float), "distance": st.distance.to_numpy(float), "ytg": st.ytg.to_numpy(float),
        "own_to": np.where(home, st.to_home, st.to_away), "opp_to": np.where(home, st.to_away, st.to_home),
        "opp_conv": (st.kind.to_numpy() == "conv").astype(float),
        "gets_2h_kick": np.where(st.period.to_numpy() <= 2,
                                 np.where(home, st.gets_2h_home, 1 - st.gets_2h_home), 0.0),
        "is_home": np.where(st.pos_home.to_numpy() == 0, 0.5, home.astype(float)),
        "kind": pd.Series(st.kind).map({k_: i for i, k_ in enumerate(KINDS)}).to_numpy(float)})
    return X[list(WP_FEATURES)], sgn


def lite_frame(st, k=K):
    pre = logit(st.pre_p_home)
    lead = st.lead_home.to_numpy(float)
    X = pd.DataFrame({"lead": lead, "lead_z": lead / np.sqrt(st.share.to_numpy() + k), "pre": pre,
                      "pre_s": pre * np.sqrt(st.share.to_numpy()), "secs_left": st.secs_left.to_numpy(),
                      "period": np.minimum(st.period.to_numpy(), 5.0),
                      "down": np.where(st.kind.to_numpy() == "snap", st.down.to_numpy(float), np.nan)})
    return X[list(LITE_FEATURES)]


# ---------------------------------------------------------------- the models

def _mono(cols, mono):
    return "(" + ",".join(str(mono.get(c, 0)) for c in cols) + ")"


def fit_xgb(X, y, mono, params, rounds, seed=0):
    p = dict(params, monotone_constraints=_mono(X.columns, mono), nthread=NTHREAD, seed=seed)
    return xgb.train(p, xgb.DMatrix(X, label=y), num_boost_round=rounds)


def fit_stern(st, k=K):
    """(a, b, c, d) of logit P(home) = a + b lead/sqrt(s+K) + c pre sqrt(s) + d pre, by maximum likelihood."""
    from scipy.optimize import minimize
    X = _stern_x(st, k)
    y = st.y.to_numpy(float)

    def nll(b):
        z = X @ b
        return np.sum(np.logaddexp(0, z) - y * z) + 1e-6 * b @ b

    def grad(b):
        return X.T @ (sigmoid(X @ b) - y) + 2e-6 * b

    return minimize(nll, np.array([0.0, 1.0, 1.0, 0.0]), jac=grad, method="BFGS").x


def _stern_x(st, k=K):
    pre = logit(st.pre_p_home)
    s = st.share.to_numpy()
    return np.column_stack([np.ones(len(st)), st.lead_home.to_numpy(float) / np.sqrt(s + k), pre * np.sqrt(s), pre])


def predict_all(models, st):
    out = {}
    if "wp" in models:
        X, sgn = wp_frame(st)
        p = models["wp"].predict(xgb.DMatrix(X))
        out["wp"] = np.where(sgn == 1, p, 1 - p)
    if "wp_lite" in models:
        out["wp_lite"] = models["wp_lite"].predict(xgb.DMatrix(lite_frame(st)))
    if "stern" in models:
        out["stern"] = sigmoid(_stern_x(st) @ models["stern"])
    return out


def fit_all(tr, params, rounds):
    X, sgn = wp_frame(tr)
    y_pos = np.where(sgn == 1, tr.y, 1 - tr.y)
    return {"wp": fit_xgb(X, y_pos, WP_MONO, params, rounds),
            "wp_lite": fit_xgb(lite_frame(tr), tr.y.to_numpy(), LITE_MONO, params, rounds),
            "stern": fit_stern(tr)}


def walk_forward(st, since, until, refit_days, train_days, params, rounds, train_every=1, models=None,
                 log=print):
    """Each model's P(home) on the states of matches kicking off in [since, until), refitted every refit_days
    on the matches settled by the block's first midnight, the last train_days days of them."""
    out = {m: pd.Series(np.nan, index=st.index) for m in ("wp", "wp_lite", "stern")}
    days = sorted(st.loc[(st.day >= since) & (st.day < until), "day"].unique())
    last = None
    for i in range(0, len(days), refit_days):
        block = days[i:i + refit_days]
        cut = pd.Timestamp(block[0])
        tr = st[(st.settled <= cut) & (st.start >= cut - pd.Timedelta(days=train_days)) & st.y.notna()]
        if train_every > 1:
            tr = tr[tr.groupby("match_code").cumcount() % train_every == 0]
        if tr.match_code.nunique() < 30:
            log(f"  {block[0]}: only {tr.match_code.nunique()} settled matches before it; skipped")
            continue
        t0 = time.time()
        fitted = fit_all(tr, params, rounds)
        te = st.index[st.day.isin(block)]
        for m, p in predict_all(fitted, st.loc[te]).items():
            out[m].loc[te] = p
        last = fitted
        log(f"  {block[0]}..{block[-1]}: fitted on {len(tr):,} states of {tr.match_code.nunique():,} matches "
            f"in {time.time() - t0:.0f}s; priced {len(te):,}")
    if models is not None and last is not None:
        models.update(last)
    return out


# ---------------------------------------------------------------- scoring

def logloss_each(y, p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def calib(y, p):
    """(intercept, slope) of y on logit p."""
    from scipy.optimize import minimize
    x, y = logit(p), np.asarray(y, float)

    def nll(b):
        z = b[0] + b[1] * x
        return np.sum(np.logaddexp(0, z) - y * z)

    def grad(b):
        r = sigmoid(b[0] + b[1] * x) - y
        return np.array([r.sum(), (r * x).sum()])

    r = minimize(nll, np.array([0.0, 1.0]), jac=grad, method="BFGS")
    return float(r.x[0]), float(r.x[1])


def auc(y, p):
    y = np.asarray(y, float)
    ranks = pd.Series(np.asarray(p, float)).rank(method="average").to_numpy()
    n1 = y.sum()
    n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def metrics(y, p):
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), EPS, 1 - EPS)
    a, b = calib(y, p)
    return dict(n=len(y), logloss=float(logloss_each(y, p).mean()), brier=float(np.mean((p - y) ** 2)),
                auc=auc(y, p), slope=b, intercept=a)


def boot_diff(d, a, b, by="day", n_boot=2000, seed=0):
    """Mean log loss of a less b, and the 95% interval resampling whole `by` groups (days or matches)."""
    diff = logloss_each(d.y.to_numpy(), d[a].to_numpy()) - logloss_each(d.y.to_numpy(), d[b].to_numpy())
    g = pd.DataFrame({"g": d[by].to_numpy(), "x": diff}).groupby("g")["x"].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    k = np.random.default_rng(seed).integers(0, len(s), size=(n_boot, len(s)))
    draws = s[k].sum(1) / c[k].sum(1)
    return float(diff.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def table(d, cols, refs, title):
    d = d.dropna(subset=list(dict.fromkeys(list(cols) + list(refs) + ["y"])))
    lines = [f"\n{title}: {len(d):,} states, {d.match_code.nunique():,} matches, {d.day.nunique()} days"]
    if not len(d) or d.y.nunique() < 2:
        return lines + ["  (nothing to score)"]
    head = f"  {'model':14s} {'logloss':>8s} {'brier':>7s} {'auc':>6s} {'slope':>6s} {'int':>6s}"
    for r in refs:
        head += f"   {'less ' + r + ': mean (days 95%) (matches 95%)':>58s}"
    lines.append(head)
    for c in dict.fromkeys(list(cols) + list(refs)):
        m = metrics(d.y, d[c])
        s = f"  {c:14s} {m['logloss']:8.4f} {m['brier']:7.4f} {m['auc']:6.3f} {m['slope']:6.2f} {m['intercept']:+6.2f}"
        for r in refs:
            if r == c:
                s += " " * 61
                continue
            mu, lo, hi = boot_diff(d, c, r, "day")
            _, mlo, mhi = boot_diff(d, c, r, "match_code")
            s += f"   {mu:+.4f} ({lo:+.4f},{hi:+.4f}) ({mlo:+.4f},{mhi:+.4f})"
        lines.append(s)
    return lines


def by_quarter(d, cols, ref):
    d = d.dropna(subset=list(cols) + [ref, "y"])
    q = np.where(d.period >= 5, "OT", "Q" + d.period.clip(upper=4).astype(int).astype(str))
    lines = [f"\n  by quarter (log loss; {ref} first)"]
    lines.append(f"  {'':4s} {'states':>8s} " + " ".join(f"{c:>9s}" for c in [ref] + list(cols)))
    for g in sorted(set(q)):
        x = d[q == g]
        lines.append(f"  {g:4s} {len(x):8,d} " + " ".join(f"{logloss_each(x.y.to_numpy(), x[c].to_numpy()).mean():9.4f}"
                                                         for c in [ref] + list(cols)))
    return lines


def calibration(d, col, bins=10):
    d = d.dropna(subset=[col, "y"])
    edges = np.quantile(d[col], np.linspace(0, 1, bins + 1))
    b = np.clip(np.searchsorted(edges, d[col], side="right") - 1, 0, bins - 1)
    g = d.groupby(b).agg(n=("y", "size"), p=(col, "mean"), y=("y", "mean"))
    return [f"\n  calibration of {col} by decile: " + "; ".join(f"{r.p:.2f}->{r.y:.2f}" for r in g.itertuples())]


def mono_check(model, st, n=2000, seed=0):
    """Share of states where a point more of the side with the ball's lead (0..+14) or a bigger pre-match edge
    ever lowers its chance; 0 under the constraints."""
    s = st.sample(min(n, len(st)), random_state=seed)
    X, _ = wp_frame(s)
    bad_lead = bad_pre = 0
    prev = None
    for step in range(15):
        Z = X.copy()
        Z["lead"] = X.lead + step
        Z["lead_z"] = Z.lead / np.sqrt(s.share.to_numpy() + K)
        p = model.predict(xgb.DMatrix(Z))
        if prev is not None:
            bad_lead += int((p < prev - 1e-9).sum())
        prev = p
    prev = None
    for step in np.linspace(-1, 1, 9):
        Z = X.copy()
        Z["pre"] = X.pre + step
        Z["pre_s"] = Z.pre * np.sqrt(s.share.to_numpy())
        p = model.predict(xgb.DMatrix(Z))
        if prev is not None:
            bad_pre += int((p < prev - 1e-9).sum())
        prev = p
    return bad_lead, bad_pre, len(s)


def read_compare(specs, st):
    """Join each NAME=CSV (match_code, message, p_home) onto the states."""
    names = []
    for spec in specs:
        name, path = spec.split("=", 1)
        t = pd.read_csv(path)
        key = "match_code" if "match_code" in t else "MATCH_CODE"
        t = t.rename(columns={key: "match_code", "p_home": name})[["match_code", "message", name]]
        t["message"] = pd.to_numeric(t.message, errors="coerce")
        st = st.merge(t.drop_duplicates(["match_code", "message"], keep="last"), on=["match_code", "message"], how="left")
        names.append(name)
    return st, names


# ---------------------------------------------------------------- run

def tune(st, since, until, refit_days, train_days, train_every, log):
    """The XGBoost settings (GRID) with the best log loss of wp on the states of [since, until), walk-forward
    as the report's fits are; the tuning fits read every train_every-th state of a match (--tune-every)."""
    w = (st.day >= since) & (st.day < until) & st.y.notna()
    best, rows = None, []
    days = sorted(st.loc[w, "day"].unique())
    log(f"tuning on {since}..{until}: {int(w.sum()):,} states, {len(days)} days")
    max_r = max(GRID["rounds"])
    for depth, mcw in itertools.product(GRID["max_depth"], GRID["min_child_weight"]):
        params = dict(PARAMS, max_depth=depth, min_child_weight=mcw)
        preds = {r: pd.Series(np.nan, index=st.index) for r in GRID["rounds"]}
        for i in range(0, len(days), refit_days):
            block = days[i:i + refit_days]
            cut = pd.Timestamp(block[0])
            tr = st[(st.settled <= cut) & (st.start >= cut - pd.Timedelta(days=train_days)) & st.y.notna()]
            if train_every > 1:
                tr = tr[tr.groupby("match_code").cumcount() % train_every == 0]
            if tr.match_code.nunique() < 30:
                continue
            X, sgn = wp_frame(tr)
            bst = fit_xgb(X, np.where(sgn == 1, tr.y, 1 - tr.y), WP_MONO, params, max_r)
            te = st.index[st.day.isin(block) & st.y.notna()]
            Xt, sg = wp_frame(st.loc[te])
            for r in GRID["rounds"]:
                p = bst.predict(xgb.DMatrix(Xt), iteration_range=(0, r))
                preds[r].loc[te] = np.where(sg == 1, p, 1 - p)
        for r in GRID["rounds"]:
            x = preds[r][w].dropna()
            ll = float(logloss_each(st.y[x.index].to_numpy(), x.to_numpy()).mean()) if len(x) else np.inf
            rows.append((depth, mcw, r, ll, len(x)))
            log(f"    depth {depth} min_child_weight {mcw} rounds {r}: log loss {ll:.5f} ({len(x):,} states)")
            if best is None or ll < best[3]:
                best = (depth, mcw, r, ll)
    return best, rows


def main(argv=None):
    global NTHREAD
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", required=True, help="the scouting export (scouting_playover.csv)")
    ap.add_argument("--prematch", help="pre-match table: MATCH_CODE (or code), p_home; walk-forward, known at "
                                       "kick-off. Matches it lacks take prod's pre-match quote from the export")
    ap.add_argument("--no-prod-fallback", action="store_true", help="drop matches the --prematch table lacks")
    ap.add_argument("--history", default=os.path.join("eAMFCalibrator", "out", "match_history.csv"),
                    help="match history for kick-off times (MATCH_CODE, SCHEDULED_START_TIME_UTC)")
    ap.add_argument("--since", required=True, help="first kick-off day scored")
    ap.add_argument("--until", required=True, help="the day after the last scored")
    ap.add_argument("--tune-until", help="choose the XGBoost settings on [since, tune-until), report after it")
    ap.add_argument("--refit-days", type=int, default=7)
    ap.add_argument("--train-days", type=int, default=120)
    ap.add_argument("--train-every", type=int, default=1, help="train on every n-th PLAY_OVER of a match")
    ap.add_argument("--tune-every", type=int, default=2, help="the tuning fits read every n-th PLAY_OVER")
    ap.add_argument("--compare", action="append", default=[], metavar="NAME=CSV",
                    help="other in-play prices to score alongside: match_code, message, p_home")
    ap.add_argument("--nthread", type=int, default=NTHREAD)
    ap.add_argument("--out", default="out_inplay_wp")
    a = ap.parse_args(argv)
    NTHREAD = a.nthread
    os.makedirs(a.out, exist_ok=True)
    report = []

    def log(line=""):
        print(line, flush=True)
        report.append(line)

    t0 = time.time()
    ex = read_export(a.export)
    log(f"inplay_wp off {a.export}: {len(ex):,} PLAY_OVER rows, {ex.match_code.nunique():,} matches "
        f"({time.time() - t0:.0f}s)")
    prematch = read_prematch(a.prematch) if a.prematch else {}
    st, notes = build_states(ex, prematch, read_starts(a.history), "none" if a.no_prod_fallback else "prod")
    del ex
    for n in notes:
        log("  " + n)
    log(f"  states: {len(st):,} on {st.match_code.nunique():,} matches, {st.day.min()} to {st.day.max()}; "
        "by kind: " + ", ".join(f"{k} {v:,}" for k, v in st.kind.value_counts().items())
        + f"; ties (no label): {st.match_code[st.y.isna()].nunique():,} matches")
    have_to = st.to_home.notna().mean()
    log(f"  timeouts known on {have_to:.0%} of states" + ("" if have_to else " (the export predates "
        "timeouts_used_a/b: the timeout features are left blank)"))
    st, comp = read_compare(a.compare, st)

    params, rounds = dict(PARAMS), ROUNDS
    report_since = a.since
    if a.tune_until:
        best, _ = tune(st, a.since, a.tune_until, a.refit_days, a.train_days, max(a.train_every, a.tune_every), log)
        if best:
            params.update(max_depth=best[0], min_child_weight=best[1])
            rounds = best[2]
            log(f"  chosen: max_depth {best[0]}, min_child_weight {best[1]}, rounds {best[2]} "
                f"(log loss {best[3]:.5f} on the tuning days); frozen for {a.tune_until}..{a.until}")
        report_since = a.tune_until
    log(f"\nwalk-forward {a.since}..{a.until}: refit every {a.refit_days} days on the last {a.train_days} days "
        f"of settled matches; XGBoost {json.dumps({k: params[k] for k in ('max_depth', 'min_child_weight', 'eta')})}, "
        f"{rounds} rounds, nthread {NTHREAD}")
    models = {}
    preds = walk_forward(st, a.since, a.until, a.refit_days, a.train_days, params, rounds, a.train_every,
                         models, log)
    for m, p in preds.items():
        st[m] = p
    st["pre"] = st.pre_p_home
    cols = ["wp", "wp_lite", "stern", "pre"]

    windows = [(a.since, a.until, "all scored days")]
    if a.tune_until:
        windows = [(a.tune_until, a.until, "REPORT (after tuning)"), (a.since, a.tune_until, "TUNE (settings chosen here)")]
    for lo, hi, name in windows:
        w = st[(st.day >= lo) & (st.day < hi)]
        log(f"\n==== {name}: {lo} to {hi} (exclusive) ====")
        out = table(w, cols, [], "every state")
        q = w[w.prod_p_home.notna()]
        out += table(q, cols + comp, ["prod_p_home"], "the PLAY_OVERs prod quoted live (prob_50)")
        if len(q.dropna(subset=["wp", "prod_p_home", "y"])):
            out += by_quarter(q, ["wp", "wp_lite", "stern"], "prod_p_home")
        for c in comp:
            out += table(w, cols + [c], [c], f"the states {c} prices")
        out += calibration(w, "wp") + calibration(q, "prod_p_home")
        for line in out:
            log(line)
    if "wp" in models:
        bl, bp, n = mono_check(models["wp"], st[st.wp.notna()] if st.wp.notna().any() else st)
        log(f"\nmonotonicity (the last fit, {n:,} states): a point more lead lowered the chance {bl} times, "
            f"a bigger pre-match edge {bp} times")
        gain = models["wp"].get_score(importance_type="gain")
        tot = sum(gain.values()) or 1
        log("trees' gain share: " + ", ".join(f"{k} {v / tot:.1%}" for k, v in sorted(gain.items(), key=lambda kv: -kv[1])))
        log("stern (last fit): a {:+.3f}, lead/sqrt(s+K) {:+.3f}, pre*sqrt(s) {:+.3f}, pre {:+.3f}".format(*models["stern"]))
    keep = ["match_code", "message", "day", "period", "clock", "kind", "home_score", "away_score", "lead_home",
            "pos_home", "down", "distance", "ytg", "to_home", "to_away", "pre_p_home", "pre_source", "y",
            "prod_p_home", "wp", "wp_lite", "stern"] + comp
    st[keep].to_csv(os.path.join(a.out, "states.csv"), index=False)
    with open(os.path.join(a.out, "report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(report) + "\n")
    with open(os.path.join(a.out, "settings.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(vars(a), params=params, rounds=rounds, K=K, kick_yards=KICK_YARDS), fh, indent=1, default=str)
    print(f"\nwritten to {a.out}: states.csv, report.txt, settings.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    sys.exit(main())
