"""v2.1's endgame against the real games: how often a game goes to overtime and how a level game was made
level, from kick-off and from the same late states, and the choices where a side picks between a tie and a
win.

    python -m eAMFModel v21-endgame --model v21_t10 --since 2026-10-01 --until 2026-10-18 --nudge --nudge-share 0.75

The real gamers seem to do what they can to keep a game out of overtime, and the sim sends about twice as
many there (v21-report's census: 0.04 a game against 0.02, Oct 1-17). Three ways in:

  from kick-off    the sim's games from kick-off (the match's gamers and teams, nudged to its expected points
                   as the stream prices) against the real games: overtime a game; a game level at the end of
                   regulation by the score that made it level (the kick after a six from 7 behind, two after a
                   six from 8 behind, a field goal from 3 behind, a six from 6 behind, a safety) and whether it
                   came in Q4's last 3:00; and the final margin's lumps (1, 2, 3, 6, 7, 8 either way), how many
                   of them in overtime.
  the late states  every real snap of Q4 with the sides LEAD_MAX or fewer points apart, played out from the
                   real state with the kick-off's nudge: overtime and the side with the ball winning, the
                   sim's against what really followed, by the time left and that side's lead, each match
                   counting the same. The sim reached the same states the real game did, so what parts here is
                   the endgame itself, not how the game got there.
  the fork         the choices between a tie and a win, real against the sim's (from kick-off and from the late
                   states): the conversion 1 behind after the six (the kick ties) and 2 behind (two ties) in Q4's
                   last 3:00 and in overtime; 4th down and a kick on downs 1-3 in the half's last minute, 3
                   behind (the field goal ties) and 1-2 behind or level (it wins), in Q4's last 2:00.

Nothing here feeds back into the model."""

import csv
import zlib
from collections import Counter, defaultdict

import numpy as np

from . import sim14 as s14, v21, v21_report

KICK_PATHS = 250              # kick-off games per receiving side
STATE_PATHS = 200
LEAD_MAX = 8
CONV_LEFT = 180.0             # the conversions' window: Q4's last 3:00 (the build's audit's)
KICK_LEFT = v21.LATE          # 4th downs and early kicks: Q4's last 2:00
TIME_BANDS = ((240.0, 120.0, "Q4 4:00-2:01"), (120.0, 60.0, "Q4 2:00-1:01"), (60.0, -1.0, "Q4 1:00-0:00"))
LEAD_BANDS = ((-8, -4, "4-8 behind"), (-3, -3, "3 behind"), (-2, -1, "1-2 behind"), (0, 0, "level"),
              (1, 3, "1-3 ahead"), (4, 8, "4-8 ahead"))
LUMPS = (1, 2, 3, 6, 7, 8)
BOOT = 1000
TIES = ("the kick after a six, from 7 behind", "two after a six, from 8 behind", "a field goal, from 3 behind",
        "a six, from 6 behind", "a safety, from 2 behind", "another way", "a score the feed missed", "no score")
# the forks: (label, decision, the deciding side's lead (after the six for a conversion), the window, the
# options' names, the option that ties)
FORKS = (("conversion, 1 behind after the six, Q4's last 3:00", "two", (-1, -1), "conv", ("kick", "two"), 0),
         ("conversion, 2 behind after the six, Q4's last 3:00", "two", (-2, -2), "conv", ("kick", "two"), 1),
         ("conversion, 1 behind after the six, overtime", "two", (-1, -1), "ot", ("kick", "two"), 0),
         ("conversion, 2 behind after the six, overtime", "two", (-2, -2), "ot", ("kick", "two"), 1),
         ("4th down, 3 behind, Q4's last 2:00", "fourth", (-3, -3), "kick", ("go", "field goal", "punt"), 1),
         ("4th down, 1-2 behind or level, Q4's last 2:00", "fourth", (-2, 0), "kick", ("go", "field goal", "punt"), None),
         ("kick on downs 1-3, 3 behind, Q4's last minute", "early_fg", (-3, -3), "kick", ("play on", "kick"), 1),
         ("kick on downs 1-3, 1-2 behind or level, Q4's last minute", "early_fg", (-2, 0), "kick", ("play on", "kick"),
          None))


def _seed(code):
    return zlib.crc32(str(code).encode())


def tie_kind(kind, lead, pts):
    """TIES' index of the score that made a game level: its kind (v21.SCORE_KINDS; the feed's TD row with
    its conversion in it, 7 or 8), the scorer's lead before it and its points."""
    kind, lead, pts = int(kind), int(lead), int(pts)
    if lead + pts != 0:
        return 5
    if (kind == v21.PAT and lead == -1) or (kind == v21.SIX and pts == 7):
        return 0
    if (kind == v21.TWO and lead == -2) or (kind == v21.SIX and pts == 8):
        return 1
    if kind == v21.KICK3:
        return 2
    if kind == v21.SIX and pts == 6:
        return 3
    if kind == v21.SAFETY:
        return 4
    return 5


def _window(win, period, left):
    period, left = np.asarray(period), np.asarray(left, float)
    if win == "ot":
        return period >= 5
    return (period == 4) & (left <= (CONV_LEFT if win == "conv" else KICK_LEFT))


def fork_counts(name_of, choice, period, left, lead):
    """{fork index: option counts} of decisions: name_of the decision's name, the rest arrays alike."""
    out = {}
    for f, (_, name, (lo, hi), win, options, _) in enumerate(FORKS):
        k = (name_of == name) & _window(win, period, left) & (lead >= lo) & (lead <= hi)
        if name == "early_fg":
            k &= np.asarray(left, float) <= v21.EARLY_LEFT
        out[f] = np.bincount(np.asarray(choice)[k].astype(np.int64), minlength=len(options))[:len(options)]
    return out


def _sim_forks(st):
    """fork_counts of a sim's logged decisions (stats["_dec_fine"])."""
    log = st.get("_dec_fine") or []
    if not log:
        return {f: np.zeros(len(FORKS[f][4])) for f in range(len(FORKS))}
    names = np.concatenate([np.full(len(x[1]), x[0], dtype=object) for x in log])
    cols = [np.concatenate([x[i] for x in log]) for i in (2, 3, 4, 5)]
    return fork_counts(names, *cols)


def _sim_ties(st):
    """Counter of TIES over a sim's games level at the end of regulation, (tie, in Q4's last 3:00)."""
    ot, ls = st["went_ot"], st["last_score"]
    out = Counter()
    for i in np.flatnonzero(ot):
        if ls[i, 0] < 0:
            out[(7, False)] += 1
        else:
            out[(tie_kind(*ls[i, [0, 2, 3]]), bool(ls[i, 1] <= CONV_LEFT))] += 1
    return out


def _lumps(margin, ot):
    """{(|margin|, in overtime): games} for LUMPS."""
    a = np.abs(np.asarray(margin))
    return Counter({(m, o): int(((a == m) & (np.asarray(ot) == o)).sum()) for m in LUMPS for o in (False, True)})


# ---------------------------------------------------------------- the real games

def real_ending(rows):
    """(went to overtime, the tie (TIES index, in Q4's last 3:00) if it did): the last score of regulation
    off the rows' scores, the scorer's lead before it and the row's kind."""
    went = any((s14._i(r.get("period")) or 0) >= 5 for r in rows)
    if not went:
        return False, None
    prev, last = None, None
    for r in rows:
        p = s14._i(r.get("period"))
        h, a = s14._i(r.get("score_p1")), s14._i(r.get("score_p2"))
        if p is None or h is None or a is None:
            continue
        if p >= 5:
            break
        if prev is not None and (h, a) != prev:
            side = 0 if h > prev[0] else 1
            pts = (h - prev[0]) if side == 0 else (a - prev[1])
            lead = (prev[0] - prev[1]) if side == 0 else (prev[1] - prev[0])
            k = r.get("play_kind")
            kind = (v21.SIX if k == "TOUCHDOWN" else (v21.PAT if pts == 1 else v21.TWO) if k == "CONVERSION"
                    else v21.KICK3 if k == "FIELD_GOAL" else v21.SAFETY if pts == 2 else -1)
            last = (kind, lead, pts, s14._f(r.get("clock_seconds")) or 0.0, p, (h, a))
        prev = (h, a)
    if last is None:
        return True, ((7 if prev in (None, (0, 0)) else 6), False)
    kind, lead, pts, clock, p, score = last
    if score[0] != score[1]:
        return True, (6, False)                          # the score that levelled it came after the feed's last row
    return True, (tie_kind(kind, lead, pts), bool(p == 4 and clock <= CONV_LEFT))


def real_forks(rows, finals):
    """fork_counts of a match's real decisions (v21.decision_points)."""
    pts = v21.decision_points(rows, final=finals)
    names, cols = [], [[], [], [], []]
    for name in ("two", "fourth", "early_fg"):
        for q in pts.get(name, []):
            names.append(name)
            for i, v in enumerate((q["choice"], q["period"], q["left"], q["lead"])):
                cols[i].append(v)
    if not names:
        return {f: np.zeros(len(FORKS[f][4])) for f in range(len(FORKS))}
    return fork_counts(np.array(names, dtype=object), *(np.array(c, float) for c in cols))


def late_states(model, rows, pair, teams):
    """[(start, the side with the ball, its lead)] at every real snap of Q4 with the sides LEAD_MAX or fewer
    apart."""
    out = []
    for _, s in v21.match_starts(model, rows, pair, teams, every=1):
        if int(s["period"][0]) != 4:
            continue
        team = int(s["team"][0])
        lead = int(s["home"][0] - s["away"][0]) * (1 if team == 0 else -1)
        if abs(lead) <= LEAD_MAX:
            out.append((s, team, lead))
    return out


# ---------------------------------------------------------------- the sims

def match_endgame(model, code, rows, pair, teams, target, finals, kick_paths=KICK_PATHS, state_paths=STATE_PATHS,
                  share=1.0):
    """One match: the sim's kick-off games and its late states' against the real game's."""
    st = {"_dec_fine": []}
    start = v21.kickoff_start(model, pair, teams)
    alpha = None
    if target is not None and getattr(model, "nudge_dir", None) is not None:
        (h, a), alpha = v21.nudge_solve(model, start, target, kick_paths, seed=_seed(code), stats=st, share=share)
    else:
        h, a = v21.simulate(model, start, kick_paths, seed=_seed(code), stats=st)
    h, a = h.ravel(), a.ravel()
    went, tie = real_ending(rows)
    out = dict(code=code, real_ot=went, real_tie=tie, real_margin=finals[0] - finals[1], real_forks=real_forks(rows, finals),
               kick_games=len(h), kick_ot=int(st["went_ot"].sum()), kick_ties=_sim_ties(st),
               kick_lumps=_lumps(h - a, st["went_ot"]), kick_forks=_sim_forks(st), states=[],
               state_forks={f: np.zeros(len(FORKS[f][4])) for f in range(len(FORKS))})
    late = late_states(model, rows, pair, teams) if state_paths else []
    if late:
        S = v21.concat_starts([s for s, _, _ in late])
        if alpha is not None:
            S["nudge"] = np.tile(alpha, (len(S["period"]), 1))
        st2 = {"_dec_fine": []}
        sh, sa = v21.simulate(model, S, state_paths, seed=_seed(code) + 2, stats=st2)
        ot = st2["went_ot"].reshape(len(late), state_paths)
        for k, (s, team, lead) in enumerate(late):
            own, other = (sh[k], sa[k]) if team == 0 else (sa[k], sh[k])
            out["states"].append(dict(code=code, clock=float(s["clock"][0]), lead=lead,
                                      real_ot=float(went), sim_ot=float(ot[k].mean()),
                                      real_win=float((finals[team] - finals[1 - team]) > 0),
                                      sim_win=float((own > other).mean())))
        out["state_forks"] = _sim_forks(st2)
    return out


def _worker(job):
    model_path, items, kick_paths, state_paths, share = job
    model = v21.Model.load(model_path)
    return [match_endgame(model, code, rows, pair, teams, target, finals, kick_paths, state_paths, share)
            for code, rows, pair, teams, target, finals in items]


def run(model_dir, matches, history, means, since=None, until=None, kick_paths=KICK_PATHS, state_paths=STATE_PATHS,
        workers=1, nudge=False, nudge_share=1.0, pool=None):
    """Every settled match of `matches` ({code: rows}) with inputs: the sim's endgame against the real one."""
    import os
    model_path = os.path.join(model_dir, v21.MODEL_FILE) if os.path.isdir(model_dir) else model_dir
    model = v21.Model.load(model_path)
    hist = {r["MATCH_CODE"]: r for r in history}
    codes = sorted(c for c in matches if c in hist and c in means and v21_report._finals(hist[c]) is not None)
    if not codes:
        raise SystemExit("no settled matches with snapshots and inputs in that window")
    if nudge and getattr(model, "nudge_dir", None) is None:
        raise SystemExit("--nudge needs a tilt build (v21-build --mode tilt)")
    items = []
    for c in codes:
        rows = [{k: r.get(k) for k in v21_report.ROW_KEYS} for r in v21._with_side(matches[c])]
        pair, teams = v21.match_people(c, matches[c], hist)
        items.append((c, rows, pair, teams, tuple(means[c]) if nudge else None, v21_report._finals(hist[c])))
    workers = max(1, min(workers, len(items)))
    jobs = [(model_path, items[k::workers], kick_paths, state_paths, nudge_share) for k in range(workers)]
    if workers == 1:
        parts = [_worker(j) for j in jobs]
    else:
        with (pool or v21_report._pool)().Pool(workers) as p:
            parts = p.map(_worker, jobs)
    res = sorted((r for part in parts for r in part), key=lambda r: r["code"])
    return dict(results=res, since=since, until=until, mode=model.mode, nudge=bool(nudge), nudge_share=nudge_share,
                kick_paths=kick_paths, state_paths=state_paths)


# ---------------------------------------------------------------- the report

def _pc(a, b):
    return f"{100 * a / b:5.1f}" if b else "    -"


def _boot_diff(per_match, B=BOOT, seed=0):
    """The mean of the matches' (sim - real) and its 95% interval over matches."""
    x = np.asarray(per_match, float)
    if not len(x):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def kickoff_lines(res):
    n = len(res)
    games = sum(r["kick_games"] for r in res)
    real_ot = sum(r["real_ot"] for r in res)
    sim_ot = sum(r["kick_ot"] for r in res)
    L = [f"  FROM KICK-OFF: {n:,} real games, {games:,} of the sim's ({games // max(1, n):,} a match)",
         f"    overtime: real {_pc(real_ot, n)}% of games ({real_ot:,}), sim {_pc(sim_ot, games)}%",
         "    level at the end of regulation, by the score that made it level, per 100 games: real / sim"
         " (of them in Q4's last 3:00)"]
    real_t, sim_t = Counter(), Counter()
    for r in res:
        if r["real_tie"] is not None:
            real_t[r["real_tie"]] += 1
        sim_t.update(r["kick_ties"])
    for i, name in enumerate(TIES):
        rr = real_t[(i, True)] + real_t[(i, False)]
        ss = sim_t[(i, True)] + sim_t[(i, False)]
        if not rr and not ss:
            continue
        L.append(f"      {name:38s} {100 * rr / n:5.2f} / {100 * ss / max(1, games):5.2f}"
                 f"   ({_pc(real_t[(i, True)], rr).strip()}% / {_pc(sim_t[(i, True)], ss).strip()}%)")
    L.append("    the final margin either way, % of games: real / sim (of them decided in overtime, % of games)")
    real_l, sim_l = Counter(), Counter()
    for r in res:
        real_l.update(_lumps([r["real_margin"]], [r["real_ot"]]))
        sim_l.update(r["kick_lumps"])
    for m in LUMPS:
        rr, ss = real_l[(m, False)] + real_l[(m, True)], sim_l[(m, False)] + sim_l[(m, True)]
        L.append(f"      by {m}   {_pc(rr, n)} / {_pc(ss, games)}   ({_pc(real_l[(m, True)], n).strip()}"
                 f" / {_pc(sim_l[(m, True)], games).strip()})")
    return L


def states_lines(res):
    st = [s for r in res for s in r["states"]]
    if not st:
        return []
    L = [f"  FROM THE SAME LATE STATES: every real snap of Q4 with the sides {LEAD_MAX} or fewer apart, {len(st):,} of"
         f" them on {len({s['code'] for s in st}):,} matches, played out from the real state; each match counting the"
         " same in a cell",
         f"    {'':16s} {'':11s} {'matches':>7s} {'overtime real/sim':>18s} {'the side with the ball wins real/sim':>37s}"]
    for hi, lo, tname in TIME_BANDS:
        for a, b, lname in LEAD_BANDS:
            cell = [s for s in st if lo < s["clock"] <= hi and a <= s["lead"] <= b]
            by = defaultdict(list)
            for s in cell:
                by[s["code"]].append(s)
            if not by:
                continue
            m = lambda k: np.mean([np.mean([s[k] for s in v]) for v in by.values()])
            L.append(f"    {tname:16s} {lname:11s} {len(by):7,d} {100 * m('real_ot'):8.1f} / {100 * m('sim_ot'):<7.1f}"
                     f" {100 * m('real_win'):18.1f} / {100 * m('sim_win'):<7.1f}")
        by = defaultdict(list)
        for s in st:
            if lo < s["clock"] <= hi:
                by[s["code"]].append(s["sim_ot"] - s["real_ot"])
        d, dlo, dhi = _boot_diff([np.mean(v) for v in by.values()])
        L.append(f"    {tname:16s} {'all':11s} {len(by):7,d}   overtime sim less real {100 * d:+.1f} points"
                 f" [{100 * dlo:+.1f}, {100 * dhi:+.1f}]")
    return L


def fork_lines(res):
    L = ["  THE FORK: the choice between a tie and a win, % real / sim from kick-off / sim from the late states"
         " (n real; the sim's own counts in brackets)"]
    for f, (label, _, _, _, options, ties) in enumerate(FORKS):
        r = sum(x["real_forks"][f] for x in res)
        k = sum(x["kick_forks"][f] for x in res)
        s = sum(x["state_forks"][f] for x in res)
        if not r.sum() and not k.sum() and not s.sum():
            continue
        parts = ", ".join(f"{o}{' (ties)' if ties == j else ''} {_pc(r[j], r.sum()).strip()} / {_pc(k[j], k.sum()).strip()}"
                          f" / {_pc(s[j], s.sum()).strip()}" for j, o in enumerate(options))
        L.append(f"    {label:58s} n {int(r.sum()):4d} ({int(k.sum()):,}; {int(s.sum()):,}): {parts}")
    return L


def report(out):
    res = out["results"]
    nudged = (f", nudged {100 * out['nudge_share']:.0f}% of the way to the inputs" if out["nudge"] else "")
    L = [f"  v2.1's ENDGAME ({out['mode']}{nudged}) against the real games, {len(res):,} matches {out['since']} to"
         f" {out['until']}: {2 * out['kick_paths']:,} kick-off games a match, {out['state_paths']:,} from each late state", ""]
    L += kickoff_lines(res) + [""] + states_lines(res) + [""] + fork_lines(res)
    return L


def write_states_csv(path, out):
    rows = [s for r in out["results"] for s in r["states"]]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, ["code", "clock", "lead", "real_ot", "sim_ot", "real_win", "sim_win"])
        w.writeheader()
        w.writerows(rows)
