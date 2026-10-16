"""v2.1's pricing time, spent as the calibrator's stream (v21_stream) spends it: a sample of matches,
each priced as one worker prices it -- the kick-off (its nudge solved: NUDGE_ITERS + 1 runs of both
sides receiving) and every in-play PLAY_OVER state at the stream's paths in one run -- each part
timed, the games and snaps simulated counted, the sim's own parts profiled, and the window's run
projected over the workers.

    python -m eAMFModel v21-timing --model v21_t10 --since 2026-10-01 --until 2026-10-14

The projection is the simulation alone: reading Snowflake and the prior's predictions come on top
(the calibrator prints them apart)."""

import cProfile
import io
import os
import pstats
import time

import numpy as np

from . import v14, v21, v21_stream as vs

DEFAULT_MATCHES = 6
COUNT_PATHS = 20              # games a state for the snaps count (a run apart, untimed)
TOP = 18                      # the profile's functions shown
# the sim's own parts (simulate's inner functions), by their cumulative time
PARTS = (("play", "a snap: timeouts, the time and distance draws, what follows"),
         ("draw_time", "  the time draw"), ("draw_play", "  the distance draw"),
         ("decide", "decisions (4th downs, kicks, conversions, timeouts)"),
         ("fourth", "  4th downs"), ("kickoff", "kick-offs"), ("touchdown", "touchdowns and conversions"),
         ("end_period", "ends of periods"))


def _snaps_a_game(model, start, n_paths=COUNT_PATHS, seed=0):
    st = {}
    v21.simulate(model, start, n_paths, seed=seed, stats=st)
    return st.get("snaps", 0) / max(1, st.get("games", 0))


def price_match(model, rows, pair, teams, n_paths, seed=0, share=vs.NUDGE_SHARE):
    """One match priced as v21_stream's worker prices it, timed: {part: seconds} and the counts."""
    out = {}
    t = time.perf_counter()
    start = v21.kickoff_start(model, pair, teams)
    nudged = getattr(model, "nudge_dir", None) is not None and share
    if nudged:
        h, a = v21.simulate(model, start, max(20, n_paths // 10), seed=seed)
        target = (float(h.mean()) + 2.0, float(a.mean()) - 2.0)        # any target: the solve's runs are fixed
        t = time.perf_counter()
        (h, a), alpha = v21.nudge_solve(model, start, target, n_paths, seed=seed, share=share)
    else:
        h, a = v21.simulate(model, start, n_paths, seed=seed)
        alpha = None
    v14._distributions(h.ravel(), a.ravel())
    out["kickoff"] = time.perf_counter() - t
    runs = v21.NUDGE_ITERS + 1 if nudged else 1
    t = time.perf_counter()
    starts = vs.state_starts(model, [vs._as_text(r) for r in sorted(rows, key=lambda r: int(r["message"]))],
                             pair, teams)
    out["states_read"] = time.perf_counter() - t
    out["states"] = len(starts)
    out["kick_games"] = 2 * n_paths * runs
    out["kick_snaps"] = _snaps_a_game(model, start, seed=seed) * out["kick_games"]
    if not starts:
        out.update(in_play=0.0, books=0.0, play_games=0, play_snaps=0.0)
        return out
    S = v21.concat_starts([s for _, s in starts])
    if alpha is not None:
        S["nudge"] = np.tile(np.asarray(alpha, float), (len(S["period"]), 1))
    t = time.perf_counter()
    sh, sa = v21.simulate(model, S, n_paths, seed=seed + 1)
    out["in_play"] = time.perf_counter() - t
    t = time.perf_counter()
    for k in range(len(starts)):
        v14._distributions(sh[k], sa[k])
    out["books"] = time.perf_counter() - t
    out["play_games"] = len(starts) * n_paths
    out["play_snaps"] = _snaps_a_game(model, S, seed=seed + 2) * out["play_games"]
    return out


def profile_lines(model, rows, pair, teams, n_paths, seed=0, top=TOP):
    """The in-play run of one match under cProfile: simulate's parts and the functions it spends in."""
    starts = vs.state_starts(model, [vs._as_text(r) for r in sorted(rows, key=lambda r: int(r["message"]))],
                             pair, teams)
    if not starts:
        return []
    S = v21.concat_starts([s for _, s in starts])
    prof = cProfile.Profile()
    prof.enable()
    v21.simulate(model, S, n_paths, seed=seed + 1)
    prof.disable()
    stats = pstats.Stats(prof, stream=io.StringIO())
    total = sum(v[2] for v in stats.stats.values()) or 1.0          # every function's own time
    cum = {}
    for (path, _, name), v in stats.stats.items():
        if path.endswith("v21.py"):
            cum[name] = cum.get(name, 0.0) + v[3]
    lines = [f"  where one match's in-play run spends its time (cProfile, {len(starts)} states x {n_paths:,} games;"
             " profiling slows it, the shares hold):",
             "  simulate's parts (cumulative, each with what it calls):"]
    for name, label in PARTS:
        if name in cum:
            lines.append(f"    {label:56s}{100 * cum[name] / total:6.1f}%")
    lines += ["", "  the functions it spends its own time in:",
              f"    {'function':52s}{'own time':>9s}{'calls':>10s}"]
    rows_ = sorted(stats.stats.items(), key=lambda kv: -kv[1][2])[:top]
    for (path, line, name), v in rows_:
        where = f"{os.path.basename(path)}:{line}" if line else path
        label = f"{name} ({where})" if not name.startswith("<") else name
        lines.append(f"    {label[:52]:52s}{100 * v[2] / total:8.1f}%{v[1]:>10,}")
    return lines


def run(model_dir, snaps, history, n_matches=DEFAULT_MATCHES, n_paths=vs.DEFAULT_PATHS, workers=None,
        seed=0, profile=True):
    """{"per": [(code, timings)], "window": matches in the window, ...} for a seeded sample of matches."""
    t = time.perf_counter()
    model = v21.Model.load(vs._model_file(model_dir))
    load = time.perf_counter() - t
    hist = {r["MATCH_CODE"]: r for r in history}
    codes = sorted(c for c in snaps if c in hist)
    pick = sorted(np.random.default_rng(seed).choice(codes, min(n_matches, len(codes)), replace=False)) if codes else []
    per = []
    for code in pick:
        pair, teams = v21.match_people(code, snaps[code], hist)
        per.append((code, price_match(model, snaps[code], pair, teams, n_paths, seed=vs._seed(code))))
    prof = []
    if profile and pick:
        code = pick[0]
        pair, teams = v21.match_people(code, snaps[code], hist)
        prof = profile_lines(model, snaps[code], pair, teams, n_paths, seed=vs._seed(code))
    return dict(per=per, window=len(codes), load=load, n_paths=n_paths, profile=prof,
                workers=workers or max(1, (os.cpu_count() or 2) - 1), cpus=os.cpu_count())


def _mins(s):
    s = int(round(s))
    return f"{s // 3600}h{(s % 3600) // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s" if s >= 60 else f"{s}s"


def report(out):
    per = out["per"]
    if not per:
        return ["  no matches with snapshots and a history row in that window"]
    n = out["n_paths"]
    lines = [f"  v2.1 pricing time: {len(per)} of the window's {out['window']:,} matches, each priced as one of the"
             f" calibrator's workers prices it ({n:,} games a state, the kick-off {2 * n:,} a run); the model loads"
             f" in {out['load']:.1f}s (once a worker)", "",
             f"  {'match':14s}{'states':>7s}{'kick-off':>10s}{'in play':>10s}{'books':>8s}{'total':>9s}"
             f"{'games':>11s}{'snaps':>12s}{'us/snap':>9s}"]
    tot = {k: 0.0 for k in ("kickoff", "in_play", "books", "states_read", "states", "kick_games", "play_games",
                            "kick_snaps", "play_snaps")}
    for code, t in per:
        secs = t["kickoff"] + t["in_play"] + t["books"] + t["states_read"]
        games, snaps = t["kick_games"] + t["play_games"], t["kick_snaps"] + t["play_snaps"]
        lines.append(f"  {code[:14]:14s}{t['states']:>7d}{t['kickoff']:>9.1f}s{t['in_play']:>9.1f}s{t['books']:>7.1f}s"
                     f"{secs:>8.1f}s{games:>11,}{snaps:>12,.0f}{1e6 * (t['kickoff'] + t['in_play']) / max(1, snaps):>9.1f}")
        for k in tot:
            tot[k] += t[k]
    m = len(per)
    secs = (tot["kickoff"] + tot["in_play"] + tot["books"] + tot["states_read"]) / m
    snaps = (tot["kick_snaps"] + tot["play_snaps"]) / m
    lines += ["", f"  a match on average: {tot['states'] / m:.0f} states, {secs:.1f}s -- the kick-off {tot['kickoff'] / m:.1f}s"
                  f" ({tot['kick_games'] / m:,.0f} games), in play {tot['in_play'] / m:.1f}s"
                  f" ({tot['play_games'] / m:,.0f} games, {tot['play_snaps'] / max(1, tot['play_games']):.1f} snaps"
                  f" each), the books {tot['books'] / m:.2f}s",
              f"  a state's {n:,} games in play: {tot['in_play'] / max(1, tot['states']):.2f}s;"
              f" a snap: {1e6 * (tot['kickoff'] + tot['in_play']) / max(1, m * snaps):.1f} microseconds",
              f"  the window's {out['window']:,} matches on {out['workers']} workers ({out['cpus']} cores here):"
              f" about {_mins(secs * out['window'] / out['workers'])} of simulation"
              f" ({_mins(secs * out['window'])} on one core)"]
    if out.get("profile"):
        lines += [""] + out["profile"]
    return lines
