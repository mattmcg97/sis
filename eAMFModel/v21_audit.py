"""v2.1's audit, down by down and by the time left: the sim against the real games, two ways.

    python -m eAMFModel v21-audit --model v21_dec --since 2026-10-01 --until 2026-10-12 --out eAMFCalibrator/out_v21_audit

  game flow   the sim's own games from kick-off (each window match with its gamers and teams) against
              the real games: snaps a game at each down in each part of the game, how often a down
              ends in a first down (or a touchdown), its yards and the clock it takes. Where the sim
              reaches a down more or less often than the real games, this is where the drives part.
  one snap    the next snap alone, drawn --state-paths times from every real state before a snap and
              set against what really came of it: the draws with the game flow taken out. By down and
              part of the game, and by down and distance: the share of each outcome (a turnover, a
              loss, no gain, a gain short of the line, a first down, a touchdown), the yards and the
              clock; and the 4th-down and early-kick choices from the same states.

The parts of the game are the halves' quarters, the second quarter of each half cut at 2:00, 1:00 and
0:30 (SLICES). Real snaps are v21.match_snaps' (a punt or a field goal is not a snap; a 4th down that
went for it is); the sim's are logged as a real one reads, PLAY_OVER to PLAY_OVER.
"""

import zlib

import numpy as np

from . import v21

SLICES = ("Q1", "Q2 to 2:00", "Q2 2:00-1:01", "Q2 1:00-0:31", "Q2 last 0:30", "Q3", "Q4 to 2:00", "Q4 2:00-1:01",
          "Q4 1:00-0:31", "Q4 last 0:30", "OT")
OUTCOMES = ("turnover", "loss", "no gain", "short", "1st down", "touchdown")
DISTS = ("1 to go", "2-3", "4-6", "7-10", "11+")
DOWNS = ("1st", "2nd", "3rd", "4th")
D, S, B, O = len(DOWNS), len(SLICES), len(DISTS), len(OUTCOMES)
FIRST, TD = 4, 5
FOURTH = ("go for it", "field goal", "punt")


def slice_of(period, clock):
    """Each snap's part of the game (SLICES)."""
    p, c = np.asarray(period), np.asarray(clock, float)
    late = np.where(c > 120, 0, np.where(c > 60, 1, np.where(c > 30, 2, 3)))
    return np.select([p == 1, p == 2, p == 3, p == 4], [0, 1 + late, 5, 6 + late], 10).astype(np.int64)


def outcome_of(kind, gain, dist, field, conv=None):
    """Each snap's outcome (OUTCOMES); gains cut at the goal line. conv: whether it moved the chains
    (the next row's down for a real snap, the sim's own for its), else gain >= dist."""
    kind, gain, dist, field = (np.asarray(x) for x in (kind, gain, dist, field))
    first = gain >= dist if conv is None else np.asarray(conv, bool)
    return np.select([kind != v21.GAIN, gain >= 100 - field, first, gain > 0, gain == 0],
                     [0, TD, FIRST, 3, 2], 1).astype(np.int64)


REMAINS = ("0-1 snaps left", "2", "3-4", "5-7", "8-12", "13+")


def tables():
    return dict(games=0.0, flow=np.zeros((D, S, O)), flow_yards=np.zeros((D, S)), flow_secs=np.zeros((D, S)),
                step=np.zeros((D, S, B, O)), step_yards=np.zeros((D, S, B)), step_secs=np.zeros((D, S, B)),
                fourth=np.zeros((S, 3)), early=np.zeros((S, 2)), ot_games=0.0, ot=np.zeros((7, 4)),
                ot_n=np.zeros((7, 4)), by_rem=np.zeros((D, len(REMAINS), O)))


OT_LEADS = v21.MARGIN_NAMES
OT_CLOCK = ("Q4 to 2:00", "Q4 2:00-1:01", "Q4 1:00-0:31", "Q4 last 0:30")


def _add(a, b):
    return {k: a[k] + b[k] for k in a}


def _put(t, key, down, sl, dist, out, gain, secs, kind, w):
    """Snaps into a table: counts by outcome, yards (gains only) and seconds, weighted w."""
    d = np.clip(np.asarray(down, np.int64) - 1, 0, D - 1)
    if key == "flow":
        np.add.at(t["flow"], (d, sl, out), w)
        np.add.at(t["flow_yards"], (d, sl), np.where(kind == v21.GAIN, gain, 0) * w)
        np.add.at(t["flow_secs"], (d, sl), secs * w)
    else:
        db = v21.dist_b(dist)
        np.add.at(t["step"], (d, sl, db, out), w)
        np.add.at(t["step_yards"], (d, sl, db), np.where(kind == v21.GAIN, gain, 0) * w)
        np.add.at(t["step_secs"], (d, sl, db), secs * w)


def _real_cols(recs):
    g = lambda k: np.array([r[k] for r in recs])
    kind, gain, dist, field = g("kind"), g("gain"), g("distance"), g("field")
    gain = np.minimum(gain, 100 - field)
    conv = np.array([r["converted"] and not r["replay"] for r in recs], bool)
    return dict(down=g("down"), sl=slice_of(g("period"), g("clock")), dist=dist,
                out=outcome_of(kind, gain, dist, field, conv), gain=gain, secs=g("seconds").astype(float), kind=kind)


def _sim_cols(log, keep=None):
    keep = np.ones(len(log["path"]), bool) if keep is None else keep
    g = lambda k: log[k][keep]
    kind, gain, dist, field = g("kind"), g("gain"), g("dist"), g("field")
    return dict(down=g("down"), sl=slice_of(g("period"), g("clock")), dist=dist,
                out=outcome_of(kind, gain, dist, field, g("conv")), gain=gain, secs=g("seconds").astype(float),
                kind=kind, rem=g("rem"))


def _worker(job):
    """Audit a list of matches: (real, the sim's game flow, the sim's next snaps)."""
    model_path, items, n_paths, state_paths, every = job
    model = v21.Model.load(model_path)
    real, flow, step = tables(), tables(), tables()
    for code, rows, pair, teams in items:
        seed = zlib.crc32(str(code).encode())
        rows = v21._with_side(rows)
        recs = [r for r in v21.match_snaps(rows) if r["period"] <= 4 + v21.MAX_OT]
        real["games"] += 1
        if recs:
            c = _real_cols(recs)
            _put(real, "flow", c["down"], c["sl"], c["dist"], c["out"], c["gain"], c["secs"], c["kind"], 1.0)
        # the sim's games from kick-off
        st = {}
        v21.simulate(model, v21.kickoff_start(model, pair, teams), n_paths, seed=seed, stats=st)
        flow["games"] += st["games"]
        flow["ot_games"] += float(st["went_ot"].sum())
        real["ot_games"] += float(any((v21.s14._i(r.get("period")) or 0) >= 5 for r in rows))
        log = v21.snap_log(st)
        if len(log["path"]):
            c = _sim_cols(log)
            _put(flow, "flow", c["down"], c["sl"], c["dist"], c["out"], c["gain"], c["secs"], c["kind"], 1.0)
        # the next snap from every real state, against what came of it
        starts = v21.match_starts(model, rows, pair, teams, every=every)
        if not starts or not state_paths:
            continue
        by_start = {r["start"]: r for r in recs}
        st_rows = [rows[i] for i, _ in starts]
        sl_start = slice_of([s["period"][0] for _, s in starts], [s["clock"][0] for _, s in starts])
        down_start = np.array([s["down"][0] for _, s in starts])
        mine = [by_start.get(v21.s14._i(r["message"])) for r in st_rows]
        got = [r for r in mine if r is not None]
        if got:
            c = _real_cols(got)
            _put(real, "step", c["down"], c["sl"], c["dist"], c["out"], c["gain"], c["secs"], c["kind"], 1.0)
            # by the snaps left the model expects at each real state (its own table, the state's own key)
            rb = v21.remain_b(model.remain_at(*v21.remain_inputs(got)))
            np.add.at(real["by_rem"], (np.clip(c["down"] - 1, 0, D - 1), rb, c["out"]), 1)
        # the real choices at the same states: 4th downs, and kicks on downs 1-3 in the half's last minute
        for j, (i, s) in enumerate(starts):
            b = rows[i + 1] if i + 1 < len(rows) else None
            if b is None or b.get("period") != rows[i].get("period"):
                continue
            p, clk, d = int(s["period"][0]), float(s["clock"][0]), int(s["down"][0])
            if d == 4:
                real["fourth"][sl_start[j], v21.PUNT if b["play_kind"] == "PUNT"
                               else v21.FG if b["play_kind"] == "FIELD_GOAL" else v21.GO] += 1
            elif (p == 2 or p >= 4) and float(v21.half_left(p, clk)) <= v21.EARLY_LEFT \
                    and 100 - int(s["y"][0]) + 17 <= v21.FG_MAX:
                real["early"][sl_start[j], int(b["play_kind"] == "FIELD_GOAL")] += 1
        S_ = v21.concat_starts([s for _, s in starts])
        st = {}
        v21.simulate(model, S_, state_paths, seed=seed + 1, stats=st, max_steps=1)
        log = v21.snap_log(st)
        if len(log["path"]):
            c = _sim_cols(log)
            _put(step, "step", c["down"], c["sl"], c["dist"], c["out"], c["gain"], c["secs"], c["kind"], 1.0 / state_paths)
            np.add.at(step["by_rem"], (np.clip(c["down"] - 1, 0, D - 1), c["rem"], c["out"]), 1.0 / state_paths)
        for name, who, k, sl, ix in st.get("_dec") or []:
            if ix is None or name not in ("fourth", "early_fg"):
                continue
            first = sl_start[ix // state_paths]
            if name == "fourth":
                np.add.at(step["fourth"], (first, k), 1.0 / state_paths)
            else:
                np.add.at(step["early"], (first, k), 1.0 / state_paths)
        # overtime: from every real Q4 state to the end, against whether the real game went to overtime
        q4 = [(j, s) for j, (i, s) in enumerate(starts) if int(s["period"][0]) == 4]
        if q4:
            went_real = float(any((v21.s14._i(r.get("period")) or 0) >= 5 for r in rows))
            st = {}
            v21.simulate(model, v21.concat_starts([s for _, s in q4]), state_paths, seed=seed + 2, stats=st)
            went = st["went_ot"].reshape(len(q4), state_paths).mean(1)
            for (j, s), w in zip(q4, went):
                side = int(s["team"][0])
                lead = int(s["home"][0] - s["away"][0]) * (1 if side == 0 else -1)
                c = float(s["clock"][0])
                cs = 0 if c > 120 else 1 if c > 60 else 2 if c > 30 else 3
                lb = int(v21.margin_b(lead))
                real["ot"][lb, cs] += went_real
                real["ot_n"][lb, cs] += 1
                step["ot"][lb, cs] += w
                step["ot_n"][lb, cs] += 1
    return real, flow, step


def run(model_dir, matches, history, n_paths=500, state_paths=50, every=1, workers=1, since=None, until=None):
    import os
    model_path = os.path.join(model_dir, v21.MODEL_FILE) if os.path.isdir(model_dir) else model_dir
    hist = {r["MATCH_CODE"]: r for r in (history or [])}
    keys = ("match_code", "message", "period", "clock_seconds", "play_kind", "offense", "down", "distance",
            "field_position", "score_p1", "score_p2", "play_messages", "team_a_side", "opening_offense",
            "timeouts_used_a", "timeouts_used_b", "timeout_after", "timeout_prev", "home_handle", "away_handle")
    items = [(c, [{k: r.get(k) for k in keys} for r in v21._with_side(rows)]) + v21.match_people(c, rows, hist)
             for c, rows in sorted(matches.items())]
    if not items:
        raise SystemExit("no matches to audit")
    workers = max(1, min(workers, len(items)))
    jobs = [(model_path, items[k::workers], n_paths, state_paths, every) for k in range(workers)]
    if workers == 1:
        parts = [_worker(j) for j in jobs]
    else:
        from . import v14
        with v14.pool_context().Pool(workers) as p:
            parts = p.map(_worker, jobs)
    real, flow, step = parts[0]
    for r, f, s in parts[1:]:
        real, flow, step = _add(real, r), _add(flow, f), _add(step, s)
    return dict(real=real, flow=flow, step=step, matches=len(items), n_paths=n_paths, state_paths=state_paths,
                every=every, since=since, until=until)


# ---------------------------------------------------------------- the report

def _pct(x, n):
    return 100.0 * x / n if n else float("nan")


def _pair(a, b, fmt="{:5.1f}"):
    f = lambda v: "  -  " if v != v else fmt.format(v)
    return f"{f(a)}/{f(b)}"


def flow_lines(real, flow):
    g_r, g_s = max(1.0, real["games"]), max(1.0, flow["games"])
    lines = ["  GAME FLOW: the sim's own games from kick-off against the real games (real / sim)"]
    total_r, total_s = real["flow"].sum((1, 2)) / g_r, flow["flow"].sum((1, 2)) / g_s
    rel = lambda a, b: f" ({100 * (b / a - 1):+.0f}%)" if a else ""
    lines.append("  snaps a game at each down: " + ", ".join(
        f"{DOWNS[d]} {total_r[d]:.2f}/{total_s[d]:.2f}{rel(total_r[d], total_s[d])}" for d in range(D)))
    for d in range(D):
        lines += ["", f"  {DOWNS[d]} down", f"  {'':16s}{'snaps a game':>14s}{'1st down or TD %':>18s}{'turnover %':>14s}"
                      f"{'yards':>14s}{'seconds':>14s}"]
        for s_ in list(range(S)) + [None]:
            r = real["flow"][d].sum(0) if s_ is None else real["flow"][d, s_]
            m = flow["flow"][d].sum(0) if s_ is None else flow["flow"][d, s_]
            nr, ns = r.sum(), m.sum()
            if not nr and not ns:
                continue
            yr = (real["flow_yards"][d].sum() if s_ is None else real["flow_yards"][d, s_]) / max(1, nr - r[0])
            ys = (flow["flow_yards"][d].sum() if s_ is None else flow["flow_yards"][d, s_]) / max(1, ns - m[0])
            sr = (real["flow_secs"][d].sum() if s_ is None else real["flow_secs"][d, s_]) / max(1, nr)
            ss = (flow["flow_secs"][d].sum() if s_ is None else flow["flow_secs"][d, s_]) / max(1, ns)
            lines.append(f"  {'all' if s_ is None else SLICES[s_]:16s}{_pair(nr / g_r, ns / g_s, '{:5.2f}'):>14s}"
                         f"{_pair(_pct(r[FIRST] + r[TD], nr), _pct(m[FIRST] + m[TD], ns)):>18s}"
                         f"{_pair(_pct(r[0], nr), _pct(m[0], ns)):>14s}{_pair(yr, ys):>14s}{_pair(sr, ss):>14s}")
    return lines


def _step_row(label, r, m, yr, ys, sr, ss):
    nr, ns = r.sum(), m.sum()
    return (f"  {label:16s}{int(round(nr)):>7d}" + "".join(f"{_pair(_pct(r[o], nr), _pct(m[o], ns)):>13s}" for o in range(O))
            + f"{_pair(yr / max(1, nr - r[0]), ys / max(1e-9, ns - m[0])):>13s}{_pair(sr / max(1, nr), ss / max(1e-9, ns)):>13s}")


def step_lines(real, step):
    lines = ["", "  ONE SNAP: the next snap drawn from every real state before a snap, against what came of it"
                 " (real / sim, % of the snaps)"]
    head = f"  {'':16s}{'n real':>7s}" + "".join(f"{o:>13s}" for o in OUTCOMES) + f"{'yards':>13s}{'seconds':>13s}"
    for d in range(D):
        lines += ["", f"  {DOWNS[d]} down, by the part of the game", head]
        for s_ in list(range(S)) + [None]:
            r = real["step"][d].sum((0, 1)) if s_ is None else real["step"][d, s_].sum(0)
            m = step["step"][d].sum((0, 1)) if s_ is None else step["step"][d, s_].sum(0)
            if not r.sum() and not m.sum():
                continue
            pick = (lambda t: t[d].sum()) if s_ is None else (lambda t: t[d, s_].sum())
            lines.append(_step_row("all" if s_ is None else SLICES[s_], r, m, pick(real["step_yards"]), pick(step["step_yards"]),
                                   pick(real["step_secs"]), pick(step["step_secs"])))
        lines += [f"  {DOWNS[d]} down, by distance", head]
        for b in range(B):
            r, m = real["step"][d, :, b].sum(0), step["step"][d, :, b].sum(0)
            if not r.sum() and not m.sum():
                continue
            lines.append(_step_row(DISTS[b], r, m, real["step_yards"][d, :, b].sum(), step["step_yards"][d, :, b].sum(),
                                   real["step_secs"][d, :, b].sum(), step["step_secs"][d, :, b].sum()))
    return lines


def rem_lines(real, step):
    lines = ["", "  ONE SNAP by the snaps left in the half the model expects at the state (the clock, running or not,"
                 " both sides' timeouts, the lead): real / sim, % of the snaps",
             f"  {'':16s}{'n real':>7s}" + "".join(f"{o:>13s}" for o in OUTCOMES)]
    for d in range(D):
        lines.append(f"  {DOWNS[d]} down")
        for b in range(len(REMAINS)):
            r, m = real["by_rem"][d, b], step["by_rem"][d, b]
            if r.sum():
                lines.append(f"  {REMAINS[b]:16s}{int(r.sum()):>7d}"
                             + "".join(f"{_pair(_pct(r[o], r.sum()), _pct(m[o], m.sum())):>13s}" for o in range(O)))
    return lines


def choice_lines(real, step):
    lines = ["", "  4th DOWN from the real states: the choice (real / sim, %)",
             f"  {'':16s}{'n real':>7s}" + "".join(f"{o:>14s}" for o in FOURTH)]
    for s_ in list(range(S)) + [None]:
        r = real["fourth"].sum(0) if s_ is None else real["fourth"][s_]
        m = step["fourth"].sum(0) if s_ is None else step["fourth"][s_]
        if r.sum():
            lines.append(f"  {'all' if s_ is None else SLICES[s_]:16s}{int(r.sum()):>7d}"
                         + "".join(f"{_pair(_pct(r[k], r.sum()), _pct(m[k], m.sum())):>14s}" for k in range(3)))
    lines += ["", "  A KICK on downs 1-3 in range, the half's last minute, from the real states (real / sim, %)",
              f"  {'':16s}{'n real':>7s}{'kick':>14s}"]
    for s_ in range(S):
        r, m = real["early"][s_], step["early"][s_]
        if r.sum():
            lines.append(f"  {SLICES[s_]:16s}{int(r.sum()):>7d}{_pair(_pct(r[1], r.sum()), _pct(m[1], m.sum())):>14s}")
    return lines


def gap_lines(real, flow, step, top=12, min_real=100):
    """The cells where the sim parts most from the real games: snaps a game (game flow) and the next
    snap's first-down-or-touchdown and turnover rates, by how many standard errors of the real share."""
    g_r, g_s = max(1.0, real["games"]), max(1.0, flow["games"])
    rows = []
    for d in range(D):
        for s_ in range(S):
            nr, ns = real["flow"][d, s_].sum(), flow["flow"][d, s_].sum()
            if nr >= min_real / 4:
                a, b = nr / g_r, ns / g_s
                se = np.sqrt(nr) / g_r                                  # a count's standard error, a game
                rows.append(((b - a) / se, f"{DOWNS[d]} down, {SLICES[s_]}: snaps a game {a:.2f} real, {b:.2f} sim"
                                           f" ({100 * (b / a - 1):+.0f}%)"))
            r, m = real["step"][d, s_].sum(0), step["step"][d, s_].sum(0)
            if r.sum() >= min_real and m.sum():
                for label, idx in (("1st down or TD", [FIRST, TD]), ("turnover", [0]), ("no gain or loss", [1, 2])):
                    p, q = r[idx].sum() / r.sum(), m[idx].sum() / m.sum()
                    se = np.sqrt(max(p * (1 - p), 1e-4) / r.sum())
                    rows.append(((q - p) / se, f"{DOWNS[d]} down, {SLICES[s_]}: next snap {label} {100 * p:.1f}% real,"
                                               f" {100 * q:.1f}% sim (n {int(r.sum()):,})"))
    rows.sort(key=lambda r: -abs(r[0]))
    return ["", f"  THE LARGEST GAPS (by standard errors of the real figure; cells with {min_real}+ real snaps):"] + \
        [f"  {z:+6.1f}  {text}" for z, text in rows[:top]]


def ot_lines(real, flow, step):
    g_r, g_s = max(1.0, real["games"]), max(1.0, flow["games"])
    lines = ["", "  OVERTIME: how often a game goes to overtime (real / sim, %)",
             f"  from kick-off: {100 * real['ot_games'] / g_r:.1f} / {100 * flow['ot_games'] / g_s:.1f}"
             f" ({int(real['ot_games'])} real games of {int(real['games'])})",
             "  from every real Q4 state, by the lead of the side with the ball and the time left",
             f"  {'':14s}" + "".join(f"{c:>20s}" for c in OT_CLOCK)]
    for lb, name in enumerate(OT_LEADS):
        n = real["ot_n"][lb]
        if not n.sum():
            continue
        cells = []
        for cs in range(4):
            if n[cs]:
                cells.append(f"{100 * real['ot'][lb, cs] / n[cs]:5.1f}/{100 * step['ot'][lb, cs] / step['ot_n'][lb, cs]:5.1f}"
                             f" ({int(n[cs])})")
            else:
                cells.append("-")
        lines.append(f"  {name:14s}" + "".join(f"{c:>20s}" for c in cells))
    return lines


def report(out):
    lines = [f"  v2.1 audit on {out['matches']:,} matches {out['since'] or ''} to {out['until'] or ''}:"
             f" {2 * out['n_paths']:,} kick-off games each; the next snap {out['state_paths']} times from every"
             f" {'' if out['every'] == 1 else str(out['every']) + 'th '}real state", ""]
    lines += gap_lines(out["real"], out["flow"], out["step"])
    lines += [""] + flow_lines(out["real"], out["flow"])
    lines += step_lines(out["real"], out["step"])
    lines += rem_lines(out["real"], out["step"])
    lines += choice_lines(out["real"], out["step"])
    lines += ot_lines(out["real"], out["flow"], out["step"])
    lines += ["", "  (snaps: a punt or a field goal is not one, a 4th down that went for it is; yards: the gain of the"
                  " snaps that kept the ball, cut at the goal line; seconds: PLAY_OVER to PLAY_OVER)"]
    return lines
