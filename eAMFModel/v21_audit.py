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
              clock; and the 4th-down and early-kick choices from the same states. Q4's last 2:00 is
              split by the lead, the real states the game's own end (v21.auto_end) covers apart:
              the sim ends the game there with no snap.
  Q4 states   from every real Q4 state to the end: how often the game went to overtime, each side's
              points still to come in regulation and, from level, who won it.

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


REMAINS = ("0-1 snaps left",) + v21.REMAIN_NAMES[1:]
LEADS3 = ("behind", "level", "ahead")
DRIVE_LEADS = ("9+ behind", "4-8 behind", "1-3 behind", "level", "1-3 ahead", "4-8 ahead", "9+ ahead")
RESULTS = ("won", "overtime", "lost")
SHORT_ROWS = ("1 to go", "2-3 to go", "4+ to go", "a half's last 2:00")
CONV_PARTS = ("the first half", "Q3", "Q4 to 3:00")
GOOD_PARTS = CONV_PARTS + ("Q4's last 3:00", "overtime")


def tables():
    return dict(games=0.0, flow=np.zeros((D, S, O)), flow_yards=np.zeros((D, S)), flow_secs=np.zeros((D, S)),
                step=np.zeros((D, S, B, O)), step_yards=np.zeros((D, S, B)), step_secs=np.zeros((D, S, B)),
                fourth=np.zeros((S, 3)), early=np.zeros((S, 2)), ot_games=0.0, ot=np.zeros((7, 4)),
                ot_n=np.zeros((7, 4)), by_rem=np.zeros((D, len(REMAINS), O)),
                late=np.zeros((4, O)), late_secs=np.zeros(4), pts_ball=np.zeros((7, 4)), pts_other=np.zeros((7, 4)),
                level=np.zeros((4, 3)), fourth_q4=np.zeros((4, 3)), early_q4=np.zeros((4, 2)),
                two_late=np.zeros((2 * v21.CONV_MARGIN + 1, 2)), two_base=np.zeros(2 * v21.CONV_MARGIN + 1),
                fourth_q4_base=np.zeros((4, 3)), early_q4_base=np.zeros((4, 2)),
                late_tb=np.zeros((3, len(v21.TIME_BUCKETS), 2)), late_src=np.zeros((3, 3, 2)),
                drives=np.zeros((len(DRIVE_LEADS), 4, len(v21.DRIVE_ENDS))),
                short_tb=np.zeros((len(SHORT_ROWS), len(v21.TIME_BUCKETS), 3)),
                two_part=np.zeros((len(CONV_PARTS), 2 * v21.CONV_MARGIN + 1, 2)),
                two_part_base=np.zeros((len(CONV_PARTS), 2 * v21.CONV_MARGIN + 1)),
                two_good=np.zeros((len(GOOD_PARTS), 2)), two_sim=np.zeros(4))


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
                out=outcome_of(kind, gain, dist, field, conv), gain=gain, secs=g("seconds").astype(float), kind=kind,
                period=g("period"), clock=g("clock").astype(float), lead=g("margin"), tb=v21.time_b(g("play_seconds")))


def _sim_cols(log, keep=None):
    keep = np.ones(len(log["path"]), bool) if keep is None else keep
    g = lambda k: log[k][keep]
    kind, gain, dist, field = g("kind"), g("gain"), g("dist"), g("field")
    return dict(down=g("down"), sl=slice_of(g("period"), g("clock")), dist=dist,
                out=outcome_of(kind, gain, dist, field, g("conv")), gain=gain, secs=g("seconds").astype(float),
                kind=kind, rem=g("rem"), period=g("period"), clock=g("clock").astype(float), lead=g("lead"),
                tb=g("tb"), src=g("src"))


def _late(t, c, w, auto=None):
    """Q4's last 2:00 by the lead (LEADS3; row 3: the states the game's own end covers)."""
    on = (c["period"] == 4) & (c["clock"] <= v21.LATE)
    row = np.sign(np.asarray(c["lead"])).astype(np.int64) + 1
    if auto is not None:
        row = np.where(auto, 3, row)
    np.add.at(t["late"], (row[on], c["out"][on]), w)
    np.add.at(t["late_secs"], row[on], c["secs"][on] * w)
    # by the time drawn (the play's own seconds) and, the sim's, where its distance draw came from
    on3 = on & (row < 3)
    td = (c["out"] == TD).astype(float)
    np.add.at(t["late_tb"], (row[on3], c["tb"][on3], 0), w)
    np.add.at(t["late_tb"], (row[on3], c["tb"][on3], 1), td[on3] * w)
    if "src" in c:
        np.add.at(t["late_src"], (row[on3], c["src"][on3], 0), w)
        np.add.at(t["late_src"], (row[on3], c["src"][on3], 1), td[on3] * w)


def real_drive_end(rows, i, final=None):
    """How the drive with the ball at row i ended (v21.DRIVE_ENDS index): the next score, kick, punt,
    change of side or the half's end after it. final: the game's (home, away) final, for a drive the
    rows end on in regulation -- the feed can miss the game's last play, not its final score."""
    _i = v21.s14._i
    a = rows[i]
    off, period = a.get("offense"), _i(a.get("period"))
    a_home = (a.get("team_a_side") or "home") != "away"
    mine_home = (off == "TEAM_A") == a_home
    score = lambda r: ((_i(r.get("score_p1")) or 0), (_i(r.get("score_p2")) or 0))
    h0, w0 = score(a)
    for j in range(i + 1, len(rows)):
        r = rows[j]
        h, w = score(r)
        d_me, d_them = (h - h0, w - w0) if mine_home else (w - w0, h - h0)
        if d_me >= 6:
            return 0
        if d_me == 3:
            return 1
        if d_them > 0:
            return 4                          # a defensive touchdown or a safety
        kind = r.get("play_kind")
        if kind == "FIELD_GOAL":
            nxt = rows[j + 1] if j + 1 < len(rows) else r
            h1, w1 = score(nxt)
            return 1 if ((h1 - h0) if mine_home else (w1 - w0)) == 3 else 2
        if kind == "PUNT":
            return 3
        if kind == "TURNOVER_ON_DOWNS":
            return 5
        if _i(r.get("period")) != period:
            if period in (1, 3):              # the quarter's end: the drive goes on
                period = _i(r.get("period"))
                continue
            return 6
        if r.get("offense") in ("TEAM_A", "TEAM_B") and r.get("offense") != off and kind in v21.s14.SNAP_KINDS:
            return 5 if "TURNOVER_ON_DOWNS" in (rows[j - 1].get("play_messages") or "") else 4
    if final is not None and period == 4 and not any((_i(r.get("period")) or 0) >= 5 for r in rows):
        pts = v21.end_points(dict(a, score_p1=rows[-1].get("score_p1"), score_p2=rows[-1].get("score_p2")),
                             None, final)
        if pts is not None and pts[1] == 0 and pts[0] >= 6:
            return 0
        if pts == (3, 0):
            return 1
    return 6


def _late_cell(cell):
    """Conversion cells of Q4's last 3:00 (not overtime's)."""
    cell = np.asarray(cell)
    return (cell >= 3) & (cell <= v21.CONV_CELLS - 2)


def _good_part(cell):
    """A conversion cell's part of the game (GOOD_PARTS): the first half, Q3, Q4 to 3:00, its last 3:00,
    overtime; CONV_PARTS are the first three."""
    cell = np.asarray(cell)
    return np.where(cell == v21.CONV_CELLS - 1, 4, np.where(cell >= 3, 3, cell)).astype(np.int64)


def _short(t, c, w):
    """2nd-4th downs by the time the play took (SHORT_ROWS: by the distance over the game, then the
    halves' last 2:00 at every distance): the snaps, those that moved the chains (a new set of downs or a
    touchdown) and those that went nowhere."""
    down, dist, out = np.asarray(c["down"]), np.asarray(c["dist"]), np.asarray(c["out"])
    p, clk, tb = np.asarray(c["period"]), np.asarray(c["clock"], float), np.asarray(c["tb"])
    later = down >= 2
    rows = (later & (dist <= 1), later & (dist >= 2) & (dist <= 3), later & (dist >= 4),
            later & ((p == 2) | (p == 4)) & (clk <= v21.LATE))
    moved = ((out == FIRST) | (out == TD)).astype(float)
    nowhere = (out == 2).astype(float)
    for k, on in enumerate(rows):
        np.add.at(t["short_tb"], (k, tb[on], 0), w)
        np.add.at(t["short_tb"], (k, tb[on], 1), moved[on] * w)
        np.add.at(t["short_tb"], (k, tb[on], 2), nowhere[on] * w)


def _side_lead(s):
    side = int(s["team"][0])
    pts = (int(s["home"][0]), int(s["away"][0]))
    return side, pts, pts[side] - pts[1 - side]


def _regulation(rows, final=None):
    """The real (home, away) score when regulation ended: the final (home's, away's) where the game did
    not go to overtime and the history has it, else the rows' (the scores only rise)."""
    if final is not None and not any((v21.s14._i(r.get("period")) or 0) >= 5 for r in rows):
        return int(final[0]), int(final[1])
    h = a = 0
    for r in rows:
        p = v21.s14._i(r.get("period"))
        if p is None or p > 4:
            continue
        h = max(h, v21.s14._i(r.get("score_p1")) or 0)
        a = max(a, v21.s14._i(r.get("score_p2")) or 0)
    return h, a


def _worker(job):
    """Audit a list of matches: (real, the sim's game flow, the sim's next snaps)."""
    model_path, items, n_paths, state_paths, every = job
    model = v21.Model.load(model_path)
    real, flow, step = tables(), tables(), tables()
    by = getattr(model, "two_good_by", None)
    flow["two_sim"] += np.r_[by if by is not None else np.full(3, model.two_good), 1.0]
    for code, rows, pair, teams, final in items:
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
        # conversions in Q4's last 3:00 by the lead after the six: the sim's games against the real one's
        for cell, lead6, k, base in st.get("_two") or []:
            on = _late_cell(cell)
            np.add.at(flow["two_late"], (lead6[on] + v21.CONV_MARGIN, k[on]), 1.0)
            np.add.at(flow["two_base"], lead6[on] + v21.CONV_MARGIN, base[on])
            # and before Q4's last 3:00: a side two scores behind kicks or goes for two after its first six
            pt = _good_part(cell)
            on = pt < len(CONV_PARTS)
            np.add.at(flow["two_part"], (pt[on], lead6[on] + v21.CONV_MARGIN, k[on]), 1.0)
            np.add.at(flow["two_part_base"], (pt[on], lead6[on] + v21.CONV_MARGIN), base[on])
        for q in v21.decision_points(rows, recs)["two"]:
            lead6 = int(np.clip(q["lead"], -v21.CONV_MARGIN, v21.CONV_MARGIN)) + v21.CONV_MARGIN
            if _late_cell(q["cell"]):
                real["two_late"][lead6, int(q["choice"])] += 1
            pt = int(_good_part(q["cell"]))
            if pt < len(CONV_PARTS):
                real["two_part"][pt, lead6, int(q["choice"])] += 1
            if q["choice"]:
                real["two_good"][pt] += (1.0, float(q.get("good", 0)))
        # the next snap from every real state, against what came of it
        starts = v21.match_starts(model, rows, pair, teams, every=every)
        if not starts or not state_paths:
            continue
        by_start = {r["start"]: r for r in recs}
        st_rows = [rows[i] for i, _ in starts]
        sl_start = slice_of([s["period"][0] for _, s in starts], [s["clock"][0] for _, s in starts])
        # Q4's last 2:00 by what a field goal does for the side with the ball (v21.fg4_of; -1: elsewhere)
        lead_q4 = np.array([int(v21.fg4_of(_side_lead(s)[2])) if int(s["period"][0]) == 4
                            and float(s["clock"][0]) <= v21.LATE else -1 for _, s in starts])
        down_start = np.array([s["down"][0] for _, s in starts])
        mine = [by_start.get(v21.s14._i(r["message"])) for r in st_rows]
        pairs = [(r, s) for (_, s), r in zip(starts, mine) if r is not None]
        got = [r for r, _ in pairs]
        if got:
            c = _real_cols(got)
            _put(real, "step", c["down"], c["sl"], c["dist"], c["out"], c["gain"], c["secs"], c["kind"], 1.0)
            # the states where the sim ends the game at once (the side ahead runs it out)
            auto = np.array([bool(v21.auto_end(s["period"][0], s["clock"][0], _side_lead(s)[2], s["down"][0],
                                               s["tos"][0][1 - _side_lead(s)[0]])) for _, s in pairs])
            _late(real, c, 1.0, auto)
            _short(real, c, 1.0)
            # by the snaps left the model expects at each real state (its own table, the state's own key)
            rb = v21.remain_b(model.remain_at(*v21.remain_inputs(got)))
            np.add.at(real["by_rem"], (np.clip(c["down"] - 1, 0, D - 1), rb, c["out"]), 1)
        # the real choices at the same states: 4th downs, and kicks on downs 1-3 in the half's last minute
        for j, (i, s) in enumerate(starts):
            b = rows[i + 1] if i + 1 < len(rows) else None
            p, clk, d = int(s["period"][0]), float(s["clock"][0]), int(s["down"][0])
            if b is None or b.get("period") != rows[i].get("period"):
                # the half's end: the score after it says whether the side kicked (v21.decision_points)
                if p not in (2, 4) or (b is None and (p != 4 or final is None)) \
                        or float(v21.half_left(p, clk)) > v21.EARLY_LEFT:
                    continue
                pts = v21.end_points(rows[i], b, final if b is None else None)
                b = dict(play_kind="FIELD_GOAL" if pts == (3, 0) else "", period=rows[i].get("period"))
                if d == 4 and b["play_kind"] != "FIELD_GOAL":
                    continue
            if d == 4:
                k = v21.PUNT if b["play_kind"] == "PUNT" else v21.FG if b["play_kind"] == "FIELD_GOAL" else v21.GO
                real["fourth"][sl_start[j], k] += 1
                if lead_q4[j] >= 0:
                    real["fourth_q4"][lead_q4[j], k] += 1
            elif (p == 2 or p >= 4) and float(v21.half_left(p, clk)) <= v21.EARLY_LEFT \
                    and 100 - int(s["y"][0]) + 17 <= v21.FG_MAX:
                real["early"][sl_start[j], int(b["play_kind"] == "FIELD_GOAL")] += 1
                if lead_q4[j] >= 0:
                    real["early_q4"][lead_q4[j], int(b["play_kind"] == "FIELD_GOAL")] += 1
        S_ = v21.concat_starts([s for _, s in starts])
        st = {}
        v21.simulate(model, S_, state_paths, seed=seed + 1, stats=st, max_steps=1)
        log = v21.snap_log(st)
        if len(log["path"]):
            c = _sim_cols(log)
            _put(step, "step", c["down"], c["sl"], c["dist"], c["out"], c["gain"], c["secs"], c["kind"], 1.0 / state_paths)
            np.add.at(step["by_rem"], (np.clip(c["down"] - 1, 0, D - 1), c["rem"], c["out"]), 1.0 / state_paths)
            _late(step, c, 1.0 / state_paths)
            _short(step, c, 1.0 / state_paths)
        for (name, who, k, sl, ix), base in zip(st.get("_dec") or [], st.get("_dec_base") or []):
            if ix is None or name not in ("fourth", "early_fg"):
                continue
            first, lq = sl_start[ix // state_paths], lead_q4[ix // state_paths]
            on = lq >= 0
            if name == "fourth":
                np.add.at(step["fourth"], (first, k), 1.0 / state_paths)
                np.add.at(step["fourth_q4"], (lq[on], k[on]), 1.0 / state_paths)
                np.add.at(step["fourth_q4_base"], lq[on], base[on] / state_paths)
            else:
                np.add.at(step["early"], (first, k), 1.0 / state_paths)
                np.add.at(step["early_q4"], (lq[on], k[on]), 1.0 / state_paths)
                np.add.at(step["early_q4_base"], lq[on], base[on] / state_paths)
        # overtime: from every real Q4 state to the end, against whether the real game went to overtime
        q4 = [(j, s) for j, (i, s) in enumerate(starts) if int(s["period"][0]) == 4]
        if q4:
            went_real = float(any((v21.s14._i(r.get("period")) or 0) >= 5 for r in rows))
            st = {}
            v21.simulate(model, v21.concat_starts([s for _, s in q4]), state_paths, seed=seed + 2, stats=st)
            went = st["went_ot"].reshape(len(q4), state_paths).mean(1)
            reg_sim = st["regulation"].reshape(len(q4), state_paths, 2)
            ends = st["drive_end"].reshape(len(q4), state_paths)
            reg_real = _regulation(rows, final)
            for k, ((j, s), w) in enumerate(zip(q4, went)):
                if int(s["phase"][0]) == v21.SNAP:   # how the drive with the ball ended
                    c_ = float(s["clock"][0])
                    cs_ = 0 if c_ > 120 else 1 if c_ > 60 else 2 if c_ > 30 else 3
                    lb_ = int(v21.margin_b(_side_lead(s)[2]))
                    real["drives"][lb_, cs_, real_drive_end(rows, starts[j][0], final)] += 1
                    e = ends[k][ends[k] >= 0]
                    if len(e):
                        step["drives"][lb_, cs_] += np.bincount(e, minlength=len(v21.DRIVE_ENDS)) / len(e)
                side, pts, lead = _side_lead(s)
                c = float(s["clock"][0])
                cs = 0 if c > 120 else 1 if c > 60 else 2 if c > 30 else 3
                lb = int(v21.margin_b(lead))
                real["ot"][lb, cs] += went_real
                real["ot_n"][lb, cs] += 1
                step["ot"][lb, cs] += w
                step["ot_n"][lb, cs] += 1
                # each side's points still to come in regulation, and from level who won it
                mine_s, theirs_s = reg_sim[k, :, side], reg_sim[k, :, 1 - side]
                real["pts_ball"][lb, cs] += reg_real[side] - pts[side]
                real["pts_other"][lb, cs] += reg_real[1 - side] - pts[1 - side]
                step["pts_ball"][lb, cs] += (mine_s - pts[side]).mean()
                step["pts_other"][lb, cs] += (theirs_s - pts[1 - side]).mean()
                if lead == 0:
                    d_real = reg_real[side] - reg_real[1 - side]
                    real["level"][cs, 0 if d_real > 0 else 1 if d_real == 0 else 2] += 1
                    d_sim = mine_s - theirs_s
                    step["level"][cs] += [(d_sim > 0).mean(), (d_sim == 0).mean(), (d_sim < 0).mean()]
    return real, flow, step


def run(model_dir, matches, history, n_paths=500, state_paths=50, every=1, workers=1, since=None, until=None):
    import os
    model_path = os.path.join(model_dir, v21.MODEL_FILE) if os.path.isdir(model_dir) else model_dir
    hist = {r["MATCH_CODE"]: r for r in (history or [])}
    keys = ("match_code", "message", "period", "clock_seconds", "play_kind", "offense", "down", "distance",
            "field_position", "score_p1", "score_p2", "play_messages", "team_a_side", "opening_offense",
            "timeouts_used_a", "timeouts_used_b", "timeout_after", "timeout_prev", "home_handle", "away_handle")
    from .v21_report import _finals
    items = [(c, [{k: r.get(k) for k in keys} for r in v21._with_side(rows)]) + v21.match_people(c, rows, hist)
             + (_finals(hist[c]) if c in hist else None,) for c, rows in sorted(matches.items())]
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
    lines += ["", "  4th DOWN in Q4's last 2:00, by what a field goal does for the side with the ball (real / sim /"
                  " the league's share in the sim's cells, before the gamers' leans and the rules, %)",
              f"  {'':20s}{'n real':>7s}" + "".join(f"{o:>20s}" for o in FOURTH)]
    for k, name in enumerate(v21.FG4_NAMES):
        r, m, b = real["fourth_q4"][k], step["fourth_q4"][k], step["fourth_q4_base"][k]
        if r.sum():
            cells = [f"{_pair(_pct(r[o], r.sum()), _pct(m[o], m.sum()))}/{_pct(b[o], m.sum()):5.1f}" for o in range(3)]
            lines.append(f"  {name:20s}{int(r.sum()):>7d}" + "".join(f"{c:>20s}" for c in cells))
    lines += ["", "  A KICK on downs 1-3 in range, the half's last minute, from the real states (real / sim, %)",
              f"  {'':16s}{'n real':>7s}{'kick':>14s}"]
    for s_ in range(S):
        r, m = real["early"][s_], step["early"][s_]
        if r.sum():
            lines.append(f"  {SLICES[s_]:16s}{int(r.sum()):>7d}{_pair(_pct(r[1], r.sum()), _pct(m[1], m.sum())):>14s}")
    for k, name in enumerate(v21.FG4_NAMES):
        r, m, b = real["early_q4"][k], step["early_q4"][k], step["early_q4_base"][k]
        if r.sum():
            lines.append(f"  {'Q4, ' + name:28s}{int(r.sum()):>7d}{_pair(_pct(r[1], r.sum()), _pct(m[1], m.sum())):>14s}"
                         f"   (the league's share in the sim's cells {_pct(b[1], m.sum()):.1f})")
    return lines


def two_lines(real, flow):
    lines = ["", "  CONVERSIONS in Q4's last 3:00, by the lead after the six: go for two (real / the sim's own games /"
                 " the league's share in the sim's cells, %; -1: a kick ties, two wins; -2: two ties)",
             f"  {'lead':>6s}{'n real':>8s}{'n sim':>10s}{'go for two':>20s}"]
    for i in range(real["two_late"].shape[0]):
        r, m = real["two_late"][i], flow["two_late"][i]
        if r.sum() >= 3:
            cell = f"{_pair(_pct(r[1], r.sum()), _pct(m[1], m.sum()))}/{_pct(flow['two_base'][i], m.sum()):5.1f}"
            lines.append(f"  {i - v21.CONV_MARGIN:>+6d}{int(r.sum()):>8d}{int(m.sum()):>10d}{cell:>20s}")
    return lines


def two_part_lines(real, flow):
    lines = ["", "  CONVERSIONS before Q4's last 3:00, by the lead after the six: go for two (real / the sim's own games /"
                 " the league's share in the sim's cells, %; n real) -- 8 behind after the six: 14 behind, the first"
                 " of two touchdowns back",
             f"  {'lead':>6s}" + "".join(f"{name:>26s}" for name in CONV_PARTS)]
    for i in range(real["two_part"].shape[1]):
        r, m, b = real["two_part"][:, i], flow["two_part"][:, i], flow["two_part_base"][:, i]
        if r.sum() < 5 or i - v21.CONV_MARGIN > 3:
            continue
        cells = [f"{_pair(_pct(r[k, 1], r[k].sum()), _pct(m[k, 1], m[k].sum()))}/{_pct(b[k], m[k].sum()):5.1f}"
                 f" ({int(r[k].sum())})" if r[k].sum() >= 3 else "-" for k in range(len(CONV_PARTS))]
        lines.append(f"  {i - v21.CONV_MARGIN:>+6d}" + "".join(f"{c:>26s}" for c in cells))
    g, sim = real["two_good"], flow["two_sim"]
    lines.append("  two-point tries made (real, by the part of the game; the sim's before Q4's last 3:00 "
                 f"{_pct(sim[0], sim[3]):.1f}%, in it {_pct(sim[1], sim[3]):.1f}%, overtime {_pct(sim[2], sim[3]):.1f}%): "
                 + ", ".join(f"{name} {_pct(g[k, 1], g[k, 0]):.1f}% of {int(g[k, 0])}" for k, name in enumerate(GOOD_PARTS)
                             if g[k, 0]))
    return lines


def short_lines(real, step):
    lines = ["", "  ONE SNAP on 2nd-4th downs by the time its play took: the share of the snaps in each, then the share"
                 " that moved the chains (a new set of downs or a touchdown), real / sim, %; moved: all of them, real /"
                 " sim / the sim's own rates on the real snaps' times (near the real: the time draw is off; near the"
                 " sim: the draw given the time)",
             f"  {'':20s}" + "".join(f"{b:>20s}" for b in v21.TIME_BUCKETS) + f"{'moved':>20s}{'nowhere':>14s}"]
    for k, name in enumerate(SHORT_ROWS):
        r, m = real["short_tb"][k], step["short_tb"][k]
        if not r[:, 0].sum() or not m[:, 0].sum():
            continue
        f = lambda v, fmt: "-" if v != v else fmt.format(v)
        cells = [f"{f(_pct(r[b, 0], r[:, 0].sum()), '{:.0f}')}/{f(_pct(m[b, 0], m[:, 0].sum()), '{:.0f}')}% "
                 f"{f(_pct(r[b, 1], r[b, 0]), '{:.0f}')}/{f(_pct(m[b, 1], m[b, 0]), '{:.0f}')}" for b in range(r.shape[0])]
        has = m[:, 0] > 0
        mix = float((r[has, 0] * m[has, 1] / m[has, 0]).sum() / max(r[has, 0].sum(), 1e-9))
        moved = f"{_pct(r[:, 1].sum(), r[:, 0].sum()):.1f}/{_pct(m[:, 1].sum(), m[:, 0].sum()):.1f}/{100 * mix:.1f}"
        nowhere = _pair(_pct(r[:, 2].sum(), r[:, 0].sum()), _pct(m[:, 2].sum(), m[:, 0].sum()))
        lines.append(f"  {name:20s}" + "".join(f"{c:>20s}" for c in cells) + f"{moved:>20s}{nowhere:>14s}")
    return lines


def late_tb_lines(real, step):
    lines = ["", "  Q4's last 2:00, one snap by the time its play took: the share of the snaps in each, then its"
                 " touchdown % (real / sim)",
             f"  {'':10s}" + "".join(f"{b:>22s}" for b in v21.TIME_BUCKETS)]
    for k, name in enumerate(LEADS3):
        r, m = real["late_tb"][k], step["late_tb"][k]
        if not r[:, 0].sum():
            continue
        f = lambda v, fmt: "-" if v != v else fmt.format(v)
        cells = [f"{f(_pct(r[b, 0], r[:, 0].sum()), '{:.0f}')}/{f(_pct(m[b, 0], m[:, 0].sum()), '{:.0f}')}% "
                 f"{f(_pct(r[b, 1], r[b, 0]), '{:.1f}')}/{f(_pct(m[b, 1], m[b, 0]), '{:.1f}')}" for b in range(r.shape[0])]
        lines.append(f"  {name:10s}" + "".join(f"{c:>22s}" for c in cells))
    lines += ["  the sim's distance draws by where they came from: share of the snaps and touchdown %"]
    for k, name in enumerate(LEADS3):
        m = step["late_src"][k]
        if m[:, 0].sum():
            lines.append(f"  {name:10s}" + ", ".join(
                f"{src} {_pct(m[j, 0], m[:, 0].sum()):.0f}% (touchdown {_pct(m[j, 1], m[j, 0]):.1f})"
                for j, src in enumerate(("the league's", "a team's own", "a gamer's own")) if m[j, 0]))
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


def late_lines(real, step):
    lines = ["", "  ONE SNAP in Q4's last 2:00, by the lead of the side with the ball (real / sim, % of the snaps)",
             f"  {'':24s}{'n real':>7s}" + "".join(f"{o:>13s}" for o in OUTCOMES) + f"{'seconds':>13s}"]
    names = LEADS3 + ("ahead, the game's own end",)
    for k, name in enumerate(names):
        r, m = real["late"][k], step["late"][k]
        if not r.sum():
            continue
        sr = real["late_secs"][k] / r.sum()
        ss = step["late_secs"][k] / m.sum() if m.sum() else float("nan")
        lines.append(f"  {name:24s}{int(r.sum()):>7d}" + "".join(
            f"{_pair(_pct(r[o], r.sum()), _pct(m[o], m.sum())):>13s}" for o in range(O)) + f"{_pair(sr, ss):>13s}")
    lines.append("  (the game's own end: the side ahead with the ball, its downs at the 40-second play clock outlasting"
                 " the clock, the defense out of timeouts -- the sim ends the game there, no snap)")
    return lines


def q4_lines(real, step):
    lines = []
    for key, label in (("pts_ball", "the side with the ball"), ("pts_other", "the other side")):
        lines += ["", f"  Q4 STATES: {label}'s points still to come in regulation, by the lead of the side with the"
                      f" ball and the time left (real / sim)", f"  {'':14s}" + "".join(f"{c:>20s}" for c in OT_CLOCK)]
        for lb, name in enumerate(OT_LEADS):
            n = real["ot_n"][lb]
            if not n.sum():
                continue
            cells = [f"{real[key][lb, cs] / n[cs]:5.2f}/{step[key][lb, cs] / step['ot_n'][lb, cs]:5.2f}" if n[cs] else "-"
                     for cs in range(4)]
            lines.append(f"  {name:14s}" + "".join(f"{c:>20s}" for c in cells))
    lines += ["", "  Q4 STATES, level: how regulation ended for the side with the ball (real / sim, %)",
              f"  {'':16s}{'n real':>7s}" + "".join(f"{o:>14s}" for o in RESULTS)]
    for cs in range(4):
        r, m = real["level"][cs], step["level"][cs]
        if r.sum():
            lines.append(f"  {OT_CLOCK[cs]:16s}{int(r.sum()):>7d}"
                         + "".join(f"{_pair(_pct(r[k], r.sum()), _pct(m[k], m.sum())):>14s}" for k in range(3)))
    lines.append("  (real states of one game share its result: the real figures move in steps of whole games)")
    return lines


def drive_lines(real, step):
    lines = ["", "  Q4 STATES: how the drive with the ball ended, by its lead and the time left (real / sim, % of"
                 " the states; clock out: the half or the game ran out on it, or it ran out the clock)",
             f"  {'':28s}{'n real':>7s}" + "".join(f"{e:>14s}" for e in v21.DRIVE_ENDS)]
    for lb, name in enumerate(DRIVE_LEADS):
        for cs in range(4):
            r, m = real["drives"][lb, cs], step["drives"][lb, cs]
            if r.sum() < 20:
                continue
            lines.append(f"  {name + ', ' + OT_CLOCK[cs]:28s}{int(r.sum()):>7d}" + "".join(
                f"{_pair(_pct(r[e], r.sum()), _pct(m[e], m.sum())):>14s}" for e in range(len(v21.DRIVE_ENDS))))
    return lines


def report(out):
    lines = [f"  v2.1 audit on {out['matches']:,} matches {out['since'] or ''} to {out['until'] or ''}:"
             f" {2 * out['n_paths']:,} kick-off games each; the next snap {out['state_paths']} times from every"
             f" {'' if out['every'] == 1 else str(out['every']) + 'th '}real state", ""]
    lines += gap_lines(out["real"], out["flow"], out["step"])
    lines += [""] + flow_lines(out["real"], out["flow"])
    lines += step_lines(out["real"], out["step"])
    lines += rem_lines(out["real"], out["step"])
    lines += short_lines(out["real"], out["step"])
    lines += late_lines(out["real"], out["step"])
    lines += late_tb_lines(out["real"], out["step"])
    lines += choice_lines(out["real"], out["step"])
    lines += two_lines(out["real"], out["flow"])
    lines += two_part_lines(out["real"], out["flow"])
    lines += ot_lines(out["real"], out["flow"], out["step"])
    lines += q4_lines(out["real"], out["step"])
    lines += drive_lines(out["real"], out["step"])
    lines += ["", "  (snaps: a punt or a field goal is not one, a 4th down that went for it is; yards: the gain of the"
                  " snaps that kept the ball, cut at the goal line; seconds: PLAY_OVER to PLAY_OVER)"]
    return lines
