"""Moneyline money before and after kick-off: does the side backed pre-match keep being backed in
play, and where does the book lose -- the pre-match price, or something seen during the game?

    python analysis/prematch_inplay.py
    python analysis/prematch_inplay.py --bets eAMFCalibrator/out/bets_sim_snapshot.csv

Per match: the side with the most pre-match moneyline stake is "the pre-match side" (when it holds at
least --lean of the pre-match stake). Every in-play moneyline bet is then "with" it (the same side) or
"against" it. Prints:
  - pre-match: the book's margin, and how the pre-match sides did (won vs prod's implied chance)
  - in play, with and against the pre-match side: stake, margin, and the same by quarter
  - matches by what the pre-match side's backers' money did in play, and the result
  - with the snapshot export (--snapshots), the game state at each in-play bet: whether the
    pre-match side was ahead, level or behind when the money came in, and where it stood after Q1
  - a check that the feed's "Home" moneyline is PLAYER_1 (the side listed first): settled Home bets
    that won, against the final scores
Writes prematch_inplay_matches.csv next to the bets file: one row a match with the matchup, start,
final, pre-match side, its stake and chance, the in-play stake with and against it by quarter, and
the book's revenue on each.
"""

import argparse
import csv
import datetime as dt
import os
from bisect import bisect_right
from collections import defaultdict


def _time(v):
    try:
        return dt.datetime.fromisoformat(str(v)[:19].replace("T", " "))
    except ValueError:
        return None


def load_scores(path, codes):
    """{match: ([times], [(period, clock, player 1, player 2)])} off the snapshot export."""
    by = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["match_code"] not in codes:
                continue
            t = _time(r.get("file_time") or r.get("publish_time"))
            if t is None or r.get("score_p1") in ("", None):
                continue
            by[r["match_code"]].append((t, r.get("period") or "", r.get("clock_seconds") or "",
                                        int(float(r["score_p1"])), int(float(r["score_p2"]))))
    out = {}
    for c, rows in by.items():
        rows.sort()
        out[c] = ([x[0] for x in rows], [x[1:] for x in rows])
    return out


def score_at(scores, code, t):
    got = scores.get(code)
    if not got or t is None:
        return None
    i = bisect_right(got[0], t) - 1
    return got[1][i] if i >= 0 else None


def lead_of(side, p1, p2):
    lead = (p1 - p2) if str(side).lower() in ("home", "player 1") else (p2 - p1)
    return "ahead" if lead > 0 else "level" if lead == 0 else "behind"


def after_q1(scores, code):
    got = scores.get(code)
    if not got:
        return None
    q1 = [x for x in got[1] if str(x[0]) == "1"]
    return q1[-1] if q1 else None


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _true(v):
    return str(v).strip().lower() in ("true", "1", "yes")


def _margin(rev, stake):
    return f"{100 * rev / stake:+6.2f}%" if stake else "     -"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bets", default=os.path.join("eAMFCalibrator", "out", "bets_sim.csv"))
    ap.add_argument("--history", default=os.path.join("eAMFCalibrator", "out", "match_history.csv"))
    ap.add_argument("--lean", type=float, default=0.6, help="share of pre-match stake that makes a side 'the' side")
    ap.add_argument("--top", type=int, default=25, help="matches listed where the book lost most in play")
    ap.add_argument("--snapshots", default=os.path.join("eAMFCalibrator", "out", "scouting_playover.csv"),
                    help="the snapshot export, for the score at each bet ('' to skip)")
    args = ap.parse_args()

    games = {}
    with open(args.history, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            games[r["MATCH_CODE"]] = r

    pre = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0.0, 0]))      # match -> side -> stake, rev, prob*stake, won
    live = defaultdict(list)
    with open(args.bets, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("market") != "moneyline" or _true(r.get("excluded")):
                continue
            stake, rev = _num(r.get("stake")), _num(r.get("revenue"))
            if _true(r.get("in_play")):
                live[r["match_code"]].append(r)
            else:
                c = pre[r["match_code"]][r.get("selection")]
                c[0] += stake
                c[1] += rev
                c[2] += stake * _num(r.get("stream_prob") or r.get("implied_prob"))
                c[3] = 1 if r.get("result") == "won" else 0

    # the feed's Home moneyline against the final score: settled Home bets that won
    home_won = [0, 0]
    for code, bets in live.items():
        for r in bets:
            if r.get("selection") == "Home" and r.get("result") == "won" and r.get("final_p1") not in ("", None):
                home_won[0] += 1
                home_won[1] += _num(r["final_p1"]) > _num(r["final_p2"])
    scores = {}
    if args.snapshots and os.path.exists(args.snapshots):
        print(f"  reading the scores off {args.snapshots}", flush=True)
        scores = load_scores(args.snapshots, set(live))
    state = defaultdict(lambda: [0.0, 0.0])            # (way, ahead/level/behind) -> stake, revenue
    q1_state = defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0, 0])   # state after Q1 -> matches, with, with rev, against, against rev, won

    tot = defaultdict(lambda: [0.0, 0.0])
    by_q = defaultdict(lambda: [0.0, 0.0])
    sides = []                       # (implied chance, won) for each pre-match side
    rows = []
    for code in set(pre) | set(live):
        g = games.get(code, {})
        p = pre.get(code, {})
        pre_stake = sum(c[0] for c in p.values())
        pre_rev = sum(c[1] for c in p.values())
        tot["pre-match"][0] += pre_stake
        tot["pre-match"][1] += pre_rev
        side = None
        if p:
            s, c = max(p.items(), key=lambda kv: kv[1][0])
            if pre_stake and c[0] / pre_stake >= args.lean:
                side = s
                sides.append((c[2] / c[0] if c[0] else 0.0, c[3]))
        row = {"match_code": code,
               "matchup": f"{g.get('PLAYER_1_HANDLE', '?')} ({g.get('PLAYER_1_TEAM', '')}) [home] v "
                          f"{g.get('PLAYER_2_HANDLE', '?')} ({g.get('PLAYER_2_TEAM', '')}) [away]",
               "event_start_utc": str(g.get("SCHEDULED_START_TIME_UTC", ""))[:16],
               "final": f"{g.get('PLAYER_1_FINAL_SCORE', '')}-{g.get('PLAYER_2_FINAL_SCORE', '')}",
               "pre_side": side or "", "pre_stake": round(pre_stake, 2), "pre_revenue": round(pre_rev, 2),
               "pre_side_share": round(p[side][0] / pre_stake, 3) if side else "",
               "pre_side_chance": round(p[side][2] / p[side][0], 3) if side and p[side][0] else "",
               "pre_side_won": p[side][3] if side else ""}
        for way in ("with", "against", "no side"):
            row[f"live_{way}_stake"] = row[f"live_{way}_revenue"] = 0.0
        for q in ("1", "2", "3", "4", "OT"):
            row[f"q{q}_with_stake"] = row[f"q{q}_against_stake"] = 0.0
        for r in live.get(code, []):
            stake, rev = _num(r.get("stake")), _num(r.get("revenue"))
            way = "no side" if side is None else ("with" if r.get("selection") == side else "against")
            row[f"live_{way}_stake"] += stake
            row[f"live_{way}_revenue"] += rev
            tot[f"in play, {way} the pre-match side" if side else "in play, no pre-match side"][0] += stake
            tot[f"in play, {way} the pre-match side" if side else "in play, no pre-match side"][1] += rev
            q = str(r.get("period") or "")
            q = q if q in ("1", "2", "3", "4") else "OT"
            sc = score_at(scores, code, _time(r.get("bet_time"))) if side else None
            if sc is not None:
                st = state[(way, lead_of(side, sc[2], sc[3]))]
                st[0] += stake
                st[1] += rev
            if side:
                row[f"q{q}_{way}_stake"] += stake
                by_q[(q, way)][0] += stake
                by_q[(q, way)][1] += rev
        q1 = after_q1(scores, code) if side else None
        row["pre_side_after_q1"] = lead_of(side, q1[2], q1[3]) if q1 else ""
        if q1:
            g1 = q1_state[row["pre_side_after_q1"]]
            g1[0] += 1
            g1[1] += row["live_with_stake"]
            g1[2] += row["live_with_revenue"]
            g1[3] += row["live_against_stake"]
            g1[4] += row["live_against_revenue"]
            g1[5] += p[side][3]
        live_rev = row["live_with_revenue"] + row["live_against_revenue"] + row["live_no side_revenue"]
        row["live_revenue"] = round(live_rev, 2)
        w, a = row["live_with_stake"], row["live_against_stake"]
        row["live_follows"] = ("" if not side or not (w + a) else
                               "follows" if w >= 0.6 * (w + a) else "switches" if a >= 0.6 * (w + a) else "both")
        rows.append({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()})

    print("\n  MONEYLINE: PRE-MATCH AGAINST IN PLAY (book's revenue; negative = the book lost)")
    print(f"  {'':42s}{'stake':>12s}{'revenue':>12s}{'margin':>9s}")
    for k in ("pre-match", "in play, with the pre-match side", "in play, against the pre-match side",
              "in play, no pre-match side"):
        s, r = tot[k]
        print(f"  {k:42s}{s:>12,.0f}{r:>12,.0f}{_margin(r, s):>9s}")

    if sides:
        n = len(sides)
        exp = sum(c for c, _ in sides) / n
        won = sum(w for _, w in sides) / n
        print(f"\n  pre-match sides ({n:,} matches with {100 * args.lean:.0f}%+ of the pre-match stake on one side):"
              f" priced to win {100 * exp:.1f}%, won {100 * won:.1f}%"
              f" ({'the pre-match money found an edge' if won > exp + 0.02 else 'no clear pre-match edge'})")

    print(f"\n  in play by quarter, with / against the pre-match side: stake, margin")
    for q in ("1", "2", "3", "4", "OT"):
        cells = []
        for way in ("with", "against"):
            s, r = by_q[(q, way)]
            cells.append(f"{way} {s:>10,.0f} {_margin(r, s)}")
        print(f"    Q{q:3s}  " + "   ".join(cells))

    if state:
        print("\n  in play by the score when the bet came in, the pre-match side ahead / level / behind:")
        for way in ("with", "against"):
            cells = []
            for st in ("ahead", "level", "behind"):
                sk, rv = state[(way, st)]
                cells.append(f"{st} {sk:>10,.0f} {_margin(rv, sk)}")
            print(f"    {way:8s} " + "   ".join(cells))
    if q1_state:
        print("\n  matches by where the pre-match side stood after Q1: in-play stake and margin with and against it")
        for st in ("ahead", "level", "behind"):
            n, ws, wr, a_s, ar, won = q1_state[st]
            if n:
                print(f"    {st:7s} {n:>4,} matches   with {ws:>10,.0f} {_margin(wr, ws)}   against {a_s:>10,.0f}"
                      f" {_margin(ar, a_s)}   pre-match side won {100 * won / n:.0f}%")
    elif args.snapshots:
        print(f"\n  (no scores read: {args.snapshots} not found or no match in it)")
    if home_won[0]:
        print(f"\n  check: of {home_won[0]:,} settled in-play Home moneyline bets that won, {100 * home_won[1] / home_won[0]:.1f}%"
              " had player 1 (listed first) ahead at the end" + (" -- Home is the side listed first" if home_won[1] > 0.9 * home_won[0]
              else " -- Home is NOT the side listed first: the matchup reads away v home"))

    groups = defaultdict(lambda: [0, 0.0, 0.0, 0])
    for r in rows:
        if r["live_follows"]:
            gg = groups[r["live_follows"]]
            gg[0] += 1
            gg[1] += r["live_with_stake"] + r["live_against_stake"]
            gg[2] += r["live_revenue"]
            gg[3] += r["pre_side_won"] or 0
    print("\n  matches by where the in-play money went (60%+ of it with or against the pre-match side):")
    for k in ("follows", "switches", "both"):
        n, s, rev, won = groups[k]
        if n:
            print(f"    {k:9s} {n:>5,} matches, in-play stake {s:>11,.0f}, margin {_margin(rev, s)},"
                  f" pre-match side won {100 * won / n:.0f}%")

    rows.sort(key=lambda r: r["live_revenue"])
    print(f"\n  the {args.top} matches where the book lost most in play")
    print(f"  {'lost':>9s} {'pre side':8s} {'chance':>6s} {'won':>3s} {'with':>9s} {'against':>9s}  {'in play':8s}"
          f" {'start (UTC)':16s} {'final':7s} matchup")
    for r in rows[:args.top]:
        print(f"  {-r['live_revenue']:>9,.0f} {str(r['pre_side'])[:8]:8s} {str(r['pre_side_chance']):>6s}"
              f" {str(r['pre_side_won']):>3s} {r['live_with_stake']:>9,.0f} {r['live_against_stake']:>9,.0f}"
              f"  {r['live_follows']:8s} {r['event_start_utc']:16s} {r['final']:7s} {r['matchup']}")

    path = os.path.join(os.path.dirname(args.bets) or ".", "prematch_inplay_matches.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  every match ({len(rows):,}) -> {path}")


if __name__ == "__main__":
    main()
