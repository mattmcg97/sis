"""The moneyline betting runs that cost the book most, with each match's matchup and start time.

A run is a match's moneyline bets on one selection with no more than --gap minutes between one
bet and the next (pre-match and in play kept apart). Ranked by the book's revenue on the run,
most lost first.

    python analysis/worst_ml_runs.py
    python analysis/worst_ml_runs.py --bets eAMFCalibrator/out/bets_sim.csv --top 50 --gap 3

Under each run its hot points: the --window-second stretches (default 30) where the book lost most,
each with its time, how far into the match, the quarter, bets, stake, loss, prod's probability and
the odds taken, and the game state where the bets file carries it. Writes worst_ml_runs.csv (every
run) and worst_ml_hotspots.csv (the hot points of the top runs) next to the bets file.
"""

import argparse
import csv
import datetime as dt
import os
from collections import defaultdict


def _time(v):
    try:
        return dt.datetime.fromisoformat(str(v)[:19].replace("T", " "))
    except ValueError:
        return None


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bets", default=os.path.join("eAMFCalibrator", "out", "bets_sim.csv"))
    ap.add_argument("--history", default=os.path.join("eAMFCalibrator", "out", "match_history.csv"))
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--gap", type=float, default=2.0, help="minutes between bets that still make one run")
    ap.add_argument("--window", type=float, default=30.0, help="seconds a hot point covers")
    ap.add_argument("--hot", type=int, default=3, help="hot points shown under each run")
    args = ap.parse_args()

    games = {}
    with open(args.history, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            games[r["MATCH_CODE"]] = r

    by = defaultdict(list)
    with open(args.bets, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("market") != "moneyline" or str(r.get("excluded", "")).lower() == "true":
                continue
            t = _time(r.get("bet_time"))
            if t is None:
                continue
            by[(r["match_code"], r.get("selection"), r.get("in_play"))].append((t, r))

    runs = []
    for (code, sel, in_play), bets in by.items():
        bets.sort(key=lambda x: x[0])
        cur = [bets[0]]
        for b in bets[1:]:
            if (b[0] - cur[-1][0]).total_seconds() / 60 <= args.gap:
                cur.append(b)
            else:
                runs.append((code, sel, in_play, cur))
                cur = [b]
        runs.append((code, sel, in_play, cur))

    out = []
    for code, sel, in_play, bets in runs:
        rows = [r for _, r in bets]
        stake = sum(_num(r.get("stake")) for r in rows)
        revenue = sum(_num(r.get("revenue")) for r in rows)
        g = games.get(code, {})
        start = _time(g.get("SCHEDULED_START_TIME_UTC"))
        first, last = bets[0][0], bets[-1][0]
        probs = [_num(r.get("stream_prob")) for r in rows if r.get("stream_prob")]
        out.append({
            "_bets": bets, "_start": start,
            "match_code": code,
            "matchup": f"{g.get('PLAYER_1_HANDLE', '?')} ({g.get('PLAYER_1_TEAM', '')}) v "
                       f"{g.get('PLAYER_2_HANDLE', '?')} ({g.get('PLAYER_2_TEAM', '')})",
            "event_start_utc": start.strftime("%Y-%m-%d %H:%M") if start else "",
            "final": f"{g.get('PLAYER_1_FINAL_SCORE', '')}-{g.get('PLAYER_2_FINAL_SCORE', '')}",
            "selection": sel, "in_play": in_play,
            "run_start_utc": first.strftime("%H:%M:%S"), "run_end_utc": last.strftime("%H:%M:%S"),
            "minutes_into_match": round((first - start).total_seconds() / 60, 1) if start else "",
            "periods": "/".join(sorted({str(r.get("period") or "") for r in rows})),
            "bets": len(rows), "stake": round(stake, 2), "book_revenue": round(revenue, 2),
            "prod_prob_first": round(probs[0], 3) if probs else "",
            "prod_prob_last": round(probs[-1], 3) if probs else "",
            "result": rows[0].get("result", ""),
        })
    out.sort(key=lambda r: r["book_revenue"])
    top = out[:args.top]

    path = os.path.join(os.path.dirname(args.bets) or ".", "worst_ml_runs.csv")
    keys = [k for k in out[0] if not k.startswith("_")]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(out)
    hot_rows = []
    for rank, r in enumerate(top, 1):
        r["_hot"] = hot_points(r, args.window)[:args.hot]
        for h in r["_hot"]:
            hot_rows.append(dict(rank=rank, match_code=r["match_code"], matchup=r["matchup"],
                                 event_start_utc=r["event_start_utc"], **h))
    hot_path = os.path.join(os.path.dirname(args.bets) or ".", "worst_ml_hotspots.csv")
    if hot_rows:
        with open(hot_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(hot_rows[0]))
            w.writeheader()
            w.writerows(hot_rows)

    print(f"{'#':>3} {'lost':>10} {'stake':>10} {'bets':>5}  {'start (UTC)':16}  {'run':17} {'min in':>6} "
          f"{'per':4} {'sel':4} {'prob':>11} {'final':7} matchup")
    for i, r in enumerate(top, 1):
        print(f"{i:>3} {-r['book_revenue']:>10,.0f} {r['stake']:>10,.0f} {r['bets']:>5}  {r['event_start_utc']:16}  "
              f"{r['run_start_utc']}-{r['run_end_utc']} {str(r['minutes_into_match']):>6} {r['periods']:4} "
              f"{str(r['selection'])[:4]:4} {str(r['prod_prob_first'])}>{str(r['prod_prob_last']):5} "
              f"{r['final']:7} {r['matchup']}")
        for h in r["_hot"]:
            state = "  ".join(f"{k}={h[k]}" for k in STATE if h.get(k) not in (None, ""))
            print(f"{'':5}hot {h['from_utc']} (+{h['into_match']} in, Q{h['period']}): lost {-h['book_revenue']:,.0f}"
                  f" on {h['bets']} bets, stake {h['stake']:,.0f}, prod {h['prod_prob']}, odds {h['odds']}"
                  + (f"  {state}" if state else ""))
    print(f"\n  every run ({len(out):,}) -> {path}\n  hot points of the top {len(top)} -> {hot_path}")


STATE = ("score", "home_score", "away_score", "lead", "clock", "game_clock", "clock_seconds", "down",
         "distance", "field_position", "possession", "backed_player", "operator")


def hot_points(run, window):
    """The run's stretches of `window` seconds, worst for the book first."""
    bets, start = run["_bets"], run["_start"]
    t0 = bets[0][0]
    slots = defaultdict(list)
    for t, r in bets:
        slots[int((t - t0).total_seconds() // window)].append((t, r))
    out = []
    for k, items in slots.items():
        rows = [r for _, r in items]
        first = items[0][0]
        probs = [_num(r.get("stream_prob")) for r in rows if r.get("stream_prob")]
        odds = [_num(r.get("odds")) for r in rows if r.get("odds")]
        into = (first - start).total_seconds() if start else None
        h = dict(from_utc=first.strftime("%H:%M:%S"), to_utc=items[-1][0].strftime("%H:%M:%S"),
                 into_match=f"{int(into // 60)}:{int(into % 60):02d}" if into is not None else "",
                 period=rows[0].get("period", ""), bets=len(rows),
                 stake=round(sum(_num(r.get("stake")) for r in rows), 2),
                 book_revenue=round(sum(_num(r.get("revenue")) for r in rows), 2),
                 prod_prob=(f"{min(probs):.2f}-{max(probs):.2f}" if probs else ""),
                 odds=(f"{min(odds):.2f}-{max(odds):.2f}" if odds else ""),
                 operators=len({r.get("operator") for r in rows if r.get("operator")}))
        for key in STATE:
            vals = [r.get(key) for r in rows if r.get(key) not in (None, "")]
            h[key] = vals[0] if vals else ""
        out.append(h)
    out.sort(key=lambda h: h["book_revenue"])
    return out


if __name__ == "__main__":
    main()
