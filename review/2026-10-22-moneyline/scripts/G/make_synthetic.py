"""A synthetic scouting PLAY_OVER export in eAMFCalibrator/scouting.py's EXPORT_FIELDS format, for testing
inplay_wp.py where the real export is not to hand.

    python make_synthetic.py --out DIR [--days 60] [--per-day 30] [--seed 1]
        -> DIR/scouting_playover.csv, DIR/prematch.csv (MATCH_CODE, p_home), DIR/history.csv,
           DIR/compare_oracleish.csv (match_code, message, p_home)

Each match is a toy game of four 240-second quarters on gamer strengths (home edge included): drives of snaps
whose gains lean on the offense's strength less the defense's, punts, field goals, touchdowns with the kick
after, turnovers, timeouts called late by the side behind, the side ahead running the clock, and a sudden-death
overtime (a tie when it runs out). The quirks the reader must handle are put in on purpose: TEAM_A is the away
side in a third of the matches (team_a_side), the scoreboard lags the touchdown's PLAY_OVER on a fifth of
them (score_p*_at_start + 6), timeouts_used_a/b reset each half, and prod's prob_50 (a Stern-style price with
noise, live on most rows) stands in for prod's in-play quote.
"""

import argparse
import csv
import math
import os
import random

import numpy as np

QUARTER = 240
FIELDS = ["match_code", "message", "file_time", "publish_time", "period", "clock_seconds", "bet_state", "play_kind",
          "scrimmage", "team_a_side", "offense", "down", "distance", "field_position", "next_start_message",
          "next_offense", "next_down", "next_distance", "next_field_position", "score_p1", "score_p2",
          "score_p1_at_start", "score_p2_at_start", "play_messages", "final_p1", "final_p2", "first_play_message",
          "opening_offense", "home_handle", "away_handle", "next_start_clock", "quoted", "timeouts_used_a",
          "timeouts_used_b"]
for _m in (50, 51, 52, 53, 54, 55):
    FIELDS += [f"line_{_m}", f"prob_{_m}", f"live_{_m}", f"outcome_{_m}"]
FIELDS += ["prematch_line_52", "prematch_prob_52", "prematch_line_54", "prematch_prob_54", "prematch_prob_50",
           "live_markets"]


def sig(x):
    return 1 / (1 + math.exp(-x))


class Game:
    def __init__(self, code, start, rng, r_home, r_away, a_side):
        self.code, self.rng, self.start = code, rng, start
        self.r = {"home": r_home, "away": r_away}
        self.a_side = a_side                                  # which side TEAM_A is
        self.team = {a_side: "TEAM_A", ("away" if a_side == "home" else "home"): "TEAM_B"}
        self.score = {"home": 0, "away": 0}
        self.used = {"home": 0, "away": 0}
        self.rows = []
        self.msg = 10
        self.t = 0.0                                         # real seconds since kick-off
        self.edge = 0.12 * (r_home - r_away) + 0.04

    def other(self, s):
        return "away" if s == "home" else "home"

    def emit(self, period, clock, kind, offense, down, dist, field, messages="", before=None, lag=False):
        clock = max(0.0, clock)
        self.msg += self.rng.randint(3, 9)
        self.t += self.rng.uniform(8, 25)
        b = before or dict(self.score)
        now = b if lag else self.score
        lead = self.score["home"] - self.score["away"]
        s = (max(0, 4 - period) * QUARTER + clock) / 960 if period <= 4 else clock / 960
        z = (lead + 14 * self.edge * s) / (11 * math.sqrt(s + 0.01)) * 1.6
        z += (0.15 if offense == "home" else -0.15) + self.rng.gauss(0, 0.12)
        prob = min(0.995, max(0.005, sig(z)))
        live = 1 if self.rng.random() < 0.8 else 0
        self.rows.append(dict(
            match_code=self.code, message=self.msg, file_time=f"{(self.start + np.timedelta64(int(self.t), 's'))}",
            period=period, clock_seconds=max(0, int(clock)), play_kind=kind, scrimmage=int(kind not in ("CONVERSION", "KICKOFF")),
            team_a_side=self.a_side, offense=self.team[offense], down=down, distance=dist, field_position=field,
            score_p1=now["home"], score_p2=now["away"], score_p1_at_start=b["home"], score_p2_at_start=b["away"],
            play_messages=messages, quoted=1,
            timeouts_used_a=self.used[self.a_side], timeouts_used_b=self.used[self.other(self.a_side)],
            prob_50=round(prob, 4), live_50=live, prob_51=round(1 - prob, 4), live_51=live))

    def tm(self, side):
        return self.team[side]

    def play(self):
        rng = self.rng
        first = "home" if rng.random() < 0.5 else "away"
        self.opening = self.tm(first)
        period, clock, off = 1, float(QUARTER), first
        y, down, dist = 25, 1, 10
        self.emit(1, clock - 5, "KICKOFF", off, 1, 10, 25, f"KICKOFF_{self.tm(self.other(off))}|POSSESSION_{self.tm(off)}")
        clock -= 5
        while True:
            if clock <= 0:
                if period == 2:
                    self.used = {"home": 0, "away": 0}
                if period == 4 and self.score["home"] != self.score["away"]:
                    break
                if period >= 5:
                    break
                period += 1
                clock = float(QUARTER)
                if period in (3, 5):
                    off = self.other(first) if period == 3 else ("home" if rng.random() < 0.5 else "away")
                    if period == 5:
                        self.used = {"home": 0, "away": 0}
                    y, down, dist = 25, 1, 10
                    clock -= 5
                    self.emit(period, clock, "KICKOFF", off, 1, 10, 25,
                              f"KICKOFF_{self.tm(self.other(off))}|POSSESSION_{self.tm(off)}")
                continue
            lead = self.score[off] - self.score[self.other(off)]
            late = period >= 4 and clock < 150
            q = 0.35 * (self.r[off] - self.r[self.other(off)]) + (0.08 if off == "home" else 0)
            ytg = 100 - y
            if down == 4 and not (late and lead < 0 and ytg > 35):
                before = dict(self.score)
                if ytg <= 37:
                    clock -= 6
                    good = rng.random() < 0.92 - 0.01 * max(0, ytg - 20)
                    if good:
                        self.score[off] += 3
                    self.emit(period, clock, "FIELD_GOAL", off, 1, 10, 35,
                              ("FIELD_GOAL_GOOD_" if good else "FIELD_GOAL_MISSED_") + self.tm(off), before)
                    if period == 5 and good:
                        break
                    if good:
                        kicker, off = off, self.other(off)
                        clock -= 4
                        y, down, dist = 25, 1, 10
                        self.emit(period, clock, "KICKOFF", off, 1, 10, 25,
                                  f"KICKOFF_{self.tm(kicker)}|POSSESSION_{self.tm(off)}")
                    else:
                        off = self.other(off)
                        y, down, dist = 100 - min(80, y + 7), 1, 10
                        self.emit(period, clock, "SCRIMMAGE", off, 1, 10, y, "POSSESSION_" + self.tm(off))
                    continue
                clock -= 10
                kicker, off = off, self.other(off)
                y, down, dist = max(5, min(40, 100 - y - 40)), 1, 10
                self.emit(period, clock, "PUNT", off, 1, 10, y, f"PUNT_{self.tm(kicker)}|POSSESSION_{self.tm(off)}")
                continue
            # the clock: the side ahead late runs it, the side behind hurries and calls timeouts
            run = rng.uniform(14, 30)
            if late and lead > 0:
                run = rng.uniform(30, 40)
            elif late and lead < 0:
                run = rng.uniform(6, 12)
                if self.used[off] < 3 and rng.random() < 0.3:
                    self.used[off] += 1
                    run = 5
            if rng.random() < 0.25:
                run = 6                                     # an incompletion stops the clock
            clock -= run
            if clock <= 0 and period in (2, 4, 5):
                continue
            if rng.random() < 0.025:                        # a turnover
                off = self.other(off)
                y, down, dist = max(1, min(95, 100 - y)), 1, 10
                self.emit(period, max(clock, 1), "SCRIMMAGE", off, 1, 10, y, "POSSESSION_" + self.tm(off))
                continue
            gain = int(round(rng.gauss(4.2 + 3.5 * q, 7)))
            if rng.random() < 0.05 + 0.03 * q:
                gain += rng.randint(15, 50)
            y += gain
            if y >= 100:
                before = dict(self.score)
                self.score[off] += 6
                lag = rng.random() < 0.2
                self.emit(period, max(clock, 1), "TOUCHDOWN", off, 1, 10, 97, "TOUCHDOWN_" + self.tm(off), before, lag)
                if period == 5:
                    break
                before = dict(self.score)
                go2 = late and self.score[off] - self.score[self.other(off)] in (-2, -1)
                if go2:
                    good = rng.random() < 0.5
                    self.score[off] += 2 if good else 0
                    m = ("TWO_POINT_CONVERSION_SUCCESSFUL_" if good else "TWO_POINT_CONVERSION_FAILED_") + self.tm(off)
                else:
                    good = rng.random() < 0.95
                    self.score[off] += 1 if good else 0
                    m = ("EXTRA_POINT_GOOD_" if good else "EXTRA_POINT_MISSED_") + self.tm(off)
                self.emit(period, max(clock, 1), "CONVERSION", off, 1, 10, 85, m, before)
                kicker, off = off, self.other(off)
                y, down, dist = 25, 1, 10
                clock -= 4
                self.emit(period, max(clock, 1), "KICKOFF", off, 1, 10, 25, f"KICKOFF_{self.tm(kicker)}|POSSESSION_{self.tm(off)}")
                continue
            y = max(1, y)
            if gain >= dist:
                down, dist = 1, min(10, 100 - y)
            else:
                down, dist = down + 1, dist - gain
            self.emit(period, max(clock, 0), "SCRIMMAGE", off, down, dist, y)
        for r in self.rows:
            r["final_p1"], r["final_p2"] = self.score["home"], self.score["away"]
            r["opening_offense"] = self.opening
        return self.rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--per-day", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)
    gamers = {f"G{i:02d}": rng.gauss(0, 1) for i in range(40)}
    names = list(gamers)
    day0 = np.datetime64("2026-07-01T00:00:00")
    pre_rows, hist_rows, comp_rows = [], [], []
    n = 0
    with open(os.path.join(a.out, "scouting_playover.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for d in range(a.days):
            date = day0 + np.timedelta64(d, "D")
            dd = str(date)[:10]
            for k in range(a.per_day):
                home, away = rng.sample(names, 2)
                code = f"AF{k + 1:03d}{dd[8:10]}{dd[5:7]}{dd[2:4]}"
                start = date + np.timedelta64(int(k * 2880 / a.per_day * 60 / 2), "s")
                side = "away" if rng.random() < 1 / 3 else "home"
                g = Game(code, start, rng, gamers[home], gamers[away], side)
                rows = g.play()
                p_true = sig(1.1 * (gamers[home] - gamers[away]) * 0.6 + 0.1)
                pre_tab = sig(math.log(p_true / (1 - p_true)) + rng.gauss(0, 0.25))
                pre_prod = sig(math.log(p_true / (1 - p_true)) * 0.8 + rng.gauss(0, 0.3))
                for r in rows:
                    r["home_handle"], r["away_handle"] = home, away
                    r["prematch_prob_50"] = round(pre_prod, 4)
                    w.writerow(r)
                    comp_rows.append((code, r["message"], min(0.999, max(0.001, sig(math.log(r["prob_50"] / (1 - r["prob_50"])) * 0.9 + rng.gauss(0, 0.2))))))
                pre_rows.append((code, round(pre_tab, 5)))
                hist_rows.append((code, str(start).replace("T", " "), home, away, rows[-1]["final_p1"], rows[-1]["final_p2"]))
                n += 1
    with open(os.path.join(a.out, "prematch.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["MATCH_CODE", "p_home"])
        w.writerows(pre_rows)
    with open(os.path.join(a.out, "history.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["MATCH_CODE", "SCHEDULED_START_TIME_UTC", "PLAYER_1_HANDLE", "PLAYER_2_HANDLE",
                    "PLAYER_1_FINAL_SCORE", "PLAYER_2_FINAL_SCORE"])
        w.writerows(hist_rows)
    with open(os.path.join(a.out, "compare_oracleish.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["match_code", "message", "p_home"])
        w.writerows(comp_rows)
    print(f"{n:,} synthetic matches -> {a.out}")


if __name__ == "__main__":
    main()
