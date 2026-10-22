import contextlib
import csv
import datetime as dt
import io
import math
import os
import random
import tempfile
import unittest
from unittest import mock

import numpy as np

from eAMFCalibrator import ml_prior as mp

T0 = dt.datetime(2026, 3, 1, 8, 0)
HISTORY_FIELDS = ["MATCH_CODE", "SPORT_CODE", "STREAM_NUMBER", "SCHEDULED_START_TIME_UTC", "PLAYER_1_HANDLE",
                  "PLAYER_1_TEAM", "PLAYER_2_HANDLE", "PLAYER_2_TEAM", "PLAYER_1_FINAL_SCORE", "PLAYER_2_FINAL_SCORE"]
GAMERS = {"ACE": 5.0, "BOB": 3.0, "CAT": 1.0, "DAN": 0.0, "EVE": -1.0, "FOX": -2.0, "GUS": -2.0, "HAL": -4.0}
TEAMS = {"Bears": 1.0, "Colts": 0.5, "Jets": -0.5, "Lions": -1.0}
HOME = {"1": 2.0, "2": 0.0}                # the home edge by stream
PACE = {"ACE": 4.0, "BOB": -3.0, "CAT": 0.0, "DAN": 2.0, "EVE": -2.0, "FOX": 5.0, "GUS": -4.0, "HAL": -2.0}
LEAGUE_TOTAL = 40.0


def planted_total(home, away):
    return LEAGUE_TOTAL + PACE[home] + PACE[away]


def league(days=60, per_day=12, noise=6.0, seed=1, unplayed=3):
    """[Match]: per_day matches a day, 45 minutes apart from 08:00, two random gamers with random teams on
    alternating streams, the home margin the planted strengths' and the total the planted paces' (each plus
    normal noise); the last `unplayed` without a result."""
    rng = random.Random(seed)
    out = []
    for d in range(days):
        for k in range(per_day):
            home, away = rng.sample(sorted(GAMERS), 2)
            th, ta = rng.choice(sorted(TEAMS)), rng.choice(sorted(TEAMS))
            stream = "1" if k % 2 == 0 else "2"
            m = HOME[stream] + GAMERS[home] - GAMERS[away] + TEAMS[th] - TEAMS[ta] + rng.gauss(0, noise)
            total = planted_total(home, away) + rng.gauss(0, noise)
            hs, as_ = max(0.0, round((total + m) / 2)), max(0.0, round((total - m) / 2))
            out.append(mp.Match(T0 + dt.timedelta(days=d, minutes=45 * k), f"AF{d:03d}{k:02d}", stream, home, away,
                                th, ta, float(hs), float(as_)))
    for i in range(len(out) - unplayed, len(out)):
        out[i] = out[i]._replace(home_score=None, away_score=None)
    return out


def coin_league(days=100, per_day=40, a=0.2, b=0.15, b_table=0.0, seed=2):
    """([Match], margins, table margins): each match's margin drawn at random and its result (1-0 or 0-1) off
    logistic(a + b x margin + b_table x the table's margin), the table's drawn at random too."""
    rng = random.Random(seed)
    matches, exp, tbl = [], [], []
    for d in range(days):
        for k in range(per_day):
            m, t = rng.uniform(-10, 10), rng.uniform(-10, 10)
            won = rng.random() < 1.0 / (1.0 + math.exp(-(a + b * m + b_table * t)))
            matches.append(mp.Match(T0 + dt.timedelta(days=d, minutes=20 * k), f"C{d:03d}{k:02d}", "1", "ACE", "BOB",
                                    "", "", 1.0 if won else 0.0, 0.0 if won else 1.0))
            exp.append(m)
            tbl.append(t)
    return matches, exp, tbl


def write_history(path, matches):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, HISTORY_FIELDS)
        w.writeheader()
        for m in matches:
            w.writerow(dict(MATCH_CODE=m.code, SPORT_CODE="AF", STREAM_NUMBER=m.stream,
                            SCHEDULED_START_TIME_UTC=f"{m.start:%Y-%m-%d %H:%M:%S}", PLAYER_1_HANDLE=m.home,
                            PLAYER_1_TEAM=m.home_team, PLAYER_2_HANDLE=m.away, PLAYER_2_TEAM=m.away_team,
                            PLAYER_1_FINAL_SCORE="" if m.home_score is None else int(m.home_score),
                            PLAYER_2_FINAL_SCORE="" if m.away_score is None else int(m.away_score)))


def write_table(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, ["MATCH_CODE", "start", "home", "away", "exp_margin", "exp_total", "p_home"])
        w.writeheader()
        w.writerows(rows)


def priced(matches, **kw):
    """{match: its row} off the ridge, the link and the total's filter (min_rows 100 unless given)."""
    exp, _ = mp.margins(matches)
    kw.setdefault("min_rows", 100)
    return {r["match"].code: r for r in mp.price(matches, exp, total=mp.totals(matches), **kw)}


class TestNoLookAhead(unittest.TestCase):
    def setUp(self):
        self.matches = league()
        m = next(x for x in self.matches if x.code == "AF05006")          # day 50, 12:30
        self.m = m
        rival = "HAL" if m.home != "HAL" else "ACE"
        x = mp.Match(m.start - dt.timedelta(minutes=30), "AFX", "3", m.home, rival, "Bears", "Jets", 50.0, 0.0)
        y = mp.Match(m.start - dt.timedelta(minutes=36), "AFY", "3", m.home, rival, "Bears", "Jets", 50.0, 0.0)
        self.matches = sorted(self.matches + [x, y], key=lambda r: (r.start, r.code))

    def changed(self, code, hs, as_):
        return [r._replace(home_score=hs, away_score=as_) if r.code == code else r for r in self.matches]

    def test_a_result_is_in_36_minutes_after_its_kick_off_and_not_before(self):
        base = priced(self.matches)[self.m.code]
        self.assertTrue(math.isfinite(base["p_home"]))
        keys = ("exp_margin", "p_home", "exp_total")
        late = priced(self.changed("AFX", 0.0, 50.0))[self.m.code]     # 30 minutes before: not in yet
        self.assertEqual([late[k] for k in keys], [base[k] for k in keys])
        lag = priced(self.changed("AFY", 0.0, 50.0))[self.m.code]      # 36 minutes before: in
        self.assertLess(lag["exp_margin"], base["exp_margin"] - 0.3)
        self.assertLess(lag["p_home"], base["p_home"])
        self.assertEqual(lag["exp_total"], base["exp_total"])          # a 50-0 and a 0-50 total the same
        more = priced(self.changed("AFY", 50.0, 40.0))[self.m.code]
        self.assertGreater(more["exp_total"], base["exp_total"] + 0.5)
        own = priced(self.changed(self.m.code, 0.0, 90.0))[self.m.code]  # its own result
        self.assertEqual([own[k] for k in keys], [base[k] for k in keys])

    def test_nothing_after_a_kick_off_moves_its_price(self):
        full = priced(self.matches)
        for code in ("AF04603", "AF04811", "AF05006", "AF05507", "AF05902"):
            kick = next(r.start for r in self.matches if r.code == code)
            cut = [r._replace(home_score=None, away_score=None) if r.start + mp.RESULT_LAG > kick
                   else r for r in self.matches]
            got = priced(cut)[code]
            self.assertTrue(math.isfinite(got["p_home"]), code)
            self.assertAlmostEqual(got["exp_margin"], full[code]["exp_margin"], places=12, msg=code)
            self.assertAlmostEqual(got["p_home"], full[code]["p_home"], places=12, msg=code)
            self.assertEqual(got["link_b"], full[code]["link_b"])
            self.assertAlmostEqual(got["exp_total"], full[code]["exp_total"], places=10, msg=code)

    def test_the_scheduled_matches_are_priced_off_every_result_in(self):
        rows = priced(self.matches)
        last = [m for m in self.matches if m.home_score is None]
        self.assertEqual(len(last), 3)
        for m in last:
            self.assertTrue(0.0 < rows[m.code]["p_home"] < 1.0)
            self.assertEqual(rows[m.code]["priced_by"], "ridge")

    def test_the_first_days_go_unpriced_until_the_link_has_its_results(self):
        rows = priced(self.matches)
        first = min(r["match"].start for r in rows.values() if math.isfinite(r["p_home"]))
        # the history's first 36 days are kept out of the links; 100 results at 12 a day take 9 days more
        self.assertEqual(first.date(), (T0 + dt.timedelta(days=36 + 9)).date())


class TestRidge(unittest.TestCase):
    def test_it_recovers_the_planted_strengths(self):
        matches = league(days=150, noise=3.0, unplayed=0)
        _, ridge = mp.margins(matches, half_life=1000.0)
        b = ridge.beta
        for g, s in GAMERS.items():
            self.assertAlmostEqual(b[ridge.gamers[g]], s, delta=0.4, msg=g)
        for t, s in TEAMS.items():
            self.assertAlmostEqual(b[ridge.teams[t]], s, delta=0.4, msg=t)
        for s, edge in HOME.items():
            self.assertAlmostEqual(b[0] + b[ridge.streams[s]], edge, delta=0.4, msg=s)

    def test_a_result_counts_half_as_much_a_half_life_on(self):
        """One gamer's two results, a half-life apart: the ridge reads them 1:2 (the penalty aside)."""
        old = mp.Match(T0, "A1", "", "ACE", "BOB", "", "", 10.0, 0.0)
        new = mp.Match(T0 + dt.timedelta(days=30), "A2", "", "ACE", "BOB", "", "", 0.0, 10.0)
        probe = mp.Match(T0 + dt.timedelta(days=31), "A3", "", "ACE", "BOB", "", "", None, None)
        exp, _ = mp.margins([old, new, probe], gamer_penalty=1e-9)
        self.assertAlmostEqual(exp[2], (10.0 * 0.5 - 10.0) / 1.5, delta=0.05)

    def test_a_team_left_blank_or_the_same_on_both_sides_counts_for_nothing(self):
        r = mp.Ridge(league(days=2))
        cols, val = r.row(mp.Match(T0, "Z", "1", "ACE", "BOB", "Bears", "Bears", None, None))
        self.assertNotIn(r.teams["Bears"], list(cols))
        cols, val = r.row(mp.Match(T0, "Z", "9", "ACE", "BOB", "", "Jets", None, None))
        self.assertEqual(dict(zip(cols.tolist(), val.tolist())),
                         {0: 1.0, r.gamers["ACE"]: 1.0, r.gamers["BOB"]: -1.0, r.teams["Jets"]: -1.0})


class TestTotal(unittest.TestCase):
    def test_the_filter_recovers_the_planted_totals(self):
        matches = league(days=150, noise=3.0, unplayed=0)
        tot = mp.totals(matches)
        last = [(t, planted_total(m.home, m.away)) for m, t in zip(matches, tot) if m.start >= T0 + dt.timedelta(days=140)]
        miss = np.array([t - p for t, p in last])
        self.assertLess(np.abs(miss).mean(), 0.6)
        self.assertLess(abs(miss.mean()), 0.3)
        spread = np.std([p for _, p in last])
        self.assertGreater(np.corrcoef([t for t, _ in last], [p for _, p in last])[0, 1], 0.97)
        self.assertGreater(spread, 3.0)

    def test_nothing_before_the_first_result_and_the_league_level_after(self):
        matches = league(days=3, unplayed=0)
        tot = mp.totals(matches)
        first_in = matches[0].start + mp.RESULT_LAG
        for m, t in zip(matches, tot):
            if m.start < first_in:
                self.assertTrue(math.isnan(t), m.code)
            else:
                self.assertTrue(20.0 < t < 60.0, m.code)

    def test_a_result_moves_the_total_through_the_gamers_and_the_league(self):
        """One 70-point result: the league's mean moves, and its two gamers' totals more than the others'."""
        base = [mp.Match(T0 + dt.timedelta(hours=k), f"B{k}", "1", g, h, "", "", 20.0, 20.0)
                for k, (g, h) in enumerate([("ACE", "BOB"), ("CAT", "DAN")] * 20)]
        probe = [mp.Match(T0 + dt.timedelta(hours=41), "P1", "1", "ACE", "CAT", "", "", None, None),
                 mp.Match(T0 + dt.timedelta(hours=41), "P2", "1", "DAN", "BOB", "", "", None, None),
                 mp.Match(T0 + dt.timedelta(hours=41), "P3", "1", "CAT", "DAN", "", "", None, None)]
        big = mp.Match(T0 + dt.timedelta(hours=40), "BIG", "1", "ACE", "BOB", "", "", 40.0, 30.0)
        before = mp.totals(base + probe)[-3:]
        after = mp.totals(base + [big] + probe)[-3:]
        self.assertAlmostEqual(before[0], 40.0, delta=0.5)
        self.assertGreater(after[2], before[2])                        # the league's level
        self.assertGreater(after[0] - before[0], after[2] - before[2])  # ACE's own matches more
        self.assertGreater(after[1] - before[1], after[2] - before[2])  # BOB's

    def test_the_points_split_the_total_by_the_margin(self):
        rows = priced(league())
        for r in rows.values():
            if math.isnan(r["exp_total"]):
                continue
            self.assertAlmostEqual(r["home_pts"] + r["away_pts"], r["exp_total"], places=9)
            self.assertAlmostEqual(r["home_pts"] - r["away_pts"], r["exp_margin"], places=9)
            self.assertAlmostEqual(r["sd_total"], 12.3 * (r["exp_total"] / 35.0) ** 0.75, places=9)
        self.assertAlmostEqual(mp.total_sd(35.0), 12.3)


class TestLink(unittest.TestCase):
    def test_the_fit_recovers_a_planted_logistic(self):
        rng = np.random.default_rng(0)
        m = rng.uniform(-10, 10, 20000)
        y = (rng.random(20000) < 1.0 / (1.0 + np.exp(-(0.2 + 0.15 * m)))).astype(float)
        a, b = mp.logistic_fit(np.column_stack([np.ones(len(m)), m]), y)
        self.assertAlmostEqual(a, 0.2, delta=0.06)
        self.assertAlmostEqual(b, 0.15, delta=0.01)

    def test_each_day_s_link_recovers_it_off_the_last_60_days(self):
        matches, exp, _ = coin_league()
        rows = mp.price(matches, exp, burn_in=0)
        last = rows[-1]
        self.assertAlmostEqual(last["link_a"], 0.2, delta=0.15)
        self.assertAlmostEqual(last["link_b"], 0.15, delta=0.03)
        p = 1.0 / (1.0 + math.exp(-(last["link_a"] + last["link_b"] * last["exp_margin"])))
        self.assertAlmostEqual(last["p_home"], p, places=12)
        days = {r["match"].start.date(): r["link_b"] for r in rows}
        first = min(d for d, b in days.items() if math.isfinite(b))
        self.assertEqual(first, (T0 + dt.timedelta(days=8)).date())    # 300 results at 40 a day: 8 days in
        self.assertEqual(len({b for b in days.values() if math.isfinite(b)}), len(days) - 8)   # one fit a day

    def test_a_tie_is_left_out_of_the_link(self):
        matches, exp, _ = coin_league(days=20)
        tie = [m._replace(home_score=1.0, away_score=1.0) if m.start.day % 3 == 0 else m for m in matches]
        a = mp.price(matches, exp, burn_in=0)
        b = mp.price(tie, exp, burn_in=0)
        self.assertNotEqual(a[-1]["link_b"], b[-1]["link_b"])
        drop = [m for m in matches if m.start.day % 3 != 0]
        keep = [e for m, e in zip(matches, exp) if m.start.day % 3 != 0]
        c = mp.price(drop, keep, burn_in=0, days=[matches[-1].start.date()])
        self.assertAlmostEqual(b[-1]["link_b"], c[-1]["link_b"], places=12)


class TestStack(unittest.TestCase):
    def setUp(self):
        self.matches, self.exp, tbl = coin_league(b=0.1, b_table=0.1)
        self.tbl = {m.code: (t, None, None) for m, t in zip(self.matches, tbl)}

    def test_without_enough_table_rows_it_falls_back_to_the_link(self):
        recent = {m.code: self.tbl[m.code] for m in self.matches[-120:]}       # 3 days: under 300
        rows = mp.price(self.matches, self.exp, recent, burn_in=0)
        self.assertEqual({r["priced_by"] for r in rows[-120:]}, {"ridge"})
        for r in rows[-120:]:
            self.assertEqual(r["p_home"], r["p_ridge"])
            self.assertTrue(math.isnan(r["stack_a"]))
            self.assertTrue(math.isfinite(r["table_margin"]))

    def test_with_enough_it_stacks_the_two_margins(self):
        from_30 = {m.code: self.tbl[m.code] for m in self.matches if m.start >= T0 + dt.timedelta(days=30)}
        rows = mp.price(self.matches, self.exp, from_30, burn_in=0)
        stacked = [r for r in rows if r["priced_by"] == "stack"]
        self.assertEqual(min(r["match"].start for r in stacked).date(), (T0 + dt.timedelta(days=38)).date())
        self.assertEqual({r["priced_by"] for r in rows if r["match"].start < T0 + dt.timedelta(days=38)} - {""},
                         {"ridge"})
        last = rows[-1]
        self.assertAlmostEqual(last["stack_b_table"], 0.1, delta=0.035)
        self.assertAlmostEqual(last["stack_b_ridge"], 0.1, delta=0.035)
        self.assertAlmostEqual(last["stack_a"], 0.2, delta=0.15)
        z = last["stack_a"] + last["stack_b_table"] * last["table_margin"] + last["stack_b_ridge"] * last["exp_margin"]
        self.assertAlmostEqual(last["p_home"], 1.0 / (1.0 + math.exp(-z)), places=12)
        self.assertNotEqual(last["p_home"], last["p_ridge"])


class TestCommand(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.history = os.path.join(self.dir.name, "history.csv")
        self.matches = league()
        write_history(self.history, self.matches)
        self.table = os.path.join(self.dir.name, "avg.csv")         # the planted margins, from day 30
        write_table(self.table, [dict(MATCH_CODE=m.code, start=f"{m.start:%Y-%m-%d %H:%M:%S}", home=m.home, away=m.away,
                                      exp_margin=HOME[m.stream] + GAMERS[m.home] - GAMERS[m.away] + TEAMS[m.home_team]
                                      - TEAMS[m.away_team], exp_total=planted_total(m.home, m.away), p_home=0.5)
                                 for m in self.matches if m.start >= T0 + dt.timedelta(days=30)])

    def tearDown(self):
        self.dir.cleanup()

    def run_cli(self, *argv):
        from eAMFCalibrator import __main__ as cli
        with contextlib.redirect_stdout(io.StringIO()):
            return cli.main(["ml-prior", *argv])

    def test_the_command_writes_one_table_for_both_of_v21_s_nudges(self):
        out = os.path.join(self.dir.name, "out")
        # the links start 45 days in (36 days left out, then 100 results at 12 a day): write from day 46
        self.assertEqual(self.run_cli("--history", self.history, "--since", "2026-04-16", "--table", self.table,
                                      "--min-rows", "100", "--out", out), 0)
        path = os.path.join(out, "ml_prior.csv")
        with open(path, newline="", encoding="utf-8") as fh:
            rd = csv.DictReader(fh)
            rows = list(rd)
            self.assertEqual(rd.fieldnames, mp.COLUMNS)
        for c in ("MATCH_CODE", "start", "home", "away", "home_team", "away_team", "stream", "exp_margin", "p_home",
                  "link_a", "link_b", "exp_total", "sd_total", "Pred_P1_Points", "Pred_P2_Points"):
            self.assertIn(c, mp.COLUMNS)
        since = [m for m in self.matches if m.start >= dt.datetime(2026, 4, 16)]
        self.assertEqual([r["MATCH_CODE"] for r in rows], [m.code for m in since])
        self.assertTrue(all(0.0 < float(r["p_home"]) < 1.0 for r in rows))
        self.assertEqual({r["priced_by"] for r in rows}, {"stack"})
        self.assertEqual([r["home_score"] for r in rows[-3:]], ["", "", ""])          # scheduled, still priced
        first = rows[0]
        self.assertEqual((first["home"], first["stream"]), (since[0].home, since[0].stream))
        from eAMFModel import table_prior, v21_stream
        table = v21_stream.ml_table(path)                                # the moneyline nudge's p_home
        self.assertEqual(set(table), {m.code for m in since})
        self.assertAlmostEqual(table[rows[0]["MATCH_CODE"]], float(rows[0]["p_home"]))
        prior = table_prior.TablePrior(path)                             # the points nudge's points
        self.assertEqual(set(prior.points), {m.code for m in since})
        for r in rows:
            h, a = prior.points[r["MATCH_CODE"]]
            self.assertEqual((h, a), (float(r["Pred_P1_Points"]), float(r["Pred_P2_Points"])))
            self.assertAlmostEqual(h + a, float(r["exp_total"]), delta=2e-4)
            self.assertAlmostEqual(h - a, float(r["exp_margin"]), delta=2e-4)
            self.assertTrue(30.0 < float(r["exp_total"]) < 50.0)
        with open(os.path.join(out, "ml_prior.txt"), encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn(f"{len(since):,} written, 2026-04-16 to 2026-04-29, 3 of them without a result yet", text)
        self.assertIn("avg.csv has", text)
        self.assertIn("p_home (the model)", text)
        self.assertIn("avg.csv's own p_home", text)
        self.assertIn("avg.csv's exp_total", text)
        self.assertIn("exp_total (the filter)", text)
        self.assertIn("2026-04", text)

    def test_the_span_written_doesnt_change_a_price(self):
        a, b = os.path.join(self.dir.name, "a"), os.path.join(self.dir.name, "b")
        self.run_cli("--history", self.history, "--min-rows", "100", "--out", a)
        self.run_cli("--history", self.history, "--since", "2026-04-20", "--until", "2026-04-25", "--min-rows", "100",
                     "--out", b)
        with open(os.path.join(a, "ml_prior.csv"), newline="", encoding="utf-8") as fh:
            whole = {r["MATCH_CODE"]: r for r in csv.DictReader(fh)}
        with open(os.path.join(b, "ml_prior.csv"), newline="", encoding="utf-8") as fh:
            part = list(csv.DictReader(fh))
        self.assertEqual(len(whole), len(self.matches))
        self.assertEqual(len(part), 5 * 12)
        for r in part:
            self.assertEqual(r, whole[r["MATCH_CODE"]])

    def test_its_defaults_and_its_stops(self):
        from eAMFCalibrator import __main__ as cli
        with tempfile.TemporaryDirectory() as d:
            write_history(os.path.join(d, "match_history.csv"), self.matches)
            with mock.patch.object(cli, "DEFAULT_OUT", d):
                self.assertEqual(self.run_cli("--min-rows", "100"), 0)
            self.assertEqual(sorted(os.listdir(os.path.join(d, "ml_prior"))), ["ml_prior.csv", "ml_prior.txt"])
            with mock.patch.object(cli, "DEFAULT_OUT", os.path.join(d, "nowhere")), self.assertRaises(SystemExit):
                self.run_cli()
        with self.assertRaises(SystemExit):
            self.run_cli("--history", self.history, "--table", os.path.join(self.dir.name, "no.csv"))
        with self.assertRaises(SystemExit):
            self.run_cli("--history", self.history, "--since", "2027-01-01", "--out", os.path.join(self.dir.name, "x"))


if __name__ == "__main__":
    unittest.main()
