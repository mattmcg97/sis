import json
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np

from eAMFModel import game as gm, trader, v14
from eAMFModel.tests.fakes import _matches, _with_handles


class TestTrader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        v14.build(_with_handles(_matches(40)), cls.tmp.name, grid_paths=60, verbose=False)
        cls.t = trader.Trader(cls.tmp.name, paths=400)
        cls.setup = {"home_player": "ALPHA", "away_player": "BRAVO", "means": [24.0, 14.0],
                     "opening_receiver": "away", "margin": 0.05}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def call(self, path, **body):
        status, reply = trader.respond(self.t, path, body)
        self.assertEqual(status, 200, reply)
        return reply

    def test_meta_lists_the_builds_players_and_says_it_has_no_pre_match_model(self):
        m = self.call("/api/meta")
        self.assertTrue({"ALPHA", "BRAVO"} <= set(m["players"]))
        self.assertFalse(m["prematch"])
        self.assertEqual(m["paths"], 400)

    def test_without_a_pre_match_model_the_expected_points_are_typed_in(self):
        r = self.call("/api/prematch", setup=self.setup)
        self.assertIn("No pre-match model", r["error"])
        self.assertEqual(set(r["home_profile"]), {"plays", "pace", "aggression", "kick"})
        status, reply = trader.respond(self.t, "/api/start", {"setup": {"home_player": "ALPHA"}})
        self.assertEqual(status, 400)

    def test_start_prices_the_kick_off_at_key_number_lines(self):
        r = self.call("/api/start", setup=self.setup)
        self.assertEqual((r["game"]["phase"], r["game"]["side"]), (gm.KICKOFF, gm.AWAY))
        for price in (r["prematch"], r["price"]):
            ml = price["moneyline"]
            self.assertAlmostEqual(ml["home"]["p"] + ml["away"]["p"], 1.0, places=6)
            self.assertGreater(ml["home"]["p"], 0.5)                     # 24 against 14
            self.assertEqual(price["spread"]["line"] % 1, 0.5)
            self.assertEqual(price["total"]["line"] % 1, 0.5)
            self.assertEqual(len(price["spread"]["ladder"]), 2 * trader.SPREAD_LADDER + 1)
            self.assertEqual(len(price["margin_dist"]["p"]), 2 * trader.MARGIN_SHOWN + 1)
            self.assertGreater(price["moneyline"]["home"]["odds"], 1.0)
        self.assertAlmostEqual(r["price"]["expected"]["home"] - r["price"]["expected"]["away"],
                               r["price"]["expected"]["margin"], places=1)

    def test_a_play_moves_the_game_and_the_price(self):
        start = self.call("/api/start", setup=self.setup)
        g = self.call("/api/play", setup=self.setup, game=start["game"], play={"type": "kickoff"},
                      kept=start["price"]["kept"])
        td = self.call("/api/play", setup=self.setup, game=g["game"],
                       play={"type": "turnover", "touchdown": True, "seconds": 5}, kept=g["price"]["kept"])
        self.assertEqual(td["game"]["home"], 6)
        self.assertGreater(td["price"]["moneyline"]["home"]["p"], g["price"]["moneyline"]["home"]["p"])
        with self.assertRaises(ValueError):
            trader.respond(self.t, "/api/play", {"setup": self.setup, "game": td["game"],
                                                 "play": {"type": "punt"}})   # a conversion is next

    def test_a_finished_game_is_settled(self):
        end = gm.Game(period=4, clock=0, home=21, away=17, phase=gm.FINAL).to_dict()
        r = self.call("/api/price", setup=self.setup, game=end)
        self.assertTrue(r["price"]["final"])
        self.assertEqual(r["price"]["moneyline"]["home"]["p"], 1.0)

    def test_the_server_serves_the_page_and_the_api(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), trader.handler_for(self.t))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            page = urllib.request.urlopen(base + "/").read().decode("utf-8")
            self.assertIn('<h1 id="title">Trader</h1>', page)
            req = urllib.request.Request(base + "/api/start", json.dumps({"setup": self.setup}).encode(),
                                         {"Content-Type": "application/json"})
            reply = json.loads(urllib.request.urlopen(req).read())
            self.assertEqual(reply["game"]["phase"], gm.KICKOFF)
            bad = urllib.request.Request(base + "/api/play", json.dumps(
                {"setup": self.setup, "game": reply["game"], "play": {"type": "punt"}}).encode())
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(bad)
            self.assertEqual(caught.exception.code, 400)
        finally:
            server.shutdown()
            server.server_close()

    def test_the_version_is_read_off_the_build(self):
        self.assertEqual(self.t.version, "v14")
        self.assertEqual(trader.version_of(self.tmp.name), "v14")
        self.assertEqual(self.call("/api/meta")["version"], "v14")
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaises(SystemExit):
                trader.version_of(empty)
        with self.assertRaises(SystemExit):
            trader.load_version("v99")

    def test_odds_carry_the_margin(self):
        self.assertEqual(trader.odds(0.5, 0.0), 2.0)
        self.assertEqual(trader.odds(0.5, 0.05), round(1 / (0.5 * 1.05), 2))


class TestV21Trader(unittest.TestCase):
    """v2.1 in the trader: a match off the prior table brings its points and drives; the kick-off's points
    nudge and tempo are solved once and carried into every state; a conversion is priced as the sim plays one."""

    @classmethod
    def setUpClass(cls):
        import csv
        import os
        from eAMFModel import v21
        from eAMFModel.tests.test_v21 import HISTORY, built
        cls.tmp = tempfile.TemporaryDirectory()
        cls.model = built("tilt")
        cls.model.save(os.path.join(cls.tmp.name, v21.MODEL_FILE))
        cls.table = os.path.join(cls.tmp.name, "avg_form4_w05.csv")
        h = HISTORY[:3]
        with open(cls.table, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["MATCH_CODE", "start", "home", "home_team", "away", "away_team", "home_pts", "away_pts",
                        "exp_drives", "league_drives"])
            for k, (r, pts, drives) in enumerate(zip(h, ((26.0, 13.0), (17.0, 17.0), (14.0, 21.0)), (11.5, 10.0, 8.5))):
                w.writerow([r["MATCH_CODE"], f"2026-10-1{k} 12:00:00", r["PLAYER_1_HANDLE"], r["PLAYER_1_TEAM"],
                            r["PLAYER_2_HANDLE"], r["PLAYER_2_TEAM"], *pts, drives, 10.0])
        cls.codes = [r["MATCH_CODE"] for r in h]
        cls.t = trader.V21Trader(cls.tmp.name, paths=300, prior=cls.table)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def call(self, path, **body):
        status, reply = trader.respond(self.t, path, body)
        self.assertEqual(status, 200, reply)
        return reply

    def setup_of(self, code):
        r = self.call("/api/prematch", setup={"match": code})
        return dict(r, match=code, opening_receiver="home", margin=0.05)

    def test_meta_and_a_match_off_the_table(self):
        m = self.call("/api/meta")
        self.assertEqual(m["version"], "v21-drives")
        self.assertTrue(m["pace"])
        self.assertEqual([x["code"] for x in m["matches"]], self.codes[::-1])        # the latest first
        self.assertEqual(m["streams"], [])
        r = self.call("/api/prematch", setup={"match": self.codes[0]})
        self.assertEqual((r["means"], r["exp_drives"], r["league_drives"]), ([26.0, 13.0], 11.5, 10.0))
        self.assertTrue(r["home_player"])
        swapped = self.call("/api/prematch", setup={"home_player": r["away_player"], "away_player": r["home_player"]})
        self.assertEqual(swapped["means"], [13.0, 26.0])                              # their meeting, ends swapped
        self.assertIn("ends swapped", swapped["note"])
        self.assertIn("league", self.call("/api/prematch", setup={"home_player": "NOBODY"})["note"])

    def test_the_pace_is_the_betting_sims(self):
        from eAMFModel import v21_stream
        drives = v21_stream.drives_table(self.table)
        level = sum(drives.values()) / len(drives)
        for code in self.codes:
            self.assertAlmostEqual(self.t._pace(self.setup_of(code)), drives[code] / level, places=9)
        typed = dict(self.setup_of(self.codes[0]), match="", exp_drives=12.0)          # typed: against its league
        self.assertAlmostEqual(self.t._pace(typed), 12.0 / 10.0 * self.t.adj, places=9)
        self.assertIsNone(self.t._pace(dict(typed, exp_drives="")))                    # blank: no pace nudge

    def test_the_kick_off_nudged_and_carried_into_the_game(self):
        fast, slow = self.setup_of(self.codes[0]), self.setup_of(self.codes[2])
        r = self.call("/api/start", setup=fast)
        v = r["prematch"]["v21"]
        self.assertEqual(len(v["nudge"]), 2)
        self.assertGreater(v["goal_drives"], v["own_drives"])                           # 11.5 against a 10 league
        self.assertGreater(r["prematch"]["moneyline"]["home"]["p"], 0.6)               # 26 against 13
        self.assertGreater(r["prematch"]["expected"]["margin"], 5)
        self.assertGreater(self.call("/api/start", setup=slow)["prematch"]["v21"]["tempo"], v["tempo"])  # below 0: quicker
        k = self.t._kickoff(fast)
        self.assertIs(self.t._kickoff(fast), k)                                         # solved once a setup
        g = self.call("/api/play", setup=fast, game=r["game"], play={"type": "kickoff"}, kept=r["price"]["kept"])
        self.assertEqual(g["game"]["phase"], gm.SCRIMMAGE)
        S = self.t.start_of(gm.Game.from_dict(g["game"]), k["pair"], k["teams"])
        self.assertEqual((int(S["phase"][0]), int(S["team"][0]), int(S["y"][0])), (1, 0, 25))
        played = self.call("/api/play", setup=fast, game=g["game"], play={"type": "play", "yards": 4})
        self.assertEqual(played["game"]["snaps"], 1)
        self.assertEqual(played["game"]["last"], "short")
        self.assertEqual(int(self.t.start_of(gm.Game.from_dict(played["game"]), k["pair"], k["teams"])["prev"][0]),
                         __import__("eAMFModel.v21", fromlist=["v21"]).SHORT)
        nudged = dict(fast, means=[13.0, 26.0])
        self.assertLess(self.call("/api/price", setup=nudged, game=played["game"])["price"]["moneyline"]["home"]["p"],
                        played["price"]["moneyline"]["home"]["p"])

    def test_a_conversion_is_the_scorers_try(self):
        from eAMFModel import v21
        setup = self.setup_of(self.codes[1])
        r = self.call("/api/start", setup=setup)
        g = self.call("/api/play", setup=setup, game=r["game"], play={"type": "kickoff"})
        td = self.call("/api/play", setup=setup, game=g["game"], play={"type": "touchdown", "seconds": 8})
        self.assertEqual((td["game"]["phase"], td["game"]["home"]), (gm.CONVERSION, 6))
        k = self.t._kickoff(setup)
        S = self.t.start_of(gm.Game.from_dict(td["game"]), k["pair"], k["teams"])
        self.assertEqual((int(S["convert"][0]), int(S["team"][0]), int(S["phase"][0])), (0, 1, v21.KICK))
        last = dict(S, period=np.array([4]), clock=np.array([1.0]))                     # the try with a second left
        h, a = v21.simulate(self.model, last, 400, seed=1)
        self.assertTrue(np.all(h >= 6))
        self.assertGreater(np.mean(np.isin(h, (7, 8))), 0.8)                             # the try, then time
        gained = td["price"]["expected"]["home"] - g["price"]["expected"]["home"]     # the six and the try, less the
        self.assertTrue(3 < gained < 8, gained)                                          # drive's own expected points

    def test_overtime_waits_for_the_toss_then_carries_who_has_had_the_ball(self):
        setup = self.setup_of(self.codes[1])
        level = gm.Game(period=4, clock=5.0, home=10, away=10, phase=gm.SCRIMMAGE, side=gm.HOME, field=25,
                        fresh=False).to_dict()
        r = self.call("/api/play", setup=setup, game=level, play={"type": "kneel"})
        self.assertEqual(r["game"]["phase"], gm.TOSS)
        self.assertTrue(r["price"]["waiting"])                                          # no price at the toss
        self.assertNotIn("moneyline", r["price"])
        r = self.call("/api/play", setup=setup, game=r["game"], play={"type": "toss", "side": "away"})
        self.assertEqual((r["game"]["phase"], r["game"]["side"]), (gm.KICKOFF, gm.AWAY))
        self.assertIn("moneyline", r["price"])
        k = self.t._kickoff(setup)                                                      # away has had the ball,
        g = gm.Game(period=5, clock=200.0, home=10, away=10, phase=gm.SCRIMMAGE, side=gm.HOME, field=30,
                    ot_had_ball=[gm.AWAY])                                              # home has it now
        S = self.t.start_of(g, k["pair"], k["teams"])
        self.assertEqual(S["ot_done"].tolist(), [[False, True]])
        self.assertEqual(int(S["ot_cur"][0]), -1)

    def test_the_moneyline_typed_or_off_its_table(self):
        import os
        ml = os.path.join(self.tmp.name, "ml.csv")
        with open(ml, "w", newline="", encoding="utf-8") as fh:
            fh.write(f"MATCH_CODE,p_home\n{self.codes[1]},0.75\n")
        t = trader.V21Trader(self.tmp.name, paths=300, prior=self.table, ml_path=ml)
        m = t.meta()
        self.assertEqual(m["version"], "v21-drives-ml")
        self.assertTrue(m["ml"])
        self.assertIn("100% to the moneyline", m["how"])
        self.assertIn("moneyline on 1", m["prior"])
        r = trader.respond(t, "/api/prematch", {"setup": {"match": self.codes[1]}})[1]
        self.assertEqual(r["ml_home"], 0.75)
        setup = dict(r, match=self.codes[1], opening_receiver="home", margin=0.05)
        v = t.price(setup)["v21"]
        self.assertEqual(v["ml"]["goal"], 0.75)
        self.assertLess(abs(v["ml"]["got"] - 0.75), 0.06)
        k = t._kickoff(setup)
        self.assertGreater(k["alpha"][0], k["alpha"][1])
        self.assertLess(abs(sum(k["books"][1] * np.arange(len(k["books"][1]))) - 34.0), 2.0)     # the total kept
        self.assertNotIn("ml", t.price(dict(setup, ml_home=""))["v21"])                           # blank: none
        self.assertLess(t._kickoff(dict(setup, ml_home=0.4))["alpha"][0], k["alpha"][0])          # typed over
        g = gm.Game.start(gm.HOME)
        g, _ = gm.apply(g, {"type": "kickoff"})
        self.assertGreater(t.price(setup, g)["moneyline"]["home"]["p"],
                           t.price(dict(setup, ml_home=""), g)["moneyline"]["home"]["p"] + 0.05)   # in play: carried
        self.assertNotIn("ml_home", self.call("/api/prematch", setup={"match": self.codes[1]}))   # no table: typed
        self.assertTrue(self.t.meta()["ml"])
        self.assertIsNotNone(self.t._ml(dict(setup, ml_home=0.6)))
        self.assertIsNone(self.t._ml(dict(setup, ml_home=1.0)))

    def test_v21_is_read_off_the_build_and_named(self):
        self.assertTrue(trader.is_v21(self.tmp.name))
        self.assertTrue(trader.is_v21(None, "v21-drives"))
        self.assertFalse(trader.is_v21(None, "v14"))
        plain = trader.V21Trader(self.tmp.name, paths=200, prior=self.table, pace=False)
        self.assertEqual(plain.version, "v21")
        self.assertIsNone(plain.price(self.setup_of(self.codes[0]))["v21"]["tempo"])


class TestEveryVersion(unittest.TestCase):
    """A new version (copied from the last) must keep everything the trader prices with."""

    def test_every_version_offers_what_the_trader_prices_with(self):
        names = trader.versions()
        self.assertIn("v14", names)
        self.assertEqual(names, sorted(names, key=lambda n: int(n[1:])))
        for v in names:
            with self.subTest(version=v):
                model, stream = trader.load_version(v)
                for name in ("price_states", "price_kickoff", "key_line", "prior_theta"):
                    self.assertTrue(callable(getattr(model, name)))
                self.assertTrue(hasattr(stream.sim, "Tables"))
                variant = model.Variant(v)
                self.assertIsNotNone(variant)


if __name__ == "__main__":
    unittest.main()
