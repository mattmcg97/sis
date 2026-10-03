import json
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer

from eAMFModel import game as gm, trader, v10
from eAMFModel.tests.fakes import _matches, _with_handles


class TestTrader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        v10.build(_with_handles(_matches(40)), cls.tmp.name, grid_paths=60, verbose=False)
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
            self.assertIn("<title>v10 Trader</title>", page)
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

    def test_odds_carry_the_margin(self):
        self.assertEqual(trader.odds(0.5, 0.0), 2.0)
        self.assertEqual(trader.odds(0.5, 0.05), round(1 / (0.5 * 1.05), 2))


if __name__ == "__main__":
    unittest.main()
