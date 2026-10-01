"""The points still to come, calibrated: the checkpoint states off the export, the fit, and its
probabilities."""

import random
import unittest

from .. import totals_calibrate as tc


def row(msg, period, clock, p1, p2, line=None, live=1, scrimmage=1, final=(30, 14), prematch=44.5):
    return dict(match_code="M1", message=str(msg), period=str(period), clock_seconds=str(clock),
                scrimmage=str(scrimmage), score_p1=str(p1), score_p2=str(p2),
                line_54="" if line is None else str(line), prob_54="0.48", live_54=str(live),
                final_p1=str(final[0]), final_p2=str(final[1]), prematch_line_54=str(prematch))


class TestStates(unittest.TestCase):

    def test_each_checkpoint_is_the_quarters_last_live_total(self):
        rows = [row(1, 1, 200, 0, 0, 44.5), row(2, 1, 10, 7, 0, 47.5),
                row(3, 1, 0, 7, 0, 46.5, live=0),            # suspended: not read
                row(4, 2, 100, 7, 3, 42.5, scrimmage=0),
                row(5, 2, 0, 14, 3, 40.5)]
        s = tc.match_states(rows)
        self.assertEqual(set(s), {"q1", "h1"})
        q1 = s["q1"]
        self.assertEqual((q1["line"], q1["points"], q1["needed"], q1["plays"]), (47.5, 7, 40.5, 2))
        self.assertEqual((q1["rest"], q1["margin"], q1["secs_left"]), (44 - 7, 7, 3 * 240 + 10))
        h1 = s["h1"]
        self.assertEqual((h1["points"], h1["plays"], h1["prematch"]), (17, 4, 44.5))

    def test_a_match_with_no_final_gives_nothing(self):
        r = row(1, 1, 0, 0, 0, 44.5)
        r["final_p1"] = ""
        self.assertEqual(tc.match_states([r]), {})


class TestFit(unittest.TestCase):

    def test_the_fit_recovers_a_shrunk_line_and_reads_its_own_residuals(self):
        rng = random.Random(1)
        ss = []
        for _ in range(400):
            needed = rng.uniform(10, 20)
            ss.append(dict(needed=needed, rest=8 + 0.5 * needed + rng.choice((-3, 0, 3))))
        fit = tc.Fit(ss, ("needed",))
        self.assertAlmostEqual(fit.beta[1], 0.5, delta=0.05)
        s = dict(needed=16.0)
        # rest = 16 +- 3 at needed 16: P(rest > 15.5) is the share of residuals above -0.5
        self.assertAlmostEqual(fit.p_over(s, 15.5), 2 / 3, delta=0.08)

    def test_held_out_matches_are_the_latest(self):
        ms = [(f"M{i}", f"2026-09-{i + 1:02d}", {}) for i in range(9)]
        train, test = tc.split(ms)
        self.assertEqual([m[0] for m in test], ["M6", "M7", "M8"])

    def test_the_signals_correction_runs_per_line(self):
        rng = random.Random(2)
        rows = []
        for i in range(200):
            pts = rng.choice((0, 7, 14, 21))
            need = 30 - pts / 3 + rng.choice((-1.5, 0, 1.5))
            final = pts + 20 + rng.choice((-5, 0, 5))
            rows.append(dict(match_code=f"M{i}", final_total=final, h1_points=pts, h1_line=pts + need,
                             h1_needed=need, h1_prob=0.5, h1_v8_needed=need - 2, h1_v8_pover=0.5,
                             q1_points=None))
        lines = tc.report_signals(rows)
        text = "\n".join(lines)
        self.assertIn("v8 corrected", text)
        self.assertIn("prod as it stands", text)
        for r in rows:                                  # the line's points to come follow the board
            r["h1_needed"] = 30 - r["h1_points"] / 3
        self.assertIn("prod corrected", "\n".join(tc.report_signals(rows)))


if __name__ == "__main__":
    unittest.main()
