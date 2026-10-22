"""bet_losers reads prod's pre-match price as its OPENING quote when pre-game quotes carry a message
count (README: 'Prod's pre-game quotes carry a message count')."""
from eAMFCalibrator import bet_checks, bet_losers as bl, bet_moments, bets
from eAMFCalibrator.tests.test_bet_losers import SROWS, SCORES, SNAPS, INFO, POS, FINALS, ml, q

# pre-game quotes carry message 0 (below the first play, message 2); opening 0.40, closing 0.62
PROD = [q(50, -3600, 0.40, 0), q(51, -3600, 0.60, 0),
        q(50, -60, 0.62, 0), q(51, -60, 0.38, 0),
        q(54, -60, 0.50, 0, 44.5), q(55, -60, 0.50, 0, 44.5)]
for m, p in ((1, 0.62), (3, 0.62), (4, 0.70), (5, 0.80), (6, 0.80)):
    PROD += [q(50, 10 * m, p, m), q(51, 10 * m, 1 - p, m)]

times = bet_checks.message_times(PROD)
checks = bet_checks.Checks(bet_checks.build_feeds(SROWS), times, bet_checks.score_index(SCORES))
checks.timelines = bet_moments.build(SROWS, times)
tl = bets.timeline(PROD)
ctx = bl.Context(tl, checks, SNAPS, SROWS, INFO, POS, {"M1": {"slow": 0.55, "fast": 0.70}})
print("Context.pre_p1 (what ml-losers uses):", ctx.pre_p1("M1"))
print("closing, first play 2:              ", bl.prematch_p1(tl, "M1", first_play=2))
print("closing via kickoff:                ", bl.prematch_p1(tl, "M1", kickoff=bl.kickoff_time(checks, "M1")))
bs = [ml(35, selection=2, revenue=10.0)]
joined = bets.join(bs, {}, {}, tl, {}, None, FINALS, checks)
r = bl.bet_row(bs[0], joined[0], ctx)
print("bet on away: pre_p", r["pre_p"], "fav", r["fav"], "drift", round(r["drift"], 3),
      "elo_fast_gap", round(r["elo_fast_gap"], 3))
m = bl.match_row("M1", FINALS["M1"], ctx, [r], [])
print("match: pre_p1", m["pre_p1"], "fav_p", m["fav_p"], "winner_pre_p", m["winner_pre_p"],
      "winner_min_p (labelled 'winner's lowest p in play')", m["winner_min_p"])
