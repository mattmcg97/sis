"""bet_inference.match_row: a drawn last other game is margin 0, the same as fresh -> 'level'."""
import datetime as dt
from eAMFCalibrator import bet_inference as bi, bet_rematches as br, bets
T0 = dt.datetime(2026, 10, 1, 18, 0)
def game(code, minutes, home, away, s1, s2):
    return dict(MATCH_CODE=code, SCHEDULED_START_TIME_UTC=str(T0 + dt.timedelta(minutes=minutes)),
                PLAYER_1_HANDLE=home, PLAYER_2_HANDLE=away, PLAYER_1_TEAM="X", PLAYER_2_TEAM="Y",
                PLAYER_1_FINAL_SCORE=str(s1), PLAYER_2_FINAL_SCORE=str(s2))
# ANN drew with BOB; CAT fresh; then ANN v CAT.  DAN drew, EVE drew; then DAN v EVE.
h = [game("G1", 0, "ANN", "BOB", 14, 14), game("G2", 60, "ANN", "CAT", 21, 7),
     game("G3", 0, "DAN", "FAY", 10, 10), game("G4", 0, "EVE", "GUS", 3, 3), game("G5", 60, "DAN", "EVE", 7, 0)]
meets = br.day_meetings(h); games = bi.session_games(meets)
for g in ("G2", "G5"):
    r = bi.match_row(meets[g], [], {}, games)
    print(g, r["home_last"], r["away_last"], "->", r["state"], r["form"])
print("states that can never occur:", [s for s in bi.STATES if s in ("drew v fresh", "fresh v drew", "drew v drew")])
