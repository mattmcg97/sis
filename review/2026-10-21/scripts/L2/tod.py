from eAMFCalibrator import player_stats as ps
# TEAM_A (home) 4th and 5 at its 40, turned over on downs; next snapshot has TEAM_B with no spot
def s(kind, off, down, field, msgs="", period=1, clock=200):
    return dict(play_kind=kind, offense=off, down=down, field_position=field, period=period,
                play_messages=msgs, clock_seconds=clock)
snaps = [s("KICKOFF", "TEAM_A", 1, 25),
         s("SCRIMMAGE", "TEAM_A", 4, 5 and 40 or None, clock=190),
         s("TURNOVER_ON_DOWNS", "TEAM_B", None, None, "TURNOVER_ON_DOWNS_TEAM_A", clock=180),
         s("SCRIMMAGE", "TEAM_B", 1, 62, clock=170)]
# fix: snapshot 1 is a 4th down at the 40
snaps[1]["down"] = 4
c = ps.play_stats(snaps, "home")
print("home tod", c["home"]["tod"], "turnovers", c["home"]["turnovers"])
# with spot known on the next snapshot
snaps[2]["field_position"] = 60; snaps[2]["down"] = 1
c = ps.play_stats(snaps, "home")
print("with spot: home tod", c["home"]["tod"], "turnovers", c["home"]["turnovers"])
