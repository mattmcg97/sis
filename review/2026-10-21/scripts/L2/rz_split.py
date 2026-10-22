from eAMFCalibrator import player_stats as ps
def s(kind, off, down, field, msgs="", clock=200):
    return dict(play_kind=kind, offense=off, down=down, field_position=field, period=1,
                play_messages=msgs, clock_seconds=clock)
def run(missing):
    snaps = [s("KICKOFF", "TEAM_A", 1, 70, clock=230),
             s("SCRIMMAGE", "TEAM_A", 1, 85, clock=220),
             s("SCRIMMAGE", "TEAM_A", 2, None if missing else 88, clock=210),   # spot missing on this one
             s("SCRIMMAGE", "TEAM_A", 3, 92, clock=200),
             s("TOUCHDOWN", "TEAM_A", None, None, "TOUCHDOWN_TEAM_A", clock=190)]
    c = ps.play_stats(snaps, "home")["home"]
    return {k: c[k] for k in ("snaps", "detail_snaps", "rz_trips", "rz_td", "third_att", "third_conv", "rz_series")}
print("all spots     :", run(False))
print("one spot gone :", run(True))
def third(missing):
    snaps = [s("KICKOFF", "TEAM_A", 1, 30, clock=230),
             s("SCRIMMAGE", "TEAM_A", 3, 40, clock=220),                         # 3rd and short
             s("SCRIMMAGE", "TEAM_A", 1, None if missing else 52, clock=210),     # made it: 1st down
             s("SCRIMMAGE", "TEAM_A", 2, 55, clock=200)]
    c = ps.play_stats(snaps, "home")["home"]
    return {k: c[k] for k in ("third_att", "third_conv", "turnovers")}
print("3rd down made, next spot known  :", third(False))
print("3rd down made, next spot missing:", third(True))
