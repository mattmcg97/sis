"""drives.build_drives has no half-time rule: the same side ending Q2 with the ball and receiving Q3."""
from eAMFCalibrator.drives import PlayRow, build_drives, classify_plays
H = "Home Team"
def show(name, plays, scores=()):
    ds = build_drives("M", plays, list(scores))
    r = classify_plays(plays, list(scores))
    print(name, "->", len(ds), "drives:", [(d.drive_number, d.period_number, [p.event_message_count for p in d.plays]) for d in ds])
    print("   reasons:", {k: v for k, v in sorted(r.items())})

# 1. no kick-spot row at half time, Q3 start further back than Q2's last row
show("no kick spot, start behind", [PlayRow(10, 2, H, 1, 10, 30), PlayRow(12, 2, H, 2, 7, 33),
                                    PlayRow(20, 3, H, 1, 10, 25), PlayRow(22, 3, H, 2, 5, 30)])
# 2. no kick-spot row, Q3 start further up the field than Q2's last row
show("no kick spot, start ahead ", [PlayRow(10, 2, H, 1, 10, 20), PlayRow(12, 2, H, 2, 7, 23),
                                    PlayRow(20, 3, H, 1, 10, 25), PlayRow(22, 3, H, 2, 5, 30)])
# 3. with the kick spot row
show("with kick spot            ", [PlayRow(10, 2, H, 1, 10, 30), PlayRow(12, 2, H, 2, 7, 33),
                                    PlayRow(19, 3, H, 1, 10, 35),
                                    PlayRow(20, 3, H, 1, 10, 25), PlayRow(22, 3, H, 2, 5, 30)])
