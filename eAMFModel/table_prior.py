"""A pre-match model read from a table: each match's expected points as a walkthrough priced them
(`python -m eAMFCalibrator walkthrough`, one CSV per variant in its priors/ directory), so a sim
can be pointed at any pre-match variant without pricing it again:

    python -m eAMFCalibrator report --candidate v21 --v21-prior-model eAMFCalibrator/out/walkthrough/priors/avg.csv
    python -m eAMFModel v21-report ... --prior-model eAMFCalibrator/out/walkthrough/priors/avg.csv

The table has MATCH_CODE and the two sides' expected points (home_pts / away_pts, or
Pred_P1_Points / Pred_P2_Points, or exp_p1 / exp_p2). Every value was priced before its match as
the walkthrough's variant would have priced it live, so the table doesn't follow results itself: a
match it doesn't have gets no expected points (the sim prices it on its own means).
"""

import csv
import os

POINT_COLUMNS = (("home_pts", "away_pts"), ("Pred_P1_Points", "Pred_P2_Points"), ("exp_p1", "exp_p2"))


def is_table(path):
    """Whether a prior given as a path is a table (a CSV file) rather than a build's directory."""
    return bool(path) and str(path).lower().endswith(".csv") and os.path.isfile(str(path))


class TablePrior:
    """Expected points a match, read off a table; the means() of nb2_prior.Prematch and the rest."""

    FOLLOWS_RESULTS = False          # priced once, as of each match's kick-off
    AS_OF = True

    def __init__(self, path):
        self.path = path
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            cols = reader.fieldnames or []
            pair = next((p for p in POINT_COLUMNS if p[0] in cols and p[1] in cols), None)
            key = "MATCH_CODE" if "MATCH_CODE" in cols else "Match Id" if "Match Id" in cols else None
            if pair is None or key is None:
                raise SystemExit(f"{path}: a prior table needs MATCH_CODE and "
                                 f"{' / '.join(f'{a}, {b}' for a, b in POINT_COLUMNS)}; it has {', '.join(cols)}")
            self.points = {}
            for r in reader:
                try:
                    self.points[r[key].strip()] = (float(r[pair[0]]), float(r[pair[1]]))
                except (TypeError, ValueError):
                    continue
        if not self.points:
            raise SystemExit(f"{path}: no expected points in the table")
        n = len(self.points)
        self.league = (sum(p[0] for p in self.points.values()) / n, sum(p[1] for p in self.points.values()) / n)
        self.meta = {"table": os.path.abspath(path), "matches": n}

    def means(self, schedule, n_sims=None, results=None):
        """{match: (home, away) expected points} for the schedule's matches the table has."""
        out = {}
        for r in schedule:
            code = str(r["MATCH_CODE"]).strip()
            if code in self.points:
                out[r["MATCH_CODE"]] = self.points[code]
        return out

    def describe(self):
        return f"the table {self.path} ({len(self.points):,} matches)"
