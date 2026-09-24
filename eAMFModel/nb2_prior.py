"""Our own pre-match model for v4: the NB2 player + team model in nb2/.

v4 takes each match's scoring level from here, not from GAMEPLAI:

  fit       nb2/NBRatingTrial.py on the match history before a cut-off --
            player and team attack/defence, stream effects, dispersion and
            score correlation (recency-weighted, 60-day half-life);
  predict   nb2/NB2_schedule_predict.py on the matches to price -- each
            side's expected points (and the rest of NB2's pre-match book);
  level     NB2's totals ran about 3% under the real ones, week after week
            (Aug 20-26, Aug 27-Sep 2, Sep 3-9: ratios 1.027, 1.028, 1.035,
            each fitted only on what came before). level_scale measures
            that on the last week before the cut-off, walk-forward, shrinks
            it toward 1 by SCALE_PRIOR matches, and v4 multiplies both
            sides' expected points by it.

The two scripts are run as they are (they need pandas and scipy), in a
working directory of their own, so nb2/ is never written to.

History rows are shaped like nb2/AMFELO.csv: MATCH_CODE, SPORT_CODE,
STREAM_NUMBER, SCHEDULED_START_TIME_UTC, PLAYER_1_HANDLE, PLAYER_1_TEAM,
PLAYER_2_HANDLE, PLAYER_2_TEAM, PLAYER_1_FINAL_SCORE, PLAYER_2_FINAL_SCORE
(`python -m eAMFCalibrator history` writes one from Snowflake). PLAYER_1
is the home side, as everywhere in the feed.
"""

import csv
import datetime as dt
import json
import os
import subprocess
import sys

NB2_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nb2")
FIT_SCRIPT = "NBRatingTrial.py"
PREDICT_SCRIPT = "NB2_schedule_predict.py"
RATINGS = {"leaderboard": "NB2_player_leaderboardsept_team_joint.csv",
           "teams": "NB2_team_ratingssept_joint.csv",
           "streams": "NB2_stream_effectssept_team_joint.csv",
           "joint": "NB2_joint_calibrationsept_team.csv"}
HISTORY_FIELDS = ["MATCH_CODE", "SPORT_CODE", "STREAM_NUMBER", "SCHEDULED_START_TIME_UTC",
                  "PLAYER_1_HANDLE", "PLAYER_1_TEAM", "PLAYER_2_HANDLE", "PLAYER_2_TEAM",
                  "PLAYER_1_FINAL_SCORE", "PLAYER_2_FINAL_SCORE"]
SCALE_DAYS = 7
SCALE_PRIOR = 200
N_SIMS = 20000


def _need_libraries():
    try:
        import pandas  # noqa: F401
        import scipy  # noqa: F401
    except ImportError:
        raise SystemExit("the NB2 pre-match model needs pandas and scipy:  "
                         "py -m pip install pandas scipy")


def load_history(path, sport="AF"):
    """AF rows of an AMFELO-shaped CSV."""
    with open(path, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("SPORT_CODE", sport) in (sport, "", None)]
    missing = [f for f in HISTORY_FIELDS if f != "SPORT_CODE" and rows and f not in rows[0]]
    if missing:
        raise SystemExit(f"{path} lacks {', '.join(missing)}")
    return rows


def _start(row):
    text = (row.get("SCHEDULED_START_TIME_UTC") or "")[:19].replace("T", " ")
    try:
        return dt.datetime.fromisoformat(text)
    except ValueError:
        return None


def _write(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _run(script, args, cwd):
    done = subprocess.run([sys.executable, os.path.join(NB2_DIR, script)] + args, cwd=cwd,
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    if done.returncode != 0:
        raise SystemExit(f"nb2/{script} failed:\n{done.stdout[-2000:]}\n{done.stderr[-2000:]}")


def fit(history, out_dir, before=None):
    """Fit NB2's ratings on the finished matches in `history` that started
    before `before` (a datetime; None: all). Writes RATINGS into out_dir.
    Returns the number of matches fitted on."""
    _need_libraries()
    os.makedirs(out_dir, exist_ok=True)
    rows = [r for r in history if r.get("PLAYER_1_FINAL_SCORE") not in ("", None)
            and (before is None or (_start(r) is not None and _start(r) < before))]
    if not rows:
        raise SystemExit("no finished matches to fit the NB2 model on")
    _write(os.path.join(out_dir, "history.csv"), rows, HISTORY_FIELDS)
    _run(FIT_SCRIPT, ["--input", "history.csv"], out_dir)
    return len(rows)


def predict(ratings_dir, schedule, n_sims=N_SIMS):
    """match -> (home points, away points) off fitted ratings, for schedule
    rows shaped like the history (finals not needed). Matches NB2 cannot
    price (an unknown player or team, a stream mismatch) are left out."""
    _need_libraries()
    if not schedule:
        return {}
    work = os.path.join(ratings_dir, "predict")
    os.makedirs(work, exist_ok=True)
    rows = [{"Player 1 Name": r["PLAYER_1_HANDLE"], "Player 1 Team": r["PLAYER_1_TEAM"],
             "Player 2 Name": r["PLAYER_2_HANDLE"], "Player 2 Team": r["PLAYER_2_TEAM"],
             "Match Id": r["MATCH_CODE"], "Stream": r.get("STREAM_NUMBER", "")} for r in schedule]
    _write(os.path.join(work, "schedule.csv"), rows, list(rows[0]))
    args = ["--schedule", "schedule.csv", "--stream-col", "Stream", "--n-sims", str(n_sims)]
    for key, name in RATINGS.items():
        args += [f"--{key}", os.path.join(os.path.abspath(ratings_dir), name)]
    _run(PREDICT_SCRIPT, args, work)
    out = {}
    with open(os.path.join(work, "NB2_joint_schedule_predictions.csv"), newline="",
              encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("Prediction_Status") == "OK":
                out[r["Match Id"]] = (float(r["Pred_P1_Points"]), float(r["Pred_P2_Points"]))
    return out


def level_scale(history, work_dir, before, days=SCALE_DAYS, prior_n=SCALE_PRIOR):
    """How far NB2's totals have run under the real ones: fitted on history
    before (before - days), scored on the `days` after, walk-forward.
    Returns (scale, matches, raw ratio); scale is the ratio shrunk toward 1."""
    start = before - dt.timedelta(days=days)
    week = [r for r in history if r.get("PLAYER_1_FINAL_SCORE") not in ("", None)
            and _start(r) is not None and start <= _start(r) < before]
    if not week:
        return 1.0, 0, None
    d = os.path.join(work_dir, "level")
    fit(history, d, before=start)
    pred = predict(d, week)
    real, expected = 0.0, 0.0
    n = 0
    for r in week:
        if r["MATCH_CODE"] in pred:
            real += float(r["PLAYER_1_FINAL_SCORE"]) + float(r["PLAYER_2_FINAL_SCORE"])
            expected += sum(pred[r["MATCH_CODE"]])
            n += 1
    if not n or expected <= 0:
        return 1.0, 0, None
    ratio = real / expected
    return 1.0 + (ratio - 1.0) * n / (n + prior_n), n, ratio


def league_average(history, before, days=30):
    """(home points, away points) over the last `days` before `before`: the
    prior for a match NB2 cannot price."""
    since = before - dt.timedelta(days=days)
    rows = [r for r in history if r.get("PLAYER_1_FINAL_SCORE") not in ("", None)
            and _start(r) is not None and since <= _start(r) < before]
    if not rows:
        return 17.5, 17.5
    home = sum(float(r["PLAYER_1_FINAL_SCORE"]) for r in rows) / len(rows)
    away = sum(float(r["PLAYER_2_FINAL_SCORE"]) for r in rows) / len(rows)
    return home, away


class Prematch:
    """A fitted NB2 pre-match model as v4 stores it: the ratings and
    level.json (scale, league average) in one directory."""

    def __init__(self, directory):
        self.directory = directory
        with open(os.path.join(directory, "level.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        self.scale = meta["scale"]
        self.league = tuple(meta["league_average"])
        self.meta = meta

    @classmethod
    def build(cls, history, directory, before):
        n = fit(history, directory, before=before)
        scale, n_scale, ratio = level_scale(history, directory, before)
        meta = {"fitted_on": n, "before": before.isoformat(), "scale": scale,
                "scale_matches": n_scale, "scale_raw_ratio": ratio,
                "league_average": list(league_average(history, before))}
        with open(os.path.join(directory, "level.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=1)
        return cls(directory)

    @staticmethod
    def exists(directory):
        return os.path.exists(os.path.join(directory, "level.json"))

    def means(self, schedule, n_sims=N_SIMS):
        """match -> (home, away) expected points, scaled; matches NB2 cannot
        price get the league average."""
        pred = predict(self.directory, schedule, n_sims)
        out = {}
        for r in schedule:
            code = r["MATCH_CODE"]
            if code in pred:
                out[code] = (pred[code][0] * self.scale, pred[code][1] * self.scale)
            else:
                out[code] = self.league           # already a real average: not scaled
        return out
