"""The glmer pre-match model (glmer/, in R) as a version's prior, in place of NB2 (nb2_prior.py):
fits the global mixed model on match results and predicts each side's expected points.

The model itself -- which features, which row weighting, the form half-life -- is whatever
glmer/fit.R's defaults and glmer/config.R say (see glmer/README.md); this only runs it. It needs
R 4.x with lme4: `Rscript glmer/install_packages.R`."""

import csv
import glob
import json
import os
import re
import shutil
import subprocess
import sys

from . import nb2_prior

NAME = "glmer"
GLMER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "glmer")
FIT_SCRIPT = "fit.R"
PREDICT_SCRIPT = "predict.R"
MODEL = "model.rds"
INFO = "model_info.json"
META = "prior.json"
HISTORY = "history.csv"
RSCRIPT = None                       # the Rscript to run; None: $RSCRIPT, the PATH, Program Files


def rscript():
    """The Rscript executable: RSCRIPT, else $RSCRIPT, else the one on the PATH, else the newest
    R under Program Files (R's Windows installer doesn't put it on the PATH)."""
    path = RSCRIPT or os.environ.get("RSCRIPT") or shutil.which("Rscript")
    if not path and sys.platform.startswith("win"):
        root = os.environ.get("ProgramFiles", r"C:\Program Files")
        found = glob.glob(os.path.join(root, "R", "R-*", "bin", "Rscript.exe"))
        version = lambda p: [int(x) for x in re.findall(r"\d+", p.split(os.sep)[-3])]
        path = max(found, key=version) if found else None
    if not path:
        raise SystemExit("the glmer pre-match model needs R: install R 4.x, run "
                         "`Rscript glmer/install_packages.R`, and put Rscript on the PATH "
                         "(or set RSCRIPT to its full path)")
    return path


def _run(script, args):
    """Run one of the glmer/ scripts."""
    done = subprocess.run([rscript(), os.path.join(GLMER_DIR, script)] + args,
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    if done.returncode != 0:
        raise SystemExit(f"glmer/{script} failed:\n{done.stdout[-2000:]}\n{done.stderr[-2000:]}")


def fit(history, out_dir, before):
    """Fit the glmer model on the finished matches before a cut-off."""
    os.makedirs(out_dir, exist_ok=True)
    rows = [r for r in history if r.get("PLAYER_1_FINAL_SCORE") not in ("", None)
            and nb2_prior._start(r) is not None and nb2_prior._start(r) < before]
    if not rows:
        raise SystemExit("no finished matches to fit the glmer model on")
    path = os.path.abspath(os.path.join(out_dir, HISTORY))
    nb2_prior._write(path, rows, nb2_prior.HISTORY_FIELDS)
    _run(FIT_SCRIPT, [f"--history={path}", f"--before={before:%Y-%m-%d %H:%M:%S}",
                      f"--out={os.path.abspath(out_dir)}"])
    return len(rows)


def predict(model_dir, schedule):
    """Each match's expected (home, away) points from a fitted model. The form features look back
    over the model's own history and any finals the schedule rows carry, each match seeing only
    matches that started before it."""
    if not schedule:
        return {}
    model_dir = os.path.abspath(model_dir)
    work = os.path.join(model_dir, "predict")
    os.makedirs(work, exist_ok=True)
    sched = os.path.join(work, "schedule.csv")
    out = os.path.join(work, "predictions.csv")
    nb2_prior._write(sched, schedule, nb2_prior.HISTORY_FIELDS)
    _run(PREDICT_SCRIPT, [f"--model={os.path.join(model_dir, MODEL)}", f"--schedule={sched}",
                          f"--history={os.path.join(model_dir, HISTORY)}", "--n-sims=0",
                          f"--out={out}"])
    got = {}
    with open(out, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("Prediction_Status") == "OK":
                got[r["MATCH_CODE"]] = (float(r["Pred_P1_Points"]), float(r["Pred_P2_Points"]))
    return got


class Prematch:
    """A fitted glmer model saved in one directory, used as nb2_prior.Prematch is."""

    def __init__(self, directory):
        """Load what the build saved: the cut-off, the league average and the model's settings."""
        self.directory = directory
        with open(os.path.join(directory, META), encoding="utf-8") as fh:
            meta = json.load(fh)
        self.scale = meta["scale"]
        self.league = tuple(meta["league_average"])
        self.meta = meta

    @classmethod
    def build(cls, history, directory, before):
        """Fit the model before a cut-off and save it. Its expected points are not rescaled: the
        form features follow the league's level (NB2's level scale corrects a lag NB2 has)."""
        n = fit(history, directory, before)
        with open(os.path.join(directory, INFO), encoding="utf-8") as fh:
            info = json.load(fh)
        meta = {"prior": NAME, "fitted_on": n, "before": before.isoformat(), "scale": 1.0,
                "scale_matches": 0, "scale_raw_ratio": None,
                "league_average": list(nb2_prior.league_average(history, before)), "model": info}
        with open(os.path.join(directory, META), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=1)
        return cls(directory)

    @staticmethod
    def exists(directory):
        """Whether a fitted model is saved in this directory."""
        return all(os.path.exists(os.path.join(directory, f)) for f in (META, MODEL, HISTORY))

    def describe(self):
        """One line on what was fitted, for a build's log."""
        m = self.meta.get("model", {})
        scalar = m.get("scalar", 1)
        weighting = f"{m.get('weighting', '?')}" + (f" x{scalar:g}" if scalar != 1 else "")
        return (f"glmer {m.get('feature_set', '?')} / {weighting} (form half-life "
                f"{m.get('form_half_life', '?')} matches, {m.get('mode', '?')} model) fitted on "
                f"{self.meta['fitted_on']:,} matches before {self.meta['before'][:10]}; expected "
                "points not rescaled")

    def means(self, schedule, n_sims=None):
        """Each match's expected (home, away) points; the league average for any the model can't
        price. `n_sims` is unused: the expected points are exact, not simulated."""
        pred = predict(self.directory, schedule)
        return {r["MATCH_CODE"]: pred.get(r["MATCH_CODE"], self.league) for r in schedule}
