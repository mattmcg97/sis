"""A pre-match model refitted every day, as it would be run live: one fit per match day, each on
every result settled before that day began, and each match priced by its own day's fit.

A version's build fits its pre-match model (NB2 or glmer) once, at the build's cut-off, so over a
window of weeks its ratings go stale where a live model's would not. `python -m eAMFModel
prior-daily` fits the days of a window into one directory and attaches it to one or more builds;
their prematch_model() then prices from it (the build's prior shrink is applied on top, as before).

    DIR/rolling.json            {"prior": "glmer", "days": ["2026-08-23", ...]}
    DIR/2026-08-23/             a fitted pre-match model, cut off at 2026-08-23 00:00
"""

import datetime as dt
import inspect
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

from . import glmer_prior, nb2_prior

PRIORS = {"nb2": nb2_prior.Prematch, "glmer": glmer_prior.Prematch}
INDEX = "rolling.json"
WORKERS = 2                       # fits run side by side (each glmer fit is its own R process)


def _day(row):
    """A schedule row's match day, or None."""
    start = nb2_prior._start(row)
    return start.date() if start is not None else None


def fit(history, prior, out_dir, since, until, workers=WORKERS, verbose=True):
    """Fit `prior` once for every day from `since` to `until` (dates), each on the finished matches
    before that day began; days already fitted in `out_dir` are kept. The days are split into
    `workers` runs of consecutive days fitted side by side; within a run each glmer fit warm-starts
    from the day before's (glmer/fit.R --start), about 2-3x quicker for the same fit. Returns the
    Rolling model."""
    os.makedirs(out_dir, exist_ok=True)
    days = [since + dt.timedelta(days=k) for k in range((until - since).days + 1)]
    model = PRIORS[prior]
    warm = "start" in inspect.signature(model.build).parameters
    n = max(1, min(workers, len(days)))
    size = -(-len(days) // n)
    runs = [days[k:k + size] for k in range(0, len(days), size)]

    def run(chunk):
        prev = None
        for day in chunk:
            path = os.path.join(out_dir, day.isoformat())
            if not model.exists(path):
                t0 = time.time()
                kw = {"start": prev} if warm and prev else {}
                model.build(history, path, dt.datetime.combine(day, dt.time()), **kw)
                if verbose:
                    print(f"  {prior} fitted before {day} ({time.time() - t0:.0f}s"
                          f"{', warm' if kw else ''})", flush=True)
            prev = path
        return chunk

    with ThreadPoolExecutor(len(runs)) as pool:
        list(pool.map(run, runs))
    with open(os.path.join(out_dir, INDEX), "w", encoding="utf-8") as fh:
        json.dump({"prior": prior, "days": [d.isoformat() for d in days]}, fh, indent=1)
    return Rolling(out_dir)


def attach(model_dir, rolling_dir):
    """Point a build at a rolling prior: its prior file (vNprior.json) records the directory. The
    build's own prior must be the same kind, since its shrink was fitted for that kind."""
    names = [f for f in os.listdir(model_dir) if f.startswith("v") and f.endswith("prior.json")]
    if len(names) != 1:
        raise SystemExit(f"{model_dir}: not a version's build (no single v*prior.json)")
    path = os.path.join(model_dir, names[0])
    with open(path, encoding="utf-8") as fh:
        meta = json.load(fh)
    with open(os.path.join(rolling_dir, INDEX), encoding="utf-8") as fh:
        kind = json.load(fh)["prior"]
    if meta.get("prior", "nb2") != kind:
        raise SystemExit(f"{model_dir} was built with --prior {meta.get('prior', 'nb2')}; this rolling"
                         f" prior is {kind}: rebuild with --prior {kind} so its shrink fits it")
    meta["rolling"] = os.path.abspath(rolling_dir)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)


def detach(model_dir):
    """Undo attach: the build prices from its own single fit again."""
    for f in os.listdir(model_dir):
        if f.startswith("v") and f.endswith("prior.json"):
            path = os.path.join(model_dir, f)
            with open(path, encoding="utf-8") as fh:
                meta = json.load(fh)
            meta.pop("rolling", None)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(meta, fh, indent=1)


class Rolling:
    """The fits of a rolling prior, used as one pre-match model: each match by the fit of its own
    day (the latest fitted day not after it; the first for a match before them all)."""

    def __init__(self, directory):
        self.directory = directory
        with open(os.path.join(directory, INDEX), encoding="utf-8") as fh:
            index = json.load(fh)
        self.kind = index["prior"]
        self.days = sorted(dt.date.fromisoformat(d) for d in index["days"])
        if not self.days:
            raise SystemExit(f"{directory}: a rolling prior with no days fitted")
        self._fits = {}
        first = self.fit_for(self.days[0])
        self.league = first.league
        self.FOLLOWS_RESULTS = getattr(first, "FOLLOWS_RESULTS", False)

    @property
    def AS_OF(self):
        return getattr(self.fit_for(self.days[0]), "AS_OF", True)

    def fit_for(self, day):
        """The fit that prices a match on `day`."""
        pick = max((d for d in self.days if d <= day), default=self.days[0]) if day else self.days[-1]
        if pick not in self._fits:
            self._fits[pick] = PRIORS[self.kind](os.path.join(self.directory, pick.isoformat()))
        return self._fits[pick]

    def describe(self):
        return (f"{self.kind} refitted every day, {len(self.days)} fits from {self.days[0]} to "
                f"{self.days[-1]} ({self.directory})")

    def means(self, schedule, n_sims=None, results=None):
        """Each match's expected (home, away) points from its own day's fit."""
        groups = {}
        for r in schedule:
            fit_ = self.fit_for(_day(r))
            groups.setdefault(id(fit_), (fit_, []))[1].append(r)
        out = {}
        for fit_, rows in groups.values():
            kw = {"results": results} if getattr(fit_, "FOLLOWS_RESULTS", False) else {}
            if n_sims is not None:
                kw["n_sims"] = n_sims
            out.update(fit_.means(rows, **kw))
        return out
