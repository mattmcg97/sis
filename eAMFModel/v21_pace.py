"""v2.1's pace when ahead: each gamer's own clock use while its side is ahead.

A tilt build's time draws are the league's real snaps for the state -- the lead among it, so the
league's slowing down ahead is there already -- reweighted by exp(theta x seconds / 10), theta the
summed tilts of the offense's gamer and team and the defense's: one pace a gamer, ahead or not.
The playstyle report found more: a gamer's slowing down ahead beyond the league's repeats from one
half of its matches to the other (r = +0.58 [+0.40, +0.73], Aug 23 - Oct 17), OFFICER some 5s a snap
slower than the league ahead, STRIDER some 6s quicker.

This adds a second tilt a gamer, used on its side's snaps while the side is PACE_LEAD or more ahead:
fitted as the build fits its own, by Newton steps on the penalized log-likelihood of the gamer's real
snaps ahead, against the league's draws for their states with the build's tilts on them, shrunk
toward none as if it had the build's TILT_PRIOR league-average snaps.

    python -m eAMFModel v21-pace-fit --model v21_t10 --until 2026-10-01

fits it on the matches before --until and scores it on the ones after (to --test-until): each ahead
snap's log-likelihood with the gamer's tilt against without it, a 95% interval over matches, and
whether the seconds it says a gamer gains ahead are the ones it gains on the later matches. It writes
pace_ahead.json beside the build; the calibrator's candidate v21-pace prices with it (v21 itself: as
before, the same draws). It reads the export alone: no Snowflake, no simulation."""

import datetime as dt
import json
import os

import numpy as np

from . import v21

PACE_FILE = "pace_ahead.json"
PACE_LEAD = v21.PACE_LEAD     # the side with the ball this many points or more ahead
BOOT = 1000


def real_snaps(model, data, hist, codes):
    """The real snaps of these matches as the build reads them (v21.match_snaps, overtime's last aside)."""
    recs = []
    for c in codes:
        rows = v21._with_side(data[c])
        pair, teams = v21.match_people(c, rows, hist)
        for r in v21.match_snaps(rows, pair, teams):
            if r["period"] <= 4 + v21.MAX_OT:
                r["code"] = c
                recs.append(r)
    return recs


def time_frame(model, recs):
    """The time draws' frame for the snaps a time tilt is fitted on (the uncut ones): the league's
    draw a state gives each time bin with the build's tilts on (H), the bins the real snaps fell in,
    each bin's seconds / 10 (phi), the side with the ball's lead, gamer and match, and the snaps'
    recency weights."""
    if model.mode != "tilt" or "time" not in (getattr(model, "etilt", None) or {}):
        raise SystemExit("v21-pace-fit needs a tilt build (v21-build --mode tilt)")
    timed = ~np.array([r["cut"] for r in recs], bool)
    sub = [r for r, k in zip(recs, timed) if k]
    feats = v21._features(sub, model)
    who, _ = v21._tilt_who(model, sub)
    pt = model.time.payload
    tbin = np.where(pt["cut"], -1, v21.time_bin(pt["seconds"]))
    H = v21.tilt_bases(model.time, feats, tbin, v21.TIME_BINS, getattr(model, "blend", 0.0))
    keep = tbin >= 0
    cnt = np.bincount(tbin[keep], minlength=v21.TIME_BINS)
    phi = (np.bincount(tbin[keep], pt["seconds"][keep].astype(float), minlength=v21.TIME_BINS)
           / np.maximum(cnt, 1) / 10.0)[:, None]
    th = sum(model.etilt["time"][k][who[k]] for k in who)                     # (snaps, 1)
    H = H * np.exp(th @ phi.T)
    seconds = np.array([min(v21.MAX_SECONDS, r["seconds"]) for r in sub], float)
    return dict(H=H, obs=v21.time_bin(seconds), phi=phi, gamer=who["gamer_off"],
                lead=np.array([r["margin"] for r in sub], float), code=np.array([r["code"] for r in sub]),
                weights=v21.recency_weights(sub))


def _probs(H, phi, delta):
    """(snaps, bins) each snap's chance of each time bin under its extra tilt delta (snaps, 1)."""
    z = np.log(np.maximum(H, 1e-300)) + delta @ phi.T
    z -= z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def fit(model, frame, prior=None):
    """(gamers + 1, 1) each gamer's extra time tilt ahead -- the league's part (what the build's draws
    for the state miss of every side's clock ahead) and the gamer's own beyond it, the gamer the build has
    not met (last) the league's alone -- the in-sample log-likelihood it gains an ahead snap, each
    gamer's ahead snaps and the league's part."""
    up = frame["lead"] >= PACE_LEAD
    prior = getattr(model, "tilt_prior", v21.TILT_PRIOR) if prior is None else prior
    n = len(model.gamers) + 1
    who = {"league": np.zeros(int(up.sum()), np.int64), "gamer": frame["gamer"][up]}
    theta, _ = v21.fit_exp_tilts(frame["H"][up], frame["obs"][up], frame["phi"], who, {"league": 2, "gamer": n},
                                 prior=prior, weights=frame["weights"][up])
    league = float(theta["league"][0, 0])
    delta = theta["gamer"] + league
    delta[-1] = league                                 # a gamer the build has not met: the league's alone
    got = scored(frame, delta)
    gain = float(np.average(got["ll"], weights=frame["weights"][up])) if len(got["ll"]) else 0.0
    return delta, gain, np.bincount(frame["gamer"][up], minlength=n), league


def scored(frame, delta):
    """What the tilts say on these snaps (its later matches): each ahead snap's log-likelihood with its
    gamer's tilt less without, and the seconds it says the snap takes more (both by snap), with the
    seconds the snap took beyond the build's draw."""
    up = frame["lead"] >= PACE_LEAD
    H, phi, obs = frame["H"][up], frame["phi"], frame["obs"][up]
    d = delta[frame["gamer"][up]]
    p0, p1 = _probs(H, phi, np.zeros_like(d)), _probs(H, phi, d)
    i = np.arange(len(obs))
    ll = np.log(np.maximum(p1[i, obs], 1e-300)) - np.log(np.maximum(p0[i, obs], 1e-300))
    sec = 10.0 * phi[:, 0]
    said = p1 @ sec - p0 @ sec
    took = sec[obs] - p0 @ sec
    return dict(ll=ll, said=said, took=took, gamer=frame["gamer"][up], code=frame["code"][up])


def _boot_mean(values, groups, B=BOOT, seed=0):
    """(mean, lo, hi): the mean a snap with a 95% interval over its groups (matches)."""
    u, gi = np.unique(groups, return_inverse=True)
    s = np.bincount(gi, weights=values, minlength=len(u))
    c = np.bincount(gi, minlength=len(u)).astype(float)
    W = np.random.default_rng(seed).multinomial(len(u), [1.0 / len(u)] * len(u), size=B).astype(float)
    d = (W @ s) / np.maximum(W @ c, 1e-9)
    return float(s.sum() / c.sum()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def _date(code, hist_row):
    from .playstyle import _date as d
    return d(code, hist_row)


def run(model_file, data, history, until, test_until=None, prior=None, min_snaps=30):
    """(the report's lines, the table for pace_ahead.json): the tilts fitted on the matches before
    `until`, scored on the ones from it to `test_until` (the export's last with None)."""
    model = v21.Model.load(model_file)
    hist = {h["MATCH_CODE"]: h for h in (history or [])}
    when = {c: _date(c, hist.get(c, {})) for c in data}
    fit_codes = sorted(c for c, d in when.items() if d is not None and d < until)
    test_codes = sorted(c for c, d in when.items() if d is not None and d >= until and (test_until is None or d < test_until))
    if not fit_codes:
        raise SystemExit(f"v21-pace-fit: no matches before {until}")
    frame = time_frame(model, real_snaps(model, data, hist, fit_codes))
    delta, gain, n_up, league = fit(model, frame, prior)
    names = {i: g for g, i in model.gamers.items()}
    fitted = scored(frame, delta)
    shift = {}
    for g in range(len(names)):
        k = fitted["gamer"] == g
        if k.any():
            shift[g] = float(fitted["said"][k].mean())
    L = [f"  pace when ahead: each gamer's own time tilt while its side is {PACE_LEAD}+ ahead, beyond the league's draws"
         f" for the state and the build's own tilts ({os.path.basename(os.path.dirname(model_file)) or model_file})",
         f"  fitted on {len(fit_codes):,} matches before {until}: {int((frame['lead'] >= PACE_LEAD).sum()):,} snaps ahead,"
         f" {sum(1 for g in shift if n_up[g] >= min_snaps):,} gamers with {min_snaps}+ of them; in-sample log-likelihood"
         f" gained an ahead snap {gain:+.4f}",
         f"  the league's part (what the build's draws miss of every side's clock ahead) {league:+.3f}; the gamers' own"
         f" beyond it: sd {np.std([delta[g, 0] - league for g in shift if n_up[g] >= min_snaps] or [0]):.3f} (shrunk like the"
         f" build's own time tilts, by {getattr(model, 'tilt_prior', v21.TILT_PRIOR):g} snaps)"]
    table = dict(until=str(until), lead=PACE_LEAD, fit_matches=len(fit_codes), gain=gain, league=league,
                 gamers={names[g]: float(delta[g, 0]) for g in shift}, seconds={names[g]: shift[g] for g in shift},
                 snaps={names[g]: int(n_up[g]) for g in shift})
    if test_codes:
        test = scored(time_frame(model, real_snaps(model, data, hist, test_codes)), delta)
        ll, lo, hi = _boot_mean(test["ll"], test["code"])
        per = []
        for g in sorted(set(test["gamer"].tolist())):
            k = test["gamer"] == g
            if k.sum() >= min_snaps and g in shift:
                per.append((shift[g], float(test["took"][k].mean()), int(k.sum())))
        r = float(np.corrcoef([a for a, _, _ in per], [b for _, b, _ in per])[0, 1]) if len(per) >= 5 else float("nan")
        slope = (float(np.polyfit([a for a, _, _ in per], [b for _, b, _ in per], 1)[0]) if len(per) >= 5
                 else float("nan"))
        L += [f"  on the {len(test_codes):,} matches after ({until} to {test_until or 'the export'}'s end), {len(test['ll']):,}"
              f" snaps ahead: log-likelihood gained an ahead snap {ll:+.5f} [{lo:+.5f}, {hi:+.5f}] (above 0: the tilts"
              " knew the gamers' clock better than the build)",
              f"  the seconds a gamer's tilt says it takes ahead beyond the build, against the seconds it took beyond the"
              f" build on those matches: {len(per)} gamers with {min_snaps}+ snaps ahead, r = {r:+.2f}, slope {slope:+.2f}"
              " (1: right; under 1: the tilts go too far)"]
        table.update(test_matches=len(test_codes), test_gain=dict(mean=ll, lo=lo, hi=hi), test_r=r, test_slope=slope)
    order = sorted((g for g in shift if n_up[g] >= min_snaps), key=lambda g: -shift[g])
    L.append(f"  the gamers ({min_snaps}+ snaps ahead), slowest ahead first: seconds a snap beyond the build, tilt, snaps ahead")
    for g in order:
        L.append(f"    {names[g][:14]:14s} {shift[g]:+6.2f}s  {delta[g, 0]:+.3f}  {n_up[g]:6,d}")
    return L, table


def write(path, table):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(table, fh, indent=1, sort_keys=True)
    return path


def load(model_dir):
    """{handle: tilt} beside a build (PACE_FILE), the league's part under "" (a gamer it lacks), or None."""
    path = os.path.join(model_dir if os.path.isdir(model_dir) else os.path.dirname(model_dir), PACE_FILE)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        got = json.load(fh)
    table = {k.strip().upper(): float(v) for k, v in got["gamers"].items()}
    table[""] = float(got.get("league", 0.0))
    return table


def side_pace(table, pair):
    """(2,) each side's extra time tilt ahead (the league's part for a gamer the table lacks)."""
    t = table or {}
    return np.array([float(t.get((g or "").strip().upper(), t.get("", 0.0))) for g in pair])


def parse_day(text):
    return dt.date.fromisoformat(text) if text else None
