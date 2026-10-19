"""v21-gamers: how far v2.1's own draws set each gamer apart.

Every gamer the build knows plays an average opponent (the unknown gamer, the unknown teams) from
kick-off, home and away, with no nudge: the points the sim gives the gamer, the points it lets in,
the margin and the win chance against that average. Beside it, what the gamer really scored a match
in the build's history (against the league's) and the gamer's offense and defense tilts read along the
strength direction, a unit a point a game: off, the points the gamer's offense adds; def, the points
the gamer's defense lets the other side add (below zero: it holds them). The spread of the sim's margins across gamers against the spread of their
real ones says how much of what tells gamers apart the draws carry; the rest, pre-match, is the
prior's, whose expected points the kick-off nudge moves each side NUDGE_SHARE of the way toward.

    python -m eAMFModel v21-gamers --model v21_t10 --gamers KILLJOY,MERLIN,HERMES"""

import csv
import os

import numpy as np

from . import v21

FIELDS = ("gamer", "matches", "real_points", "sim_for", "sim_against", "sim_margin", "sim_win", "off_strength",
          "def_strength")


def strength_along(model, theta):
    """A tilt read along the strength direction, in points a game (the nudge's unit)."""
    d = np.asarray(getattr(model, "nudge_dir", None) if getattr(model, "nudge_dir", None) is not None else [], float)
    if not len(d) or not np.any(d):
        return float("nan")
    return float(np.asarray(theta, float) @ d / (d @ d))


def run(model, n_paths=1000, seed=0):
    """[dict] a row per gamer (FIELDS), and the summary lines."""
    names = sorted(model.gamers, key=model.gamers.get)
    pts = getattr(model, "_gamer_points", {}) or {}
    league = np.mean([p for v in pts.values() for p in v]) if pts else float("nan")
    starts = []
    for g in names:
        starts.append(v21.kickoff_start(model, (g, ""), ("", "")))     # the gamer at home
        starts.append(v21.kickoff_start(model, ("", g), ("", "")))     # and away
    S = v21.concat_starts(starts)
    h, a = v21.simulate(model, S, n_paths, seed=seed)
    et = getattr(model, "etilt", {}).get("dist", {})
    rows = []
    for k, g in enumerate(names):
        i = model.gamers[g]
        mine = np.concatenate([h[4 * k:4 * k + 2].ravel(), a[4 * k + 2:4 * k + 4].ravel()])
        theirs = np.concatenate([a[4 * k:4 * k + 2].ravel(), h[4 * k + 2:4 * k + 4].ravel()])
        own = pts.get(i, [])
        rows.append(dict(gamer=g, matches=len(own), real_points=float(np.mean(own) - league) if own else float("nan"),
                         sim_for=float(mine.mean()), sim_against=float(theirs.mean()),
                         sim_margin=float((mine - theirs).mean()),
                         sim_win=float(np.mean(mine > theirs) + 0.5 * np.mean(mine == theirs)),
                         off_strength=strength_along(model, et["gamer_off"][i]) if "gamer_off" in et else float("nan"),
                         def_strength=strength_along(model, et["gamer_def"][i]) if "gamer_def" in et else float("nan")))
    return rows, summary(model, rows, league)


def summary(model, rows, league):
    m = np.array([r["sim_margin"] for r in rows])
    w = np.array([r["sim_win"] for r in rows])
    f = np.array([r["sim_for"] for r in rows])
    real = np.array([r["real_points"] for r in rows])
    n = np.array([r["matches"] for r in rows])
    ok = ~np.isnan(real) & (n >= 20)
    L = [f"  {len(rows)} gamers, each against an average opponent from kick-off (no nudge): the sim's margin sd "
         f"{m.std():.2f} points ({m.min():+.1f} to {m.max():+.1f}), its win chance {w.min():.2f} to {w.max():.2f} "
         f"(sd {w.std():.3f})",
         f"  the tilts: shrunk toward the league by {getattr(model, 'tilt_prior', v21.TILT_PRIOR):g} snaps' worth, "
         f"each snap's weight halving every {v21.TILT_HALF_LIFE:g} days; pre-match the kick-off nudge then moves each "
         "side most of the way to the prior's expected points"]
    if ok.sum() >= 5:
        b = float(np.polyfit(real[ok], f[ok] - f[ok].mean(), 1)[0])
        L.append(f"  the {int(ok.sum())} gamers with 20+ matches: real points a match against the league's {league:.1f}, sd "
                 f"{real[ok].std():.2f}; the sim's points for them, sd {f[ok].std():.2f}; the sim follows a point of real "
                 f"scoring by {b:+.2f} (1: all of it)")
    return L


def lines_for(rows, gamers):
    """The table's lines for the named gamers (all of them, sim margin first, where none are named)."""
    want = {g.strip().upper() for g in gamers} if gamers else None
    pick = [r for r in rows if want is None or r["gamer"] in want]
    pick.sort(key=lambda r: -r["sim_margin"])
    L = [f"  {'gamer':14s} {'matches':>7s} {'real pts':>9s} {'sim for':>8s} {'against':>8s} {'margin':>7s} {'win':>6s} "
         f"{'off':>6s} {'def':>6s}"]
    for r in pick:
        L.append(f"  {r['gamer']:14s} {r['matches']:7d} {r['real_points']:+9.2f} {r['sim_for']:8.2f} {r['sim_against']:8.2f} "
                 f"{r['sim_margin']:+7.2f} {r['sim_win']:6.3f} {r['off_strength']:+6.2f} {r['def_strength']:+6.2f}")
    if want:
        missing = sorted(want - {r["gamer"] for r in pick})
        if missing:
            L.append("  not in the build: " + ", ".join(missing))
    return L


def write(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, FIELDS)
        w.writeheader()
        w.writerows(rows)
    return path
