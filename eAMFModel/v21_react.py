"""v2.1's in-play strength update: each side's offense nudge moved, state by state, by what the match
so far says of it, so a side outplaying its price is priced up for the rest of the game.

The nudge is v2.1's own unit of strength: a side's offense tilt moved along the strength direction,
one unit a point a game (v21.nudge_solve). Pre-match it is solved once, toward the prior's expected
points; with an update every in-play state gets its own, the kick-off's plus what the game has shown
by then. Three ways of reading the game:

  snapT  every real snap the side has played, scored against the draw the sim makes in its place:
         the league's real snaps for the same situation (the group, its blended levels and its window
         on the field, as v21.simulate draws them), tilted by the side's strength. The snap's
         features (PHI_DIST: yards, a touchdown, a turnover, a first down, a loss) against the draw's
         mean along the strength direction is the score, the draw's variance along it the
         information; their sums over the side's snaps, one Newton step from the kick-off shrunk by
         a prior precision of 1 / T^2 (v14's react, in v2.1's terms).
  ptsT   the side's points so far against its kick-off mean x the share of a game's points that has
         come by then (the build's points_share): a normal posterior of its points a game, the game's
         own spread of points (the kick-off's simulated finals) the noise, T the prior's sd.
  wpaC   every snap looked up in a win-probability table (react_wp.json beside the model, fitted by
         v21-react-fit on real games): the side's chance of winning after the snap less before it, the
         chance it added; the side's nudge C points a game for every whole chance its snaps added. A
         snap is weighed by how far it moved the match, not by how far it was from its situation's
         usual.

(T for snap and pts: the prior's sd in points a game, how far the side's strength today may be from
the kick-off's.)

Gamer by gamer (gsnapT, gptsT, gwpaC): the approach's nudge x the gamer's own scale, the slope of what
the gamer's rest of the game did against what the approach said on real games, shrunk toward the
pooled slope (react_gamers.json beside the model, written by v21-react-fit; a gamer it has none for
takes the pooled).

The calibrator names them as candidates: --candidate v21,v21-snap4,v21-pts6 prices the same build
three ways. No update (v21 itself) leaves the nudge as it was, so its prices are unchanged."""

import re

import numpy as np

from . import feed, v21

_NAME = re.compile(r"^(g?(?:snap|pts|wpa))(\d+(?:\.\d+)?)$")
VARIANTS = ("snap2", "snap4", "snap8", "pts4", "pts8", "wpa10", "wpa20")    # a spread to set against each other
WP_FILE = "react_wp.json"
GAMERS_FILE = "react_gamers.json"          # each gamer's own scale on each approach (v21-react-fit)


def spec_of(tag):
    """'snap4' -> ('snap', 4.0); None where the tag is no variant."""
    m = _NAME.match(str(tag or "").strip().lower())
    if not m or float(m.group(2)) <= 0:
        return None
    return m.group(1), float(m.group(2))


def variant_of(name):
    """'v21-snap4' -> ('snap', 4.0); None for v21 itself or a build's own tag ('v21-glmer')."""
    _, _, tag = str(name or "").partition("-")
    return spec_of(tag)


def side_theta(model, pair=("", ""), teams=("", "")):
    """(2, D) each side's distance tilt with the ball before any nudge (its gamer's and team's, the
    other side's defense); None where the build has none (not a tilt build)."""
    st = v21._start(model, 1, pair, teams)
    th = v21._etilts(model, "dist", st["gamer"], st["team_ix"])
    return None if th is None else np.asarray(th[0], float)


def observed_phi(recs):
    """(snaps, D) each real snap's distance features, as the build gave its own records them."""
    kind = np.array([r["kind"] for r in recs])
    gain = np.array([r["gain"] for r in recs], np.int64)
    field = np.array([r["field"] for r in recs], np.int64)
    first = np.array([bool(r.get("converted")) and not r.get("replay") for r in recs], bool)
    td = (kind == v21.GAIN) & (gain >= 100 - field)
    return v21.phi_dist_of(kind, gain, td, first & (kind == v21.GAIN))


def draw_sets(model, recs):
    """Each real snap's draw as the sim makes it: a list, one per snap, of (weight, records) for every
    level its draw can come from -- the group's level weights where the build blends them, else the one
    group -- the records the field window of each holds (the whole group without a window)."""
    b = model.dist
    feats = v21._features(recs, model)
    cols = {f: feats[f] for f in b.features}
    field = np.array([r["field"] for r in recs], np.int64)
    if getattr(model, "blend", 0.0):
        C = b.chain(cols)
        W = v21.dist_weights(model, C, field)
    else:
        C = b.bin_of(cols)[:, None]
        W = (C >= 0).astype(float)
    L = C.shape[1]
    pairs = [(i, li) for i in range(len(recs)) for li in range(L) if C[i, li] >= 0 and W[i, li] > 0]
    out = [[] for _ in recs]
    if not pairs:
        return out
    ii = np.array([p[0] for p in pairs])
    gg = np.array([C[i, li] for i, li in pairs], np.int64)
    share = getattr(model, "field", 0.0) or 0.0
    if share:
        order, _ = b.by_value("field")
        nmin = v21.field_nmin(model, np.array([li == L - 1 for _, li in pairs]))
        lo, hi = b.window(gg, field[ii], v21.field_radius(share, field[ii]), "field", nmin)
        recs_of = [order[l0:h0] for l0, h0 in zip(lo, hi)]
    else:
        recs_of = [np.arange(b.start[g], b.start[g] + b.count[g]) for g in gg]
    for (i, li), idx in zip(pairs, recs_of):
        if len(idx):
            out[i].append((float(W[i, li]), idx))
    return out


def snap_terms(model, recs, theta, sets=None):
    """(score, information) of each real snap along the strength direction under its side's tilt
    theta (2, D): the snap's features less its draw's mean, and the draw's variance (the levels mixed
    on their weights)."""
    d = np.asarray(model.nudge_dir, float)
    phi = model.dist.payload["phi"]
    sets = draw_sets(model, recs) if sets is None else sets
    obs = observed_phi(recs) @ d
    u, info = np.zeros(len(recs)), np.zeros(len(recs))
    for i, (r, levels) in enumerate(zip(recs, sets)):
        if not levels:
            continue
        th = theta[r["side"]]
        wsum = m1 = m2 = 0.0
        for w, idx in levels:
            P = phi[idx].astype(float)
            z = P @ th
            p = np.exp(z - z.max())
            p /= p.sum()
            x = P @ d
            m1 += w * float(p @ x)
            m2 += w * float(p @ (x * x))
            wsum += w
        m1, m2 = m1 / wsum, m2 / wsum
        u[i], info[i] = obs[i] - m1, max(0.0, m2 - m1 * m1)
    return u, info


def _clip(alpha, delta):
    return np.clip(alpha[None, :] + delta, -v21.NUDGE_MAX, v21.NUDGE_MAX) - alpha[None, :]


def _sums_by(messages, known, side, *values):
    """Each value summed over a side's snaps known by each message: [(states, 2)] per value."""
    o = np.argsort(known, kind="stable")
    k = np.searchsorted(known[o], np.asarray(messages, np.int64), side="right")
    out = []
    for v in values:
        c = np.zeros((len(known) + 1, 2))
        c[1:] = np.cumsum(np.eye(2)[side[o]] * np.asarray(v, float)[o, None], axis=0)
        out.append(c[k])
    return out


def _snaps(rows, pair, teams):
    return [r for r in v21.match_snaps(rows, pair, teams) if r["period"] <= 4 + v21.MAX_OT]


def snap_sums(model, rows, messages, alpha=None, pair=("", ""), teams=("", "")):
    """((states, 2) score, (states, 2) information): each side's snaps known by each PLAY_OVER message
    (a snap's result is the next PLAY_OVER), under its tilt with the kick-off's nudge alpha; None where
    the build has no tilts or the match no snaps."""
    theta = side_theta(model, pair, teams)
    recs = _snaps(rows, pair, teams) if theta is not None else []
    if not recs or not len(messages):
        return None
    a0 = np.zeros(2) if alpha is None else np.asarray(alpha, float)
    th = theta + a0[:, None] * np.asarray(model.nudge_dir, float)[None, :]
    u, info = snap_terms(model, recs, th)
    side = np.array([r["side"] for r in recs], np.int64)
    known = np.array([r["message"] for r in recs], np.int64)
    return tuple(_sums_by(messages, known, side, u, info))


def snap_updates(model, rows, messages, alpha, tau, pair=("", ""), teams=("", "")):
    """(states, 2) each side's nudge change at each PLAY_OVER message: score over information plus
    1 / tau^2 (snap_sums)."""
    got = snap_sums(model, rows, messages, alpha, pair, teams)
    if got is None:
        return np.zeros((len(messages), 2))
    U, I = got
    return _clip(np.zeros(2) if alpha is None else np.asarray(alpha, float), U / (I + 1.0 / tau ** 2))


# ---------------------------------------------------------------- the win-probability table (wpa)

WP_EPS = 1.0 / 48               # a 20-second floor on the share of regulation left
WP_FEATURES = ("possession", "margin", "prior", "field", "2nd down", "3rd down", "4th down", "constant")


def time_left(period, clock):
    """The share of regulation left (0 in overtime)."""
    played = (min(int(period), 4) - 1) * v21.QUARTER + (v21.QUARTER - float(clock))
    return 0.0 if int(period) >= 5 else float(np.clip(1.0 - played / (4 * v21.QUARTER), 0.0, 1.0))


def wp_features(t, margin, prior, field, down):
    """(n, 8) the win-probability model's features from the side with the ball: the margin, the field
    (its own goal 0, the other's 1) and the down over the root of the time left, the pre-match logit
    times it (strength counts for as much of the game as is left), and a constant."""
    t, margin, prior = np.asarray(t, float), np.asarray(margin, float), np.asarray(prior, float)
    field, down = np.asarray(field, float), np.asarray(down, np.int64)
    r = np.sqrt(t + WP_EPS)
    return np.stack([1.0 / r, margin / r, prior * r, (field / 100.0 - 0.5) / r,
                     (down == 2) / r, (down == 3) / r, (down == 4) / r, np.ones_like(t)], axis=1)


def _logit(p):
    p = min(0.98, max(0.02, float(p)))
    return float(np.log(p / (1 - p)))


def prior_logit(rows):
    """The home side's pre-match logit off prod's pre-match moneyline in the export (0: none)."""
    for r in rows[:1]:
        try:
            return _logit(float(r.get("prematch_prob_50")))
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def wp_state(row, home_prior=0.0):
    """(side with the ball, t, its margin, its pre-match logit, field, down) at a PLAY_OVER, a kick-off
    (a score, a conversion to come: the point after counted) as the receiver's 1st and 10 at its 25;
    None where the state is not known."""
    from . import playover
    from .state import HOME
    st, _ = playover.state_for(row)
    if st is None or st.offense is None or st.period is None or st.clock_seconds is None:
        return None
    home, away = float(st.home_score), float(st.away_score)
    if st.pending_conversion is not None:
        if st.pending_conversion == HOME:
            home += 1.0
        else:
            away += 1.0
    on = st.offense == HOME
    margin = home - away if on else away - home
    kick = st.down is None or st.pending_conversion is not None or not st.has_snap
    field, down = (25, 1) if kick else (st.field_position, st.down)
    return (0 if on else 1, time_left(st.period, st.clock_seconds), margin,
            home_prior if on else -home_prior, field, down)


def wp_of(coef, state, side):
    """The win chance of `side` (0 home, 1 away) at a wp_state; None where the state is None."""
    if state is None:
        return None
    on, t, margin, prior, field, down = state
    z = float(wp_features([t], [margin], [prior], [field], [down])[0] @ np.asarray(coef, float))
    p = 1.0 / (1.0 + np.exp(-z))
    return p if side == on else 1.0 - p


def snap_wpa(rows, coef, pair=("", ""), teams=("", "")):
    """(side, known message, the win chance the snap added its side) for every real snap: the side's
    chance at the PLAY_OVER after the snap less at the one it started from (wp_of)."""
    prior = prior_logit(rows)
    by_msg = {feed._i(r.get("message")): r for r in rows}
    out = []
    for rec in _snaps(rows, pair, teams):
        a, b = by_msg.get(rec["start"]), by_msg.get(rec["message"])
        if a is None or b is None:
            continue
        before, after = wp_of(coef, wp_state(a, prior), rec["side"]), wp_of(coef, wp_state(b, prior), rec["side"])
        if before is not None and after is not None:
            out.append((rec["side"], rec["message"], after - before))
    return out


def wpa_sums(rows, messages, coef, pair=("", ""), teams=("", "")):
    """(states, 2) the win chance each side's snaps known by each PLAY_OVER message added."""
    got = snap_wpa(rows, coef, pair, teams)
    if not got or not len(messages):
        return np.zeros((len(messages), 2))
    side, known, add = (np.array(x) for x in zip(*got))
    return _sums_by(messages, known.astype(np.int64), side.astype(np.int64), add)[0]


def load_gamers(model_dir):
    """{approach: {"setting", "k" (the pooled scale), "gamers": {handle: scale}}} beside a build
    (GAMERS_FILE), or None."""
    import json
    import os
    path = os.path.join(model_dir if os.path.isdir(model_dir) else os.path.dirname(model_dir), GAMERS_FILE)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def gamer_scale(table, approach, pair):
    """(2,) each side's own scale on an approach (the pooled where the gamer has none)."""
    t = (table or {}).get(approach) or {}
    k = float(t.get("k", 1.0))
    own = t.get("gamers") or {}
    return np.array([float(own.get((g or "").strip().upper(), k)) for g in pair])


def load_wp(model_dir):
    """The win-probability table's coefficients beside a build (WP_FILE), or None."""
    import json
    import os
    path = os.path.join(model_dir if os.path.isdir(model_dir) else os.path.dirname(model_dir), WP_FILE)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        got = json.load(fh)
    return tuple(float(x) for x in got["coef"])


def elapsed_share(model, period, clock):
    """The share of a game's points that has come by this clock (the build's points_share; the clock's
    own share of regulation where the build has none)."""
    share = getattr(model, "share", None)
    if share is None or not len(share):
        played = (min(int(period), 4) - 1) * v21.QUARTER + (v21.QUARTER - float(clock))
        return 1.0 if period >= 5 else float(np.clip(played / (4 * v21.QUARTER), 0.0, 1.0))
    return float(np.clip(1.0 - share[int(v21.share_slice(period, clock))], 0.0, 1.0))


def points_updates(model, starts, alpha, tau, kick):
    """(states, 2) each side's nudge change off its points so far: tau^2 (points - mean x share) /
    (the game's variance + tau^2 x share), the kick-off's simulated mean and variance (kick)."""
    out = np.zeros((len(starts), 2))
    if kick is None or not len(starts):
        return out
    mu, var = np.asarray(kick["mean"], float), np.maximum(np.asarray(kick["var"], float), 1.0)
    for k, s in enumerate(starts):
        f = elapsed_share(model, int(s["period"][0]), float(s["clock"][0]))
        y = np.array([float(s["home"][0]), float(s["away"][0])])
        out[k] = tau ** 2 * (y - mu * f) / (var + tau ** 2 * f)
    a0 = np.zeros(2) if alpha is None else np.asarray(alpha, float)
    return _clip(a0, out)


def updates(model, rows, sel, alpha, spec, pair=("", ""), teams=("", ""), kick=None, wp=None, gamers=None):
    """(states, 2) each side's nudge change at each selected state ([(PLAY_OVER message, start)]) under
    a variant (spec_of); zeros without one, or where the build has no strength direction. wp: the
    win-probability table's coefficients (load_wp), for wpa; gamers: each gamer's own scale (load_gamers),
    for the g- variants."""
    out = np.zeros((len(sel), 2))
    if not spec or not sel or getattr(model, "nudge_dir", None) is None:
        return out
    kind, x = spec[0], spec[1]
    if kind.startswith("g"):
        if gamers is None or kind[1:] not in gamers:
            raise ValueError(f"the {kind} update needs each gamer's own scale (v21-react-fit's {GAMERS_FILE})")
        a0 = np.zeros(2) if alpha is None else np.asarray(alpha, float)
        base = updates(model, rows, sel, alpha, (kind[1:], x), pair, teams, kick, wp)
        return _clip(a0, base * gamer_scale(gamers, kind[1:], pair)[None, :])
    if kind == "snap":
        return snap_updates(model, rows, [m for m, _ in sel], alpha, x, pair, teams)
    if kind == "pts":
        return points_updates(model, [s for _, s in sel], alpha, x, kick)
    if kind == "wpa":
        if wp is None:
            raise ValueError("the wpa update needs the win-probability table (v21-react-fit)")
        a0 = np.zeros(2) if alpha is None else np.asarray(alpha, float)
        return _clip(a0, x * wpa_sums(rows, [m for m, _ in sel], wp, pair, teams))
    raise ValueError(f"no such in-play update {kind!r}")
