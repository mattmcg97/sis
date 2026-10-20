"""v2.1's report: the pre-match expected points (the inputs) against what the sim makes of the same
matches, from kick-off and from every step of the real games, and the sim's draws against the real
snaps -- where the two part shows what the draws are missing. Nothing here feeds back into the model.

    python -m eAMFModel v21-report --model v21_full --prior-model v14_model --since 2026-10-01 --until 2026-10-12

For every settled match in the window with snapshots:
  kick-off   each side's expected points (the prior's own, or with --inputs priced the build's layered
             means) against the sim's mean from kick-off with the match's gamers and teams, and the real
             finals: overall, by band of the side's expected points, by band of the expected margin, by
             gamer and by team, and graded (moneyline log loss and Brier, margin and total RPS) for the
             prior's own score model and the sim.
  the margin the final margin from kick-off, the sim's and the prior's own score model against the
             real finals: how often each band (a tie, 1-3, 4-7, 8-14, 15+) and each key margin, the spread
             against the real misses, where each final falls in the distribution, and how the favourite
             wins -- whether the sim's margin is too wide, with too few close games
  each step  from every --every-th real PLAY_OVER before a snap: each side's points still to come --
             the input's (its expected points times the league's share of a game's points still to come
             at that clock, model.share), the sim's from the real state, and the real ones -- by part of
             the game, by favourite and underdog, and by the side's lead at the time.
  the draws  the kick-off sims' snaps against the real games': events a game, points by quarter, the
             time draw (time buckets by the play before) and the distance draw (snap classes by time
             bucket).
"""

import csv
import math
import zlib

import numpy as np

from . import sim14 as s14, v13, v21

DEFAULT_PATHS = 1000          # kick-off games per receiving side
STATE_PATHS = 200
EVERY = 4
STEPS = ("Q1 to 2:00", "Q1 last 2:00", "Q2 to 2:00", "Q2 last 2:00", "Q3 to 2:00", "Q3 last 2:00",
         "Q4 to 2:00", "Q4 last 2:00", "OT")
LEAD_EDGES = np.array([-8.5, -0.5, 0.5, 8.5])
LEAD_NAMES = ("behind 9+", "behind 1-8", "level", "ahead 1-8", "ahead 9+")
EVENTS = ("snaps", "touchdowns", "field_goal_tries", "punts", "turnovers", "defensive_tds", "safeties",
          "kick_tds", "timeouts", "overtime")
MIN_GAMER_MATCHES = 5
ROW_KEYS = ("match_code", "message", "period", "clock_seconds", "play_kind", "offense", "down", "distance",
            "field_position", "score_p1", "score_p2", "play_messages", "team_a_side", "opening_offense",
            "timeouts_used_a", "timeouts_used_b", "timeout_after", "timeout_prev", "home_handle", "away_handle")


def step_of(period, clock):
    period, clock = np.asarray(period), np.asarray(clock, float)
    return np.where(period >= 5, 8, (np.minimum(period, 4) - 1) * 2 + (clock <= v21.LATE)).astype(np.int64)


def _seed(code):
    return zlib.crc32(str(code).encode())


def _census(stats, n_who=1):
    """A sim's snaps, events and decisions counted as the report reads them."""
    log = v21.snap_log(stats)
    out = {"games": stats.get("games", 0), "events": {k: stats.get(k, 0) for k in EVENTS},
           "dec": v21.decision_census(stats, n_who),
           "q_points": stats["q_points"].sum(0) if "q_points" in stats else np.zeros(5),
           "time": np.zeros((6, 6)), "cls": np.zeros((6, s14.N_CLASSES)), "secs": np.zeros((6, 2))}
    if len(log["path"]):
        tb = v21.time_b(log["seconds"])
        prev = log["prev"].astype(np.int64)
        np.add.at(out["time"], (prev, tb), 1)
        cls = s14.snap_class(log["kind"], log["gain"], log["dist"], log["field"])
        np.add.at(out["cls"], (v21.time_b(log["drawn"]), cls), 1)
        np.add.at(out["secs"], (prev, 0), log["seconds"])
        np.add.at(out["secs"], (prev, 1), 1)
    return out


def _add(a, b):
    if a is None:
        return b
    out = {k: a[k] + b[k] for k in ("games", "q_points", "time", "cls", "secs")}
    out["events"] = {k: a["events"][k] + b["events"][k] for k in EVENTS}
    out["dec"] = {}
    for name in set(a.get("dec", {})) | set(b.get("dec", {})):
        x, y = a.get("dec", {}).get(name), b.get("dec", {}).get(name)
        out["dec"][name] = y if x is None else x if y is None else {k: x[k] + y[k] for k in x}
    return out


def real_census(matches, model=None, history=None):
    """The real games' snaps, events and (with the model, for its gamers) decisions, counted as
    _census counts the sim's."""
    out = {"games": 0, "events": {k: 0 for k in EVENTS}, "q_points": np.zeros(5), "time": np.zeros((6, 6)),
           "cls": np.zeros((6, s14.N_CLASSES)), "secs": np.zeros((6, 2))}
    for rows in matches.values():
        rows = v21._with_side(rows)
        recs = v21.match_snaps(rows)
        if not recs:
            continue
        out["games"] += 1
        ev = out["events"]
        ev["snaps"] += len(recs)
        ev["turnovers"] += sum(r["kind"] == v21.TURNOVER for r in recs)
        ev["defensive_tds"] += sum(r["kind"] == v21.DEF_TD for r in recs)
        ev["touchdowns"] += sum(1 for r in recs if r.get("read_off_score"))   # no TOUCHDOWN row: the score's
        ev["timeouts"] += sum(r.get("timeout_role") is not None for r in recs)
        for a, b in zip(rows, rows[1:]):
            k = b["play_kind"]
            ev["touchdowns"] += k == "TOUCHDOWN"
            ev["field_goal_tries"] += k == "FIELD_GOAL"
            ev["punts"] += k == "PUNT" and a.get("down") == "4" and "SAFETY" not in (b.get("play_messages") or "")
            ev["safeties"] += "SAFETY" in (b.get("play_messages") or "") and k == "PUNT"
        ev["kick_tds"] += sum(1 for k in s14.kick_records(rows) if k[2] >= 100)
        periods = [s14._i(r["period"]) for r in rows if s14._i(r.get("period"))]
        ev["overtime"] += bool(periods) and max(periods) >= 5
        last = {}
        for r in rows:
            p, h, a = s14._i(r.get("period")), s14._i(r.get("score_p1")), s14._i(r.get("score_p2"))
            if None not in (p, h, a):
                last[min(p, 5)] = h + a
        before = 0
        for q in range(1, 6):
            if q in last:
                out["q_points"][q - 1] += last[q] - before
                before = last[q]
        tb = v21.time_b([r["seconds"] for r in recs])
        prev = np.array([r["prev_cls"] for r in recs])
        np.add.at(out["time"], (prev, tb), 1)
        cls = s14.snap_class(np.array([r["kind"] for r in recs]), np.array([r["gain"] for r in recs]),
                             np.array([r["distance"] for r in recs]), np.array([r["field"] for r in recs]))
        np.add.at(out["cls"], (v21.time_b([r["play_seconds"] for r in recs]), cls), 1)
        np.add.at(out["secs"], (prev, 0), [r["seconds"] for r in recs])
        np.add.at(out["secs"], (prev, 1), 1)
    out["dec"] = v21.real_decision_census(matches, model, history) if model is not None else {}
    return out


def _worker(job):
    """Price a list of matches: kick-off and every --every-th real state."""
    model_path, items, n_paths, state_paths, every, share = job
    model = v21.Model.load(model_path)
    out, census = {}, None
    for code, rows, pair, teams, target in items:
        st = {}
        start = v21.kickoff_start(model, pair, teams)
        alpha = None
        if target is not None and getattr(model, "nudge_dir", None) is not None:
            (h, a), alpha = v21.nudge_solve(model, start, target, n_paths, seed=_seed(code), stats=st, share=share)
        else:
            h, a = v21.simulate(model, start, n_paths, seed=_seed(code), stats=st)
        h, a = h.ravel(), a.ravel()
        census = _add(census, _census(st, len(model.gamers) + 1))
        got = dict(kick=(float(h.mean()), float(a.mean())), book=v13._distributions(h, a),
                   p_home=float(((h > a).sum() + 0.5 * (h == a).sum()) / len(h)), states=[],
                   nudge=None if alpha is None else tuple(float(x) for x in alpha))
        starts = v21.match_starts(model, rows, pair, teams, every=every) if state_paths else []
        if starts:
            S = v21.concat_starts([s for _, s in starts])
            if alpha is not None:
                S["nudge"] = np.tile(alpha, (len(S["period"]), 1))
            sh, sa = v21.simulate(model, S, state_paths, seed=_seed(code) + 1)
            for k, (i, s) in enumerate(starts):
                got["states"].append(dict(row=i, period=int(s["period"][0]), clock=float(s["clock"][0]),
                                          score=(int(s["home"][0]), int(s["away"][0])),
                                          sim=(float(sh[k].mean()), float(sa[k].mean())),
                                          p_home=float(((sh[k] > sa[k]).sum() + 0.5 * (sh[k] == sa[k]).sum())
                                                       / state_paths)))
        out[code] = got
    return out, census


def run(model_dir, matches, history, means, since=None, until=None, n_paths=DEFAULT_PATHS,
        state_paths=STATE_PATHS, every=EVERY, workers=1, prior_books=None, pool=None, nudge=False, nudge_share=1.0):
    """Per match: its row, finals, the inputs, the sim's kick-off and its states; and the census of the
    sim's draws and the real games'. `means` {code: (home, away)} are the inputs; `prior_books`
    {code: (margin pmf, total pmf)} the prior's own price, for grading. nudge (a tilt build): each side's
    offense nudged along the strength direction to meet the inputs at kick-off (v21.nudge_solve), the
    same nudge carried into its real states; nudge_share: that share of the way from the sim's own means."""
    import os
    model_path = os.path.join(model_dir, v21.MODEL_FILE) if os.path.isdir(model_dir) else model_dir
    model = v21.Model.load(model_path)
    hist = {r["MATCH_CODE"]: r for r in history}
    codes = [c for c in matches if c in hist and c in means and _finals(hist[c]) is not None]
    codes.sort(key=lambda c: (str(hist[c].get("SCHEDULED_START_TIME_UTC", "")), c))
    if not codes:
        raise SystemExit("no settled matches with snapshots and inputs in that window")
    if nudge and getattr(model, "nudge_dir", None) is None:
        raise SystemExit("--nudge needs a tilt build (v21-build --mode tilt)")
    items = [(c, [{k: r.get(k) for k in ROW_KEYS} for r in v21._with_side(matches[c])])
             + v21.match_people(c, matches[c], hist) + (tuple(means[c]) if nudge else None,) for c in codes]
    workers = max(1, min(workers, len(items)))
    jobs = [(model_path, items[k::workers], n_paths, state_paths, every, nudge_share) for k in range(workers)]
    if workers == 1:
        parts = [_worker(j) for j in jobs]
    else:
        with (pool or _pool)().Pool(workers) as p:
            parts = p.map(_worker, jobs)
    sims, census = {}, None
    for got, c in parts:
        sims.update(got)
        census = _add(census, c)
    results = []
    for c in codes:
        r = hist[c]
        results.append(dict(code=c, row=r, finals=_finals(r), means=tuple(means[c]), people=v21.match_people(c, matches[c], hist),
                            prior_book=(prior_books or {}).get(c), **sims[c]))
    return dict(results=results, census=census, real=real_census({c: matches[c] for c in codes}, model, history),
                gamer_names={i: g for g, i in model.gamers.items()}, share=model.share, mode=model.mode, since=since, until=until, every=every,
                n_paths=n_paths, state_paths=state_paths, nudge=bool(nudge), nudge_share=float(nudge_share))


def _pool():
    from . import v14
    return v14.pool_context()


def _finals(r):
    try:
        return float(r["PLAYER_1_FINAL_SCORE"]), float(r["PLAYER_2_FINAL_SCORE"])
    except (KeyError, TypeError, ValueError):
        return None


# ---------------------------------------------------------------- the inputs

def prior_inputs(model_dir, rows, history, which="raw"):
    """({code: (home, away) expected points}, {code: (margin pmf, total pmf)}) off a v13/v14 build's
    pre-match model -- its own means (raw) or the means the build prices with (priced) -- or a prior table
    (a walkthrough variant's CSV), and the prior's own score model on them (a table's, the one each of its
    rows carries)."""
    from . import prior_check, table_prior
    if table_prior.is_table(model_dir):              # a walkthrough variant's table
        pre = table_prior.TablePrior(model_dir)
    else:
        version = prior_check.version_of(model_dir)
        m, st = prior_check._modules(version)
        pre = m.prematch_model(st.model_paths(model_dir)[0])
    if pre is None:
        raise SystemExit(f"{model_dir}: no pre-match model to read the expected points from")
    base = prior_check.base_prior(pre)
    means = (base if which == "raw" else pre).means(rows, results=history)
    books = {}
    own = getattr(pre, "score", {})                    # a table's own score model, match by match
    for r in rows:
        c = r["MATCH_CODE"]
        if c in means:
            score = own.get(str(c).strip())
            books[c] = (prior_check.table_books(means[c], score, seed=_seed(c)) if score else
                        prior_check.raw_books(means[c], *prior_check.sigmas(base, prior_check._start(r).date()),
                                              seed=_seed(c)))
    return means, books


# ---------------------------------------------------------------- the report

def _band_edges(values, n=5):
    v = np.asarray(values, float)
    return np.unique(np.quantile(v, np.linspace(0, 1, n + 1)[1:-1])) if len(v) else np.array([])


def _bands(values, edges):
    return np.searchsorted(edges, np.asarray(values, float), side="right")


def _band_label(i, edges):
    lo = f"{edges[i - 1]:.1f}" if i > 0 else ""
    hi = f"{edges[i]:.1f}" if i < len(edges) else ""
    return f"<{hi}" if not lo else f"{lo}+" if not hi else f"{lo}-{hi}"


def _sides(results):
    """Per side of every match: input, sim, real points for and against, gamer and team."""
    out = []
    for x in results:
        for s in (0, 1):
            out.append(dict(input=x["means"][s], sim=x["kick"][s], real=x["finals"][s],
                            input_vs=x["means"][1 - s], sim_vs=x["kick"][1 - s], real_vs=x["finals"][1 - s],
                            gamer=x["people"][0][s], team=x["people"][1][s]))
    return out


def _row(label, n, a, b, c):
    return f"  {label:22s}{n:>7,d}{a:>9.2f}{b:>9.2f}{c:>9.2f}{b - a:>+9.2f}{c - a:>+9.2f}"


HEAD = f"  {'':22s}{'n':>7s}{'input':>9s}{'sim':>9s}{'real':>9s}{'sim-in':>9s}{'real-in':>9s}"


def kickoff_lines(results):
    sides = _sides(results)
    inp, sim, real = (np.array([s[k] for s in sides]) for k in ("input", "sim", "real"))
    lines = ["  KICK-OFF: each side's expected points (input) against the sim's mean and the real points", HEAD,
             _row("every side", len(sides), inp.mean(), sim.mean(), real.mean())]
    tot = lambda key: np.array([x[key][0] + x[key][1] for x in results])
    mar = lambda key: np.array([x[key][0] - x[key][1] for x in results])
    lines.append(_row("match total", len(results), tot("means").mean(), tot("kick").mean(), tot("finals").mean()))
    lines.append(_row("home margin", len(results), mar("means").mean(), mar("kick").mean(), mar("finals").mean()))
    if len(inp) > 2 and inp.std() > 0:
        slope = np.polyfit(inp, sim, 1)[0]
        slope_r = np.polyfit(inp, real, 1)[0]
        lines.append(f"  a point more expected is {slope:.2f} more in the sim, {slope_r:.2f} more in the real games"
                     " (1: the input followed in full; the sim has no strength, so its gamers and teams carry it all)")
    edges = _band_edges(inp)
    lines += ["", "  by the side's expected points", HEAD]
    b = _bands(inp, edges)
    for i in range(len(edges) + 1):
        k = b == i
        if k.any():
            lines.append(_row(_band_label(i, edges), int(k.sum()), inp[k].mean(), sim[k].mean(), real[k].mean()))
    m_in, m_sim, m_real = mar("means"), mar("kick"), mar("finals")
    edges = _band_edges(m_in)
    lines += ["", "  by the expected home margin", HEAD]
    b = _bands(m_in, edges)
    for i in range(len(edges) + 1):
        k = b == i
        if k.any():
            lines.append(_row(_band_label(i, edges), int(k.sum()), m_in[k].mean(), m_sim[k].mean(), m_real[k].mean()))
    return lines


MARGIN_BANDS = (("tie", 0, 0), ("1-3", 1, 3), ("4-7", 4, 7), ("8-14", 8, 14), ("15+", 15, 999))
KEY_MARGINS = (1, 2, 3, 6, 7, 8)
FAV_EDGES = np.array([2.0, 4.0, 7.0])
FAV_NAMES = ("fav by <2", "fav by 2-4", "fav by 4-7", "fav by 7+")


def _margin_pmf(book):
    """(margins, pmf) off a book's margin pmf (index MARGIN_MAX is a tie)."""
    from . import v13
    pmf = np.asarray(book[0], float)
    return np.arange(len(pmf)) - v13.MARGIN_MAX, pmf


def _spreads(book):
    """(margin sd, total sd) of a book, or blanks when there is none."""
    if book is None:
        return "", ""
    mg, mp = _margin_pmf(book)
    tp = np.asarray(book[1], float)
    tk = np.arange(len(tp))
    m_mu, t_mu = float((mg * mp).sum()), float((tk * tp).sum())
    return (round(float(np.sqrt(((mg - m_mu) ** 2 * mp).sum())), 3),
            round(float(np.sqrt(((tk - t_mu) ** 2 * tp).sum())), 3))


def _pit(margins, pmf, y, u):
    """Where the real margin y falls in the distribution, a tie within its own mass split by u."""
    return float(pmf[margins < y].sum() + u * pmf[margins == y].sum())


def margin_lines(results):
    """The final margin from kick-off: how often each band (either way), the key margins, the spread of the
    distribution against the real misses, where each real final falls in it, and how the favourite wins --
    the sim's and the prior's own score model against the real finals."""
    have = [x for x in results if x.get("book") is not None]
    if not have:
        return []
    names = ["v21 sim"] + (["prior"] if all(x.get("prior_book") is not None for x in have) else [])
    books = {"v21 sim": [x["book"] for x in have], "prior": [x.get("prior_book") for x in have]}
    y = np.array([x["finals"][0] - x["finals"][1] for x in have], float)
    u = np.array([(_seed(x["code"]) % 10007) / 10007 for x in have])
    n = len(have)
    lines = ["", f"  THE FINAL MARGIN from kick-off: {n:,} matches (overtime in the real finals and the sim's)",
             f"  {'either way':22s}" + "".join(f"{b[0]:>8s}" for b in MARGIN_BANDS)]
    stats = {}
    for name in names:
        bands, keys, sd, mean, pit, fav = np.zeros(len(MARGIN_BANDS)), np.zeros(len(KEY_MARGINS)), [], [], [], []
        for k, bk in enumerate(books[name]):
            mg, p = _margin_pmf(bk)
            a = np.abs(mg)
            bands += [p[(a >= lo) & (a <= hi)].sum() for _, lo, hi in MARGIN_BANDS]
            keys += [p[a == v].sum() for v in KEY_MARGINS]
            mu = float((mg * p).sum())
            mean.append(mu)
            sd.append(float(np.sqrt(((mg - mu) ** 2 * p).sum())))
            pit.append(_pit(mg, p, y[k], u[k]))
            s = 1.0 if mu >= 0 else -1.0                          # the favourite's side, as this book has it
            fm = s * mg
            fav.append((abs(mu), p[(fm >= 1) & (fm <= 7)].sum(), p[fm >= 8].sum(), p[fm <= -1].sum(),
                        1 <= s * y[k] <= 7, s * y[k] >= 8, s * y[k] <= -1))
        stats[name] = dict(bands=bands / n, keys=keys / n, sd=np.array(sd), mean=np.array(mean), pit=np.array(pit),
                           fav=np.array(fav, float))
        lines.append(f"  {name + ' expects':22s}" + "".join(f"{100 * v:>7.1f}%" for v in stats[name]["bands"]))
    real = [np.mean((np.abs(y) >= lo) & (np.abs(y) <= hi)) for _, lo, hi in MARGIN_BANDS]
    lines.append(f"  {'real':22s}" + "".join(f"{100 * v:>7.1f}%" for v in real))
    lines += ["", f"  {'the key margins':22s}" + "".join(f"{v:>8d}" for v in KEY_MARGINS)]
    for name in names:
        lines.append(f"  {name + ' expects':22s}" + "".join(f"{100 * v:>7.1f}%" for v in stats[name]["keys"]))
    lines.append(f"  {'real':22s}" + "".join(f"{100 * np.mean(np.abs(y) == v):>7.1f}%" for v in KEY_MARGINS))
    lines += ["", "  the spread: each match's own margin sd against the real misses' (finals less its mean); the"
                  " sim too wide where its sd is above the misses'"]
    for name in names:
        s = stats[name]
        miss = y - s["mean"]
        lines.append(f"  {name:22s} its sd {s['sd'].mean():.2f}, the misses' sd {miss.std():.2f} (mean {miss.mean():+.2f}),"
                     f" the real final inside its central 50% {np.mean(np.abs(s['pit'] - 0.5) < 0.25):.1%}, 80%"
                     f" {np.mean(np.abs(s['pit'] - 0.5) < 0.4):.1%}")
    lines += ["", "  where each real final falls in the distribution, by tenth (10% each when right; too wide piles"
                  " them in the middle, too narrow in the ends)"]
    for name in names:
        h, _ = np.histogram(stats[name]["pit"], bins=10, range=(0, 1))
        lines.append(f"  {name:22s}" + " ".join(f"{100 * v / n:5.1f}" for v in h))
    lines += ["", "  the favourite (each book's own): wins by 1-7, wins by 8+, loses -- expected against real, by its"
                  " expected margin", f"  {'':12s}{'':>7s}{'':>3s}{'wins 1-7':>16s}{'wins 8+':>16s}{'loses':>16s}"]
    for name in names:
        f = stats[name]["fav"]
        b = np.searchsorted(FAV_EDGES, f[:, 0], side="right")
        for i, label in enumerate(FAV_NAMES):
            k = b == i
            if k.sum() < 5:
                continue
            e, r = f[k, 1:4].mean(0), f[k, 4:7].mean(0)
            lines.append(f"  {label:12s}{int(k.sum()):>7,d}  " + "".join(
                f"{100 * e[j]:>7.1f} {100 * r[j]:>6.1f}  " for j in range(3)) + f"({name})")
    return lines


def people_lines(results, key="gamer", top=15, min_matches=MIN_GAMER_MATCHES):
    sides = _sides(results)
    by = {}
    for s in sides:
        if s[key]:
            by.setdefault(s[key], []).append(s)
    rows = []
    for who, ss in by.items():
        if len(ss) < min_matches:
            continue
        f = lambda k: float(np.mean([s[k] for s in ss]))
        rows.append((who, len(ss), f("input"), f("sim"), f("real"), f("input_vs"), f("sim_vs"), f("real_vs")))
    if not rows:
        return []
    rows.sort(key=lambda r: r[3] - r[2])
    head = (f"  {key:16s}{'n':>5s}{'in for':>8s}{'sim-in':>8s}{'real-in':>8s}{'in vs':>8s}{'sim-in':>8s}"
            f"{'real-in':>8s}")
    fmt = lambda r: (f"  {r[0][:16]:16s}{r[1]:>5d}{r[2]:>8.1f}{r[3] - r[2]:>+8.1f}{r[4] - r[2]:>+8.1f}"
                     f"{r[5]:>8.1f}{r[6] - r[5]:>+8.1f}{r[7] - r[5]:>+8.1f}")
    lines = ["", f"  by {key} ({min_matches}+ matches): points for and against, the sim's and the real games' less the"
                 " input; the sim furthest under its input first, then furthest over", head]
    show = rows if len(rows) <= 2 * top else rows[:top] + [None] + rows[-top:]
    lines += [fmt(r) if r else "  ..." for r in show]
    gap = np.array([r[3] - r[2] for r in rows])
    lines.append(f"  spread of sim-in over {len(rows)} {key}s: sd {gap.std():.2f}"
                 f" (real-in sd {np.std([r[4] - r[2] for r in rows]):.2f})")
    return lines


def _state_rows(results, share):
    """Per side of every state: step, lead, favourite, input, sim and real points still to come."""
    out = []
    for x in results:
        mu, fin = x["means"], x["finals"]
        for s_ in x["states"]:
            sl = int(v21.share_slice(s_["period"], s_["clock"]))
            step = int(step_of(s_["period"], s_["clock"]))
            for side in (0, 1):
                sc = s_["score"][side]
                out.append((step, s_["score"][side] - s_["score"][1 - side], mu[side] > mu[1 - side],
                            mu[side] * share[sl], s_["sim"][side] - sc, fin[side] - sc))
    return np.array(out, float).reshape(-1, 6)


def step_lines(results, share):
    a = _state_rows(results, share)
    if not len(a):
        return []
    lines = ["", "  EACH STEP: from the real states, each side's points still to come -- the input's (expected points"
                 " times the league's share still to come), the sim's and the real ones", HEAD]
    for i, name in enumerate(STEPS):
        k = a[:, 0] == i
        if k.any():
            lines.append(_row(name, int(k.sum()), a[k, 3].mean(), a[k, 4].mean(), a[k, 5].mean()))
    for fav, label in ((1, "favourite (the side with more expected points)"), (0, "underdog")):
        lines += ["", f"  the {label}", HEAD]
        for i, name in enumerate(STEPS):
            k = (a[:, 0] == i) & (a[:, 2] == fav)
            if k.any():
                lines.append(_row(name, int(k.sum()), a[k, 3].mean(), a[k, 4].mean(), a[k, 5].mean()))
    lines += ["", "  by the side's lead at the time (Q2 on)", HEAD]
    lb = np.searchsorted(LEAD_EDGES, a[:, 1], side="right")
    for i, name in enumerate(LEAD_NAMES):
        k = (lb == i) & (a[:, 0] >= 2)
        if k.any():
            lines.append(_row(name, int(k.sum()), a[k, 3].mean(), a[k, 4].mean(), a[k, 5].mean()))
    return lines


def grade_lines(results):
    from . import prior_check
    have = [x for x in results if x.get("prior_book") is not None]
    if not have:
        return []
    graded = [dict(finals=x["finals"], books={"prior": x["prior_book"], "v21": x["book"]}) for x in have]
    summary, _ = prior_check.grade(graded, methods=("prior", "v21"))
    lines = ["", "  GRADED from kick-off (the prior's own score model on the inputs against the sim)",
             f"  {'':28s}{'prior':>10s}{'v21 sim':>10s}"]
    lines.append(f"  {'moneyline log loss':28s}{summary['prior']['log_loss']:>10.4f}{summary['v21']['log_loss']:>10.4f}")
    lines.append(f"  {'moneyline Brier':28s}{summary['prior']['brier']:>10.4f}{summary['v21']['brier']:>10.4f}")
    for name in ("margin", "total"):
        for key, label, fmt in (("rps", "RPS", "{:.3f}"), ("bias", "result less expected", "{:+.2f}"),
                                ("in80", "inside central 80%", "{:.1%}")):
            lines.append(f"  {name + ' ' + label:28s}" + "".join(
                f"{fmt.format(summary[m][name][key]):>10s}" for m in ("prior", "v21")))
    d, se = prior_check.paired(summary, "v21", "prior")
    lines.append(f"  log loss, sim less prior: {d:+.4f} +/- {se:.4f} (paired standard error)")
    return lines


def census_lines(census, real):
    if census is None or not real["games"]:
        return []
    g_s, g_r = max(1, census["games"]), max(1, real["games"])
    lines = ["", "  THE DRAWS: the sim's kick-off games against the real games of the window",
             f"  {'a game':28s}{'real':>9s}{'sim':>9s}"]
    for k in EVENTS:
        lines.append(f"  {k.replace('_', ' '):28s}{real['events'][k] / g_r:>9.2f}{census['events'][k] / g_s:>9.2f}")
    for q, name in enumerate(("Q1", "Q2", "Q3", "Q4", "OT")):
        lines.append(f"  {'points ' + name:28s}{real['q_points'][q] / g_r:>9.2f}{census['q_points'][q] / g_s:>9.2f}")
    lines += ["", "  the time draw: share of snaps by time, after each kind of play before (real / sim)",
              f"  {'':18s}{'snaps':>13s}{'mean secs':>13s}" + "".join(f"{b:>13s}" for b in v21.TIME_BUCKETS)]
    for p, name in enumerate(v21.PREV_NAMES):
        r, s = real["time"][p], census["time"][p]
        if not r.sum() and not s.sum():
            continue
        ms = lambda c: c["secs"][p, 0] / max(1, c["secs"][p, 1])
        lines.append(f"  {name:18s}{100 * r.sum() / max(1, real['time'].sum()):>6.1f}/{100 * s.sum() / max(1, census['time'].sum()):<6.1f}"
                     f"{ms(real):>6.1f}/{ms(census):<6.1f}"
                     + "".join(f"{100 * r[b] / max(1, r.sum()):>6.1f}/{100 * s[b] / max(1, s.sum()):<6.1f}" for b in range(6)))
    lines += ["", "  the distance draw: share of snaps by class, within each time bucket (real / sim)",
              f"  {'':8s}" + "".join(f"{c:>13s}" for c in s14.SNAP_CLASSES)]
    for b, name in enumerate(v21.TIME_BUCKETS):
        r, s = real["cls"][b], census["cls"][b]
        if not r.sum() and not s.sum():
            continue
        lines.append(f"  {name:8s}" + "".join(f"{100 * r[c] / max(1, r.sum()):>6.1f}/{100 * s[c] / max(1, s.sum()):<6.1f}"
                                            for c in range(s14.N_CLASSES)))
    return lines


def decision_lines(census, real, names=None):
    """Each decision's choices, real against the sim's kick-off games, by slice of the state; and for
    the gamers with enough real decisions, their own share of the option LEAN_SHOWN names."""
    if census is None or not real.get("dec"):
        return []
    names = names or {}
    g_s, g_r = max(1, census["games"]), max(1, real["games"])
    lines = ["", "  THE DECISIONS: share of each choice, real / sim (the sim's kick-off games)"]
    for name, spec in v21.DECISIONS.items():
        r, s_ = real["dec"].get(name), census["dec"].get(name)
        if r is None or not r["slices"].sum():
            continue
        s_ = s_ or {"slices": np.zeros_like(r["slices"]), "who": np.zeros_like(r["who"])}
        lines += ["", f"  {spec['label']} (by the {spec['who']}): {r['slices'].sum() / g_r:.2f} a game real,"
                      f" {s_['slices'].sum() / g_s:.2f} sim",
                  f"  {'':28s}{'n real':>8s}" + "".join(f"{o[:15]:>16s}" for o in spec["options"])]
        for i, label in enumerate(v21.SLICE_NAMES[name]):
            rr, ss = r["slices"][i], s_["slices"][i]
            if not rr.sum() and not ss.sum():
                continue
            lines.append(f"  {label[:28]:28s}{int(rr.sum()):>8d}" + "".join(
                f"{100 * rr[k] / max(1, rr.sum()):>9.1f}/{100 * ss[k] / max(1, ss.sum()):<6.1f}"
                for k in range(len(spec["options"]))))
        opt, need = v21.LEAN_SHOWN[name]
        n_r = r["who"].sum(1)
        ok = [i for i in np.flatnonzero(n_r >= need) if i in names and s_["who"][i].sum()]
        if len(ok) >= 3:
            share_r = np.array([r["who"][i, opt] / n_r[i] for i in ok])
            share_s = np.array([s_["who"][i, opt] / s_["who"][i].sum() for i in ok])
            order = np.argsort(-share_r)
            pick = list(order[:6]) + list(order[-6:]) if len(order) > 12 else list(order)
            corr = np.corrcoef(share_r, share_s)[0, 1] if share_r.std() > 0 and share_s.std() > 0 else float("nan")
            lines.append(f"  {spec['options'][opt]}, by gamer ({need}+ real; {len(ok)} gamers; real and sim shares"
                         f" correlate {corr:.2f}):")
            lines.append("    " + ", ".join(f"{names[ok[j]]} {100 * share_r[j]:.0f}/{100 * share_s[j]:.0f}"
                                            f" ({int(n_r[ok[j]])})" for j in pick))
    return lines


def nudge_lines(results):
    """The nudges the sim took to meet the inputs (a unit: a point a side in the sim)."""
    got = np.array([r["nudge"] for r in results if r.get("nudge") is not None], float).reshape(-1, 2)
    if not len(got):
        return []
    x = got.ravel()
    return [f"  NUDGE: each side's offense moved along the strength direction to meet its expected points"
            f" (a unit: a point a side in the sim): mean {x.mean():+.2f}, sd {x.std():.2f}, {100 * np.mean(np.abs(x) >= v21.NUDGE_MAX - 1e-9):.1f}%"
            " at the limit; home {:+.2f}, away {:+.2f}".format(got[:, 0].mean(), got[:, 1].mean()), ""]


def report(out):
    res = out["results"]
    nudged = (", nudged to the inputs" if out.get("nudge_share", 1.0) == 1.0
              else f", nudged {100 * out.get('nudge_share', 1.0):.0f}% of the way to the inputs") if out.get("nudge") else ""
    lines = [f"  v2.1 ({out['mode']}{nudged}) on {len(res):,} settled"
             f" matches {out['since']} to {out['until']}: {2 * out['n_paths']:,} kick-off games each,"
             f" {out['state_paths']:,} from every {out['every']} real state", ""]
    lines += nudge_lines(res)
    lines += kickoff_lines(res)
    lines += grade_lines(res)
    lines += margin_lines(res)
    lines += people_lines(res, "gamer")
    lines += people_lines(res, "team", top=10, min_matches=MIN_GAMER_MATCHES)
    lines += step_lines(res, out["share"])
    lines += census_lines(out["census"], out["real"])
    lines += decision_lines(out["census"], out["real"], out.get("gamer_names"))
    lines += ["", "  (input: the pre-match expected points; nothing here is fitted back into v2.1 -- the gaps show"
                  " where its draws miss)"]
    return lines


def write_csv(path, out):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["MATCH_CODE", "start", "home", "away", "home_team", "away_team", "input_home", "input_away",
                    "sim_home", "sim_away", "real_home", "real_away", "sim_p_home",
                    "sim_margin_sd", "sim_total_sd", "prior_margin_sd", "prior_total_sd"])
        for x in out["results"]:
            (g1, g2), (t1, t2) = x["people"]
            w.writerow([x["code"], x["row"].get("SCHEDULED_START_TIME_UTC", ""), g1, g2, t1, t2,
                        *(round(v, 3) for v in x["means"]), *(round(v, 3) for v in x["kick"]),
                        *x["finals"], round(x["p_home"], 4),
                        *_spreads(x.get("book")), *_spreads(x.get("prior_book"))])


def write_states_csv(path, out):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["MATCH_CODE", "row", "period", "clock", "step", "home_score", "away_score", "input_home_left",
                    "input_away_left", "sim_home_left", "sim_away_left", "real_home_left", "real_away_left",
                    "sim_p_home"])
        share = out["share"]
        for x in out["results"]:
            for s_ in x["states"]:
                sl = int(v21.share_slice(s_["period"], s_["clock"]))
                h, a = s_["score"]
                w.writerow([x["code"], s_["row"], s_["period"], s_["clock"], STEPS[int(step_of(s_["period"], s_["clock"]))],
                            h, a, round(x["means"][0] * share[sl], 3), round(x["means"][1] * share[sl], 3),
                            round(s_["sim"][0] - h, 3), round(s_["sim"][1] - a, 3),
                            x["finals"][0] - h, x["finals"][1] - a, round(s_["p_home"], 4)])
