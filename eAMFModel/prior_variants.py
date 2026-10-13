"""v14 priced off different pre-match priors, side by side on a few chosen matches.

`v14-variants DIR` makes light copies of a v14 build that differ only in the prior they price with.
Each copy shares the build's tables, grid, profiles and ratings; only the prior file and shrink differ:

    DIR_raw       the same glmer fits, without the prior shrink
    DIR_layered   the same glmer fits and shrink, with the post-game form layer on top (form_layer.py)
    DIR_nb2       NB2 instead of glmer: fitted at the build's cut-off, refitted every day after it
                  (rolling_prior.py) and shrunk by its own fit_shrink, as the glmer prior is

The build itself is the shrunk glmer. Each copy can then stand in the calibrator as its own candidate
(`--candidate v14,v14-raw=DIR_raw,...`), so the same bets are re-priced by every prior.

`v14-priors` lays out a list of matches as each prices them: prod's closing pre-match prices (and
the expected margin and total they imply), each prior's expected points as priced (and as the bare
prior, before the shrink and layers), the v14 kick-off simulation's expected points and prices off
them at prod's lines, the result, and -- given the calibrator's bets_sim.csv -- what each kept of
the bets on that match.
"""

import csv
import datetime as dt
import json
import math
import os
import shutil
from statistics import NormalDist

import numpy as np

from . import form_layer, nb2_prior, players, v14, v14_stream

VARIANTS = ("raw", "layered", "nb2")
SHARED = ("v14tables.npz", "v14grid.npz", "v14players.json", "v14big.json")
DEFAULT_PATHS = 4000
IMPLIED_SD = form_layer.NOISE           # a match's margin and total sd about the price (fitted)


def _prior_meta(model_dir):
    with open(os.path.join(model_dir, v14.PRIOR_FILE), encoding="utf-8") as fh:
        return json.load(fh)


def _shrink(model_dir):
    path = os.path.join(model_dir, v14.SHRINK_FILE)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _copy_shared(model_dir, out):
    os.makedirs(out, exist_ok=True)
    for name in SHARED:
        src = os.path.join(model_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out, name))


def _write_meta(out, meta):
    with open(os.path.join(out, v14.PRIOR_FILE), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)


def cut_off(model_dir):
    """The build's cut-off: its shrink's, else its own pre-match fit's."""
    shrink = _shrink(model_dir)
    if shrink and shrink.get("before"):
        return dt.datetime.fromisoformat(shrink["before"][:19])
    meta = _prior_meta(model_dir)
    kind = meta.get("prior", "nb2")
    own = os.path.join(model_dir, kind)
    if v14.PRIORS[kind].exists(own):
        return dt.datetime.fromisoformat(v14.PRIORS[kind](own).meta["before"][:19])
    raise SystemExit(f"{model_dir}: no shrink or pre-match fit to read the build's cut-off from")


def make(model_dir, history, variants=VARIANTS, workers=2, until=None, daily=None):
    """Make the variants' directories beside `model_dir`: {name: directory}. `daily(directory,
    history, prior, since, until, workers)` fits and attaches the NB2 variant's days (the CLI's
    _daily); without it the NB2 variant prices from its single fit."""
    model_dir = os.path.normpath(model_dir)
    meta = _prior_meta(model_dir)
    if meta.get("prior", "nb2") != "glmer":
        raise SystemExit(f"{model_dir}: built on {meta.get('prior', 'nb2')}, not glmer; the variants"
                         " are the glmer build's")
    shrink = _shrink(model_dir)
    out = {}
    for name in variants:
        d = f"{model_dir}_{name}"
        _copy_shared(model_dir, d)
        if name in ("raw", "layered") and not meta.get("rolling"):
            # priced from the build's single fit (no daily fits attached): the copy needs that fit too
            own, copy = os.path.join(model_dir, meta["prior"]), os.path.join(d, meta["prior"])
            if os.path.isdir(own):
                shutil.copytree(own, copy, dirs_exist_ok=True)
        if name == "raw":
            if os.path.exists(os.path.join(d, v14.SHRINK_FILE)):
                os.remove(os.path.join(d, v14.SHRINK_FILE))
            _write_meta(d, {k: v for k, v in meta.items() if k != form_layer.SETTING})
        elif name == "layered":
            _write_meta(d, dict(meta, **{form_layer.SETTING: True}))
            if shrink:
                shutil.copy2(os.path.join(model_dir, v14.SHRINK_FILE), os.path.join(d, v14.SHRINK_FILE))
        elif name == "nb2":
            _make_nb2(model_dir, d, history, shrink, workers, until, daily)
        else:
            raise ValueError(f"no variant {name!r} (there are {', '.join(VARIANTS)})")
        out[name] = d
    return out


def _make_nb2(model_dir, d, history, shrink, workers, until, daily):
    """The NB2 variant: its fit at the build's cut-off, its own shrink, and (with `daily`) a fit for
    every day after it."""
    before = cut_off(model_dir)
    own = os.path.join(d, "nb2")
    if not nb2_prior.Prematch.exists(own) or \
            nb2_prior.Prematch(own).meta.get("before", "")[:19] != before.isoformat()[:19]:
        print(f"  nb2: fitting at the build's cut-off, {before:%Y-%m-%d}", flush=True)
        nb2_prior.Prematch.build(history, own, before)
    _write_meta(d, {"prior": "nb2"})
    pre = nb2_prior.Prematch(own)
    if shrink and not os.path.exists(os.path.join(d, v14.SHRINK_FILE)):
        print("  nb2: its own prior shrink (NB2 refitted at the start of each stretch)", flush=True)
        got = v14.fit_shrink(history, "nb2", before, (shrink["total_centre"], shrink["margin_centre"]),
                             d, scale=pre.scale)
        if got is not None:
            with open(os.path.join(d, v14.SHRINK_FILE), "w", encoding="utf-8") as fh:
                json.dump(got, fh, indent=1)
    if daily is not None:
        daily(d, history, "nb2", before.date(), until, workers)


# -- the side-by-side table ------------------------------------------------------------------------

def _finals(r):
    try:
        return float(r["PLAYER_1_FINAL_SCORE"]), float(r["PLAYER_2_FINAL_SCORE"])
    except (KeyError, TypeError, ValueError):
        return None


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def prod_prices(snapshots, codes):
    """{match: prod's closing pre-match prices off the snapshot export: ML home, spread (line, P home
    covers), total (line, P over)}."""
    want, out = set(codes), {}
    with open(snapshots, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            code = r.get("match_code")
            if code in want and code not in out and r.get("prematch_prob_50"):
                out[code] = dict(ml=_f(r.get("prematch_prob_50")),
                                 spread=(_f(r.get("prematch_line_52")), _f(r.get("prematch_prob_52"))),
                                 total=(_f(r.get("prematch_line_54")), _f(r.get("prematch_prob_54"))))
    return out


def implied(line, p, sd):
    """The mean a normal with this sd needs to put P(X > line) at p."""
    if line is None or p is None or not 0 < p < 1:
        return None
    return line + sd * NormalDist().inv_cdf(p)


def bets_by_match(path, labels, codes):
    """{match: {"stake": .., "prod": revenue, label: revenue, ...}} off a calibrator bets_sim.csv:
    the settled bets every label re-priced (a lone candidate's columns carry no suffix)."""
    want, out = set(codes), {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        cols = reader.fieldnames or []
        suffix = {lb: (f"_{lb}" if f"candidate_revenue_{lb}" in cols else "") for lb in labels}
        missing = [lb for lb in labels if not suffix[lb] and "candidate_revenue" not in cols]
        if missing or (sum(1 for s in suffix.values() if not s) > 1):
            raise SystemExit(f"{path}: no re-priced columns for {', '.join(lb for lb in labels if not suffix[lb])}"
                             " -- run the calibrator's bets with every model as a candidate")
        for r in reader:
            code = r.get("match_code")
            if code not in want or str(r.get("excluded")) == "True" or r.get("result") not in ("won", "lost", "push"):
                continue
            revs = []
            for lb in labels:
                s = suffix[lb]
                if str(r.get(f"candidate_ok{s}")) != "True" or _f(r.get(f"candidate_revenue{s}")) is None:
                    break
                revs.append(_f(r[f"candidate_revenue{s}"]))
            else:
                m = out.setdefault(code, {"stake": 0.0, "prod": 0.0, **{lb: 0.0 for lb in labels}})
                m["stake"] += _f(r["stake"]) or 0.0
                m["prod"] += _f(r["revenue"]) or 0.0
                for lb, v in zip(labels, revs):
                    m[lb] += v
    return out


def _prof(model_dir, rows):
    tables_path, _ = v14_stream.model_paths(model_dir)
    book, ratings = v14.players_book(tables_path), v14.big_ratings(tables_path)
    teams = v14.match_teams(rows)
    out = {}
    for r in rows:
        pair = (r["PLAYER_1_HANDLE"].strip().upper(), r["PLAYER_2_HANDLE"].strip().upper())
        prof = (book.profile(pair[0]), book.profile(pair[1])) if book else (players.Profile(), players.Profile())
        if ratings:
            prof = v14.with_big(prof, ratings, pair, *v14.side_teams(pair, teams.get(r["MATCH_CODE"])))
        out[r["MATCH_CODE"]] = prof
    return out


def _bare(pre):
    while hasattr(pre, "pre"):
        pre = pre.pre
    return pre


def _book_stats(mp_, tp, prod):
    """Expected home, away, total, margin and P(home win) off a margin and total distribution, and
    its prices at prod's spread and total lines."""
    mid = v14.MARGIN_MAX
    k = np.arange(len(mp_)) - mid
    margin, total = float((k * mp_).sum()), float((np.arange(len(tp)) * tp).sum())
    out = dict(home=(total + margin) / 2, away=(total - margin) / 2, total=total, margin=margin,
               p_home=float(mp_[mid + 1:].sum() / max(1e-12, 1 - mp_[mid])))
    if prod:
        if prod["spread"][0] is not None:
            out["p_spread"] = float(v14.market_prob(52, prod["spread"][0], mp_, tp))
        if prod["total"][0] is not None:
            out["p_over"] = float(v14.market_prob(54, prod["total"][0], mp_, tp))
    return out


def compare(models, history, codes, snapshots=None, bets_csv=None, n_paths=DEFAULT_PATHS, seed=0):
    """Per match, in the list's order: its history row, finals, prod's prices, and for each model
    (label -> directory) the prior's expected points as priced and bare, and the kick-off sim's."""
    by_code = {r["MATCH_CODE"]: r for r in history}
    rows = [by_code[c] for c in codes if c in by_code]
    missing = [c for c in codes if c not in by_code]
    if missing:
        print(f"  not in the history (refresh it?): {', '.join(missing)}")
    prod = prod_prices(snapshots, codes) if snapshots and os.path.exists(snapshots) else {}
    out = [dict(row=r, finals=_finals(r), prod=prod.get(r["MATCH_CODE"]), models={}) for r in rows]
    for label, model_dir in models.items():
        tables_path, grid_path = v14_stream.model_paths(model_dir)
        tables, grid = v14.sim.Tables.load(tables_path), v14.PriorGrid.load(grid_path)
        pre = v14.prematch_model(tables_path)
        if pre is None:
            raise SystemExit(f"{model_dir}: no pre-match model")
        priced = pre.means(rows, results=history)
        bare = _bare(pre)
        raw = priced if bare is pre else bare.means(rows, results=history)
        prof = _prof(model_dir, rows)
        variant = v14.Variant("v14")
        rng = np.random.default_rng(seed)
        for x in out:
            code = x["row"]["MATCH_CODE"]
            if code not in priced:
                continue
            mp_, tp = v14_stream.kickoff_book(tables, grid, variant, code, n_paths, rng, prof[code], priced[code])
            x["models"][label] = dict(prior=tuple(priced[code]), bare=tuple(raw.get(code, priced[code])),
                                      sim=_book_stats(mp_, tp, x["prod"]))
    if bets_csv:
        money = bets_by_match(bets_csv, list(models), codes)
        for x in out:
            x["bets"] = money.get(x["row"]["MATCH_CODE"])
    return out


def _pct(v):
    return "      -" if v is None else f"{100 * v:6.1f}%"


def _num(v, fmt="{:6.1f}"):
    return "     -" if v is None else fmt.format(v)


def report(results, labels):
    """Lines of text: each match, then the list as a whole."""
    lines = []
    sd_m, sd_t = IMPLIED_SD
    head = (f"    {'':18s}{'home':>7s}{'away':>7s}{'total':>7s}{'margin':>8s}{'P home':>9s}{'P cover':>9s}"
            f"{'P over':>9s}{'kept':>9s}")
    for x in results:
        r, f, p = x["row"], x["finals"], x["prod"]
        code = r["MATCH_CODE"]
        res = f"{f[0]:.0f}-{f[1]:.0f}" if f else "?"
        spread = f"{p['spread'][0]:+g}" if p and p["spread"][0] is not None else "-"
        total = f"{p['total'][0]:g}" if p and p["total"][0] is not None else "-"
        bets = x.get("bets")
        lines += ["", f"  {code}  {r['PLAYER_1_HANDLE']} ({r.get('PLAYER_1_TEAM', '')}) v {r['PLAYER_2_HANDLE']}"
                      f" ({r.get('PLAYER_2_TEAM', '')}), {r['SCHEDULED_START_TIME_UTC'][:16]}: final {res}"
                      f" (home first); prod's lines: home's margin over {spread}, total {total}"
                      + (f"; {bets['stake']:,.0f} staked" if bets else ""), head]
        if p:
            m_ = implied(p["spread"][0], p["spread"][1], sd_m)
            t_ = implied(p["total"][0], p["total"][1], sd_t)
            h_ = (t_ + m_) / 2 if m_ is not None and t_ is not None else None
            a_ = (t_ - m_) / 2 if m_ is not None and t_ is not None else None
            kept = bets["prod"] / bets["stake"] if bets and bets["stake"] else None
            lines.append(f"    {'prod (implied)':18s}{_num(h_, '{:7.1f}')}{_num(a_, '{:7.1f}')}{_num(t_, '{:7.1f}')}"
                         f"{_num(m_, '{:+8.1f}')}{_pct(p['ml']):>9s}{_pct(p['spread'][1]):>9s}{_pct(p['total'][1]):>9s}"
                         f"{_pct(kept):>9s}")
        for lb in labels:
            got = x["models"].get(lb)
            if not got:
                lines.append(f"    {lb:18s}  not priced")
                continue
            ph, pa = got["prior"]
            bh, ba = got["bare"]
            s = got["sim"]
            kept = bets[lb] / bets["stake"] if bets and bets["stake"] else None
            if (round(bh, 2), round(ba, 2)) != (round(ph, 2), round(pa, 2)):
                lines.append(f"    {lb + ' bare prior':18s}{bh:7.1f}{ba:7.1f}{bh + ba:7.1f}{bh - ba:+8.1f}")
            lines.append(f"    {lb + ' prior':18s}{ph:7.1f}{pa:7.1f}{ph + pa:7.1f}{ph - pa:+8.1f}")
            lines.append(f"    {lb + ' sim':18s}{s['home']:7.1f}{s['away']:7.1f}{s['total']:7.1f}{s['margin']:+8.1f}"
                         f"{_pct(s['p_home']):>9s}{_pct(s.get('p_spread')):>9s}{_pct(s.get('p_over')):>9s}"
                         f"{_pct(kept):>9s}")
    lines += ["", "  the list as a whole (moneyline graded on the results, level ones left out):"]
    lines.append(f"    {'':18s}{'ML Brier':>10s}{'margin err':>12s}{'total err':>11s}{'kept':>9s}")
    rows = [x for x in results if x["finals"]]
    staked = sum(x["bets"]["stake"] for x in results if x.get("bets"))

    def summary(name, p_home, margin, total, revenue):
        ml = [(p, 1.0 if f[0] > f[1] else 0.0) for p, f in zip(p_home, (x["finals"] for x in rows))
              if p is not None and f[0] != f[1]]
        brier = sum((p - w) ** 2 for p, w in ml) / len(ml) if ml else None
        me = [abs((f[0] - f[1]) - m) for m, f in zip(margin, (x["finals"] for x in rows)) if m is not None]
        te = [abs((f[0] + f[1]) - t) for t, f in zip(total, (x["finals"] for x in rows)) if t is not None]
        lines.append(f"    {name:18s}{_num(brier, '{:10.4f}')}{_num(sum(me) / len(me) if me else None, '{:12.1f}')}"
                     f"{_num(sum(te) / len(te) if te else None, '{:11.1f}')}"
                     f"{_pct(revenue / staked if staked and revenue is not None else None):>9s}")

    prod_rows = [x["prod"] for x in rows]
    summary("prod", [p["ml"] if p else None for p in prod_rows],
            [implied(*p["spread"], sd_m) if p else None for p in prod_rows],
            [implied(*p["total"], sd_t) if p else None for p in prod_rows],
            sum(x["bets"]["prod"] for x in results if x.get("bets")) if staked else None)
    for lb in labels:
        got = [x["models"].get(lb) for x in rows]
        summary(lb + " sim", [g["sim"]["p_home"] if g else None for g in got],
                [g["sim"]["margin"] if g else None for g in got], [g["sim"]["total"] if g else None for g in got],
                sum(x["bets"][lb] for x in results if x.get("bets")) if staked else None)
    lines += ["", f"  prod (implied): the mean a normal with sd {sd_m:g} (margin) / {sd_t:g} (total) needs to give"
                  " prod's closing pre-match spread and total prices; P cover / P over at prod's lines; kept: the"
                  " book's revenue over stake on the match's bets every model re-priced (in play too)"]
    return lines


def write_csv(path, results, labels):
    keys = ["MATCH_CODE", "home", "away", "final_home", "final_away", "prod_ml", "prod_spread_line",
            "prod_spread_p", "prod_total_line", "prod_total_p", "stake", "prod_revenue"]
    for lb in labels:
        keys += [f"{lb}_{k}" for k in ("bare_home", "bare_away", "prior_home", "prior_away", "sim_home",
                                         "sim_away", "sim_p_home", "sim_p_cover", "sim_p_over", "revenue")]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(keys)
        for x in results:
            r, f, p, b = x["row"], x["finals"] or (None, None), x["prod"] or {}, x.get("bets") or {}
            row = [r["MATCH_CODE"], r["PLAYER_1_HANDLE"], r["PLAYER_2_HANDLE"], *f, p.get("ml"),
                   *(p.get("spread") or (None, None)), *(p.get("total") or (None, None)),
                   b.get("stake"), b.get("prod")]
            for lb in labels:
                g = x["models"].get(lb)
                if not g:
                    row += [None] * 10
                    continue
                s = g["sim"]
                row += [*(round(v, 3) for v in g["bare"]), *(round(v, 3) for v in g["prior"]),
                        round(s["home"], 3), round(s["away"], 3), round(s["p_home"], 4),
                        None if s.get("p_spread") is None else round(s["p_spread"], 4),
                        None if s.get("p_over") is None else round(s["p_over"], 4), b.get(lb)]
            w.writerow(row)
