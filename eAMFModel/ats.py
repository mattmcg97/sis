"""ats: the spread against the moneyline -- prod's lines and moneyline against the real finals, the league
and gamer by gamer.

Pre-match, off each match's prod quotes before the first play (the export's prematch_line_52,
prematch_prob_52, prematch_prob_50: the home margin the spread is set at, prod's chance the home side
covers it, and of a home win):

  the line       how often prod's favourite covers its line (pushes out; 50% when the line is the median
                 of the margin), and the real margin less the line, its mean against its median
  the moneyline  how often the favourite wins against prod's moneyline, and the two set against each
                 other: a favourite wins whenever it covers and also when it wins by less than the line
                 (a 1.5-point favourite by exactly 1), so prod's moneyline less its cover chance should
                 be how often that happens -- less, and the moneyline is too flat for the line it sits on

In play, every PLAY_OVER of regulation with a live moneyline and spread: the favourite at that moment
against prod's moneyline and its line, by price and half. Each match counts for its own rows; every
interval is 95% over matches.

Gamer by gamer: each match from the gamer's side -- its margin less prod's line (against the spread),
whether it covered, and its win less prod's moneyline chance -- and whether that repeats, the matches
split by alternate days and the two halves' means correlated across gamers (playstyle's test). A
gamer prod prices short every time covers more in both halves; noise does not.

    python -m eAMFModel ats --snapshots eAMFCalibrator/out/scouting_playover.csv \\
        --history eAMFCalibrator/out/match_history.csv --out eAMFCalibrator/out/ats

It reads the export alone: no Snowflake, no simulation."""

import csv
import math
import os
from collections import defaultdict

import numpy as np

from . import v21
from .playstyle import _ci, _date, _n, shrunk, split_half, tau2

MIN_MATCHES = 40
MIN_HALF = 15
BOOT = 1000
LINE_BANDS = ((0.5, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 5.0), (5.0, 99.0))
PRICE_BANDS = ((0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 0.98))


def _f(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _p(v):
    x = _f(v)
    if x is None:
        return None
    x = x / 100.0 if x > 1.0 else x
    return x if 0.0 < x < 1.0 else None


def read(data, history, since=None, until=None):
    """The matches (one each: prod's pre-match line, cover chance and moneyline, the final) and the
    in-play rows (one a PLAY_OVER of regulation with both prices live)."""
    hist = {h["MATCH_CODE"]: h for h in (history or [])}
    when = {c: _date(c, hist[c]) for c in data if c in hist}
    codes = sorted((c for c, d in when.items() if d is not None and v21.match_final(hist[c]) is not None
                    and (since is None or d >= since) and (until is None or d < until)), key=lambda c: (when[c], c))
    M, IP = defaultdict(list), defaultdict(list)
    for m, c in enumerate(codes):
        rows = data[c]
        pair, _ = v21.match_people(c, rows, hist)
        fin = v21.match_final(hist[c])
        margin = fin[0] - fin[1]
        r0 = rows[0] if rows else {}
        line, pc, p = _f(r0.get("prematch_line_52")), _p(r0.get("prematch_prob_52")), _p(r0.get("prematch_prob_50"))
        if line is not None and p is not None and abs(line) <= 40:
            for k, v in (("match", m), ("half", when[c].toordinal() % 2), ("home", pair[0]), ("away", pair[1]),
                         ("line", line), ("pc", np.nan if pc is None else pc), ("p", p), ("margin", margin)):
                M[k].append(v)
        for r in rows:
            try:
                period = int(float(r.get("period") or 0))
            except ValueError:
                continue
            if not 1 <= period <= 4 or str(r.get("live_50")) not in ("1", "1.0") or str(r.get("live_52")) not in ("1", "1.0"):
                continue
            p50, l52, p52 = _p(r.get("prob_50")), _f(r.get("line_52")), _p(r.get("prob_52"))
            if p50 is None or l52 is None or p52 is None:
                continue
            for k, v in (("match", m), ("period", period), ("p", p50), ("line", l52), ("pc", p52), ("margin", margin)):
                IP[k].append(v)
    arr = lambda d: {k: (np.asarray(v, object) if k in ("home", "away") else np.asarray(v, float)) for k, v in d.items()}
    return dict(pre=arr(M), live=arr(IP), codes=codes, since=when[codes[0]] if codes else None,
                until=when[codes[-1]] if codes else None)


def _boot(match, values, B=BOOT, seed=0):
    """(mean, lo, hi) of `values` over rows, with a 95% interval over their matches."""
    v = np.asarray(values, float)
    ok = np.isfinite(v)
    if not ok.any():
        return float("nan"), float("nan"), float("nan")
    u, mi = np.unique(np.asarray(match)[ok], return_inverse=True)
    s = np.bincount(mi, weights=v[ok], minlength=len(u))
    c = np.bincount(mi, minlength=len(u)).astype(float)
    W = np.random.default_rng(seed).multinomial(len(u), [1.0 / len(u)] * len(u), size=B).astype(float)
    lo, hi = _ci((W @ s) / np.maximum(W @ c, 1e-9))
    return float(s.sum() / c.sum()), lo, hi


def favourite(line, p, pc, margin):
    """Each row from prod's favourite's side: (its line, moneyline chance, cover chance, margin)."""
    home = p >= 0.5
    s = np.where(home, 1.0, -1.0)
    return line * s, np.where(home, p, 1 - p), np.where(home, pc, 1 - pc), margin * s


def prematch_lines(pre, B=BOOT):
    """The league pre-match: the favourite (by the line, level lines out) against its line and prod's
    moneyline, overall and by the line."""
    line = pre["line"]
    keep = line != 0
    s = np.sign(line[keep])
    fL, fm = line[keep] * s, pre["margin"][keep] * s
    fp = np.where(s > 0, pre["p"][keep], 1 - pre["p"][keep])
    fpc = np.where(s > 0, pre["pc"][keep], 1 - pre["pc"][keep])
    match = pre["match"][keep]
    cover = np.where(fm == fL, np.nan, (fm > fL).astype(float))
    win = (fm > 0).astype(float)
    short = ((fm > 0) & (fm <= fL)).astype(float)           # won, by less than the line (or on it)
    out = dict(n=int(keep.sum()), cover=_boot(match, cover, B), cover_said=float(np.nanmean(fpc)),
               won=_boot(match, win, B), won_said=float(fp.mean()), won_less=_boot(match, win - fp, B),
               short=_boot(match, short, B), short_said=float(np.nanmean(fp - fpc)),
               miss_mean=float((fm - fL).mean()), miss_median=float(np.median(fm - fL)), bands=[])
    for lo, hi in LINE_BANDS:
        k = (fL >= lo) & (fL < hi)
        if k.sum() < 20:
            continue
        out["bands"].append(dict(lo=lo, hi=hi, n=int(k.sum()), line=float(fL[k].mean()), mean=float(fm[k].mean()),
                                 median=float(np.median(fm[k])), cover=_boot(match[k], cover[k], B),
                                 cover_said=float(np.nanmean(fpc[k])), said=float(fp[k].mean()), won=_boot(match[k], win[k], B),
                                 short=float(short[k].mean()), short_said=float(np.nanmean(fp[k] - fpc[k]))))
    return out


def live_lines(live, B=BOOT):
    """In play: the favourite at each PLAY_OVER against prod's moneyline and its line, by price and half."""
    if not len(live.get("p", [])):
        return []
    fL, fp, fpc, fm = favourite(live["line"], live["p"], live["pc"], live["margin"])
    cover = np.where(fm == fL, np.nan, (fm > fL).astype(float))
    win = (fm > 0).astype(float)
    out = []
    for half, keep in (("1st half", live["period"] <= 2), ("2nd half", live["period"] >= 3)):
        for lo, hi in PRICE_BANDS:
            k = keep & (fp >= lo) & (fp < hi)
            if k.sum() < 50:
                continue
            m = live["match"][k]
            out.append(dict(half=half, lo=lo, hi=hi, rows=int(k.sum()), matches=len(np.unique(m)),
                            said=float(fp[k].mean()), won=_boot(m, win[k], B), less=_boot(m, win[k] - fp[k], B),
                            cover_said=float(fpc[k].mean()), cover=_boot(m, cover[k], B)))
    return out


def gamers(pre, min_matches=MIN_MATCHES, min_half=MIN_HALF, B=BOOT):
    """Each gamer's matches from its side: its margin less prod's line, how often it covered, its wins less
    prod's moneyline chance; the halves of its matches (alternate days) for the trait test."""
    by = defaultdict(lambda: defaultdict(list))
    for i in range(len(pre["line"])):
        for side, g in ((1.0, pre["home"][i]), (-1.0, pre["away"][i])):
            if not g:
                continue
            L, mg = pre["line"][i] * side, pre["margin"][i] * side
            p = pre["p"][i] if side > 0 else 1 - pre["p"][i]
            d = by[g]
            d["match"].append(pre["match"][i])
            d["half"].append(pre["half"][i])
            d["ats"].append(mg - L)
            d["cover"].append(np.nan if mg == L else float(mg > L))
            d["over"].append(float(mg > 0) - p)
            d["line"].append(L)
    rows = []
    for g, d in sorted(by.items()):
        if len(d["match"]) < min_matches:
            continue
        a = {k: np.asarray(v, float) for k, v in d.items()}
        ats = _boot(a["match"], a["ats"], B, seed=len(rows))
        over = _boot(a["match"], a["over"], B, seed=len(rows) + 1)
        halves = [a["half"] == h for h in (0, 1)]
        half = lambda k, h: float(a[k][halves[h]].mean()) if halves[h].sum() >= min_half else float("nan")
        rows.append(dict(gamer=g, matches=len(a["match"]), line=float(a["line"].mean()), ats=ats[0], ats_lo=ats[1],
                         ats_hi=ats[2], ats_se=float(a["ats"].std() / math.sqrt(len(a["ats"]))),
                         cover=float(np.nanmean(a["cover"])), over=over[0], over_lo=over[1], over_hi=over[2],
                         over_se=float(a["over"].std() / math.sqrt(len(a["over"]))),
                         ats_a=half("ats", 0), ats_b=half("ats", 1), over_a=half("over", 0), over_b=half("over", 1)))
    return rows


def run(data, history, since=None, until=None, min_matches=MIN_MATCHES, min_half=MIN_HALF, B=BOOT):
    got = read(data, history, since, until)
    if not len(got["pre"].get("line", [])):
        raise SystemExit("ats: no settled matches with prod's pre-match prices in the window")
    rows = gamers(got["pre"], min_matches, min_half, B)
    out = dict(pre=prematch_lines(got["pre"], B), live=live_lines(got["live"], B), gamers=rows, matches=len(got["codes"]),
               priced=int(len(got["pre"]["line"])), live_rows=int(len(got["live"].get("p", []))),
               since=got["since"], until=got["until"], min_half=min_half)
    for key in ("ats", "over"):
        t2, mean = tau2([r[key] for r in rows], [r[f"{key}_se"] for r in rows])
        sh, sd = shrunk([r[key] for r in rows], [r[f"{key}_se"] for r in rows], t2, mean)
        for r, a, b in zip(rows, sh, sd):
            r[f"{key}_shrunk"], r[f"{key}_sd"] = float(a), float(b)
        out[f"{key}_tau"], out[f"{key}_mean"] = math.sqrt(t2), mean
        out[f"{key}_trait"] = split_half([r[f"{key}_a"] for r in rows], [r[f"{key}_b"] for r in rows])
        passes = bool(np.isfinite(out[f"{key}_trait"]["lo"]) and out[f"{key}_trait"]["lo"] > 0)
        out[f"{key}_passes"] = passes
        for r in rows:
            z = (r[f"{key}_shrunk"] - mean) / r[f"{key}_sd"] if passes and r[f"{key}_sd"] else 0.0
            r[f"{key}_group"] = ("" if not passes else "beats prod" if z > 1.2816 else "short of prod" if z < -1.2816
                                 else "as priced")
    return out


def _iv(x, p=2, pct=False):
    """A signed value and its interval (pct: in points of a hundred)."""
    k = 100.0 if pct else 1.0
    return f"{_n(k * x[0], p)} [{_n(k * x[1], p)}, {_n(k * x[2], p)}]"


def _rate(x, p=1):
    """A rate in % with its interval."""
    f = lambda v: f"{100 * v:.{p}f}" if np.isfinite(v) else "-"
    return f"{f(x[0])}% [{f(x[1])}, {f(x[2])}]"


def report(out):
    pre = out["pre"]
    L = [f"  ATS: prod's lines and moneyline against the real finals -- {out['since']} to {out['until']}, {out['matches']:,}"
         f" matches, {out['priced']:,} with prod's pre-match prices, {out['live_rows']:,} in-play PLAY_OVERs with both live;"
         " 95% intervals over matches", "",
         "  PRE-MATCH: prod's favourite (by its line; level lines out) against the line and the moneyline",
         f"    covered its line {_rate(pre['cover'])} (prod's cover chance {100 * pre['cover_said']:.1f}%); won"
         f" {_rate(pre['won'])} against prod's moneyline {100 * pre['won_said']:.1f}% (won less said"
         f" {_iv(pre['won_less'], 1, True)} points)",
         f"    the real margin less the line: mean {pre['miss_mean']:+.2f}, median {pre['miss_median']:+.2f} (both 0: the line"
         " the margin's middle)",
         f"    won without covering (by less than the line, or on it) {_rate(pre['short'])}; prod's moneyline less its"
         f" cover chance {100 * pre['short_said']:.1f}% (less than what happens: the moneyline too flat for its own line)",
         f"    {'favourite by':13s} {'n':>5s} {'line':>6s} {'mean':>6s} {'median':>7s} {'covered [95%]':>24s} {'said':>6s}"
         f" {'won [95%]':>24s} {'said':>6s} {'won, not covered':>17s} {'said':>6s}"]
    for b in pre["bands"]:
        L.append(f"    {b['lo']:4.1f} to {b['hi']:<4.1f}  {b['n']:5,d} {b['line']:+6.2f} {b['mean']:+6.2f} {b['median']:+7.1f}"
                 f" {_rate(b['cover']):>24s} {100 * b['cover_said']:6.1f} {_rate(b['won']):>24s}"
                 f" {100 * b['said']:6.1f} {100 * b['short']:17.1f} {100 * b['short_said']:6.1f}")
    L += ["", "  IN PLAY: the favourite at every PLAY_OVER (prod's moneyline above 50%) against its moneyline and its line",
          f"    {'half':9s} {'price':11s} {'rows':>8s} {'matches':>7s} {'said':>6s} {'won [95%]':>24s} {'won less said':>24s}"
          f" {'cover said':>10s} {'covered [95%]':>24s}"]
    for b in out["live"]:
        L.append(f"    {b['half']:9s} {b['lo']:.2f}-{b['hi']:.2f}  {b['rows']:8,d} {b['matches']:7,d} {100 * b['said']:6.1f}"
                 f" {_rate(b['won']):>24s} {_iv(b['less'], 1, True):>24s} {100 * b['cover_said']:10.1f}"
                 f" {_rate(b['cover']):>24s}")
    for key, title in (("ats", "AGAINST THE SPREAD: each gamer's margin less prod's line"),
                       ("over", "THE MONEYLINE: each gamer's wins less prod's moneyline chance")):
        t = out[f"{key}_trait"]
        L += ["", f"  {title}",
              f"    is it a trait? the halves of each gamer's matches (alternate days), {t['n']} gamers with {out['min_half']}+"
              f" matches in each: r = {_n(t['r'], 2)} [{_n(t['lo'], 2)}, {_n(t['hi'], 2)}] (0: noise); the spread of the true"
              f" values across gamers (tau) {out[f'{key}_tau']:.3f}",
              ("    -> a trait: grouped below (80% sure off the gamers' mean)" if out[f"{key}_passes"] else
               "    -> not a trait: no groups"),
              f"    {'gamer':14s} {'matches':>7s} {'line':>6s} {'margin less line':>24s} {'covered':>8s} {'wins less said':>24s}"
              f" {'shrunk':>7s} {'half A':>7s} {'half B':>7s}  group"]
        for r in sorted(out["gamers"], key=lambda r: -r[f"{key}_shrunk"]):
            L.append(f"    {r['gamer'][:14]:14s} {r['matches']:7,d} {r['line']:+6.2f}"
                     f" {_iv((r['ats'], r['ats_lo'], r['ats_hi'])):>24s} {100 * r['cover']:7.1f}%"
                     f" {_iv((r['over'], r['over_lo'], r['over_hi']), 1, True):>24s} {_n(r[f'{key}_shrunk'] * (1 if key == 'ats' else 100), 2, 7)}"
                     f" {_n(r[f'{key}_a'] * (1 if key == 'ats' else 100), 2, 7)} {_n(r[f'{key}_b'] * (1 if key == 'ats' else 100), 2, 7)}"
                     f"  {r[f'{key}_group']}")
    return L


def write(out_dir, out):
    os.makedirs(out_dir, exist_ok=True)
    lines = report(out)
    with open(os.path.join(out_dir, "ats.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    keys = ("gamer", "matches", "line", "ats", "ats_lo", "ats_hi", "cover", "over", "over_lo", "over_hi", "ats_shrunk",
            "over_shrunk", "ats_a", "ats_b", "over_a", "over_b", "ats_group", "over_group")
    rnd = lambda v: (round(v, 5) if isinstance(v, float) and np.isfinite(v) else "" if isinstance(v, float) else v)
    with open(os.path.join(out_dir, "ats_gamers.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(keys)
        for r in out["gamers"]:
            w.writerow([rnd(r[k]) for k in keys])
    return lines
