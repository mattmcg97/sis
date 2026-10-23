"""The bet moments as in-play states: one row per (match, prod message) of the in-play moneyline bets,
with the state at the bet, prod's and v2.1's P(home) at that message, the pre-match probabilities, and
the result. Plus a wider table of training states (moneyline and spread bets, any moment).

    python scripts/G/moments.py            -> out/G/moments_ml.csv, out/G/states_train.csv, out/G/moments.txt

Sources (all read-only):
  eAMFCalibrator/out_bets_late/bets_sim.csv   every bet Oct 1-18 with the feed's state at bet time (quarter,
        clock_left, down, the ABSOLUTE score band, moment, moved, price age), prod's quote of the bet's
        selection one lag back (stream_prob; market 50 = Home, 51 = Away, a fair book: 50 + 51 = 1), and
        v2.1's at the same message (candidate_prob_v21: the form-prior build, 2,000 games a snapshot;
        candidate_prob_v21-late: the late Q4 correction since removed).
  eAMFCalibrator/out_money/bets_money_points.zip   the same bets (bet_id) with `script` = home's lead x the
        share of regulation left at the bet's seen time (bet time less the operator's lag of +1 / -2 s), and
        p = prod's market-50 price. Home's signed lead = script / share left (share from bets_sim's quarter
        and clock): the only signed score in the container. Overtime has share 0, so no lead there.
  features.csv (code, start, s1, s2, prod_p_home), preds/A_best.csv, preds/C_strength.csv: pre-match.
"""

import os
import sys

import numpy as np
import pandas as pd

ML = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ML)
import harness as H  # noqa: E402

ROOT = "/home/user/sis/eAMFCalibrator"
OUT = os.path.join(ML, "out", "G")
BETS = os.path.join(ROOT, "out_bets_late", "bets_sim.csv")
MONEY = os.path.join(OUT, "data", "money", "bets_money_points.csv")   # unzipped from out_money/
QUARTER = 240.0
Q = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4, "OT": 5}
KEEP = ["bet_id", "match_code", "bet_time", "in_play", "market", "selection", "message", "stream_prob",
        "final_p1", "final_p2", "excluded", "match_phase", "moment", "feed_suspended", "moved",
        "price_age_seconds", "quarter", "clock_left", "down", "score_margin", "state_ok",
        "candidate_prob_v21", "candidate_ok_v21", "candidate_prob_v21-late", "latency_seconds", "stake"]


def _money():
    if not os.path.exists(MONEY):
        import zipfile
        os.makedirs(os.path.dirname(MONEY), exist_ok=True)
        with zipfile.ZipFile(os.path.join(ROOT, "out_money", "bets_money_points.zip")) as z:
            z.extract("bets_money_points.csv", os.path.dirname(MONEY))
    m = pd.read_csv(MONEY, usecols=["bet_id", "t", "p", "script", "group"], low_memory=False)
    m["bet_id"] = m.bet_id.astype(str)
    return m.rename(columns={"p": "money_p_home", "t": "seen"})


def band_ok(lead, band):
    a = np.abs(lead)
    return np.where(band == "level (0-2)", a <= 2, np.where(band == "1 score (3-8)", (a >= 3) & (a <= 8),
                    np.where(band == "2+ scores (9+)", a >= 9, False)))


def load_bets():
    b = pd.read_csv(BETS, usecols=KEEP, low_memory=False)
    b["bet_id"] = b.bet_id.astype(str)
    b = b[(b.in_play == True) & b.market.isin(["moneyline", "spread"])].copy()   # noqa: E712
    b = b.merge(_money(), on="bet_id", how="left")
    b["period"] = b.quarter.map(Q)
    b["secs_left"] = np.where(b.period <= 4, (4 - b.period) * QUARTER + b.clock_left, 0.0)
    b["share_left"] = b.secs_left / (4 * QUARTER)
    raw = np.where(b.share_left > 0, b.script / b.share_left.where(b.share_left > 0), np.nan)
    b["lead_raw"] = raw
    b["lead"] = np.round(raw)
    b["lead_ok"] = (np.isfinite(raw) & (np.abs(raw - np.round(raw)) <= 0.25)
                    & band_ok(np.round(raw), b.score_margin.to_numpy()))
    b["bet_time"] = pd.to_datetime(b.bet_time, format="mixed")
    home = b.selection == "Home"
    b["prod_home"] = np.where(b.market == "moneyline", np.where(home, b.stream_prob, 1 - b.stream_prob), np.nan)
    b["v21_home"] = np.where(b.market == "moneyline", np.where(home, b.candidate_prob_v21, 1 - b.candidate_prob_v21), np.nan)
    b["v21late_home"] = np.where(b.market == "moneyline",
                                 np.where(home, b["candidate_prob_v21-late"], 1 - b["candidate_prob_v21-late"]), np.nan)
    return b


def prematch():
    f = pd.read_csv(H.FEATURES, usecols=["code", "start", "s1", "s2", "prod_p_home"])
    f = f[f.start >= "2026-09-01"].copy()
    for name in ("A_best", "C_strength"):
        p = pd.read_csv(os.path.join(ML, "preds", f"{name}.csv")).rename(columns={"p_home": f"pre_{name}"})
        f = f.merge(p, on="code", how="left")
    return f.rename(columns={"code": "match_code", "prod_p_home": "pre_prod"})


def v21_kickoff(b_all):
    """v2.1's own pre-match P(home): its price on the match's last pre-match moneyline bet."""
    pm = b_all[(b_all.in_play == False) & (b_all.market == "moneyline") & b_all.candidate_prob_v21.notna()].copy()  # noqa: E712
    pm["bet_time"] = pd.to_datetime(pm.bet_time, format="mixed")
    pm = pm.sort_values("bet_time")
    pm["pre_v21"] = np.where(pm.selection == "Home", pm.candidate_prob_v21, 1 - pm.candidate_prob_v21)
    pm["pre_prod_bet"] = np.where(pm.selection == "Home", pm.stream_prob, 1 - pm.stream_prob)
    return pm.groupby("match_code")[["pre_v21", "pre_prod_bet"]].last().reset_index()


def last_scorer(timeline, q):
    """Each of q's states' last scorer (+1 home, -1 away, 0 none seen or a half since) and the seconds since,
    off the match's earlier bet states (timeline: match_code, bet_time, lead, secs_left), strictly before the
    state's bet time. A proxy for possession the bets do not carry: the side that conceded last receives."""
    by = {c: (g.bet_time.to_numpy(), g.lead.to_numpy(), g.secs_left.to_numpy())
          for c, g in timeline.sort_values("bet_time").groupby("match_code", sort=False)}
    sc = np.zeros(len(q))
    since = np.full(len(q), np.nan)
    for i, (code, t, lead, s) in enumerate(zip(q.match_code, q.bet_time.to_numpy(), q.lead.to_numpy(),
                                               q.secs_left.to_numpy())):
        if code not in by:
            continue
        bt, bl, bs = by[code]
        j = int(np.searchsorted(bt, t, side="left")) - 1
        first = s
        while j >= 0 and bl[j] == lead:
            first = bs[j]
            j -= 1
        if j >= 0:
            sc[i] = np.sign(lead - bl[j])
            since[i] = first - s
        elif lead != 0:
            sc[i] = np.sign(lead)
        if first > 480 >= s:
            sc[i] = 0.0
    return sc, since


def dedupe(b):
    """One row per (match, prod message): the bet nearest the message (least price age), its state."""
    b = b.sort_values(["match_code", "message", "price_age_seconds", "bet_time"], na_position="last")
    g = b.groupby(["match_code", "message"], sort=False)
    first = g.head(1).copy()
    agg = g.agg(n_bets=("bet_id", "size"), stake_all=("stake", "sum"),
                prod_home_sd=("prod_home", "std"), v21_home_sd=("v21_home", "std")).reset_index()
    return first.merge(agg, on=["match_code", "message"], how="left")


def main():
    os.makedirs(OUT, exist_ok=True)
    allb = pd.read_csv(BETS, usecols=KEEP, low_memory=False)
    pre = prematch().merge(v21_kickoff(allb), on="match_code", how="left")
    b = load_bets()
    lines = [f"bets in play, moneyline + spread: {len(b):,} on {b.match_code.nunique():,} matches; "
             f"joined to the money file's lead: {b.script.notna().mean():.1%}; lead read cleanly "
             f"(whole number within 0.25, inside the band): {b.lead_ok.mean():.1%}"]
    lines.append("  lead rounding |raw - round| > 0.02: "
                 f"{(np.abs(b.lead_raw - b.lead) > 0.02).mean():.1%} (the clock at the money's seen time "
                 "against bets_sim's at bet time)")
    # training states: every ML / spread bet's state, any moment, deduped per (match, message)
    tr = b[b.message.notna() & b.period.between(1, 4) & b.lead_ok & b.secs_left.notna()].copy()
    tr["is_ml"] = tr.market == "moneyline"
    tr = tr.sort_values(["match_code", "message", "is_ml", "price_age_seconds"], ascending=[True, True, False, True])
    tr = tr.groupby(["match_code", "message"], sort=False).head(1)
    tr = tr.merge(pre, on="match_code", how="left")
    timeline = b[b.lead_ok & b.period.between(1, 4)][["match_code", "bet_time", "lead", "secs_left"]]
    tr["last_scorer"], tr["since_score"] = last_scorer(timeline, tr)
    tr["y"] = np.where(tr.final_p1 > tr.final_p2, 1.0, np.where(tr.final_p1 < tr.final_p2, 0.0, np.nan))
    cols = ["match_code", "message", "start", "bet_time", "market", "period", "clock_left", "secs_left", "down",
            "last_scorer", "since_score",
            "moment", "feed_suspended", "moved", "lead", "score_margin", "y", "final_p1", "final_p2",
            "pre_A_best", "pre_C_strength", "pre_prod", "pre_v21"]
    tr[cols].to_csv(os.path.join(OUT, "states_train.csv"), index=False)
    lines.append(f"training states (ML + spread bets, Q1-Q4, lead read, one per match and message): {len(tr):,} "
                 f"on {tr.match_code.nunique():,} matches")

    ml = b[(b.market == "moneyline") & b.message.notna() & b.stream_prob.notna()].copy()
    lines.append(f"moneyline in-play bets with prod's price: {len(ml):,}; moments (match, message): "
                 f"{ml.groupby(['match_code', 'message']).ngroups:,}")
    mo = dedupe(ml).merge(pre, on="match_code", how="left")
    mo["last_scorer"], mo["since_score"] = last_scorer(timeline, mo)
    mo["y"] = np.where(mo.final_p1 > mo.final_p2, 1.0, np.where(mo.final_p1 < mo.final_p2, 0.0, np.nan))
    mo["day"] = mo.start.str[:10]
    checks = {
        "in Q1-Q4 (overtime has no lead)": mo.period.between(1, 4),
        "lead read cleanly": mo.lead_ok,
        "feed did not move between the price and the bet (moved = nothing)": mo.moved == "nothing",
        "v2.1 priced the same state as prod (candidate_ok_v21)": mo.candidate_ok_v21 == True,  # noqa: E712
        "not excluded by the bet checks": mo.excluded != True,  # noqa: E712
        "v2.1 price present": mo.v21_home.notna(),
        "pre-match A_best / C_strength / prod present": mo.pre_A_best.notna() & mo.pre_C_strength.notna() & mo.pre_prod.notna(),
        "not a tie": mo.y.notna(),
    }
    keep = pd.Series(True, index=mo.index)
    lines.append(f"\nmoments: {len(mo):,}; the filters, in turn:")
    for k, v in checks.items():
        keep &= v.fillna(False).astype(bool)
        lines.append(f"  {k:72s} {int(keep.sum()):7,d}")
    mo["clean"] = keep
    mo["broad"] = (mo.period.between(1, 4) & mo.lead_ok & (mo.candidate_ok_v21 == True) & (mo.excluded != True)  # noqa: E712
                   & mo.v21_home.notna() & mo.pre_A_best.notna() & mo.pre_prod.notna() & mo.y.notna())
    lines.append(f"  broad set (any `moved`, the state read at the bet): {int(mo.broad.sum()):,}")
    d = (mo.prod_home - mo.money_p_home).abs()
    lines.append(f"\nprod's P(home) from the bet's side against the money file's market-50 price: "
                 f"median |diff| {d.median():.4f}, 99th pct {d.quantile(.99):.4f}")
    lines.append(f"prod's price spread within a message (sd over its bets): mean {mo.prod_home_sd.mean():.4f}; "
                 f"v2.1's: {mo.v21_home_sd.mean():.6f}")
    out = ["match_code", "message", "day", "start", "bet_time", "period", "clock_left", "secs_left", "down",
           "last_scorer", "since_score",
           "moment", "feed_suspended", "moved", "price_age_seconds", "lead", "lead_raw", "score_margin",
           "prod_home", "money_p_home", "v21_home", "v21late_home", "y", "final_p1", "final_p2", "n_bets",
           "stake_all", "pre_A_best", "pre_C_strength", "pre_prod", "pre_v21", "pre_prod_bet", "clean", "broad"]
    mo[out].to_csv(os.path.join(OUT, "moments_ml.csv"), index=False)
    c = mo[mo.clean]
    lines.append(f"\nclean moments: {len(c):,} on {c.match_code.nunique():,} matches, {c.day.nunique()} days "
                 f"({c.day.min()} to {c.day.max()})")
    lines.append("  by quarter: " + ", ".join(f"Q{int(k)} {v:,}" for k, v in c.period.value_counts().sort_index().items()))
    lines.append("  by moment: " + ", ".join(f"{k} {v:,}" for k, v in c.moment.value_counts().items()))
    lines.append("  |lead|: " + ", ".join(f"{k} {v:,}" for k, v in c.lead.abs().clip(upper=9).value_counts().sort_index().items())
                 + " (9 = 9+)")
    lines.append("  by day: " + ", ".join(f"{k[5:]} {v}" for k, v in c.day.value_counts().sort_index().items()))
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(OUT, "moments.txt"), "w") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()
