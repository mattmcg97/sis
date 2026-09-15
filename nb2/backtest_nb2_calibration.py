#!/usr/bin/env python3
"""
Walk-forward calibration backtest for the NB2 pre-match rating model.

Fits the same joint player+team NB2 model as NBRatingTrial.py, but only
on a chronological TRAIN split, then prices the held-out TEST matches
purely out-of-sample (as if predicting before they happened) and checks
calibration against what actually happened -- the same exercise as the
in-play GAMEPLAI calibration check, applied to the pre-match model.

A single chronological train/test split (not full rolling walk-forward)
is used for speed: fitting is a nontrivial optimisation, so refitting at
every week/month would be expensive. This still avoids the circularity
of fitting on all data and checking fit against that same data, which
would be optimistic.

Reuses the actual statistical machinery from NBRatingTrial.py (NB2
log-likelihood, gradients) and NB2_schedule_predict.py (joint
Gamma-Poisson simulation) via import, rather than reimplementing it, so
this is testing the real model, not an approximation of it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import NBRatingTrial as fitmod
import NB2_schedule_predict as pricemod

TRAIN_FRACTION = 0.75
HALF_LIFE_DAYS = 60.0
N_SIMS = 20000
SEED = 260909

SD_PLAYER = 0.5
SD_TEAM = 0.25
SD_STREAM = 0.15
SD_BETA = 1.0
MU_LOG_ALPHA = -1.8
SD_LOG_ALPHA = 0.4
DISPERSION_SCALE_MIN = 0.05
DISPERSION_SCALE_MAX = 3.0
SHARED_FRACTION_MAX = 0.98


def load_clean(path):
    df = pd.read_csv(path)
    required = [
        "SCHEDULED_START_TIME_UTC", "STREAM_NUMBER",
        "PLAYER_1_HANDLE", "PLAYER_2_HANDLE",
        "PLAYER_1_TEAM", "PLAYER_2_TEAM",
        "PLAYER_1_FINAL_SCORE", "PLAYER_2_FINAL_SCORE",
    ]
    df = df.dropna(subset=required).copy()
    df["SCHEDULED_START_TIME_UTC"] = pd.to_datetime(df["SCHEDULED_START_TIME_UTC"], errors="coerce", utc=True)
    for c in ["PLAYER_1_FINAL_SCORE", "PLAYER_2_FINAL_SCORE"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["SCHEDULED_START_TIME_UTC", "PLAYER_1_FINAL_SCORE", "PLAYER_2_FINAL_SCORE"]).copy()
    df = df[(df["PLAYER_1_FINAL_SCORE"] >= 0) & (df["PLAYER_2_FINAL_SCORE"] >= 0)].copy()
    df["PLAYER_1_FINAL_SCORE"] = df["PLAYER_1_FINAL_SCORE"].astype(int)
    df["PLAYER_2_FINAL_SCORE"] = df["PLAYER_2_FINAL_SCORE"].astype(int)
    df["PLAYER_1_HANDLE"] = df["PLAYER_1_HANDLE"].astype(str).str.strip().str.upper()
    df["PLAYER_2_HANDLE"] = df["PLAYER_2_HANDLE"].astype(str).str.strip().str.upper()
    df["PLAYER_1_TEAM"] = df["PLAYER_1_TEAM"].astype(str).str.strip()
    df["PLAYER_2_TEAM"] = df["PLAYER_2_TEAM"].astype(str).str.strip()
    return df.sort_values("SCHEDULED_START_TIME_UTC").reset_index(drop=True)


def fit_nb2(df):
    """Fit the joint player+team NB2 model on df. Mirrors NBRatingTrial.py's main()."""
    df = df.copy()
    stream_num = pd.to_numeric(df["STREAM_NUMBER"], errors="coerce")
    df["_stream_key"] = stream_num if stream_num.notna().all() else df["STREAM_NUMBER"].astype(str)
    streams = sorted(df["_stream_key"].unique().tolist())

    players = sorted(pd.unique(df[["PLAYER_1_HANDLE", "PLAYER_2_HANDLE"]].to_numpy().ravel()).tolist())
    teams = sorted(pd.unique(df[["PLAYER_1_TEAM", "PLAYER_2_TEAM"]].to_numpy().ravel()).tolist())
    player_index = {p: i for i, p in enumerate(players)}
    team_index = {t: i for i, t in enumerate(teams)}
    stream_index = {s: i for i, s in enumerate(streams)}
    n_players, n_teams, n_streams, n_matches = len(players), len(teams), len(streams), len(df)

    A = df["PLAYER_1_FINAL_SCORE"].to_numpy(dtype=float)
    B = df["PLAYER_2_FINAL_SCORE"].to_numpy(dtype=float)
    p1 = df["PLAYER_1_HANDLE"].map(player_index).to_numpy(dtype=int)
    p2 = df["PLAYER_2_HANDLE"].map(player_index).to_numpy(dtype=int)
    t1 = df["PLAYER_1_TEAM"].map(team_index).to_numpy(dtype=int)
    t2 = df["PLAYER_2_TEAM"].map(team_index).to_numpy(dtype=int)
    st = df["_stream_key"].map(stream_index).to_numpy(dtype=int)

    latest_time = df["SCHEDULED_START_TIME_UTC"].max()
    age_days = (latest_time - df["SCHEDULED_START_TIME_UTC"]).dt.total_seconds().to_numpy() / 86400.0
    weights_raw = np.power(2.0, -age_days / HALF_LIFE_DAYS) if HALF_LIFE_DAYS > 0 else np.ones(n_matches)
    weights = weights_raw / np.mean(weights_raw)

    i_beta = 0
    cursor = 1
    s_pa = slice(cursor, cursor + max(n_players - 1, 0)); cursor = s_pa.stop
    s_pd = slice(cursor, cursor + max(n_players - 1, 0)); cursor = s_pd.stop
    s_ta = slice(cursor, cursor + max(n_teams - 1, 0)); cursor = s_ta.stop
    s_td = slice(cursor, cursor + max(n_teams - 1, 0)); cursor = s_td.stop
    s_s = slice(cursor, cursor + max(n_streams - 1, 0)); cursor = s_s.stop
    s_la = slice(cursor, cursor + n_players); cursor = s_la.stop
    n_params = cursor

    def unpack(theta):
        beta0 = theta[i_beta]
        pa = fitmod.centred_from_free(theta[s_pa]) if n_players > 1 else np.zeros(1)
        pd_ = fitmod.centred_from_free(theta[s_pd]) if n_players > 1 else np.zeros(1)
        ta = fitmod.centred_from_free(theta[s_ta]) if n_teams > 1 else np.zeros(1)
        td = fitmod.centred_from_free(theta[s_td]) if n_teams > 1 else np.zeros(1)
        se = fitmod.centred_from_free(theta[s_s]) if n_streams > 1 else np.zeros(1)
        la = theta[s_la]
        alpha = np.exp(np.clip(la, -12, 4))
        return beta0, pa, pd_, ta, td, se, la, alpha

    def objective(theta):
        beta0, pa, pd_, ta, td, se, la, alpha = unpack(theta)
        etaA = np.clip(beta0 + se[st] + pa[p1] + ta[t1] - pd_[p2] - td[t2], -15, 15)
        etaB = np.clip(beta0 + se[st] + pa[p2] + ta[t2] - pd_[p1] - td[t1], -15, 15)
        muA = np.exp(etaA); muB = np.exp(etaB)
        llA = fitmod.nb2_logpmf(A, muA, alpha[p1])
        llB = fitmod.nb2_logpmf(B, muB, alpha[p2])
        value = -np.sum(weights * (llA + llB))
        value += 0.5 * (beta0 / SD_BETA) ** 2
        value += 0.5 * np.sum((pa / SD_PLAYER) ** 2)
        value += 0.5 * np.sum((pd_ / SD_PLAYER) ** 2)
        value += 0.5 * np.sum((ta / SD_TEAM) ** 2)
        value += 0.5 * np.sum((td / SD_TEAM) ** 2)
        value += 0.5 * np.sum((se / SD_STREAM) ** 2)
        value += 0.5 * np.sum(((la - MU_LOG_ALPHA) / SD_LOG_ALPHA) ** 2)

        g_etaA = fitmod.dloglik_deta(A, muA, alpha[p1])
        g_etaB = fitmod.dloglik_deta(B, muB, alpha[p2])
        g_laA = fitmod.dloglik_dlogalpha(A, muA, alpha[p1])
        g_laB = fitmod.dloglik_dlogalpha(B, muB, alpha[p2])

        g_beta = -np.sum(weights * (g_etaA + g_etaB))
        g_pa = np.zeros(n_players); g_pd = np.zeros(n_players)
        g_ta = np.zeros(n_teams); g_td = np.zeros(n_teams)
        g_s = np.zeros(n_streams); g_la = np.zeros(n_players)
        np.add.at(g_pa, p1, -weights * g_etaA)
        np.add.at(g_pa, p2, -weights * g_etaB)
        np.add.at(g_pd, p2, +weights * g_etaA)
        np.add.at(g_pd, p1, +weights * g_etaB)
        np.add.at(g_ta, t1, -weights * g_etaA)
        np.add.at(g_ta, t2, -weights * g_etaB)
        np.add.at(g_td, t2, +weights * g_etaA)
        np.add.at(g_td, t1, +weights * g_etaB)
        np.add.at(g_s, st, -weights * (g_etaA + g_etaB))
        np.add.at(g_la, p1, -weights * g_laA)
        np.add.at(g_la, p2, -weights * g_laB)

        g_beta += beta0 / SD_BETA ** 2
        g_pa += pa / SD_PLAYER ** 2
        g_pd += pd_ / SD_PLAYER ** 2
        g_ta += ta / SD_TEAM ** 2
        g_td += td / SD_TEAM ** 2
        g_s += se / SD_STREAM ** 2
        g_la += (la - MU_LOG_ALPHA) / SD_LOG_ALPHA ** 2

        grad = np.empty(n_params, dtype=float)
        grad[i_beta] = g_beta
        if n_players > 1:
            grad[s_pa] = fitmod.free_gradient_from_centred(g_pa)
            grad[s_pd] = fitmod.free_gradient_from_centred(g_pd)
        if n_teams > 1:
            grad[s_ta] = fitmod.free_gradient_from_centred(g_ta)
            grad[s_td] = fitmod.free_gradient_from_centred(g_td)
        if n_streams > 1:
            grad[s_s] = fitmod.free_gradient_from_centred(g_s)
        grad[s_la] = g_la
        return float(value), grad

    theta0 = np.zeros(n_params, dtype=float)
    theta0[i_beta] = np.log(max(np.mean(np.concatenate([A, B])), 1e-3))
    theta0[s_la] = MU_LOG_ALPHA
    if n_streams > 1:
        overall = max(np.mean(np.concatenate([A, B])), 1e-6)
        raw = []
        for s in streams:
            mask = df["_stream_key"].to_numpy() == s
            vals = np.concatenate([A[mask], B[mask]])
            raw.append(np.log(max(vals.mean(), 1e-6) / overall))
        raw = np.asarray(raw); raw -= raw.mean(); theta0[s_s] = raw[:-1]

    bounds = [(None, None)] * n_params
    for j in range(s_la.start, s_la.stop):
        bounds[j] = (-6.0, 2.0)

    res = minimize(objective, theta0, jac=True, method="L-BFGS-B", bounds=bounds,
                    options={"maxiter": 5000, "maxfun": 250000, "ftol": 1e-10, "gtol": 1e-6, "maxls": 50})
    if not res.success:
        raise RuntimeError(f"NB2 optimisation failed: {res.message}")

    beta0, pa, pd_, ta, td, se, la, alpha = unpack(res.x)
    beta0 = float(beta0)

    etaA = beta0 + se[st] + pa[p1] + ta[t1] - pd_[p2] - td[t2]
    etaB = beta0 + se[st] + pa[p2] + ta[t2] - pd_[p1] - td[t1]
    muA = np.exp(np.clip(etaA, -15, 15)); muB = np.exp(np.clip(etaB, -15, 15))
    resA = A - muA; resB = B - muB

    joint_map = {}
    global_num = global_den = 0.0
    for s, idx in stream_index.items():
        mask = st == idx; ww = weights_raw[mask]
        aA = alpha[p1[mask]]; aB = alpha[p2[mask]]
        mA = muA[mask]; mB = muB[mask]; eA = resA[mask]; eB = resB[mask]
        num = float(np.sum(ww * ((eA ** 2 - mA) + (eB ** 2 - mB))))
        den = float(np.sum(ww * (aA * mA ** 2 + aB * mB ** 2)))
        scale = float(np.clip(num / den, DISPERSION_SCALE_MIN, DISPERSION_SCALE_MAX)) if den > 0 else 1.0
        eaA = scale * aA; eaB = scale * aB
        cap = mA * mB * np.minimum(eaA, eaB)
        cnum = float(np.sum(ww * eA * eB)); cden = float(np.sum(ww * cap))
        frac = float(np.clip(cnum / cden, 0, SHARED_FRACTION_MAX)) if cden > 0 else 0.0
        joint_map[s] = {"Dispersion_Scale": scale, "Shared_Gamma_Fraction": frac}
        global_num += num; global_den += den
    gscale = float(np.clip(global_num / global_den, DISPERSION_SCALE_MIN, DISPERSION_SCALE_MAX)) if global_den > 0 else 1.0
    eaA = gscale * alpha[p1]; eaB = gscale * alpha[p2]
    cap = muA * muB * np.minimum(eaA, eaB)
    gcovden = float(np.sum(weights_raw * cap))
    gfrac = float(np.clip(np.sum(weights_raw * resA * resB) / gcovden, 0, SHARED_FRACTION_MAX)) if gcovden > 0 else 0.0
    joint_map["GLOBAL"] = {"Dispersion_Scale": gscale, "Shared_Gamma_Fraction": gfrac}

    stream_eff_map = {s: float(se[idx]) for s, idx in stream_index.items()}

    return {
        "beta0": beta0, "player_index": player_index, "team_index": team_index,
        "pa": pa, "pd": pd_, "ta": ta, "td": td,
        "alpha": alpha, "stream_eff_map": stream_eff_map, "joint_map": joint_map,
    }


def predict_match(model, p1_name, p2_name, t1_name, t2_name, stream_key, n_sims, rng):
    pi = model["player_index"]; ti = model["team_index"]
    if p1_name not in pi or p2_name not in pi or t1_name not in ti or t2_name not in ti:
        return None
    i1, i2 = pi[p1_name], pi[p2_name]
    j1, j2 = ti[t1_name], ti[t2_name]
    stream_eff = model["stream_eff_map"].get(stream_key, 0.0)
    jcal = model["joint_map"].get(stream_key, model["joint_map"]["GLOBAL"])
    beta0 = model["beta0"]
    pa, pd_, ta, td, alpha = model["pa"], model["pd"], model["ta"], model["td"], model["alpha"]

    eta1 = beta0 + stream_eff + pa[i1] + ta[j1] - pd_[i2] - td[j2]
    eta2 = beta0 + stream_eff + pa[i2] + ta[j2] - pd_[i1] - td[j1]
    mu1 = float(np.exp(np.clip(eta1, -15, 15)))
    mu2 = float(np.exp(np.clip(eta2, -15, 15)))
    alpha1 = float(jcal["Dispersion_Scale"] * alpha[i1])
    alpha2 = float(jcal["Dispersion_Scale"] * alpha[i2])
    shared_fraction = float(jcal["Shared_Gamma_Fraction"])

    s1, s2, _ = pricemod.joint_nb2_draws(mu1, alpha1, mu2, alpha2, shared_fraction, n_sims, rng)
    p1_reg = float(np.mean(s1 > s2))
    tie = float(np.mean(s1 == s2))
    p1_fair = p1_reg + 0.5 * tie
    return {
        "mu1": mu1, "mu2": mu2, "p1_fair_ml": p1_fair,
        "median_total_pred": float(np.median(s1.astype(int) + s2.astype(int))),
    }


def main():
    path = Path(__file__).parent / "AMFELO.csv"
    df = load_clean(path)
    n = len(df)
    split = int(n * TRAIN_FRACTION)
    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()
    print(f"Total matches: {n}")
    print(f"Train: {len(train)}  ({train['SCHEDULED_START_TIME_UTC'].min()} -> {train['SCHEDULED_START_TIME_UTC'].max()})")
    print(f"Test:  {len(test)}  ({test['SCHEDULED_START_TIME_UTC'].min()} -> {test['SCHEDULED_START_TIME_UTC'].max()})")

    print("\nFitting NB2 model on TRAIN only...")
    model = fit_nb2(train)
    print(f"  players={len(model['player_index'])} teams={len(model['team_index'])}")

    stream_num = pd.to_numeric(test["STREAM_NUMBER"], errors="coerce")
    test = test.copy()
    test["_stream_key"] = stream_num if stream_num.notna().all() else test["STREAM_NUMBER"].astype(str)

    rng = np.random.default_rng(SEED)
    preds = []
    skipped_unknown = 0
    print(f"\nPricing {len(test)} held-out TEST matches out-of-sample...")
    for _, row in test.iterrows():
        p1n, p2n = row["PLAYER_1_HANDLE"], row["PLAYER_2_HANDLE"]
        t1n, t2n = row["PLAYER_1_TEAM"], row["PLAYER_2_TEAM"]
        stream_key = row["_stream_key"]
        out = predict_match(model, p1n, p2n, t1n, t2n, stream_key, N_SIMS, rng)
        if out is None:
            skipped_unknown += 1
            continue
        actual_total = row["PLAYER_1_FINAL_SCORE"] + row["PLAYER_2_FINAL_SCORE"]
        p1_won = row["PLAYER_1_FINAL_SCORE"] > row["PLAYER_2_FINAL_SCORE"]
        preds.append({
            "p1_fair_ml": out["p1_fair_ml"], "p1_won": p1_won,
            "pred_total": out["mu1"] + out["mu2"], "actual_total": actual_total,
        })

    print(f"Predicted (out-of-sample): {len(preds)}   Skipped (unseen player/team in train): {skipped_unknown}")

    pred_df = pd.DataFrame(preds)

    print("\n=== Moneyline calibration (out-of-sample TEST) ===")
    pred_df["bucket"] = pd.qcut(pred_df["p1_fair_ml"], 10, duplicates="drop")
    calib = pred_df.groupby("bucket", observed=True).agg(
        n=("p1_won", "size"), predicted=("p1_fair_ml", "mean"), realized=("p1_won", "mean")
    ).reset_index()
    calib["gap_pp"] = 100 * (calib["realized"] - calib["predicted"])
    print(calib.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    brier = float(np.mean((pred_df["p1_fair_ml"] - pred_df["p1_won"].astype(float)) ** 2))
    eps = 1e-9
    p = pred_df["p1_fair_ml"].clip(eps, 1 - eps)
    y = pred_df["p1_won"].astype(float)
    logloss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    print(f"\nBrier score: {brier:.4f}   Log loss: {logloss:.4f}")
    print("(Brier: 0=perfect, 0.25=uninformative always-50% baseline; lower is better)")

    print("\n=== Total points calibration (out-of-sample TEST) ===")
    err = pred_df["actual_total"] - pred_df["pred_total"]
    print(f"Mean predicted total: {pred_df['pred_total'].mean():.2f}   Mean actual total: {pred_df['actual_total'].mean():.2f}")
    print(f"Mean error (actual - predicted): {err.mean():+.3f}   RMSE: {np.sqrt((err ** 2).mean()):.3f}")
    corr = np.corrcoef(pred_df["pred_total"], pred_df["actual_total"])[0, 1]
    print(f"Correlation(predicted total, actual total): {corr:.3f}")


if __name__ == "__main__":
    main()
