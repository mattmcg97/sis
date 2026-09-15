#!/usr/bin/env python3
"""
Exploratory recency-weighted NB2 player + team model with joint score covariance calibration.
Produced September 2026. This model was developed to explore the predictability of the Madden game,
not be a production pricing engine now that we have 25k games. There is a seperate script to predict pre-match off a schedule csv.

Mean score model
----------------
For player i using team t_i against player j using team t_j in stream s:

    log(mu_i) = beta0 + stream[s]
                + player_attack[i] + team_attack[t_i]
                - player_defense[j] - team_defense[t_j]

    log(mu_j) = beta0 + stream[s]
                + player_attack[j] + team_attack[t_j]
                - player_defense[i] - team_defense[t_i]

N.B. - I have used all of the eAMF data available to me in Snowflake - right from the very beginning of the product.
At the start there was definitely an initiation period for the players, with some crazy scoring, so we may want to look at walk-forward
windows for the mean scoring model that exclude the initial period of excess volatility.

All player, team and stream effects use sum-to-zero constraints.  This makes the
team effects identifiable separately from gamer effects because the data contain
substantial crossover of gamers between teams.

On the score scale the effects interact multiplicatively.  For example, the
attacking multiplier for player i using team t is

    exp(player_attack[i]) * exp(team_attack[t]).

Likewise the defending side is the product of player and team suppression
multipliers.

Explicit player x team residual/chemistry terms are NOT used in the
forecast by default: chronological holdout testing on this dataset showed that they overfit this
sample, so some optimisation/tuning will probably needed on that bit. A player-team diagnostics CSV is exported but I
have had no real time to get stuck into it.

Marginal NB2 score model
------------------------
    Var(Y_i) = mu_i + alpha_i * mu_i^2

where alpha_i is scorer/player-specific.

Joint covariance layer
----------------------
The fitted alpha values are variance-calibrated by stream.  A shared Gamma game
factor then creates positive score covariance while preserving NB2 marginals.

Outputs
-------
- player leaderboard
- team attack/defence ratings
- stream scoring effects
- joint dispersion/covariance calibration
- player-team residual diagnostics (not used in pricing)


"""

import argparse
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import digamma, gammaln


def parse_args():
    p = argparse.ArgumentParser(description="Fit recency-weighted player+team NB2 model with joint covariance calibration.")
    p.add_argument("--input", default="AMFELO.csv")
    p.add_argument("--output", default="NB2_player_leaderboardsept_team_joint.csv")
    p.add_argument("--team-output", default="NB2_team_ratingssept_joint.csv")
    p.add_argument("--stream-output", default="NB2_stream_effectssept_team_joint.csv")
    p.add_argument("--joint-output", default="NB2_joint_calibrationsept_team.csv")
    p.add_argument("--chemistry-output", default="NB2_player_team_diagnosticssept.csv")
    p.add_argument("--half-life-days", type=float, default=60.0)
    p.add_argument("--sd-player", type=float, default=0.5, help="Prior SD for player attack/defence")
    p.add_argument("--sd-team", type=float, default=0.25, help="Prior SD for team attack/defence")
    p.add_argument("--sd-stream", type=float, default=0.15)
    p.add_argument("--sd-beta", type=float, default=1.0)
    p.add_argument("--mu-log-alpha", type=float, default=-1.8)
    p.add_argument("--sd-log-alpha", type=float, default=0.4)
    p.add_argument("--dispersion-scale-min", type=float, default=0.05)
    p.add_argument("--dispersion-scale-max", type=float, default=3.0)
    p.add_argument("--shared-fraction-max", type=float, default=0.98)
    p.add_argument("--maxiter", type=int, default=5000)
    p.add_argument("--maxfun", type=int, default=250000)
    p.add_argument("--ftol", type=float, default=1e-10)
    p.add_argument("--gtol", type=float, default=1e-6)
    p.add_argument("--no-approx-se", action="store_true")
    return p.parse_args()

#Core Functions

def nb2_logpmf(y, mu, alpha):
    mu = np.clip(mu, 1e-10, 1e8)
    alpha = np.clip(alpha, 1e-8, 10.0)
    r = 1.0 / alpha
    return (
        gammaln(y + r) - gammaln(r) - gammaln(y + 1.0)
        + r * np.log(r) - r * np.log(r + mu)
        + y * np.log(mu) - y * np.log(r + mu)
    )


def dloglik_deta(y, mu, alpha):
    return (y - mu) / (1.0 + alpha * mu)


def dloglik_dlogalpha(y, mu, alpha):
    alpha = np.clip(alpha, 1e-8, 10.0)
    r = 1.0 / alpha
    dll_dr = (
        digamma(y + r) - digamma(r)
        + np.log(r) + 1.0 - np.log(r + mu)
        - (r + y) / (r + mu)
    )
    return -r * dll_dr


def centred_from_free(free):
    if len(free) == 0:
        return np.zeros(1)
    return np.concatenate([free, [-np.sum(free)]])


def free_gradient_from_centred(full_grad):
    if len(full_grad) <= 1:
        return np.array([], dtype=float)
    return full_grad[:-1] - full_grad[-1]


def weighted_ess(w):
    den = float(np.sum(w ** 2))
    return float(np.sum(w) ** 2 / den) if den > 0 else 0.0


def main():
    args = parse_args()
    df = pd.read_csv(args.input)

    """Generalised cleaning and preparation phase"""

    required = [
        "SCHEDULED_START_TIME_UTC", "STREAM_NUMBER",
        "PLAYER_1_HANDLE", "PLAYER_2_HANDLE",
        "PLAYER_1_TEAM", "PLAYER_2_TEAM",
        "PLAYER_1_FINAL_SCORE", "PLAYER_2_FINAL_SCORE",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=required).copy()
    df["SCHEDULED_START_TIME_UTC"] = pd.to_datetime(df["SCHEDULED_START_TIME_UTC"], errors="coerce", utc=True)
    for c in ["PLAYER_1_FINAL_SCORE", "PLAYER_2_FINAL_SCORE"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    """
    As I mentioned in my overview - we do need to check that all matches do actually reach their conclusion
    I haven't systematically excluded cancelled matches or ones where there may have been a console problem.
    That is a important issue because they would skew scoring quite badly.
    """

    df = df.dropna(subset=["SCHEDULED_START_TIME_UTC", "PLAYER_1_FINAL_SCORE", "PLAYER_2_FINAL_SCORE"]).copy()

    df = df[(df["PLAYER_1_FINAL_SCORE"] >= 0) & (df["PLAYER_2_FINAL_SCORE"] >= 0)].copy()
    if not np.allclose(df["PLAYER_1_FINAL_SCORE"], np.round(df["PLAYER_1_FINAL_SCORE"])) or \
       not np.allclose(df["PLAYER_2_FINAL_SCORE"], np.round(df["PLAYER_2_FINAL_SCORE"])):
        raise ValueError("Final scores must be integer-valued for NB2.")

    df["PLAYER_1_FINAL_SCORE"] = df["PLAYER_1_FINAL_SCORE"].astype(int)
    df["PLAYER_2_FINAL_SCORE"] = df["PLAYER_2_FINAL_SCORE"].astype(int)

    df["PLAYER_1_HANDLE"] = df["PLAYER_1_HANDLE"].astype(str).str.strip().str.upper()
    df["PLAYER_2_HANDLE"] = df["PLAYER_2_HANDLE"].astype(str).str.strip().str.upper()

    df["PLAYER_1_TEAM"] = df["PLAYER_1_TEAM"].astype(str).str.strip()
    df["PLAYER_2_TEAM"] = df["PLAYER_2_TEAM"].astype(str).str.strip()
    df = df.sort_values("SCHEDULED_START_TIME_UTC").reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid rows remain.")

    stream_num = pd.to_numeric(df["STREAM_NUMBER"], errors="coerce")
    if stream_num.notna().all():
        df["_stream_key"] = stream_num
        streams = sorted(df["_stream_key"].unique().tolist())
    else:
        df["_stream_key"] = df["STREAM_NUMBER"].astype(str)
        streams = sorted(df["_stream_key"].unique().tolist())

    """Organising the central components of the model; players, teams and streams into lists"""

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

    """Applying the half-life approach"""

    latest_time = df["SCHEDULED_START_TIME_UTC"].max()
    age_days = (latest_time - df["SCHEDULED_START_TIME_UTC"]).dt.total_seconds().to_numpy() / 86400.0
    if args.half_life_days > 0:
        weights_raw = np.power(2.0, -age_days / args.half_life_days)
    else:
        weights_raw = np.ones(n_matches, dtype=float)
    weights = weights_raw / np.mean(weights_raw)
    ess = weighted_ess(weights_raw)

    # beta | player A | player D | team A | team D | stream | log-alpha(player)
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
        pa = centred_from_free(theta[s_pa]) if n_players > 1 else np.zeros(1)
        pd_ = centred_from_free(theta[s_pd]) if n_players > 1 else np.zeros(1)
        ta = centred_from_free(theta[s_ta]) if n_teams > 1 else np.zeros(1)
        td = centred_from_free(theta[s_td]) if n_teams > 1 else np.zeros(1)
        se = centred_from_free(theta[s_s]) if n_streams > 1 else np.zeros(1)
        la = theta[s_la]
        alpha = np.exp(np.clip(la, -12, 4))
        return beta0, pa, pd_, ta, td, se, la, alpha

    def objective(theta):
        beta0, pa, pd_, ta, td, se, la, alpha = unpack(theta)
        etaA = np.clip(beta0 + se[st] + pa[p1] + ta[t1] - pd_[p2] - td[t2], -15, 15)
        etaB = np.clip(beta0 + se[st] + pa[p2] + ta[t2] - pd_[p1] - td[t1], -15, 15)
        muA = np.exp(etaA)
        muB = np.exp(etaB)
        llA = nb2_logpmf(A, muA, alpha[p1])
        llB = nb2_logpmf(B, muB, alpha[p2])
        value = -np.sum(weights * (llA + llB))

        value += 0.5 * (beta0 / args.sd_beta) ** 2
        value += 0.5 * np.sum((pa / args.sd_player) ** 2)
        value += 0.5 * np.sum((pd_ / args.sd_player) ** 2)
        value += 0.5 * np.sum((ta / args.sd_team) ** 2)
        value += 0.5 * np.sum((td / args.sd_team) ** 2)
        value += 0.5 * np.sum((se / args.sd_stream) ** 2)
        value += 0.5 * np.sum(((la - args.mu_log_alpha) / args.sd_log_alpha) ** 2)

        g_etaA = dloglik_deta(A, muA, alpha[p1])
        g_etaB = dloglik_deta(B, muB, alpha[p2])
        g_laA = dloglik_dlogalpha(A, muA, alpha[p1])
        g_laB = dloglik_dlogalpha(B, muB, alpha[p2])

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

        g_beta += beta0 / args.sd_beta ** 2
        g_pa += pa / args.sd_player ** 2
        g_pd += pd_ / args.sd_player ** 2
        g_ta += ta / args.sd_team ** 2
        g_td += td / args.sd_team ** 2
        g_s += se / args.sd_stream ** 2
        g_la += (la - args.mu_log_alpha) / args.sd_log_alpha ** 2

        grad = np.empty(n_params, dtype=float)
        grad[i_beta] = g_beta
        if n_players > 1:
            grad[s_pa] = free_gradient_from_centred(g_pa)
            grad[s_pd] = free_gradient_from_centred(g_pd)
        if n_teams > 1:
            grad[s_ta] = free_gradient_from_centred(g_ta)
            grad[s_td] = free_gradient_from_centred(g_td)
        if n_streams > 1:
            grad[s_s] = free_gradient_from_centred(g_s)
        grad[s_la] = g_la
        return float(value), grad

    theta0 = np.zeros(n_params, dtype=float)
    theta0[i_beta] = np.log(max(np.mean(np.concatenate([A, B])), 1e-3))
    theta0[s_la] = args.mu_log_alpha
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

    res = minimize(
        objective, theta0, jac=True, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": args.maxiter, "maxfun": args.maxfun, "ftol": args.ftol,
                 "gtol": args.gtol, "maxls": 50},
    )
    if not res.success:
        raise RuntimeError(f"NB2 optimisation failed: {res.message}; nit={res.nit}; nfev={res.nfev}")

    beta0, pa, pd_, ta, td, se, la, alpha = unpack(res.x)
    player_rating = pa + pd_
    player_skill = 0.5 * player_rating
    player_pace = 0.5 * (pa - pd_)
    team_rating = ta + td
    team_skill = 0.5 * team_rating
    team_pace = 0.5 * (ta - td)

    # Approximate player/team rating SE from numerical Hessian of analytic gradient.
    player_se = np.full(n_players, np.nan)
    team_se = np.full(n_teams, np.nan)
    if not args.no_approx_se:
        try:
            xhat = np.asarray(res.x, dtype=float)
            H = np.empty((n_params, n_params), dtype=float)
            steps = np.cbrt(np.finfo(float).eps) * np.maximum(1.0, np.abs(xhat))
            for j in range(n_params):
                xp=xhat.copy(); xm=xhat.copy(); xp[j]+=steps[j]; xm[j]-=steps[j]
                lo,hi=bounds[j]
                if hi is not None and xp[j] > hi: xp[j]=xhat[j]
                if lo is not None and xm[j] < lo: xm[j]=xhat[j]
                if xp[j] != xhat[j] and xm[j] != xhat[j]:
                    H[:,j]=(objective(xp)[1]-objective(xm)[1])/(xp[j]-xm[j])
                elif xp[j] != xhat[j]:
                    H[:,j]=(objective(xp)[1]-objective(xhat)[1])/(xp[j]-xhat[j])
                else:
                    H[:,j]=(objective(xhat)[1]-objective(xm)[1])/(xhat[j]-xm[j])
            H=0.5*(H+H.T)
            cov=np.linalg.pinv(H, rcond=1e-10, hermitian=True)
            for k in range(n_players):
                g=np.zeros(n_params)
                if n_players > 1:
                    if k < n_players-1:
                        g[s_pa.start+k]=1.0; g[s_pd.start+k]=1.0
                    else:
                        g[s_pa]=-1.0; g[s_pd]=-1.0
                player_se[k]=np.sqrt(max(float(g@cov@g),0.0))
            for k in range(n_teams):
                g=np.zeros(n_params)
                if n_teams > 1:
                    if k < n_teams-1:
                        g[s_ta.start+k]=1.0; g[s_td.start+k]=1.0
                    else:
                        g[s_ta]=-1.0; g[s_td]=-1.0
                team_se[k]=np.sqrt(max(float(g@cov@g),0.0))
        except Exception as exc:
            warnings.warn(f"Could not calculate approximate SEs: {exc}")

    # Fitted means used by all diagnostics/calibration.
    etaA = beta0 + se[st] + pa[p1] + ta[t1] - pd_[p2] - td[t2]
    etaB = beta0 + se[st] + pa[p2] + ta[t2] - pd_[p1] - td[t1]
    muA = np.exp(np.clip(etaA, -15, 15)); muB = np.exp(np.clip(etaB, -15, 15))
    resA=A-muA; resB=B-muB

    # Player output.
    games = df["PLAYER_1_HANDLE"].value_counts().add(df["PLAYER_2_HANDLE"].value_counts(), fill_value=0).astype(int)
    wg = pd.Series(0.0, index=players)
    for handles in [df["PLAYER_1_HANDLE"], df["PLAYER_2_HANDLE"]]:
        tmp=pd.Series(weights_raw,index=handles).groupby(level=0).sum(); wg=wg.add(tmp,fill_value=0)
    a1=df[["SCHEDULED_START_TIME_UTC","PLAYER_1_HANDLE","_stream_key"]].rename(columns={"PLAYER_1_HANDLE":"Player","_stream_key":"Stream"})
    a2=df[["SCHEDULED_START_TIME_UTC","PLAYER_2_HANDLE","_stream_key"]].rename(columns={"PLAYER_2_HANDLE":"Player","_stream_key":"Stream"})
    app=pd.concat([a1,a2]).sort_values("SCHEDULED_START_TIME_UTC"); latest_player=app.groupby("Player",as_index=False).tail(1).set_index("Player")
    baseline_mu=float(np.exp(beta0)); baseline_sd=np.sqrt(baseline_mu+alpha*baseline_mu**2)
    player_rows=[]
    for p,idx in player_index.items():
        player_rows.append({
            "Player":p,"Attack":pa[idx],"Defense":pd_[idx],"NB2_Rating":player_rating[idx],
            "Skill_Log_Component":player_skill[idx],"Pace_Log_Component":player_pace[idx],
            "Attack_Multiplier":float(np.exp(pa[idx])),"Defense_Opponent_Score_Multiplier":float(np.exp(-pd_[idx])),
            "Pace_Multiplier":float(np.exp(player_pace[idx])),"Approx_Rating_SE":player_se[idx],
            "Approx_Rating_Lower95":player_rating[idx]-1.96*player_se[idx] if np.isfinite(player_se[idx]) else np.nan,
            "Approx_Rating_Upper95":player_rating[idx]+1.96*player_se[idx] if np.isfinite(player_se[idx]) else np.nan,
            "Alpha":alpha[idx],"Baseline_Score_SD":baseline_sd[idx],"Games":int(games.get(p,0)),
            "Recency_Weighted_Games":float(wg.get(p,0)),"Current_Stream":latest_player.loc[p,"Stream"],
            "Last_Played_UTC":latest_player.loc[p,"SCHEDULED_START_TIME_UTC"],
        })
    player_df=pd.DataFrame(player_rows).sort_values(["NB2_Rating","Games"],ascending=[False,False]).reset_index(drop=True)
    player_df.insert(0,"Rank",np.arange(1,len(player_df)+1)); player_df.to_csv(args.output,index=False)

    # Team output and team usage.
    team_games=df["PLAYER_1_TEAM"].value_counts().add(df["PLAYER_2_TEAM"].value_counts(),fill_value=0).astype(int)
    team_wg=pd.Series(0.0,index=teams)
    for tc in [df["PLAYER_1_TEAM"],df["PLAYER_2_TEAM"]]:
        tmp=pd.Series(weights_raw,index=tc).groupby(level=0).sum(); team_wg=team_wg.add(tmp,fill_value=0)
    team_rows=[]
    for t,idx in team_index.items():
        team_rows.append({
            "Team":t,"Attack":ta[idx],"Defense":td[idx],"Team_Rating":team_rating[idx],
            "Skill_Log_Component":team_skill[idx],"Pace_Log_Component":team_pace[idx],
            "Attack_Multiplier":float(np.exp(ta[idx])),
            "Defense_Opponent_Score_Multiplier":float(np.exp(-td[idx])),
            "Pace_Multiplier":float(np.exp(team_pace[idx])),"Approx_Rating_SE":team_se[idx],
            "Approx_Rating_Lower95":team_rating[idx]-1.96*team_se[idx] if np.isfinite(team_se[idx]) else np.nan,
            "Approx_Rating_Upper95":team_rating[idx]+1.96*team_se[idx] if np.isfinite(team_se[idx]) else np.nan,
            "Games":int(team_games.get(t,0)),"Recency_Weighted_Games":float(team_wg.get(t,0)),
        })
    team_df=pd.DataFrame(team_rows).sort_values("Team_Rating",ascending=False).reset_index(drop=True)
    team_df.insert(0,"Rank",np.arange(1,len(team_df)+1)); team_df.to_csv(args.team_output,index=False)

    # Stream output.
    stream_rows=[]
    for s,idx in stream_index.items():
        mask=st==idx; scores=np.concatenate([A[mask],B[mask]])
        stream_rows.append({"Stream":s,"Matches":int(mask.sum()),"Raw_Mean_Score":float(scores.mean()),
                            "Stream_Log_Scoring_Effect":float(se[idx]),"Scoring_Multiplier_vs_Global":float(np.exp(se[idx])),
                            "Model_Baseline_Mean_Score":float(np.exp(beta0+se[idx]))})
    stream_df=pd.DataFrame(stream_rows).sort_values("Stream").reset_index(drop=True); stream_df.to_csv(args.stream_output,index=False)

    # Joint dispersion/covariance calibration by stream.
    joint_rows=[]; global_num=global_den=0.0
    for s,idx in stream_index.items():
        mask=st==idx; ww=weights_raw[mask]
        aA=alpha[p1[mask]];aB=alpha[p2[mask]];mA=muA[mask];mB=muB[mask];eA=resA[mask];eB=resB[mask]
        num=float(np.sum(ww*((eA**2-mA)+(eB**2-mB))));den=float(np.sum(ww*(aA*mA**2+aB*mB**2)))
        scale=float(np.clip(num/den,args.dispersion_scale_min,args.dispersion_scale_max)) if den>0 else 1.0
        eaA=scale*aA;eaB=scale*aB;varA=mA+eaA*mA**2;varB=mB+eaB*mB**2
        cap=mA*mB*np.minimum(eaA,eaB); cnum=float(np.sum(ww*eA*eB)); cden=float(np.sum(ww*cap))
        frac=float(np.clip(cnum/cden,0,args.shared_fraction_max)) if cden>0 else 0.0
        cov=frac*cap; mvar=varA+varB-2*cov; tvar=varA+varB+2*cov;ws=float(np.sum(ww))
        corr=float(np.corrcoef(eA,eB)[0,1]) if np.std(eA)>0 and np.std(eB)>0 else np.nan
        joint_rows.append({
            "Stream":s,"Matches":int(mask.sum()),"Recency_ESS":weighted_ess(ww),"Dispersion_Scale":scale,
            "Shared_Gamma_Fraction":frac,"Observed_Residual_Correlation":corr,
            "Observed_Score_Residual_Variance_Sum":float(np.sum(ww*(eA**2+eB**2))/ws),
            "Model_Score_Residual_Variance_Sum":float(np.sum(ww*(varA+varB))/ws),
            "Observed_Margin_Residual_Variance":float(np.sum(ww*(eA-eB)**2)/ws),
            "Model_Margin_Residual_Variance":float(np.sum(ww*mvar)/ws),
            "Observed_Total_Residual_Variance":float(np.sum(ww*(eA+eB)**2)/ws),
            "Model_Total_Residual_Variance":float(np.sum(ww*tvar)/ws),
            "Observed_Score_Covariance":float(np.sum(ww*eA*eB)/ws),"Model_Score_Covariance":float(np.sum(ww*cov)/ws),
        });global_num+=num;global_den+=den
    gscale=float(np.clip(global_num/global_den,args.dispersion_scale_min,args.dispersion_scale_max)) if global_den>0 else 1.0
    eaA=gscale*alpha[p1];eaB=gscale*alpha[p2];cap=muA*muB*np.minimum(eaA,eaB)
    gcovden=float(np.sum(weights_raw*cap));gfrac=float(np.clip(np.sum(weights_raw*resA*resB)/gcovden,0,args.shared_fraction_max)) if gcovden>0 else 0.0
    varA=muA+eaA*muA**2;varB=muB+eaB*muB**2;cov=gfrac*cap;ws=float(np.sum(weights_raw))
    joint_rows.append({
        "Stream":"GLOBAL","Matches":n_matches,"Recency_ESS":ess,"Dispersion_Scale":gscale,"Shared_Gamma_Fraction":gfrac,
        "Observed_Residual_Correlation":float(np.corrcoef(resA,resB)[0,1]),
        "Observed_Score_Residual_Variance_Sum":float(np.sum(weights_raw*(resA**2+resB**2))/ws),
        "Model_Score_Residual_Variance_Sum":float(np.sum(weights_raw*(varA+varB))/ws),
        "Observed_Margin_Residual_Variance":float(np.sum(weights_raw*(resA-resB)**2)/ws),
        "Model_Margin_Residual_Variance":float(np.sum(weights_raw*(varA+varB-2*cov))/ws),
        "Observed_Total_Residual_Variance":float(np.sum(weights_raw*(resA+resB)**2)/ws),
        "Model_Total_Residual_Variance":float(np.sum(weights_raw*(varA+varB+2*cov))/ws),
        "Observed_Score_Covariance":float(np.sum(weights_raw*resA*resB)/ws),"Model_Score_Covariance":float(np.sum(weights_raw*cov)/ws),
    })
    joint_df=pd.DataFrame(joint_rows);joint_df.to_csv(args.joint_output,index=False)

    # Player x team residual diagnostics. These are intentionally NOT part of the forecast.
    scorer=pd.concat([
        pd.DataFrame({"Player":df["PLAYER_1_HANDLE"],"Team":df["PLAYER_1_TEAM"],"Y":A,"Mu":muA,"W":weights_raw}),
        pd.DataFrame({"Player":df["PLAYER_2_HANDLE"],"Team":df["PLAYER_2_TEAM"],"Y":B,"Mu":muB,"W":weights_raw}),
    ],ignore_index=True)
    defender=pd.concat([
        pd.DataFrame({"Player":df["PLAYER_2_HANDLE"],"Team":df["PLAYER_2_TEAM"],"Yopp":A,"Muopp":muA,"W":weights_raw}),
        pd.DataFrame({"Player":df["PLAYER_1_HANDLE"],"Team":df["PLAYER_1_TEAM"],"Yopp":B,"Muopp":muB,"W":weights_raw}),
    ],ignore_index=True)
    def attack_diag(g):
        ya=float(np.sum(g.W*g.Y)); mp=float(np.sum(g.W*g.Mu)); eps=1e-9
        return pd.Series({"Games":len(g),"Recency_Weighted_Games":float(g.W.sum()),"Weighted_Actual_Points":ya,
                          "Weighted_Predicted_Points":mp,"Residual_Attack_Log_Effect":float(np.log((ya+eps)/(mp+eps)))})
    def defense_diag(g):
        ya=float(np.sum(g.W*g.Yopp)); mp=float(np.sum(g.W*g.Muopp)); eps=1e-9
        return pd.Series({"Def_Games":len(g),"Def_Recency_Weighted_Games":float(g.W.sum()),"Weighted_Actual_Points_Allowed":ya,
                          "Weighted_Predicted_Points_Allowed":mp,"Residual_Defense_Log_Effect":float(-np.log((ya+eps)/(mp+eps)))})
    ad=scorer.groupby(["Player","Team"]).apply(attack_diag,include_groups=False).reset_index()
    dd=defender.groupby(["Player","Team"]).apply(defense_diag,include_groups=False).reset_index()
    chem=ad.merge(dd,on=["Player","Team"],how="outer")
    chem["Residual_Combined_Rating"]=chem["Residual_Attack_Log_Effect"]+chem["Residual_Defense_Log_Effect"]
    chem["Well_Observed_40plus"]=(chem["Games"].fillna(0)>=40)&(chem["Def_Games"].fillna(0)>=40)
    chem=chem.sort_values(["Well_Observed_40plus","Residual_Combined_Rating"],ascending=[False,False])
    chem.to_csv(args.chemistry_output,index=False)

    # Useful crossover diagnostic.
    pt=pd.concat([
        df[["PLAYER_1_HANDLE","PLAYER_1_TEAM"]].rename(columns={"PLAYER_1_HANDLE":"Player","PLAYER_1_TEAM":"Team"}),
        df[["PLAYER_2_HANDLE","PLAYER_2_TEAM"]].rename(columns={"PLAYER_2_HANDLE":"Player","PLAYER_2_TEAM":"Team"}),
    ])
    diversity=pt.groupby("Player")["Team"].nunique()

    print("\nNB2 PLAYER + TEAM JOINT MODEL")
    print("="*78)
    print(f"Matches:                 {n_matches:,}")
    print(f"Players:                 {n_players}")
    print(f"Teams:                   {n_teams}")
    print(f"Streams:                 {n_streams}  {streams}")
    print(f"Date range:              {df['SCHEDULED_START_TIME_UTC'].min()} -> {latest_time}")
    print(f"Half-life days:          {args.half_life_days:g}")
    print(f"Recency ESS:             {ess:,.1f}")
    print(f"Median teams per player: {diversity.median():.0f}; all {n_teams} teams: {(diversity==n_teams).sum()} players")
    print(f"Global baseline score:   {np.exp(beta0):.3f}")
    print(f"Final penalised NLL:     {res.fun:.3f}")
    print(f"Optimizer iterations:    {res.nit}; evaluations: {res.nfev}")
    print("\nTEAM RATINGS")
    print(team_df[["Rank","Team","Attack","Defense","Team_Rating","Attack_Multiplier","Defense_Opponent_Score_Multiplier","Pace_Log_Component","Games"]].to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    print("\nJOINT CALIBRATION")
    print(joint_df[["Stream","Dispersion_Scale","Shared_Gamma_Fraction","Observed_Residual_Correlation","Observed_Margin_Residual_Variance","Model_Margin_Residual_Variance"]].to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    print("\nOutputs:")
    for f in [args.output,args.team_output,args.stream_output,args.joint_output,args.chemistry_output]: print(" ",f)


if __name__ == "__main__":
    main()
