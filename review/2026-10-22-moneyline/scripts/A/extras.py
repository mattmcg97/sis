"""The link's extras for one margin source: every column a candidate term reads, built walk-forward.

    X = build(df, m)          # df with the source in column m; returns a frame of extra columns
    TERMS[name](X, m) -> list of term arrays added to [m] (or replacing it)
"""
import numpy as np
import pandas as pd

import common as C


def nz(x):
    x = np.asarray(x, float)
    return np.where(np.isfinite(x), x, 0.0)


def build(df, m, fast=None, total=None, var=None):
    """Extra columns for source m. fast: a fast-form margin column (for the form gap), total: an expected
    total column, var: the source's own uncertainty column (points^2)."""
    X = pd.DataFrame(index=df.index)
    X["m"] = df[m].to_numpy(float)
    rm = C.rematch_features(df, m)
    X = X.join(rm)
    X["rm_ind"] = (rm.rm_n > 0).astype(float)
    X["meet2"] = (df.meeting == 2).astype(float)
    X["meet3p"] = (df.meeting >= 3).astype(float)
    sf = C.session_features(df, m)
    X = X.join(sf)
    X["sess_win"] = nz(df.p1_session_win - 0.5) - nz(df.p2_session_win - 0.5)
    X["sess_margin"] = nz(df.p1_session_pf - df.p1_session_pa) - nz(df.p2_session_pf - df.p2_session_pa)
    X["win3"] = nz(df.p1_win_3 - 0.5) - nz(df.p2_win_3 - 0.5)
    X["s2"] = (df.stream == 2).astype(float)
    X["s3"] = (df.stream == 3).astype(float)
    X["games_min"] = np.minimum(df.p1_games.fillna(0), df.p2_games.fillna(0)).to_numpy(float)
    X["new"] = (X.games_min < 30).astype(float)
    X["h2h"] = df.h2h_margin.fillna(0).to_numpy(float)
    if fast is not None:
        X["fast_gap"] = (df[fast] - df[m]).to_numpy(float)
    if total is not None:
        X["tot"] = df[total].to_numpy(float)
    if var is not None:
        X["var"] = df[var].to_numpy(float)
    X["ewma_gap"] = (((df.p1_ewma_pf - df.p1_ewma_pa) - (df.p2_ewma_pf - df.p2_ewma_pa)) / 2 - df[m]).to_numpy(float)
    return X


def terms(X, spec, c_unc=0.0, t0=34.0):
    """The link's term list for a spec: a tuple of extra names; the margin is always first (possibly
    transformed by the 'unc' or 'tot' extras)."""
    m = X.m.to_numpy()
    if "unc" in spec and c_unc:
        m = m / np.sqrt(1.0 + c_unc * X["var"].to_numpy())
    if "unc_g" in spec and c_unc:
        m = m / np.sqrt(1.0 + c_unc / np.maximum(X.games_min.to_numpy(), 1.0))
    if "tot" in spec:
        m = m * np.sqrt(t0 / np.clip(X.tot.to_numpy(), 15.0, 60.0))
    out = [m]
    for s in spec:
        if s in ("unc", "unc_g", "tot"):
            continue
        if s == "rematch":
            out += [X.rm_res.to_numpy()]
        elif s == "rematch_full":
            out += [X.rm_res.to_numpy(), X.rm_win.to_numpy(), X.rm_ind.to_numpy()]
        elif s == "rm_win":
            out += [X.rm_win.to_numpy()]
        elif s == "meeting":
            out += [X.meet2.to_numpy(), X.meet3p.to_numpy()]
        elif s == "sess_gap":
            out += [X.sf_gap.to_numpy()]
        elif s == "sess_win":
            out += [X.sess_win.to_numpy()]
        elif s == "sess_margin":
            out += [X.sess_margin.to_numpy()]
        elif s == "win3":
            out += [X.win3.to_numpy()]
        elif s == "fast_gap":
            out += [X.fast_gap.to_numpy()]
        elif s == "ewma_gap":
            out += [np.where(np.isfinite(X.ewma_gap.to_numpy()), X.ewma_gap.to_numpy(), 0.0)]
        elif s == "stream":
            out += [X.s2.to_numpy(), X.s3.to_numpy()]
        elif s == "stream_x":
            out += [X.s2.to_numpy(), X.s3.to_numpy(), m * X.s2.to_numpy(), m * X.s3.to_numpy()]
        elif s == "new_x":
            out += [m * X.new.to_numpy()]
        elif s == "h2h":
            out += [X.h2h.to_numpy()]
        elif s == "tot_x":
            out += [m * (X.tot.to_numpy() - t0) / t0]
        else:
            raise KeyError(s)
    return out
