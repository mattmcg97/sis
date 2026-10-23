"""Tune online rating models on TUNE (2026-06-01..07-31) log loss by Nelder-Mead over transformed params.
Usage: python tune_dyn.py <name>   (configs below). Writes out/B/tune_dyn_<name>.json"""
import common as C
import json, sys, time
import numpy as np
from scipy.optimize import minimize
import dyn

H = C.H
d = C.slim()
A = dyn._arrays(d)
W = (H.in_window(d, H.TUNE) & d.y.notna()).to_numpy()
Y = d.y.to_numpy()[W]


def ll(p):
    p = np.clip(p[W], 1e-6, 1 - 1e-6)
    return float(-np.mean(Y * np.log(p) + (1 - Y) * np.log(1 - p)))


# name: (function, {param: (init, kind)}, fixed kwargs);  kind: 'log' (positive), 'lin'
CONFIGS = {
    "elo": (dyn.elo, {"K": (0.08, "log"), "h": (0.05, "lin")}, {}),
    "elo_new": (dyn.elo, {"K": (0.08, "log"), "Knew": (0.1, "log"), "nc": (20, "log"), "h": (0.05, "lin"),
                          "r0": (-0.1, "lin")}, {}),
    "elo_team": (dyn.elo, {"K": (0.08, "log"), "Knew": (0.1, "log"), "nc": (20, "log"), "h": (0.05, "lin"),
                           "r0": (-0.1, "lin"), "Kt": (0.01, "log")}, {}),
    "elo_team_gt": (dyn.elo, {"K": (0.08, "log"), "Knew": (0.1, "log"), "nc": (20, "log"), "h": (0.05, "lin"),
                              "r0": (-0.1, "lin"), "Kt": (0.01, "log"), "Kgt": (0.02, "log")}, {}),
    "elo_team_tau": (dyn.elo, {"K": (0.08, "log"), "Knew": (0.1, "log"), "nc": (20, "log"), "h": (0.05, "lin"),
                               "r0": (-0.1, "lin"), "Kt": (0.01, "log"), "tau": (200, "log")}, {}),
    "elo_team_sh": (dyn.elo, {"K": (0.08, "log"), "Knew": (0.1, "log"), "nc": (20, "log"), "h": (0.05, "lin"),
                              "r0": (-0.1, "lin"), "Kt": (0.01, "log"), "Kh": (0.002, "log")}, {"stream_home": True}),
    "mov_elo": (dyn.elo, {"K": (0.08, "log"), "Knew": (0.1, "log"), "nc": (20, "log"), "h": (0.05, "lin"),
                          "r0": (-0.1, "lin"), "Kt": (0.01, "log"), "mova": (0.5, "lin"), "movc": (3, "log")}, {}),
    "mov_elo_ac": (dyn.elo, {"K": (0.08, "log"), "Knew": (0.1, "log"), "nc": (20, "log"), "h": (0.05, "lin"),
                             "r0": (-0.1, "lin"), "Kt": (0.01, "log"), "mova": (0.5, "lin"), "movc": (3, "log"),
                             "ac": (0.5, "log")}, {}),
    "kal": (dyn.kalman, {"v0": (0.3, "log"), "q_day": (0.002, "log"), "h": (0.05, "lin")}, {}),
    "kal_r0": (dyn.kalman, {"v0": (0.3, "log"), "q_day": (0.002, "log"), "q_game": (0.0005, "log"),
                            "h": (0.05, "lin"), "r0": (-0.1, "lin")}, {}),
    "kal_team": (dyn.kalman, {"v0": (0.3, "log"), "q_day": (0.002, "log"), "q_game": (0.0005, "log"),
                              "h": (0.05, "lin"), "r0": (-0.1, "lin"), "vt0": (0.02, "log"), "qt_day": (0.0002, "log")}, {}),
    "kal_team_gt": (dyn.kalman, {"v0": (0.3, "log"), "q_day": (0.002, "log"), "q_game": (0.0005, "log"),
                                 "h": (0.05, "lin"), "r0": (-0.1, "lin"), "vt0": (0.02, "log"),
                                 "qt_day": (0.0002, "log"), "vgt0": (0.02, "log"), "qgt_day": (0.0002, "log")}, {}),
    "kal_team_noint": (dyn.kalman, {"v0": (0.3, "log"), "q_day": (0.002, "log"), "q_game": (0.0005, "log"),
                                    "h": (0.05, "lin"), "r0": (-0.1, "lin"), "vt0": (0.02, "log"),
                                    "qt_day": (0.0002, "log")}, {"integrate": False}),
    "kal_team_probit": (dyn.kalman, {"v0": (0.3, "log"), "q_day": (0.002, "log"), "q_game": (0.0005, "log"),
                                     "h": (0.05, "lin"), "r0": (-0.1, "lin"), "vt0": (0.02, "log"),
                                     "qt_day": (0.0002, "log")}, {"upd": "probit"}),
    "kal_team_sh": (dyn.kalman, {"v0": (0.3, "log"), "q_day": (0.002, "log"), "q_game": (0.0005, "log"),
                                 "h": (0.05, "lin"), "r0": (-0.1, "lin"), "vt0": (0.02, "log"),
                                 "qt_day": (0.0002, "log"), "hv0": (0.005, "log")}, {"stream_home": True}),
    "mov_kal_team": (dyn.kalman, {"v0": (0.3, "log"), "q_day": (0.002, "log"), "q_game": (0.0005, "log"),
                                  "h": (0.05, "lin"), "r0": (-0.1, "lin"), "vt0": (0.02, "log"),
                                  "qt_day": (0.0002, "log"), "mova": (0.3, "lin"), "movc": (3, "log")}, {}),
    "kal2": (dyn.kalman2, {"v0": (1.5, "log"), "q_day": (0.001, "log"), "q_game": (0.0001, "log"),
                           "h": (0.01, "lin"), "r0": (-0.08, "lin"), "vt0": (0.02, "log"), "qt_day": (0.0002, "log"),
                           "vf": (0.02, "log"), "tauf": (0.2, "log")}, {}),
    "kal_team_pair": (dyn.kalman, {"v0": (3.5, "log"), "q_day": (0.002, "log"), "h": (0.01, "lin"),
                                   "r0": (-0.01, "lin"), "vt0": (4.0, "log"), "qt_day": (0.00002, "log"),
                                   "vp0": (0.05, "log"), "qp_day": (0.0001, "log")}, {}),
    "kal3": (dyn.kalman3, {"v0": (3.66, "log"), "q_day": (0.0022, "log"), "h": (0.01, "lin"), "r0": (-0.14, "lin"),
                           "vt0": (17.0, "log"), "qt_day": (1e-5, "log"), "vf": (0.067, "log"), "tauf": (0.35, "log"),
                           "hv0": (0.01, "log"), "vp0": (0.05, "log"), "qp_day": (1e-4, "log")}, {}),
    "mov_kal3": (dyn.kalman3, {"v0": (3.66, "log"), "q_day": (0.0022, "log"), "h": (0.01, "lin"), "r0": (-0.14, "lin"),
                               "vt0": (17.0, "log"), "qt_day": (1e-5, "log"), "vf": (0.067, "log"),
                               "tauf": (0.35, "log"), "hv0": (0.01, "log"), "vp0": (0.05, "log"),
                               "qp_day": (1e-4, "log"), "mova": (0.5, "lin"), "movc": (3, "log")}, {}),
    "kfull": (dyn.kalman_full, {"v0": (0.3, "log"), "q_day": (0.002, "log"), "h": (0.05, "lin"),
                                "vt0": (0.05, "log"), "qt_day": (0.0002, "log")}, {"hv0": 0.0}),
}


def unpack(x, spec, fixed):
    kw = dict(fixed)
    for (k, (init, kind)), xi in zip(spec.items(), x):
        kw[k] = float(np.exp(xi)) if kind == "log" else float(xi)
    if "mova" in kw:
        kw["mov"] = (kw.pop("mova"), kw.pop("movc"))
    return kw


def tune(name, maxfev=300, init=None):
    maxfev = max(maxfev, 60 * len(CONFIGS[name][1]))
    fn, spec, fixed = CONFIGS[name]
    if init:
        spec = {k: (init.get(k, v[0]), v[1]) for k, v in spec.items()}
    x0 = np.array([np.log(v[0]) if v[1] == "log" else v[0] for v in spec.values()])
    hist = []

    def f(x):
        kw = unpack(x, spec, fixed)
        try:
            val = ll(fn(d, A=A, **kw))
        except (OverflowError, FloatingPointError, ValueError):
            val = 1.0
        if not np.isfinite(val):
            val = 1.0
        hist.append(val)
        if len(hist) % 25 == 0:
            print(f"  {name} {len(hist)}: best {min(hist):.5f}", flush=True)
        return val

    t = time.time()
    step = np.where([v[1] == "log" for v in spec.values()], 0.7, 0.05)
    sim = np.vstack([x0] + [x0 + np.eye(len(x0))[i] * step[i] for i in range(len(x0))])
    r = minimize(f, x0, method="Nelder-Mead", options=dict(maxfev=maxfev, initial_simplex=sim, xatol=1e-3, fatol=1e-6))
    kw = unpack(r.x, spec, fixed)
    out = dict(name=name, fn=fn.__name__, kw=kw, tune_ll=float(r.fun), nfev=int(r.nfev), secs=time.time() - t)
    json.dump(out, open(f"{C.OUT}/tune_dyn_{name}.json", "w"), indent=1, default=float)
    print(json.dumps(out, default=float), flush=True)
    return out


if __name__ == "__main__":
    for name in sys.argv[1:]:
        tune(name)
