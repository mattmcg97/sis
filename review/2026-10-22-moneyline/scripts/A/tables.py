"""The walkthrough / form prior tables, with agent A's own walkthrough run (2026-07-25 to 08-31, out/A/walkthrough)
put in front of the repo's (2026-09-01 to 10-18) where it exists. The form table's margin before September is
rebuilt exactly as form_prior makes it: avg's margin moved half way to the drift-4 rating (checked on Sep-Oct:
max difference 0.04 points); its own moneyline is left blank there (it needs NB2's score model).
"""
import os

import numpy as np
import pandas as pd

import common as C

H = C.H
EXT = os.path.join(C.OUT, "walkthrough", "priors")
NAMES = {"avg": "avg.csv", "avg_plain": "avg-plain.csv", "nb2": "nb2.csv", "nb2_infer": "nb2_infer.csv",
         "nb2_both": "nb2_both.csv", "glmer": "glmer.csv", "glmer_both": "glmer_both.csv"}
LATE = ("2026-09-22", "2026-10-19")
SEP_TUNE = ("2026-09-01", "2026-09-22")


def have_ext():
    return all(os.path.exists(os.path.join(EXT, f)) for f in NAMES.values())


def add_ext(df):
    """df (from C.load(tables=True)) with the extended rows filled in before 2026-09-01; a column tbl_ext
    marks rows priced by agent A's own walkthrough run."""
    df["tbl_ext"] = False
    if not have_ext():
        return df
    for name, f in NAMES.items():
        t = pd.read_csv(os.path.join(EXT, f), usecols=["MATCH_CODE", "start", "exp_margin", "exp_total", "p_home"])
        t = t[t.start < "2026-09-01"].set_index("MATCH_CODE")
        idx = df.code.isin(t.index) & (df.day < "2026-09-01")
        codes = df.loc[idx, "code"]
        for col, src in (("margin", "exp_margin"), ("total", "exp_total"), ("p", "p_home")):
            df.loc[idx, f"tbl_{name}_{col}"] = t.loc[codes, src].to_numpy()
        df.loc[idx, "tbl_ext"] = True
    idx = df.tbl_ext
    df.loc[idx, "tbl_form_margin"] = df.loc[idx, "tbl_avg_margin"] + 0.5 * (df.loc[idx, "rating_margin"]
                                                                           - df.loc[idx, "tbl_avg_margin"])
    df.loc[idx, "tbl_form_total"] = df.loc[idx, "tbl_avg_total"]
    df.loc[idx, "tbl_meeting"] = df.loc[idx, "meeting"]
    return df
