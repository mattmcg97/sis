"""Scoring rules and reliability summaries.

Probabilities are 0-1 here. GAMEPLAI publishes PROBABILITY on a 0-100
scale, so the pipeline divides by 100 on the way in -- see
pipeline.to_unit_probability.

Three numbers, because they answer different questions:

  Brier      mean squared error. Overall accuracy, rewards sharpness as
             well as calibration, so a well-calibrated but timid model
             still scores poorly.
  Log loss   punishes confident mistakes far harder than Brier. A model
             that says 99% and is wrong shows up here first.
  ECE        expected calibration error: average |realized - predicted|
             across probability bins, weighted by bin population. This is
             the one that isolates calibration from sharpness, and the one
             to read when asking "can I trust the number on the tin?".

Clipping for log loss is at 1e-6: this feed does publish 0.00 and values
above 99, and an unclipped log(0) on a single wrong row would take the
whole run to infinity.
"""

import math

LOG_LOSS_CLIP = 1e-6


def brier_score(pairs):
    """pairs: iterable of (predicted_probability, realized_outcome_bool)."""
    pairs = list(pairs)
    if not pairs:
        return None
    return sum((p - (1.0 if y else 0.0)) ** 2 for p, y in pairs) / len(pairs)


def log_loss(pairs):
    pairs = list(pairs)
    if not pairs:
        return None
    total = 0.0
    for p, y in pairs:
        p = min(max(p, LOG_LOSS_CLIP), 1.0 - LOG_LOSS_CLIP)
        total += -math.log(p) if y else -math.log(1.0 - p)
    return total / len(pairs)


def reliability_bins(pairs, n_bins):
    """Equal-width probability bins with predicted vs realized in each."""
    pairs = list(pairs)
    bins = [{"lo": i / n_bins, "hi": (i + 1) / n_bins, "n": 0,
             "pred_sum": 0.0, "wins": 0} for i in range(n_bins)]
    for p, y in pairs:
        idx = min(int(p * n_bins), n_bins - 1)
        b = bins[idx]
        b["n"] += 1
        b["pred_sum"] += p
        b["wins"] += 1 if y else 0
    for b in bins:
        b["predicted"] = b["pred_sum"] / b["n"] if b["n"] else None
        b["realized"] = b["wins"] / b["n"] if b["n"] else None
        b["gap"] = (b["realized"] - b["predicted"]) if b["n"] else None
    return bins


def expected_calibration_error(pairs, n_bins):
    pairs = list(pairs)
    if not pairs:
        return None
    total = len(pairs)
    ece = 0.0
    for b in reliability_bins(pairs, n_bins):
        if b["n"]:
            ece += (b["n"] / total) * abs(b["gap"])
    return ece


def summarize(pairs, n_bins):
    pairs = list(pairs)
    if not pairs:
        return {"n": 0, "brier": None, "log_loss": None, "ece": None,
                "mean_predicted": None, "realized": None, "gap": None}
    mean_pred = sum(p for p, _ in pairs) / len(pairs)
    realized = sum(1 for _, y in pairs if y) / len(pairs)
    return {
        "n": len(pairs),
        "brier": brier_score(pairs),
        "log_loss": log_loss(pairs),
        "ece": expected_calibration_error(pairs, n_bins),
        "mean_predicted": mean_pred,
        "realized": realized,
        "gap": realized - mean_pred,
    }
