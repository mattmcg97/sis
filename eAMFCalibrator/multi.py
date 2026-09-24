"""Several candidates against prod in one report (`--candidate v4,v5`).

Each candidate is paired with prod in its own pass -- the pairing engine
compares two streams -- and the report sets the passes side by side, so
every table reads real, prod, then each candidate.

Two things keep the columns comparable:

  * one population: every side is cut to the snapshots (match, drive,
    market) that all of them paired, so prod's figures and the real
    outcomes are the same whichever candidate a column belongs to;
  * one question per table: a model that quotes its own lines (v4, v5) is
    read twice off one simulation -- at prod's line, where its probability
    answers prod's question (the calibration tables), and at its own line
    (the line tables: whose line landed nearer the result). A candidate
    that is a table of quotes has only its own quotes, used for both.

Repeated queries are answered from memory (config.FETCH_CACHE) and a model
is simulated once per match for both of its readings, so each extra pass
costs little.
"""

from . import config, directional, indrive, prematch, snowflake_io


def display_name(stream):
    """'MODEL:v4' -> 'v4'; a table keeps its name."""
    if snowflake_io.is_model(stream):
        name, lines = snowflake_io.model_version(stream)
        return name if lines is None else f"{name}@{lines}"
    return stream


def _pass(stream, verbose=True):
    config.STREAMS["candidate"] = stream
    sink, closing = indrive.Sink(), prematch.Sink()
    pairs, stats, header, scan = directional.run(verbose=verbose, sink=sink,
                                                 prematch_sink=closing)
    return {"pairs": pairs, "stats": stats, "header": header, "scan": scan,
            "indrive": sink.summary(), "prematch": closing.summary(),
            "prematch_rows": [prematch.row(o) for o in closing.observations]}


def run(candidates, verbose=True):
    """[side] per candidate stream, each with its "prob" pass (the
    candidate at prod's line where it has lines of its own) and its "line"
    pass (as it quotes)."""
    saved, cached = config.STREAMS["candidate"], config.FETCH_CACHE
    config.FETCH_CACHE = True
    sides = []
    try:
        for stream in candidates:
            name = display_name(stream)
            if len(candidates) == 1:
                # one candidate keeps the name the reports always gave it
                # (--candidate-label, the model version, else "candidate")
                from . import labels
                config.STREAMS["candidate"] = stream
                name = labels.candidate_label()
            if verbose:
                print(f"\nPairing prod against {name}")
            side = {"name": name, "stream": stream}
            side["line"] = _pass(stream, verbose)
            if snowflake_io.has_own_lines(stream) and snowflake_io.model_version(stream)[1] is None:
                if verbose:
                    print(f"  ... and {name} read at prod's line")
                side["prob"] = _pass(stream + "@prod", verbose)
            else:
                side["prob"] = side["line"]
            sides.append(side)
    finally:
        config.STREAMS["candidate"] = saved
        config.FETCH_CACHE = cached
        snowflake_io.clear_fetch_cache()
    return sides


def pair_key(pair):
    return (pair.match_code, pair.drive_number, pair.market_id)


def common_population(sides, which):
    """Cut every side's `which` pairs to the keys all sides paired.
    Returns the number of keys dropped for want of every side."""
    keys = None
    for side in sides:
        mine = {pair_key(p) for p in side[which]["pairs"]}
        keys = mine if keys is None else keys & mine
    keys = keys or set()
    dropped = 0
    for side in sides:
        before = len(side[which]["pairs"])
        side[f"{which}_pairs"] = [p for p in side[which]["pairs"] if pair_key(p) in keys]
        dropped = max(dropped, before - len(side[f"{which}_pairs"]))
    return dropped


def build(sides, n_bootstrap=2000):
    """Each side's report on the common population: "prob_full" off its
    prod-line pairs, "line_full" off its own-line pairs."""
    dropped = {"prob": common_population(sides, "prob"),
               "line": common_population(sides, "line")}
    for side in sides:
        side["prob_full"] = directional.build_full_report(side["prob_pairs"], n_bootstrap)
        if side["line"] is side["prob"]:
            side["line_full"] = side["prob_full"]
        else:
            side["line_full"] = directional.build_full_report(side["line_pairs"], n_bootstrap)
    return dropped
