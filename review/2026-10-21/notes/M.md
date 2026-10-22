# M pages and command lines
Scripts: scratchpad/sweep/M (t1–t3.py, race.js, serve.py, work/). No HIGH.

M1 BUG VERIFIED trader page: play can vanish when clock runs out while priced. trader.html:500-517 send, :537-542 run-out timer, :659-667 & :550-554 Enter/space. Requests not queued; send() takes current history entry as base; Busy blocks mouse only. Playwright: Q1 0:03, clock started, "Play ▸" to 40 → history "... home play +15, first down (field 40)", "clock to 0:00; end of Q1 (period 2, field 25)" → +15 gone. Double Enter/Space same any time. Fix: queue requests; timer skips clock-out while pending.

M2 BUG VERIFIED OT: who already had ball dropped when state becomes sim start (= B-M2). trader.py:437-455 start_of ignores game.ot_had_ball; v21.simulate 2848-2850; match_starts/state_starts no OT history. OT 7-7, home punted, away 1st & 10 own 25, 20k paths: trader P(home) 0.489 vs patched (home done) 0.349 → 14-pt ML error mid-OT, in trader and betting sim's in-play OT books. game.py only checks OT finish at period end.

M3 METHOD VERIFIED(size) trader nudge-share default 1.0 vs betting sim 0.75. trader.py:268, :546, __main__ trader --nudge-share 1.0 (help "as the betting sim was run"); docstring trader.py:262 says NUDGE_SHARE (0.75); config.V21_NUDGE_SHARE 0.75 (eAMFCalibrator/config.py:105); v21_stream.NUDGE_SHARE; replay_ui.DEFAULT_SHARE 0.75. README October table 75% beat 100% ML log loss (0.6806 vs 0.6842). Same match prior 26-13: share 1.0 ML 0.917 margin 11.2; 0.75 → 0.838, 8.1. (Note: N-M2 says Oct 12–18 drives runs were at 1.0 — need to reconcile which share the headline used.)

M4 BUG VERIFIED trader ignores pace table betting sim reads first. trader.py:288-292 drives_table(tables) passes prior table as explicit path; no --drives. Stream: drives_table(drives_path, prior_tables) → --v21-drives / $EAMF_V21_DRIVES first. $EAMF_V21_DRIVES set: AF000 pace 0.80 stream vs 1.15 trader. Fix: drives_table(None, tables) or --drives.

L1 BUG VERIFIED replay chart solves nudge at 400 paths, step at 1000. replay_ui.py:425-427 price_all passes CHART_PATHS into self.nudge. Step nudge [5.49, 0.04] ML 0.823; chart [5.14, 0.15] 0.814. Fix: solve once at page paths. (My code from this session.)
L2 BUG VERIFIED trader blank Margin % breaks Start & later plays. trader.html:396-402, :450; trader.py:219 float(None) TypeError.
L3 BUG REASONED replay: match code not in --history loads without prod quotes/teams. replay_ui.py:105-115, 260-278: quote window defaults 2025-12-31→2026-01-03; no team tilts; follows-results prior gets whole history incl. this & later matches. README example --history AMFELO_20261013.csv → any later code hits it. Fix: refuse/warn or start from feed's first row.
L4 BUG REASONED trader post-safety state priced as normal kickoff (game.py:179-181).
L5 BUG REASONED trader --version accepts any v21* ignoring suffix (trader.py:478-481, 547-549): v21-snap4, v21-drives0.5 run as v21-drives at 1.0.
L6 BUG REASONED v21-build / v21-sweep silently skip missing explicit --history.
L7 BUG REASONED trader Edit state forces fresh=false, keeps last clock class (trader.html:617-622).
Sound: trader home/away, price display, kick-off nudge & tempo carry, conversion & kick-off starts, cache key; CLI wiring read.
