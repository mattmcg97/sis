# B v21 simulation (v21.py 2335–end, v21_rules, game.py)
Scripts: scratchpad/sweep/B (models on _matches(n=80, seed=3)).

M1 BUG VERIFIED auto_end a play clock early when clock stopped (same as A-M1). v21.py:640, simulate 3409. (5-down)*PLAY_CLOCK > clock; prev never checked; no 2-min warning. Q4 0:35 leader 4th & 6 off stopped clock (prev NOTHING), defense no timeouts → sim ends game 200/200 paths; real: leader must snap 4th down, trailer gets ball. 1st & 10 at 2:20 after kickoff: sim ends at once; real snaps 2:20, 2:00 (warning), 1:18, 0:36 → 4th down ~30s left. Stopped-clock cut-offs ~126/84/42/0 vs sim 160/120/80/40. Impact: late-Q4 no-timeout stopped-clock states (incl. trailer just scored & kicked off) leader ML too high, Under too likely; few points win prob in those states. Fix: running_of(prev) into auto_end; (4-down)*PLAY_CLOCK + kneel seconds when stopped; Q4 >2:00 warning.

M2 METHOD VERIFIED OT starts carry no possession history. v21.py:2847-2850 simulate ot_done set-up; match_starts 2733, v21_stream.state_starts, trader.start_of no field. ot_done inferred from score: leader had ball, nobody else. Level OT after one possession = fresh OT. OT 14-14 away on ball (after home punted): 4,000 paths, 24% both scored, 7.7% away FG then home win — impossible under README rule. Trader try start home FG then away TD 20-17: home scored again 44%, won 44%, correct 0%. Real OT ~2.7% of games. Fix: had-the-ball flags from OT rows / game.py ot_had_ball into start.

L1 BUG VERIFIED missed FG: sim ball at line of scrimmage, game.py spot of kick. v21.py:3031 max(20, 100-y) vs game.py:322 KICK_BEHIND=7 (test_game.py:51). y=70: sim own 30, game.py 37. NFL/Madden spot of kick; sim14 fitted fg_after from kick_decisions' after (v21 discards). ~0.3 EP per miss, ~0.1 pts/game toward kicker. Fix: fit spot from after.
L2 BUG VERIFIED punt-return TDs → DEF_TD on 4th-down go plays; sim never returns punt for TD. v21.py:1227-1237 snap_records. Scouting classifies punt-return TD as PUNT; _scored fallback logs 4th-down snap DEF_TD (read_off_score). kick_decisions logs same play as punt. Blocked FG TDs same. Inflates DEF_TD on go plays. Small. Fix: skip read-off-score records whose next row is PUNT/FIELD_GOAL; kick-return TDs into punt draw.
L3 BUG REASONED OT possession tracking misses kick-return TDs & defensive scores. v21.py:3388-3399, 2979 walk.
L4 BUG VERIFIED game.py OT never ends mid-period, always kicks to home; trader drops ot_had_ball. game.py:204-220, 212; trader.py:437. Trader only.
L5 BUG REASONED no opening receiver → second-half kick deterministically to away. v21.py:2745-2746 match_starts (first ... else 0 → kick2=1); v21_stream.state_starts. README 460-463 says coin per path. ~a possession of margin when blank; frequency unknown.
L6 METHOD REASONED play crossing 2:00 never brings 2-min warning. v21.py:3267; match_starts prev 2761. ~20% of halves lose a stoppage, ~25s.
L7 BUG REASONED free kick after safety priced as ordinary kickoff (state_starts SAFETY → KICK; game.py:180). Rare.
L8 BUG REASONED try start at 0:00 kicks off before period ends. v21.py:3377-3403. Trader only.
L9 BUG REASONED match_starts distance "0" → 10. v21.py:2773.
Sound: v21_rules tree/binomial/mask; starts' side handling (TEAM_A away matched); scoring arithmetic; second-half kick; seed reproducibility; build reports.
