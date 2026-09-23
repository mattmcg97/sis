"""Every number the model runs on, with where it came from.

The model has no learned weights. What it has is a handful of structural
constants, each fitted to data in `fit.py` and recorded here with its
provenance, plus the one dial that separates the versions: how much the
pre-match view is trusted once the game starts (`prior_strength`).
"""

from dataclasses import dataclass, field, replace

from .clock import ClockParams
from .drive import DriveParams


@dataclass(frozen=True)
class Params:
    # Drives per team per regulation game. Maximum likelihood on 23,952 AF
    # finals (nb2/AMFELO.csv) under the possession model of fit.py: 4.80,
    # with the game's drive count normal, sd 2.14. Separately, 262 matches
    # of directional_pairs give 46% TD / 11% FG per drive, which is what the
    # same fit reads off the finals (45.1% / 10.3%). Two independent sources
    # agreeing is the check on the unit: a "drive" here is a real possession.
    drives_per_team: float = 4.80

    # Touchdown conversions. Finals fit: 6 only 5.1%, 8 (two-point) 4.3%;
    # drive data 4.4% / 3.6%.
    q6: float = 0.051
    q8: float = 0.043

    # Overtime settles on a field goal: the finals fit puts all of a
    # regulation tie's mass on a 3-point win (ot_fg = 1.0 at the bound).
    ot_margin: int = 3

    # On average half of the current drive's time is still to run, when
    # the feed cannot say how old the drive is.
    current_drive_share: float = 0.5

    # The clock a whole drive needs, in average drives' worth, and its
    # spread as a share of that: the chance the current drive is cut off
    # by the end of the half (pricer.Model._in_time). drive_time 0 turns
    # it off.
    drive_time: float = 1.2
    drive_time_cv: float = 0.8
    # How far the drive in progress follows the half's falling scoring rate
    # (clock.intensity, capped at 1): 0 not at all, 1 fully.
    current_intensity: float = 1.0

    # End of the game (pricer.Model._endgame). Late = fewer than this many
    # drives' worth of the game left, second half only.
    late_drives: float = 3.0
    # A leader on the ball late runs the clock: its drive uses this many
    # more drives' worth of time (fitted to 0: once the drive's own clock
    # and the half's falling scoring are in, extra run-off did not help)...
    kill_time: float = 0.0
    # ...and ahead by kill_lead or more it kneels, scoring this much less.
    kill_lead: int = 9
    kill_score: float = 0.8
    # ...ahead by less, it plays safe, scoring this much less (fitted on
    # SCOUTING_FULL: in the last two minutes a leader on the ball scored 2.4
    # more points on average where the model had 4.5; 0.5 took the Q4 totals
    # Brier from 0.248 to 0.237 against prod's 0.243 on training matches)...
    run_score: float = 0.5
    # ...and a trailer on the ball late hurries: its drive uses this many
    # drives' worth of clock less, leaving more for what follows.
    hurry: float = 0.0

    # Variance of the drive count, per expected drive, on top of what the
    # clock's own uncertainty adds. The finals fit's count variance is
    # 2.14^2 = 4.58 over 9.6 drives. On the half clock nothing of it is
    # clock uncertainty at kickoff (the whole game is ahead for certain),
    # so all of it is the game's own: 4.58 / 9.6 = 0.48. (On the quarter
    # clock the period-length spread takes 1.69 of it, leaving 0.30.)
    count_dispersion: float = 0.48

    # Mean points of a scoring drive: 0.46 TDs at ~6.99 and 0.11 FGs at 3,
    # over 0.57 scores.
    points_per_score: float = 6.2

    # Gamma prior shape on each side's scoring rate, in scores. Large =
    # hold the pre-match number, small = follow the game. See strength.py.
    prior_strength: float = 16.0
    # 1 = price at the posterior mean. Spreading the strength over quantiles
    # (3, 5, 7) widens the pre-match margin sd from 10.9 to 14+ at k = 4,
    # far past the 10.2 the finals show, and drags the mean off the line
    # through the curvature of the drive chain, so it is off by default.
    n_slices: int = 1

    # A 1st-and-10 at exactly this spot is a kickoff, not a snap: it is
    # priced as a drive about to start from `drive.start_field`.
    kickoff_field: int = 35

    clock: ClockParams = field(default_factory=ClockParams)
    drive: DriveParams = field(default_factory=DriveParams)

    def with_(self, **changes):
        return replace(self, **changes)


# Versions. Add one by naming what differs from the base.
VERSIONS = {
    # Anchored: the pre-match number is worth sixteen scores of evidence,
    # so four touchdowns above expectation move a side's rate about 25%.
    # On 262 matches (moneyline, the clean read) this sits level with prod;
    # trusting the scoreboard any more cost Brier -- see README.
    "v1": Params(prior_strength=16.0),
    # Reactive: four scores' worth, so the game takes over four times as
    # fast. The "a player playing well can't stay a pre-match underdog" view.
    "v2": Params(prior_strength=4.0),
}


def version(name):
    try:
        return VERSIONS[name.lower()]
    except KeyError:
        raise KeyError(f"unknown model version {name!r}; "
                       f"have {', '.join(sorted(VERSIONS))}") from None
