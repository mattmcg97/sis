"""The league's 4th-down and field-goal curves, fitted to real decisions in SCOUTING_FULL. The
simulations start from them and refit go_coef and fg_kick_coef on their own build's 4th downs."""

import math
from dataclasses import dataclass

FIELD = 100


@dataclass(frozen=True)
class DriveParams:
    # 4th down, fitted to 7,400 real 4th-down decisions and 3,000 field-goal attempts in
    # SCOUTING_FULL. Madden players go for it far more than the NFL: 4th-and-1 from their own 35
    # about 85% of the time, 4th-and-6 about half the time. Logistic in the log of the distance
    # and the field position (see go_probability); a kick is a field goal rather than a punt from
    # about the opponent's 45 on; and the kickers are good -- 95% at 45 yards, 50% near 63.
    go_coef: tuple = (-1.4844, -0.1921, 12.7965, -10.4955, -2.3300, 0.1111, 1.1210)
    fg_kick_coef: tuple = (-24.8943, 47.7511)
    make_coef: tuple = (1.7102, 14.8557, -27.0659)
    fg_max_distance: int = 75       # nothing is tried from further than this


def kick_distance(field_position):
    """A field goal's distance from the line of scrimmage."""
    return FIELD - field_position + 17


def _sigmoid(z):
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def fg_make(params, field_position):
    """P(a field goal from here is good): kick distance is 117 - field."""
    distance = kick_distance(field_position)
    if distance > params.fg_max_distance:
        return 0.0
    d = distance / 100.0
    a, b, c = params.make_coef
    return _sigmoid(a + b * d + c * d * d)


def go_probability(params, field_position, distance, aggression=0.0):
    """P(going for it on 4th down) for an average player, shifted in log odds by the player's
    `aggression`."""
    u = field_position / 100.0
    lt = math.log(max(1, distance))
    red = 1.0 if field_position >= 80 else 0.0
    c = params.go_coef
    z = c[0] + c[1] * lt + c[2] * u + c[3] * u * u + c[4] * lt * u + c[5] * red + c[6] * lt * red
    return _sigmoid(z + aggression)


def field_goal_share(params, field_position):
    """P(a 4th-down kick is a field goal rather than a punt)."""
    a, b = params.fg_kick_coef
    return _sigmoid(a + b * field_position / 100.0)
