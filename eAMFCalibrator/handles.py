"""Do the PLAYER_1 / PLAYER_2 handles stay pinned to the same team?

Everything the calibrator buckets on is read in the PLAYER_1 frame. The
score difference is p1 minus p2, the possession flag is Home or Away, and
each market resolves by asking which of the two won. If the handles swap
sides part-way through a match, all three invert from that point and the
match still looks perfectly well-formed: every row parses, every bucket
fills, and the numbers land in the wrong cells. That is worse than missing
data, because nothing downstream can tell.

Scoring only goes up in this sport, so a swap leaves a mark: one side's
cumulative total goes DOWN. That is the whole check, with two labels that
separate a swap from ordinary feed noise.

  MIRROR      the pair (p1, p2) becomes exactly (p2, p1). Nothing but a
              swap does that, so these are the ones to trust.
  REGRESSION  a total went down some other way -- a swap landing on the
              same message as a score, a rescinded score after review, or
              a feed glitch. Worth looking at, not proof on its own.

Two blind spots, both worth stating rather than discovering later:

  * A match crossed from its very first message never regresses, because
    there is nothing to regress from. It is simply mirrored throughout,
    and SCORE_CHANGES alone cannot see it. So the last running total is
    also checked against SCORE_ENDGAME, a different table: if those two
    disagree in mirror image, the whole match is crossed.
  * A swap while the score is level is invisible here, and stays invisible
    until the next score. Nothing in the score feed can date it.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Optional, Tuple


MIRROR = "mirror"
REGRESSION = "regression"
FINAL_MIRRORED = "final_mirrored"
FINAL_MISMATCH = "final_mismatch"

# The kinds that mean the handles moved. FINAL_MISMATCH is deliberately not
# one of them: the two tables disagreeing is worth knowing, but a missing
# late score explains it just as well as a swap does.
FLIP_KINDS = (MIRROR, REGRESSION, FINAL_MIRRORED)

KIND_ORDER = (MIRROR, REGRESSION, FINAL_MIRRORED, FINAL_MISMATCH)

# Width of the widest kind name, so the text tables can size their column
# from the labels rather than a number that drifts when one is added.
KIND_TITLES = {
    MIRROR: "Mirrored",
    REGRESSION: "Regression",
    FINAL_MIRRORED: "Final mirrored",
    FINAL_MISMATCH: "Final mismatch",
}

KIND_WIDTH = max(len(title) for title in KIND_TITLES.values()) + 2


@dataclass(frozen=True)
class ScoreAnomaly:
    match_code: str
    event_message_count: Optional[int]  # None where there is no one message
    kind: str
    before: Optional[Tuple[int, int]] = None
    after: Optional[Tuple[int, int]] = None
    # The possession kinds are not about two score states, so they carry a
    # rendered phrase instead of a pair of totals.
    detail: Optional[str] = None

    @property
    def is_flip(self):
        return self.kind in FLIP_KINDS

    @property
    def is_final_check(self):
        """True for the SCORE_ENDGAME cross-check, which has no message."""
        return self.kind in (FINAL_MIRRORED, FINAL_MISMATCH)

    @property
    def where(self):
        """Where in the match the flag sits, for a table column.

        Three cases, and they are not the same: a message count, the final
        cross-check which compares two tables rather than two moments, and
        a verdict on the match as a whole.
        """
        if self.event_message_count is not None:
            return f"{self.event_message_count:,}"
        return "final" if self.is_final_check else "match"

    @property
    def joiner(self):
        """The in-play kinds are a transition; the final kinds are a
        comparison between two tables, which a rightward arrow would
        misdescribe."""
        return "vs" if self.is_final_check else "->"

    def describe(self):
        if self.detail is not None:
            return self.detail
        b1, b2 = self.before
        a1, a2 = self.after
        return f"{b1}-{b2} {self.joiner} {a1}-{a2}"


def scan_match(match_code, scores, final=None):
    """Every score anomaly in one match, in message order.

    `scores` are ScoreRows; `final` is the (p1, p2) from SCORE_ENDGAME, or
    None when the match has not settled. Cumulative columns arrive sparse,
    so each side carries its last seen value forward independently -- the
    same rule drives.score_at reads them under.
    """
    anomalies = []
    p1, p2 = 0, 0
    for row in sorted(scores, key=lambda s: s.event_message_count):
        new_p1 = p1 if row.p1_cumulative is None else row.p1_cumulative
        new_p2 = p2 if row.p2_cumulative is None else row.p2_cumulative
        if (new_p1, new_p2) != (p1, p2):
            kind = None
            # An exact mirror is always also a regression, so it has to be
            # tested first or every swap would be filed as the vaguer kind.
            if p1 != p2 and (new_p1, new_p2) == (p2, p1):
                kind = MIRROR
            elif new_p1 < p1 or new_p2 < p2:
                kind = REGRESSION
            if kind is not None:
                anomalies.append(ScoreAnomaly(
                    match_code=match_code,
                    event_message_count=row.event_message_count,
                    kind=kind, before=(p1, p2), after=(new_p1, new_p2)))
        p1, p2 = new_p1, new_p2

    if final is not None and final[0] is not None and final[1] is not None:
        f1, f2 = final
        if (p1, p2) != (f1, f2):
            mirrored = f1 != f2 and (p1, p2) == (f2, f1)
            anomalies.append(ScoreAnomaly(
                match_code=match_code,
                event_message_count=None,
                kind=FINAL_MIRRORED if mirrored else FINAL_MISMATCH,
                before=(p1, p2), after=(f1, f2)))

    return anomalies


class Scan:
    """Accumulates anomalies across match chunks.

    Threaded through the chunk loop the same way `stats` is, so the scan
    covers the whole window rather than whichever chunk ran last.
    """

    def __init__(self):
        self.matches = 0
        self.anomalies = []
        self._flipped = set()

    def add(self, match_code, scores, final=None):
        """Scan one match. Returns True if its handles look stable."""
        self.matches += 1
        found = scan_match(match_code, scores, final)
        self.anomalies.extend(found)
        if any(a.is_flip for a in found):
            self._flipped.add(match_code)
            return False
        return True

    @property
    def flipped_matches(self):
        return frozenset(self._flipped)

    def summary(self):
        by_match = {}
        for anomaly in self.anomalies:
            by_match.setdefault(anomaly.match_code, []).append(anomaly)
        counts = Counter(a.kind for a in self.anomalies)
        matches_by_kind = {
            kind: len({a.match_code for a in self.anomalies if a.kind == kind})
            for kind in KIND_ORDER
        }
        return {
            "matches": self.matches,
            "matches_flagged": len(by_match),
            "matches_flipped": len(self._flipped),
            "matches_clean": self.matches - len(by_match),
            "flipped_matches": sorted(self._flipped),
            "counts": {kind: counts.get(kind, 0) for kind in KIND_ORDER},
            "matches_by_kind": matches_by_kind,
            "by_match": by_match,
            "anomalies": sorted(
                self.anomalies,
                key=lambda a: (a.match_code, a.event_message_count or 0)),
        }


def empty_summary():
    """The shape `summary()` returns when nothing was scanned."""
    return Scan().summary()
