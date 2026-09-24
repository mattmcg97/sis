"""What the HTML reports call the candidate.

The suite compares prod against "the candidate". When an eAMFModel version
stands in for it (--candidate v3), the reports should say so: every
visible "candidate" / "cand" becomes the version's name. The name comes from
--candidate-label, else the model version, else it stays "candidate".

Only what a reader sees is renamed -- text between tags and title=""
tooltips. Tags, classes, data attributes, scripts and styles are left as
they are, so nothing the page runs on changes.
"""

import re

from . import config

_PIECES = re.compile(r"(<script\b.*?</script>|<style\b.*?</style>|<[^>]+>)",
                     re.IGNORECASE | re.DOTALL)
_TITLE_ATTR = re.compile(r'(\btitle=")([^"]*)(")')


def candidate_label():
    """The candidate's display name."""
    if getattr(config, "CANDIDATE_LABEL", None):
        return config.CANDIDATE_LABEL
    stream = config.STREAMS.get("candidate", "") or ""
    if stream.upper().startswith("MODEL:"):
        return stream.split(":", 1)[1] or "candidate"
    return "candidate"


def _words(text, label):
    upper = label.upper() if label.islower() else label
    for pattern, repl in (
            (r"\b[Tt]he candidate's\b", f"{label}'s"),
            (r"\b[Tt]he candidate\b", label),
            (r"\bCANDIDATE\b", upper),
            (r"\bCandidate\b", label),
            (r"\bcandidate\b", label),
            (r"\bCAND\b", upper),
            (r"\bCand\b", label),
            (r"\bcand\b", label),
            (r"\bC gap\b", f"{label} gap")):
        text = re.sub(pattern, repl, text)
    return text


def relabel(document, label=None):
    """The HTML with the candidate called `label` wherever a reader sees it."""
    label = label or candidate_label()
    if label.lower() == "candidate":
        return document
    out = []
    for piece in _PIECES.split(document):
        if not piece:
            continue
        if piece.startswith("<"):
            head = piece[:7].lower()
            if head.startswith("<script") or head.startswith("<style"):
                out.append(piece)
            else:
                out.append(_TITLE_ATTR.sub(
                    lambda m: m.group(1) + _words(m.group(2), label) + m.group(3), piece))
        else:
            out.append(_words(piece, label))
    return "".join(out)


def default_report_name(base):
    """eamf_report.html -> eamf_report_v3.html when a named model stands in."""
    label = candidate_label()
    if label.lower() == "candidate":
        return base
    stem, dot, ext = base.rpartition(".")
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", label)
    return f"{stem}_{safe}.{ext}" if dot else f"{base}_{safe}"
