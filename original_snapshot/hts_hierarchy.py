"""
hts_hierarchy.py
----------------
Hierarchy-aware evaluation for U.S. 10-digit HTS codes.

Implements Decision 2 of the design-lock: the severity ladder respects the
8-digit U.S. rate line (the legally operative duty level) and treats the
national extension as two hierarchical PAIRS (digits 7-8, then 9-10), not as
four independent digits.

U.S. HTS structure of a 10-digit code:
    digits 1-2   chapter
    digits 1-4   heading
    digits 1-6   subheading      (international HS)
    digits 1-8   rate line       (U.S. tariff / duty level)
    digits 1-10  statistical     (full national code)

This module has NO dataset or API dependency and is fully unit-tested, so it
can be trusted before the sprint begins.
"""
from __future__ import annotations
import re
from dataclasses import dataclass

# Prefix lengths that matter, best(shortest) -> full.
LEVELS = (2, 4, 6, 8, 10)
LEVEL_NAME = {2: "chapter", 4: "heading", 6: "subheading", 8: "rate_line", 10: "full"}

# Severity ladder, best -> worst. ordinal 0 = exact match, 5 = wrong chapter.
SEVERITY_ORDER = (
    ("correct_10",        "Exact match at the full 10-digit code."),
    ("wrong_stat_suffix", "Correct through 8 digits (rate line); only the statistical suffix (9-10) differs."),
    ("wrong_rate_line",   "Correct through 6 digits (subheading); the U.S. rate line (7-8) is wrong."),
    ("wrong_subheading",  "Correct through 4 digits (heading); the subheading (5-6) is wrong."),
    ("wrong_heading",     "Correct chapter only; the heading (3-4) is wrong."),
    ("wrong_chapter",     "Wrong chapter: an entirely unrelated part of the schedule."),
)
SEVERITY_ORDINAL = {name: i for i, (name, _) in enumerate(SEVERITY_ORDER)}

# Weighting schemes for the sensitivity analysis (each sums to 1.0).
# Credit is nested/paired: a level scores only if every digit up to it is correct.
WEIGHT_SCHEMES = {
    "balanced":         {2: 0.15, 4: 0.20, 6: 0.30, 8: 0.20, 10: 0.15},
    "subheading_heavy": {2: 0.10, 4: 0.15, 6: 0.40, 8: 0.20, 10: 0.15},
    "extension_heavy":  {2: 0.10, 4: 0.15, 6: 0.25, 8: 0.25, 10: 0.25},
}


def normalize_hts(code) -> str:
    """Return a clean 10-digit HTS string.

    Strips dots/spaces and zero-pads on the left to 10 digits, which recovers
    leading zeros lost when a code like 0101.21.0010 is read from a CSV as an
    integer. Raises ValueError if the result is not exactly 10 digits.
    """
    if code is None:
        raise ValueError("HTS code is None")
    digits = re.sub(r"\D", "", str(code))
    # Recover a single leading zero lost to integer conversion (e.g. chapter-01
    # code read as 101210010 -> 0101210010). HTS has no chapter 00, so at most
    # one zero can be lost; a genuinely shorter code is an error, not padded.
    if len(digits) == 9:
        digits = "0" + digits
    if len(digits) != 10:
        raise ValueError(f"HTS code must be 10 digits (got {len(digits)}): {code!r}")
    return digits


def deepest_correct_level(pred: str, gold: str) -> int:
    """Longest matching prefix length among LEVELS (0 if even the chapter is wrong)."""
    p, g = normalize_hts(pred), normalize_hts(gold)
    best = 0
    for n in LEVELS:
        if p[:n] == g[:n]:
            best = n
        else:
            break
    return best


def prefix_correct(pred: str, gold: str) -> dict[int, bool]:
    """Boolean prefix correctness at each of 2/4/6/8/10 digits."""
    p, g = normalize_hts(pred), normalize_hts(gold)
    return {n: (p[:n] == g[:n]) for n in LEVELS}


@dataclass(frozen=True)
class Severity:
    tier: str
    ordinal: int           # 0 best .. 5 worst
    deepest_correct: int   # 0/2/4/6/8/10
    description: str


def severity(pred: str, gold: str) -> Severity:
    """Classify a prediction into one of the six severity tiers (Decision 2)."""
    d = deepest_correct_level(pred, gold)
    tier = {10: "correct_10", 8: "wrong_stat_suffix", 6: "wrong_rate_line",
            4: "wrong_subheading", 2: "wrong_heading", 0: "wrong_chapter"}[d]
    return Severity(tier, SEVERITY_ORDINAL[tier], d, dict(SEVERITY_ORDER)[tier])


def partial_credit(pred: str, gold: str, scheme: str = "balanced") -> float:
    """Hierarchical partial-credit score in [0, 1] under a named weight scheme.

    Nested/paired: the weight for a level is earned only when the whole prefix
    up to that level is correct. The rate-line (7-8) and statistical (9-10)
    extensions are scored as pairs, never as loose individual digits.
    """
    if scheme not in WEIGHT_SCHEMES:
        raise KeyError(f"Unknown weight scheme {scheme!r}; options: {list(WEIGHT_SCHEMES)}")
    w = WEIGHT_SCHEMES[scheme]
    correct = prefix_correct(pred, gold)
    return round(sum(w[n] for n in LEVELS if correct[n]), 6)


def evaluate(pred: str, gold: str) -> dict:
    """Full per-case evaluation record (retrospective; requires the gold code)."""
    sev = severity(pred, gold)
    rec = {
        "pred": normalize_hts(pred),
        "gold": normalize_hts(gold),
        "exact_10": sev.tier == "correct_10",
        "exact_6": prefix_correct(pred, gold)[6],
        "severity_tier": sev.tier,
        "severity_ordinal": sev.ordinal,
        "deepest_correct": sev.deepest_correct,
        "prefix_correct": prefix_correct(pred, gold),
    }
    for scheme in WEIGHT_SCHEMES:
        rec[f"partial_{scheme}"] = partial_credit(pred, gold, scheme)
    return rec
