# src/analysis/quality_gate_v1.py
"""
Quality gate for Friday shortlist.

Purpose:
- Give every Thursday fixture a 0..1 quality score based on flags and odds matching.
- Allow Friday to filter out "garbage" fixtures before making picks.

This file is intentionally dependency-free (stdlib only).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class QualityResult:
    score: float
    reasons: Tuple[str, ...]


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return hi if x > hi else lo if x < lo else x


def fixture_quality_score(fixture: Dict[str, Any]) -> QualityResult:
    """
    Returns:
        QualityResult(score=0..1, reasons=(...))

    Expected fixture shape (from Thursday v3):
        fixture["flags"] : dict with booleans:
            odds_matched, value_missing, history_missing, style_missing, confidence_low, etc.
        fixture["odds_match"] : dict with:
            matched: bool, score: float (0..1), strict_ok: bool (optional)
    """
    flags = fixture.get("flags") or {}
    odds_match = fixture.get("odds_match") or {}

    reasons = []

    # Start from "odds match" score if present, else from flags.odds_matched
    om_score = odds_match.get("score")
    if isinstance(om_score, (int, float)):
        base = float(om_score)
    else:
        base = 1.0 if flags.get("odds_matched") is True else 0.0

    score = base

    # Hard-ish penalties: missing critical data
    if flags.get("value_missing"):
        score -= 0.22
        reasons.append("value_missing")
    if flags.get("history_missing"):
        score -= 0.18
        reasons.append("history_missing")
    if flags.get("style_missing"):
        score -= 0.18
        reasons.append("style_missing")

    # Odds strictness / mismatch
    if odds_match.get("matched") is False:
        score -= 0.30
        reasons.append("odds_not_matched")
    if odds_match.get("strict_ok") is False:
        score -= 0.10
        reasons.append("odds_strict_fail")

    # Confidence (if Thursday flagged it)
    if flags.get("confidence_low"):
        score -= 0.12
        reasons.append("confidence_low")

    # If Thursday says it's unstable, we don't trust it.
    if flags.get("prob_instability"):
        score -= 0.10
        reasons.append("prob_instability")

    # Gentle nudge: if we have a confidence scalar use it.
    conf = fixture.get("confidence")
    if isinstance(conf, (int, float)):
        # center around 0.5; boost good confidence slightly
        score += (float(conf) - 0.5) * 0.10

    return QualityResult(score=_clamp(score), reasons=tuple(reasons))
