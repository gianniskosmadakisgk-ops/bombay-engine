# src/analysis/quality_gate_v1.py
from __future__ import annotations

from typing import Any, Dict


def fixture_quality_score(fx: Dict[str, Any]) -> float:
    """
    Returns quality score in [0, 1].
    Uses flags + odds_match.score + confidence + missing-data penalties.
    """
    flags = fx.get("flags", {}) or {}
    odds_match = fx.get("odds_match", {}) or {}

    odds_matched = bool(flags.get("odds_matched", False))
    odds_score = float(odds_match.get("score", 0.0) or 0.0)  # 0..1 (if present)

    confidence = float(flags.get("confidence", 0.0) or 0.0)  # 0..1
    prob_instability = float(flags.get("prob_instability", 0.0) or 0.0)  # 0..?

    # Missing penalties
    missing_pen = 0.0
    if flags.get("value_missing", False):
        missing_pen += 0.20
    if flags.get("history_missing", False):
        missing_pen += 0.15
    if flags.get("style_missing", False):
        missing_pen += 0.15

    # Instability penalty (soft)
    # 0.00 -> 0 penalty, 0.05 -> ~0.10 penalty, 0.10 -> ~0.20 penalty
    instab_pen = min(0.20, prob_instability * 2.0)

    # If odds are not matched, quality is basically trash for Friday selection
    if not odds_matched:
        return 0.0

    # Base blend: odds quality + confidence
    # odds_score is heavy because mismatches/nulls were killing you
    base = 0.65 * odds_score + 0.35 * confidence

    score = base - missing_pen - instab_pen

    # Clamp
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


def passes_quality_gate(fx: Dict[str, Any], portfolio: str) -> bool:
    """
    Default thresholds for the "8.5" target.
    Core: 0.70
    Fun:  0.60
    Draw: 0.70
    """
    q = fixture_quality_score(fx)

    portfolio = (portfolio or "").lower().strip()
    if portfolio in ("core", "corebet"):
        return q >= 0.70
    if portfolio in ("fun", "funbet"):
        return q >= 0.60
    if portfolio in ("draw", "drawbet"):
        return q >= 0.70

    # safe default
    return q >= 0.70
