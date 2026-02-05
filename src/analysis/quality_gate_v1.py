# src/analysis/quality_gate_v1.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class QualityResult:
    score: float
    reasons: Tuple[str, ...]


def fixture_quality_score(fx: Dict[str, Any]) -> QualityResult:
    """
    Quality score 0..1 based on Thursday fixture flags.
    Hard penalties for missing/odds mismatch, soft penalties for instability.
    """
    reasons = []

    flags = fx.get("flags") or {}
    odds_match = fx.get("odds_match") or {}

    score = 1.0

    # Odds matching is critical
    if not bool(flags.get("odds_matched", False)):
        score -= 0.55
        reasons.append("odds_not_matched")

    om_score = odds_match.get("score")
    if isinstance(om_score, (int, float)):
        if om_score < 0.60:
            score -= 0.20
            reasons.append("odds_match_low")
        elif om_score < 0.75:
            score -= 0.10
            reasons.append("odds_match_mid")

    # Missing core data
    if bool(flags.get("value_missing", False)) or bool(flags.get("value_mismatch", False)):
        score -= 0.20
        reasons.append("value_missing_or_mismatch")

    if bool(flags.get("history_missing", False)):
        score -= 0.20
        reasons.append("history_missing")

    if bool(flags.get("style_missing", False)):
        score -= 0.20
        reasons.append("style_missing")

    # Confidence bonus/penalty
    conf = flags.get("confidence")
    if isinstance(conf, (int, float)):
        if conf >= 0.75:
            score += 0.03
            reasons.append("high_confidence")
        elif conf <= 0.45:
            score -= 0.10
            reasons.append("low_confidence")

    # Instability penalty
    inst = flags.get("prob_instability")
    if isinstance(inst, (int, float)):
        if inst > 0.06:
            score -= 0.15
            reasons.append("prob_instability_high")
        elif inst > 0.03:
            score -= 0.08
            reasons.append("prob_instability_mid")

    # Clamp
    score = max(0.0, min(1.0, float(score)))

    return QualityResult(score=score, reasons=tuple(reasons))
