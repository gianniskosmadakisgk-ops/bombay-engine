# src/analysis/quality_gate_v1.py
# ============================================================
# QUALITY GATE v1 — Production
#
# Goal:
#   Give each Thursday fixture a quality score 0..1 using existing flags.
#   Friday can then drop low-quality fixtures BEFORE building candidates.
#
# Inputs expected in fixture:
#   - odds_match: { matched: bool, score: float }
#   - flags: {
#       confidence: float|None,
#       odds_strict_ok: bool|None,
#       value_missing: bool|None,
#       history_missing: bool|None,
#       style_missing: bool|None,
#       prob_instability: float|None
#     }
# ============================================================

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Tuple

def _sf(x, d=None):
    try:
        return float(x)
    except Exception:
        return d

@dataclass(frozen=True)
class QualityResult:
    score: float
    reasons: Tuple[str, ...]

def fixture_quality_score(fx: Dict[str, Any]) -> QualityResult:
    """
    Returns score in [0..1] and reasons for penalties.
    This is intentionally conservative: missing signals reduce score.
    """
    reasons = []

    flags = (fx.get("flags") or {})
    om = (fx.get("odds_match") or {})

    # ---- odds match baseline ----
    matched = bool(om.get("matched"))
    om_score = _sf(om.get("score"), 0.0) or 0.0

    if not matched:
        reasons.append("odds_not_matched")
        # if odds don't match, quality is near-zero
        return QualityResult(score=0.0, reasons=tuple(reasons))

    # base from odds_match.score (already 0..1)
    q = max(0.0, min(1.0, om_score))

    # ---- confidence ----
    conf = _sf(flags.get("confidence"), None)
    if conf is None:
        # missing confidence = small penalty (unknown)
        q *= 0.95
        reasons.append("confidence_missing")
    else:
        # push low confidence down hard
        if conf < 0.40:
            q *= 0.70; reasons.append("confidence_low")
        elif conf < 0.50:
            q *= 0.85; reasons.append("confidence_mid")
        else:
            q *= 1.00

    # ---- missing data flags ----
    # Any missing inputs should penalize; this is exactly what you want.
    if flags.get("value_missing") is True:
        q *= 0.80; reasons.append("value_missing")
    if flags.get("history_missing") is True:
        q *= 0.80; reasons.append("history_missing")
    if flags.get("style_missing") is True:
        q *= 0.80; reasons.append("style_missing")

    # If flags are absent entirely, treat as unknown penalty
    if "value_missing" not in flags:
        q *= 0.95; reasons.append("value_flag_missing")
    if "history_missing" not in flags:
        q *= 0.95; reasons.append("history_flag_missing")
    if "style_missing" not in flags:
        q *= 0.95; reasons.append("style_flag_missing")

    # ---- strict odds (optional) ----
    # If strict flag exists and is False -> penalty
    # (Do NOT hard fail here; Friday can choose to enforce threshold)
    strict_ok = flags.get("odds_strict_ok")
    if strict_ok is False:
        q *= 0.90; reasons.append("odds_strict_failed")

    # ---- instability (optional) ----
    gap = _sf(flags.get("prob_instability"), None)
    if gap is None:
        q *= 0.97; reasons.append("instability_missing")
    else:
        if gap > 0.25:
            q *= 0.80; reasons.append("instability_high")
        elif gap > 0.18:
            q *= 0.90; reasons.append("instability_mid")

    q = max(0.0, min(1.0, q))
    return QualityResult(score=round(q, 4), reasons=tuple(reasons))
