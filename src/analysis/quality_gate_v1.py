# filename: src/analysis/quality_gate_v1.py
# ============================================================
# QUALITY GATE v1.0 — Friday Pre-filter
#
# Goal:
#   Cut "garbage fixtures" before Friday pickers touch them.
#   Uses existing Thursday fields: odds_match, flags.confidence,
#   flags.*_missing, flags.odds_strict_ok, flags.prob_instability.
#
# Output:
#   - fixture_quality_score(fx) -> 0..1
#   - fixture_passes_quality(fx, threshold) -> bool
#
# Notes:
#   - Missing fields are handled safely (never crash).
#   - Score is conservative: missing data penalizes.
# ============================================================

from __future__ import annotations
from typing import Dict, Any, Optional

def _safe_float(v, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default

def _safe_bool(v, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "y", "ok"):
        return True
    if s in ("false", "0", "no", "n"):
        return False
    return default

def fixture_quality_score(
    fx: Dict[str, Any],
    *,
    use_strict_bonus: bool = True,
    use_instability_penalty: bool = True,
) -> float:
    """
    Compute 0..1 quality score from fixture's meta fields.
    """
    flags = (fx or {}).get("flags") or {}
    odds_match = (fx or {}).get("odds_match") or {}

    # --- odds match (biggest signal)
    matched = _safe_bool(odds_match.get("matched"), False)
    om_score = _safe_float(odds_match.get("score"), 0.0)
    om_score = max(0.0, min(1.0, om_score))
    odds_part = (om_score if matched else 0.0)

    # --- confidence (if missing, we don't kill it, but we don't reward it either)
    conf = flags.get("confidence")
    conf = _safe_float(conf, 0.0) if conf is not None else 0.0
    conf = max(0.0, min(1.0, conf))

    # --- missing data penalties
    style_missing = _safe_bool(flags.get("style_missing"), False)
    history_missing = _safe_bool(flags.get("history_missing"), False)
    value_missing = _safe_bool(flags.get("value_missing"), False)

    missing_cnt = int(style_missing) + int(history_missing) + int(value_missing)
    # 0 missing => 1.00 ; 1 => 0.75 ; 2 => 0.45 ; 3 => 0.15
    missing_factor = {0: 1.00, 1: 0.75, 2: 0.45, 3: 0.15}.get(missing_cnt, 0.15)

    # --- strict odds bonus (small)
    strict_ok = _safe_bool(flags.get("odds_strict_ok"), False)
    strict_bonus = 0.03 if (use_strict_bonus and strict_ok) else 0.0

    # --- instability penalty (small but real)
    # prob_instability ~ 0..1 ; bigger = worse
    inst = flags.get("prob_instability")
    inst = _safe_float(inst, 0.0) if inst is not None else 0.0
    inst = max(0.0, min(1.0, inst))
    inst_penalty = (0.10 * inst) if use_instability_penalty else 0.0

    # Weighted score
    # odds=0.50, confidence=0.25, completeness=0.25
    score = (
        0.50 * odds_part +
        0.25 * conf +
        0.25 * missing_factor
    )
    score = score + strict_bonus - inst_penalty

    # Clamp 0..1
    score = max(0.0, min(1.0, score))
    return round(score, 4)

def fixture_passes_quality(
    fx: Dict[str, Any],
    threshold: float,
) -> bool:
    if threshold is None:
        return True
    try:
        th = float(threshold)
    except Exception:
        return True
    sc = fixture_quality_score(fx)
    return sc >= th
