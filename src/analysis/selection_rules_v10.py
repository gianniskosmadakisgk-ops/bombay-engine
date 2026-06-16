# src/analysis/selection_rules_v10.py
"""
Bombay Selection Rules v10

Purpose
-------
This file is the V10 selection filter layer.

It does NOT fetch data.
It does NOT calculate probabilities.
It receives Thursday candidates and decides:

- candidate rating
- Core eligibility
- Fun eligibility
- SuperFun eligibility
- favourite trap rejection
- artificial fill protection
- market exposure control

Naming:
- CoreBet   = singles
- FunBet    = system
- SuperFun  = system / lottery
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# CONFIG
# ============================================================

@dataclass(frozen=True)
class CoreSelectionConfig:
    odds_min: float = 1.55
    odds_max: float = 1.95

    min_confirmation: float = 0.78
    min_quality: float = 0.64
    min_value_pct: float = 3.0

    min_prob_side: float = 0.51
    min_prob_total: float = 0.52

    max_unders: int = 1
    max_same_market: int = 3
    max_total_markets: int = 4

    reject_slow_fav_trap: bool = True
    reject_odds_grade_below: str = "B"


@dataclass(frozen=True)
class FunSelectionConfig:
    odds_min: float = 1.80
    odds_max: float = 2.45

    min_confirmation: float = 0.72
    min_quality: float = 0.60
    min_value_pct: float = 2.0

    min_prob: float = 0.42

    target_n: int = 6
    max_n: int = 7

    max_same_market: int = 3
    max_unders: int = 2
    max_core_overlap: int = 2


@dataclass(frozen=True)
class SuperFunSelectionConfig:
    min_n: int = 10
    target_n: int = 12
    max_n: int = 12

    min_confirmation: float = 0.66
    min_quality: float = 0.56
    min_value_pct: float = 0.0

    max_unders: int = 3
    max_same_market: int = 5


CORE_CFG = CoreSelectionConfig()
FUN_CFG = FunSelectionConfig()
SUPERFUN_CFG = SuperFunSelectionConfig()


# ============================================================
# HELPERS
# ============================================================

def sf(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def si(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def odds_grade_rank(grade: str) -> int:
    g = str(grade or "").strip().upper()
    return {
        "A": 4,
        "B": 3,
        "C": 2,
        "D": 1,
    }.get(g, 0)


def grade_at_least(grade: str, minimum: str) -> bool:
    return odds_grade_rank(grade) >= odds_grade_rank(minimum)


def is_total_market(market: str) -> bool:
    return str(market or "") in ("Over 2.5", "Under 2.5")


def is_under(market: str) -> bool:
    return str(market or "") == "Under 2.5"


def is_side(market: str) -> bool:
    return str(market or "") in ("Home (1)", "Away (2)")


def fixture_key(row: Dict[str, Any]) -> Any:
    return row.get("fixture_id")


def market_key(row: Dict[str, Any]) -> str:
    return str(row.get("market") or "")


def chrono_key(row: Dict[str, Any]) -> Tuple[str, str]:
    return str(row.get("date") or ""), str(row.get("time_gr") or "")


def strength_key(row: Dict[str, Any]) -> Tuple[float, float, float, float]:
    return (
        sf(row.get("selection_score")),
        sf(row.get("confirmation_score")),
        sf(row.get("quality")),
        sf(row.get("value_pct")),
    )


# ============================================================
# CANDIDATE SCORING
# ============================================================

def market_probability_ok(row: Dict[str, Any], mode: str) -> bool:
    market = market_key(row)
    prob = sf(row.get("prob"))

    if mode == "core":
        if is_side(market):
            return prob >= CORE_CFG.min_prob_side
        if is_total_market(market):
            return prob >= CORE_CFG.min_prob_total
        return False

    return prob >= FUN_CFG.min_prob


def favourite_trap(row: Dict[str, Any]) -> bool:
    """
    Reject the exact thing that hurt us:
    favourite sides without enough separation / style / value.

    It is intentionally simple.
    The Thursday engine already calculated slow_fav_trap.
    """
    if bool(row.get("slow_fav_trap")):
        return True

    market = market_key(row)
    if not is_side(market):
        return False

    prob_gap = sf(row.get("prob_gap"))
    value_pct = sf(row.get("value_pct"))
    confirmation = sf(row.get("confirmation_score"))

    if prob_gap < 0.10 and value_pct < 8.0:
        return True

    if confirmation < 0.76 and value_pct < 6.0:
        return True

    return False


def over_preferred_over_side(side_row: Dict[str, Any], over_row: Optional[Dict[str, Any]]) -> bool:
    """
    If same fixture has a strong Over and a shaky side,
    prefer Over. User observation: Overs confirm easier than weak sides.
    """
    if over_row is None:
        return False

    if market_key(side_row) not in ("Home (1)", "Away (2)"):
        return False

    if market_key(over_row) != "Over 2.5":
        return False

    if bool(over_row.get("over_style_ok")) is False:
        return False

    over_conf = sf(over_row.get("confirmation_score"))
    side_conf = sf(side_row.get("confirmation_score"))
    over_val = sf(over_row.get("value_pct"))
    side_val = sf(side_row.get("value_pct"))

    if over_conf >= side_conf + 0.03 and over_val >= side_val:
        return True

    if over_val >= side_val + 6.0 and over_conf >= 0.76:
        return True

    return False


def selection_score(row: Dict[str, Any]) -> float:
    """
    One practical score for ordering candidates.

    Value matters, but not blindly.
    Confirmation + quality keep us away from fake value.
    """
    prob = sf(row.get("prob"))
    value_pct = sf(row.get("value_pct"))
    confirmation = sf(row.get("confirmation_score"))
    quality = sf(row.get("quality"))
    grade_rank = odds_grade_rank(str(row.get("odds_match_grade") or ""))

    score = (
        confirmation * 45.0
        + quality * 25.0
        + prob * 20.0
        + min(value_pct, 25.0) * 1.2
        + grade_rank * 2.0
    )

    if favourite_trap(row):
        score -= 18.0

    if market_key(row) == "Over 2.5" and bool(row.get("over_style_ok")):
        score += 4.0

    if market_key(row) == "Under 2.5":
        score -= 3.0

    return round(score, 4)


def money_rating(row: Dict[str, Any]) -> str:
    conf = sf(row.get("confirmation_score"))
    q = sf(row.get("quality"))
    v = sf(row.get("value_pct"))
    p = sf(row.get("prob"))
    grade = str(row.get("odds_match_grade") or "").upper()

    if favourite_trap(row):
        return "NORMAL"

    if conf >= 0.86 and q >= 0.75 and v >= 8.0 and p >= 0.54 and grade in ("A", "B"):
        return "ELITE"

    if conf >= 0.80 and q >= 0.68 and v >= 5.0 and p >= 0.51 and grade in ("A", "B"):
        return "STRONG"

    return "NORMAL"


def enrich_candidate(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    out["selection_score"] = selection_score(out)
    out["money_rating"] = money_rating(out)
    out["is_favourite_trap"] = favourite_trap(out)
    return out


# ============================================================
# CORE ELIGIBILITY
# ============================================================

def eligible_for_core(row: Dict[str, Any]) -> Tuple[bool, str]:
    odds = sf(row.get("odds"))
    market = market_key(row)

    if not (CORE_CFG.odds_min <= odds <= CORE_CFG.odds_max):
        return False, "odds_out_of_core_range"

    if not grade_at_least(str(row.get("odds_match_grade") or ""), CORE_CFG.reject_odds_grade_below):
        return False, "odds_grade_too_low"

    if sf(row.get("confirmation_score")) < CORE_CFG.min_confirmation:
        return False, "confirmation_too_low"

    if sf(row.get("quality")) < CORE_CFG.min_quality:
        return False, "quality_too_low"

    if sf(row.get("value_pct")) < CORE_CFG.min_value_pct:
        return False, "value_too_low"

    if not market_probability_ok(row, "core"):
        return False, "probability_too_low"

    if market == "Over 2.5" and bool(row.get("over_style_ok")) is False:
        return False, "over_style_not_ok"

    if CORE_CFG.reject_slow_fav_trap and favourite_trap(row):
        return False, "favourite_trap"

    if market not in ("Home (1)", "Away (2)", "Over 2.5", "Under 2.5"):
        return False, "unsupported_market"

    return True, "ok"


def select_core(candidates: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    enriched = [enrich_candidate(c) for c in candidates]

    by_fixture_market: Dict[Any, Dict[str, Dict[str, Any]]] = {}
    for c in enriched:
        fid = fixture_key(c)
        by_fixture_market.setdefault(fid, {})[market_key(c)] = c

    eligible: List[Dict[str, Any]] = []
    rejected: Dict[str, int] = {}

    for c in enriched:
        ok, reason = eligible_for_core(c)
        if not ok:
            rejected[reason] = rejected.get(reason, 0) + 1
            continue

        if is_side(market_key(c)):
            over_row = by_fixture_market.get(fixture_key(c), {}).get("Over 2.5")
            if over_preferred_over_side(c, over_row):
                rejected["side_yielded_to_better_over"] = rejected.get("side_yielded_to_better_over", 0) + 1
                continue

        eligible.append(c)

    eligible.sort(key=strength_key, reverse=True)

    selected: List[Dict[str, Any]] = []
    used_fixtures = set()
    market_counts: Dict[str, int] = {}
    totals_count = 0
    unders_count = 0

    for c in eligible:
        fid = fixture_key(c)
        market = market_key(c)

        if fid in used_fixtures:
            continue

        if market_counts.get(market, 0) >= CORE_CFG.max_same_market:
            continue

        if is_total_market(market):
            if totals_count >= CORE_CFG.max_total_markets:
                continue
            if is_under(market) and unders_count >= CORE_CFG.max_unders:
                continue

        selected.append(c)
        used_fixtures.add(fid)
        market_counts[market] = market_counts.get(market, 0) + 1

        if is_total_market(market):
            totals_count += 1
        if is_under(market):
            unders_count += 1

        if len(selected) >= 5:
            break

    selected.sort(key=chrono_key)

    debug = {
        "mode": "core_v10",
        "candidates_total": len(candidates),
        "eligible_before_exposure": len(eligible),
        "selected": len(selected),
        "rejected": rejected,
        "market_counts": market_counts,
        "totals_count": totals_count,
        "unders_count": unders_count,
        "rules": {
            "odds_range": [CORE_CFG.odds_min, CORE_CFG.odds_max],
            "min_confirmation": CORE_CFG.min_confirmation,
            "min_quality": CORE_CFG.min_quality,
            "min_value_pct": CORE_CFG.min_value_pct,
            "max_same_market": CORE_CFG.max_same_market,
            "max_total_markets": CORE_CFG.max_total_markets,
            "max_unders": CORE_CFG.max_unders,
        },
    }

    return selected, debug


# ============================================================
# FUN ELIGIBILITY / SELECTION
# ============================================================

def eligible_for_fun(row: Dict[str, Any]) -> Tuple[bool, str]:
    odds = sf(row.get("odds"))
    market = market_key(row)

    if not (FUN_CFG.odds_min <= odds <= FUN_CFG.odds_max):
        return False, "odds_out_of_fun_range"

    if not grade_at_least(str(row.get("odds_match_grade") or ""), "C"):
        return False, "odds_grade_too_low"

    if sf(row.get("confirmation_score")) < FUN_CFG.min_confirmation:
        return False, "confirmation_too_low"

    if sf(row.get("quality")) < FUN_CFG.min_quality:
        return False, "quality_too_low"

    if sf(row.get("value_pct")) < FUN_CFG.min_value_pct:
        return False, "value_too_low"

    if not market_probability_ok(row, "fun"):
        return False, "probability_too_low"

    if market == "Over 2.5" and bool(row.get("over_style_ok")) is False:
        return False, "over_style_not_ok"

    if favourite_trap(row):
        return False, "favourite_trap"

    if market not in ("Home (1)", "Away (2)", "Over 2.5", "Under 2.5"):
        return False, "unsupported_market"

    return True, "ok"


def select_fun(candidates: List[Dict[str, Any]], core_fixture_ids: Optional[set] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    core_fixture_ids = core_fixture_ids or set()
    enriched = [enrich_candidate(c) for c in candidates]

    eligible: List[Dict[str, Any]] = []
    rejected: Dict[str, int] = {}

    for c in enriched:
        ok, reason = eligible_for_fun(c)
        if ok:
            eligible.append(c)
        else:
            rejected[reason] = rejected.get(reason, 0) + 1

    eligible.sort(key=strength_key, reverse=True)

    selected: List[Dict[str, Any]] = []
    used_fixtures = set()
    market_counts: Dict[str, int] = {}
    unders_count = 0
    core_overlap = 0

    def try_add(c: Dict[str, Any]) -> bool:
        nonlocal unders_count, core_overlap

        fid = fixture_key(c)
        market = market_key(c)

        if fid in used_fixtures:
            return False

        if market_counts.get(market, 0) >= FUN_CFG.max_same_market:
            return False

        if is_under(market) and unders_count >= FUN_CFG.max_unders:
            return False

        if fid in core_fixture_ids and core_overlap >= FUN_CFG.max_core_overlap:
            return False

        selected.append(c)
        used_fixtures.add(fid)
        market_counts[market] = market_counts.get(market, 0) + 1

        if is_under(market):
            unders_count += 1

        if fid in core_fixture_ids:
            core_overlap += 1

        return True

    non_overlap = [c for c in eligible if fixture_key(c) not in core_fixture_ids]
    overlap = [c for c in eligible if fixture_key(c) in core_fixture_ids]

    for bucket in (non_overlap, overlap):
        for c in bucket:
            if len(selected) >= FUN_CFG.target_n:
                break
            try_add(c)

    if len(selected) == FUN_CFG.target_n:
        extra_pool = [c for c in non_overlap + overlap if fixture_key(c) not in used_fixtures]
        extra_pool.sort(key=strength_key, reverse=True)

        for c in extra_pool:
            if len(selected) >= FUN_CFG.max_n:
                break

            if sf(c.get("confirmation_score")) < 0.82:
                continue
            if sf(c.get("quality")) < 0.68:
                continue
            if sf(c.get("value_pct")) < 5.0:
                continue

            try_add(c)

    selected.sort(key=chrono_key)

    debug = {
        "mode": "fun_v10",
        "candidates_total": len(candidates),
        "eligible": len(eligible),
        "selected": len(selected),
        "rejected": rejected,
        "market_counts": market_counts,
        "unders_count": unders_count,
        "core_overlap": core_overlap,
        "rules": {
            "odds_range": [FUN_CFG.odds_min, FUN_CFG.odds_max],
            "target_n": FUN_CFG.target_n,
            "max_n": FUN_CFG.max_n,
            "max_core_overlap": FUN_CFG.max_core_overlap,
        },
    }

    return selected, debug


# ============================================================
# SUPERFUN SELECTION
# ============================================================

def select_superfun(core: List[Dict[str, Any]], fun: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    SuperFun is built only from Core + Fun.

    Core has priority.
    Fun fills the rest.
    """
    pool: List[Dict[str, Any]] = []

    for row in core:
        x = dict(row)
        x["source"] = "core"
        pool.append(x)

    for row in fun:
        x = dict(row)
        x["source"] = "fun"
        pool.append(x)

    best_by_fixture: Dict[Any, Dict[str, Any]] = {}

    for row in pool:
        fid = fixture_key(row)
        if fid is None:
            continue

        current = best_by_fixture.get(fid)
        if current is None:
            best_by_fixture[fid] = row
            continue

        if current.get("source") == "core":
            continue

        if row.get("source") == "core":
            best_by_fixture[fid] = row
            continue

        if strength_key(row) > strength_key(current):
            best_by_fixture[fid] = row

    unique = list(best_by_fixture.values())
    unique = [enrich_candidate(x) for x in unique]

    eligible = []
    rejected = {
        "confirmation_too_low": 0,
        "quality_too_low": 0,
    }

    for row in unique:
        if sf(row.get("confirmation_score")) < SUPERFUN_CFG.min_confirmation:
            rejected["confirmation_too_low"] += 1
            continue
        if sf(row.get("quality")) < SUPERFUN_CFG.min_quality:
            rejected["quality_too_low"] += 1
            continue
        eligible.append(row)

    eligible.sort(key=lambda x: (x.get("source") == "core", *strength_key(x)), reverse=True)

    selected: List[Dict[str, Any]] = []
    market_counts: Dict[str, int] = {}
    unders_count = 0

    for row in eligible:
        if len(selected) >= SUPERFUN_CFG.max_n:
            break

        market = market_key(row)

        if market_counts.get(market, 0) >= SUPERFUN_CFG.max_same_market:
            continue

        if is_under(market) and unders_count >= SUPERFUN_CFG.max_unders:
            continue

        selected.append(row)
        market_counts[market] = market_counts.get(market, 0) + 1

        if is_under(market):
            unders_count += 1

    selected.sort(key=chrono_key)

    debug = {
        "mode": "superfun_v10_from_core_plus_fun",
        "input_core": len(core),
        "input_fun": len(fun),
        "unique_fixtures": len(unique),
        "eligible": len(eligible),
        "selected": len(selected),
        "rejected": rejected,
        "market_counts": market_counts,
        "unders_count": unders_count,
        "rules": {
            "min_n": SUPERFUN_CFG.min_n,
            "target_n": SUPERFUN_CFG.target_n,
            "max_n": SUPERFUN_CFG.max_n,
            "max_same_market": SUPERFUN_CFG.max_same_market,
            "max_unders": SUPERFUN_CFG.max_unders,
        },
    }

    return selected, debug


# ============================================================
# TEST OUTPUT
# ============================================================

def selection_rules_summary() -> Dict[str, Any]:
    return {
        "version": "selection_rules_v10",
        "core": CORE_CFG.__dict__,
        "fun": FUN_CFG.__dict__,
        "superfun": SUPERFUN_CFG.__dict__,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(selection_rules_summary(), ensure_ascii=False, indent=2))
