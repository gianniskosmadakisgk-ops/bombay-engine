# src/analysis/bankroll_policy_v10.py
"""
Bombay Bankroll Policy v10

Purpose
-------
Central money-management rules for Bombay V10.

Important naming:
- CoreBet   = singles only
- FunBet    = system bet only
- SuperFun  = system / lottery bet only
- DrawBet name is retired

This file does NOT select picks.
It only decides:
- bankroll defaults
- weekly exposure targets
- hard caps
- Core single stake sizing
- FunBet system stake
- SuperFun system stake
- system k/n rules
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# CONFIG
# ============================================================

@dataclass(frozen=True)
class CorePolicy:
    bankroll_default: float = 1000.0
    target_exposure: float = 160.0
    hard_cap: float = 180.0

    elite_stake: float = 50.0
    strong_stake: float = 40.0
    normal_stake: float = 30.0

    min_picks: int = 2
    soft_target_picks: int = 4
    max_picks: int = 5


@dataclass(frozen=True)
class SystemPolicy:
    bankroll_default: float
    target_stake: float
    hard_cap: float

    min_picks: int
    soft_target_picks: int
    max_picks: int


CORE_POLICY = CorePolicy()

FUN_POLICY = SystemPolicy(
    bankroll_default=500.0,
    target_stake=60.0,
    hard_cap=70.0,
    min_picks=6,
    soft_target_picks=6,
    max_picks=7,
)

SUPERFUN_POLICY = SystemPolicy(
    bankroll_default=500.0,
    target_stake=50.0,
    hard_cap=60.0,
    min_picks=10,
    soft_target_picks=12,
    max_picks=12,
)


# ============================================================
# SAFE HELPERS
# ============================================================

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def round_money(value: Any) -> float:
    return round(safe_float(value, 0.0), 2)


# ============================================================
# CORE STAKING
# ============================================================

def core_stake_from_rating(rating: str) -> float:
    """
    CoreBet is singles only.

    Ratings expected:
    - ELITE
    - STRONG
    - NORMAL

    Unknown rating falls back to NORMAL.
    """
    r = str(rating or "").strip().upper()

    if r == "ELITE":
        return CORE_POLICY.elite_stake

    if r == "STRONG":
        return CORE_POLICY.strong_stake

    return CORE_POLICY.normal_stake


def core_rating_from_score(
    confirmation_score: float,
    quality: float,
    value_pct: float,
    prob: float,
    odds_grade: str,
    slow_fav_trap: bool = False,
) -> str:
    """
    Converts candidate metrics into a money rating.

    This is intentionally strict.
    We do not want 7 fake “strong” bets.
    """
    conf = safe_float(confirmation_score)
    q = safe_float(quality)
    v = safe_float(value_pct)
    p = safe_float(prob)
    grade = str(odds_grade or "").upper().strip()

    if slow_fav_trap:
        return "NORMAL"

    if (
        conf >= 0.86
        and q >= 0.75
        and v >= 8.0
        and p >= 0.54
        and grade in ("A", "B")
    ):
        return "ELITE"

    if (
        conf >= 0.80
        and q >= 0.68
        and v >= 5.0
        and p >= 0.51
        and grade in ("A", "B")
    ):
        return "STRONG"

    return "NORMAL"


def apply_core_exposure_cap(lines: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Receives already-ranked Core lines.
    Adds stakes.
    Stops before hard cap.

    Expected order:
    strongest first.
    """
    selected: List[Dict[str, Any]] = []
    exposure = 0.0

    for line in lines:
        if len(selected) >= CORE_POLICY.max_picks:
            break

        rating = str(line.get("money_rating") or line.get("rating") or "NORMAL").upper()
        stake = core_stake_from_rating(rating)

        if exposure + stake > CORE_POLICY.hard_cap:
            continue

        row = dict(line)
        row["money_rating"] = rating
        row["stake"] = round_money(stake)

        selected.append(row)
        exposure += stake

    debug = {
        "policy": "core_singles_v10",
        "bankroll_default": CORE_POLICY.bankroll_default,
        "target_exposure": CORE_POLICY.target_exposure,
        "hard_cap": CORE_POLICY.hard_cap,
        "min_picks": CORE_POLICY.min_picks,
        "soft_target_picks": CORE_POLICY.soft_target_picks,
        "max_picks": CORE_POLICY.max_picks,
        "selected_picks": len(selected),
        "open": round_money(exposure),
    }

    return selected, debug


# ============================================================
# SYSTEM BET RULES
# ============================================================

def fun_system_shape(n: int) -> Tuple[int, str]:
    """
    FunBet system shape.

    Current preferred logic:
    - 6 picks => 3/6
    - 7 picks => 4/7

    Why:
    FunBet is not lottery. It is controlled upside.
    """
    n = safe_int(n)

    if n >= 7:
        return 4, "4/7"

    if n >= 6:
        return 3, "3/6"

    if n > 0:
        return max(1, n - 3), f"{max(1, n - 3)}/{n}"

    return 0, "0/0"


def superfun_system_shape(n: int) -> Tuple[int, str]:
    """
    SuperFun system shape.

    Current preferred logic:
    - 10 picks => 6/10
    - 11 picks => 7/11
    - 12 picks => 8/12

    This is lottery/upside, but not random garbage.
    """
    n = safe_int(n)

    if n >= 12:
        return 8, "8/12"

    if n == 11:
        return 7, "7/11"

    if n == 10:
        return 6, "6/10"

    if n > 0:
        return max(1, n - 4), f"{max(1, n - 4)}/{n}"

    return 0, "0/0"


def system_columns(n: int, k: int) -> int:
    n = safe_int(n)
    k = safe_int(k)

    if n <= 0 or k <= 0 or k > n:
        return 0

    return int(comb(n, k))


def build_system_bet(
    picks: List[Dict[str, Any]],
    mode: str,
    requested_stake: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Builds system metadata only.

    mode:
    - fun
    - superfun
    """
    mode_clean = str(mode or "").strip().lower()

    if mode_clean == "fun":
        policy = FUN_POLICY
        k, target = fun_system_shape(len(picks))
    elif mode_clean == "superfun":
        policy = SUPERFUN_POLICY
        k, target = superfun_system_shape(len(picks))
    else:
        raise ValueError(f"Unknown system mode: {mode}")

    n = len(picks)
    columns = system_columns(n, k)

    stake_total = safe_float(requested_stake, policy.target_stake)
    stake_total = clamp(stake_total, 0.0, policy.hard_cap)

    stake_per_column = round(stake_total / columns, 4) if columns > 0 else 0.0

    return {
        "mode": mode_clean,
        "n": n,
        "k": k,
        "target": target,
        "columns": columns,
        "stake_total": round_money(stake_total),
        "stake": round_money(stake_total),
        "stake_per_column": stake_per_column,
        "min_hits": k,
        "policy": {
            "bankroll_default": policy.bankroll_default,
            "target_stake": policy.target_stake,
            "hard_cap": policy.hard_cap,
            "min_picks": policy.min_picks,
            "soft_target_picks": policy.soft_target_picks,
            "max_picks": policy.max_picks,
        },
    }


# ============================================================
# BANKROLL DEFAULTS
# ============================================================

def default_bankrolls() -> Dict[str, float]:
    return {
        "core": CORE_POLICY.bankroll_default,
        "fun": FUN_POLICY.bankroll_default,
        "superfun": SUPERFUN_POLICY.bankroll_default,
        "draw": SUPERFUN_POLICY.bankroll_default,
    }


def normalize_history_bankrolls(history: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """
    Reads current bankrolls from history.
    Falls back to V10 defaults.

    Supports old key:
    - draw

    Maps it to:
    - superfun
    """
    defaults = default_bankrolls()

    if not isinstance(history, dict):
        return defaults

    bc = history.get("bankroll_current") or {}
    if not isinstance(bc, dict):
        return defaults

    core = safe_float(bc.get("core"), defaults["core"])
    fun = safe_float(bc.get("fun"), defaults["fun"])

    superfun = bc.get("superfun")
    if superfun is None:
        superfun = bc.get("draw")

    superfun = safe_float(superfun, defaults["superfun"])

    return {
        "core": round_money(core),
        "fun": round_money(fun),
        "superfun": round_money(superfun),
        "draw": round_money(superfun),
    }


# ============================================================
# OUTPUT SUMMARY
# ============================================================

def bankroll_policy_summary() -> Dict[str, Any]:
    return {
        "version": "bankroll_policy_v10",
        "naming": {
            "core": "CoreBet singles",
            "fun": "FunBet system",
            "superfun": "SuperFun system / lottery",
            "draw": "retired alias for superfun",
        },
        "core": {
            "bankroll_default": CORE_POLICY.bankroll_default,
            "target_exposure": CORE_POLICY.target_exposure,
            "hard_cap": CORE_POLICY.hard_cap,
            "stakes": {
                "ELITE": CORE_POLICY.elite_stake,
                "STRONG": CORE_POLICY.strong_stake,
                "NORMAL": CORE_POLICY.normal_stake,
            },
            "min_picks": CORE_POLICY.min_picks,
            "soft_target_picks": CORE_POLICY.soft_target_picks,
            "max_picks": CORE_POLICY.max_picks,
        },
        "fun": {
            "bankroll_default": FUN_POLICY.bankroll_default,
            "target_stake": FUN_POLICY.target_stake,
            "hard_cap": FUN_POLICY.hard_cap,
            "system_rules": {
                "6": "3/6",
                "7": "4/7",
            },
            "min_picks": FUN_POLICY.min_picks,
            "soft_target_picks": FUN_POLICY.soft_target_picks,
            "max_picks": FUN_POLICY.max_picks,
        },
        "superfun": {
            "bankroll_default": SUPERFUN_POLICY.bankroll_default,
            "target_stake": SUPERFUN_POLICY.target_stake,
            "hard_cap": SUPERFUN_POLICY.hard_cap,
            "system_rules": {
                "10": "6/10",
                "11": "7/11",
                "12": "8/12",
            },
            "min_picks": SUPERFUN_POLICY.min_picks,
            "soft_target_picks": SUPERFUN_POLICY.soft_target_picks,
            "max_picks": SUPERFUN_POLICY.max_picks,
        },
    }


if __name__ == "__main__":
    import json

    print(json.dumps(bankroll_policy_summary(), ensure_ascii=False, indent=2))
