# src/analysis/learning_memory_v1.py
"""
Bombay Learning Memory v1

Reads:
- logs/tuesday_history_v3.json

Writes:
- logs/market_performance_memory.json

Purpose:
Creates performance memory by:
- market
- league
- odds band
- confidence band
- market + league
- market + odds band

This is the first learning layer for Bombay 2.0.
"""

from __future__ import annotations

import os
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

HISTORY_PATH = os.getenv(
    "TUESDAY_HISTORY_PATH",
    os.path.join(PROJECT_ROOT, "logs", "tuesday_history_v3.json")
)

OUTPUT_PATH = os.getenv(
    "MARKET_MEMORY_PATH",
    os.path.join(PROJECT_ROOT, "logs", "market_performance_memory.json")
)


# ------------------------------------------------------------
# SAFE HELPERS
# ------------------------------------------------------------

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


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def atomic_write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def norm_market(market: Any) -> str:
    m = str(market or "").strip()

    aliases = {
        "home": "Home (1)",
        "1": "Home (1)",
        "home (1)": "Home (1)",
        "away": "Away (2)",
        "2": "Away (2)",
        "away (2)": "Away (2)",
        "draw": "Draw (X)",
        "x": "Draw (X)",
        "draw (x)": "Draw (X)",
        "over": "Over 2.5",
        "over 2.5": "Over 2.5",
        "under": "Under 2.5",
        "under 2.5": "Under 2.5",
    }

    return aliases.get(m.lower(), m or "unknown")


def odds_band(odds: Any) -> str:
    o = sf(odds, 0.0)

    if o <= 0:
        return "unknown"
    if o < 1.70:
        return "1.00-1.69"
    if o < 1.90:
        return "1.70-1.89"
    if o < 2.10:
        return "1.90-2.09"
    if o < 2.40:
        return "2.10-2.39"
    return "2.40+"


def confidence_band(value: Any) -> str:
    c = sf(value, 0.0)

    if c <= 0:
        return "unknown"

    # handles both 0.82 and 82
    if c <= 1.0:
        c *= 100.0

    if c < 70:
        return "<70"
    if c < 75:
        return "70-74"
    if c < 80:
        return "75-79"
    if c < 85:
        return "80-84"
    if c < 90:
        return "85-89"
    return "90+"


def value_band(value_pct: Any) -> str:
    v = sf(value_pct, 0.0)

    if v < 0:
        return "<0"
    if v < 3:
        return "0-2.9"
    if v < 6:
        return "3-5.9"
    if v < 10:
        return "6-9.9"
    if v < 15:
        return "10-14.9"
    return "15+"


def pick_result(row: Dict[str, Any]) -> Optional[bool]:
    """
    Returns:
    - True for win
    - False for loss
    - None if unknown/void/pending
    """
    for key in ("hit", "won", "success", "is_win"):
        if key in row:
            v = row.get(key)
            if isinstance(v, bool):
                return v
            if str(v).lower() in ("1", "true", "yes", "win", "won", "hit"):
                return True
            if str(v).lower() in ("0", "false", "no", "loss", "lost", "miss"):
                return False

    result = str(
        row.get("result")
        or row.get("status")
        or row.get("outcome")
        or ""
    ).strip().lower()

    if result in ("win", "won", "hit", "green", "success"):
        return True
    if result in ("loss", "lost", "miss", "red", "fail", "failed"):
        return False

    return None


def profit_for_pick(row: Dict[str, Any], won: Optional[bool]) -> float:
    """
    Uses explicit profit if available.
    Otherwise calculates:
    win: stake * (odds - 1)
    loss: -stake
    unknown: 0
    """
    for key in ("profit", "pnl", "net", "return_net"):
        if key in row and row.get(key) is not None:
            return sf(row.get(key), 0.0)

    if won is None:
        return 0.0

    stake = sf(row.get("stake") or row.get("stake_total") or row.get("amount"), 1.0)
    odds = sf(row.get("odds"), 0.0)

    if stake <= 0:
        stake = 1.0

    if won:
        if odds > 1.0:
            return stake * (odds - 1.0)
        return stake

    return -stake


def stake_for_pick(row: Dict[str, Any]) -> float:
    stake = sf(row.get("stake") or row.get("stake_total") or row.get("amount"), 1.0)
    return stake if stake > 0 else 1.0


# ------------------------------------------------------------
# HISTORY EXTRACTION
# ------------------------------------------------------------

PICK_LIST_KEYS = {
    "picks",
    "singles",
    "system_pool",
    "played",
    "settled",
    "selections",
    "core",
    "funbet",
    "superfun",
    "drawbet",
}


def looks_like_pick(obj: Dict[str, Any]) -> bool:
    has_market = "market" in obj
    has_odds = "odds" in obj
    has_match = "match" in obj or "home" in obj or "away" in obj or "fixture_id" in obj
    has_result = any(k in obj for k in ("result", "status", "outcome", "hit", "won", "success", "profit", "pnl"))

    return bool(has_market and (has_odds or has_match) and has_result)


def collect_picks(obj: Any) -> List[Dict[str, Any]]:
    """
    Recursively searches Tuesday history for pick-like dicts.
    This makes the script tolerant to old/new JSON formats.
    """
    found: List[Dict[str, Any]] = []

    if isinstance(obj, dict):
        if looks_like_pick(obj):
            found.append(obj)

        for key, value in obj.items():
            if key in PICK_LIST_KEYS or isinstance(value, (dict, list)):
                found.extend(collect_picks(value))

    elif isinstance(obj, list):
        for item in obj:
            found.extend(collect_picks(item))

    return found


# ------------------------------------------------------------
# AGGREGATION
# ------------------------------------------------------------

def empty_bucket() -> Dict[str, Any]:
    return {
        "picks": 0,
        "settled": 0,
        "wins": 0,
        "losses": 0,
        "stake": 0.0,
        "profit": 0.0,
        "hit_rate": None,
        "roi": None,
    }


def add_to_bucket(bucket: Dict[str, Any], row: Dict[str, Any]) -> None:
    won = pick_result(row)
    stake = stake_for_pick(row)
    profit = profit_for_pick(row, won)

    bucket["picks"] += 1
    bucket["stake"] += stake
    bucket["profit"] += profit

    if won is True:
        bucket["settled"] += 1
        bucket["wins"] += 1
    elif won is False:
        bucket["settled"] += 1
        bucket["losses"] += 1


def finalize_bucket(bucket: Dict[str, Any]) -> Dict[str, Any]:
    settled = si(bucket.get("settled"))
    wins = si(bucket.get("wins"))
    stake = sf(bucket.get("stake"))
    profit = sf(bucket.get("profit"))

    out = dict(bucket)
    out["stake"] = round(stake, 2)
    out["profit"] = round(profit, 2)

    if settled > 0:
        out["hit_rate"] = round((wins / settled) * 100.0, 2)
    else:
        out["hit_rate"] = None

    if stake > 0:
        out["roi"] = round((profit / stake) * 100.0, 2)
    else:
        out["roi"] = None

    return out


def add_group(groups: Dict[str, Dict[str, Any]], key: str, row: Dict[str, Any]) -> None:
    key = str(key or "unknown")
    if key not in groups:
        groups[key] = empty_bucket()
    add_to_bucket(groups[key], row)


def finalize_groups(groups: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {
        key: finalize_bucket(value)
        for key, value in sorted(groups.items(), key=lambda x: x[0])
    }


def build_memory(history: Any) -> Dict[str, Any]:
    picks = collect_picks(history)

    markets: Dict[str, Dict[str, Any]] = {}
    leagues: Dict[str, Dict[str, Any]] = {}
    odds_bands: Dict[str, Dict[str, Any]] = {}
    confidence_bands: Dict[str, Dict[str, Any]] = {}
    value_bands: Dict[str, Dict[str, Any]] = {}
    market_by_league: Dict[str, Dict[str, Any]] = {}
    market_by_odds_band: Dict[str, Dict[str, Any]] = {}
    market_by_confidence_band: Dict[str, Dict[str, Any]] = {}

    for row in picks:
        market = norm_market(row.get("market"))
        league = str(row.get("league") or "unknown")
        ob = odds_band(row.get("odds"))
        cb = confidence_band(row.get("confirmation_score") or row.get("confidence") or row.get("prob"))
        vb = value_band(row.get("value_pct"))

        add_group(markets, market, row)
        add_group(leagues, league, row)
        add_group(odds_bands, ob, row)
        add_group(confidence_bands, cb, row)
        add_group(value_bands, vb, row)
        add_group(market_by_league, f"{market} | {league}", row)
        add_group(market_by_odds_band, f"{market} | {ob}", row)
        add_group(market_by_confidence_band, f"{market} | {cb}", row)

    return {
        "title": "Bombay Market Performance Memory V1",
        "version": "learning_memory_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "history_path": HISTORY_PATH,
            "picks_detected": len(picks),
        },
        "summary": finalize_bucket({
            **empty_bucket(),
            "picks": 0,
            "settled": 0,
            "wins": 0,
            "losses": 0,
            "stake": 0.0,
            "profit": 0.0,
        }),
        "markets": finalize_groups(markets),
        "leagues": finalize_groups(leagues),
        "odds_bands": finalize_groups(odds_bands),
        "confidence_bands": finalize_groups(confidence_bands),
        "value_bands": finalize_groups(value_bands),
        "market_by_league": finalize_groups(market_by_league),
        "market_by_odds_band": finalize_groups(market_by_odds_band),
        "market_by_confidence_band": finalize_groups(market_by_confidence_band),
    }


def patch_summary(memory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Builds correct global summary from all market buckets.
    """
    total = empty_bucket()

    for bucket in memory.get("markets", {}).values():
        total["picks"] += si(bucket.get("picks"))
        total["settled"] += si(bucket.get("settled"))
        total["wins"] += si(bucket.get("wins"))
        total["losses"] += si(bucket.get("losses"))
        total["stake"] += sf(bucket.get("stake"))
        total["profit"] += sf(bucket.get("profit"))

    memory["summary"] = finalize_bucket(total)
    return memory


def main() -> None:
    if not os.path.exists(HISTORY_PATH):
        out = {
            "title": "Bombay Market Performance Memory V1",
            "version": "learning_memory_v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "error",
            "message": "history_file_missing",
            "history_path": HISTORY_PATH,
        }
        atomic_write_json(OUTPUT_PATH, out)
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    history = load_json(HISTORY_PATH)
    memory = build_memory(history)
    memory = patch_summary(memory)
    memory["status"] = "ok"

    atomic_write_json(OUTPUT_PATH, memory)

    print(json.dumps({
        "status": "ok",
        "output_path": OUTPUT_PATH,
        "history_path": HISTORY_PATH,
        "picks_detected": memory["source"]["picks_detected"],
        "summary": memory["summary"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
