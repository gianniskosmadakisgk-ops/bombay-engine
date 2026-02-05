# src/analysis/friday_shortlist_v3.py
from __future__ import annotations

import json
import os
import glob
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from quality_gate_v1 import fixture_quality_score, passes_quality_gate


LOGS_DIR = os.getenv("BOMBAY_LOGS_DIR", os.path.join(os.path.dirname(__file__), "..", "logs"))
LOGS_DIR = os.path.abspath(LOGS_DIR)

THURSDAY_REPORT_BASENAME = "thursday_report_v3.json"
FRIDAY_OUT_BASENAME = "friday_shortlist_v3.json"


# --- Config (ταιριάζει με αυτά που μου είπες) ---

CORE_MAX_TICKETS_TOTAL = 7  # max 7 στοιχήματα σύνολο (singles + (optional) double ticket)

# Double rule: μόνο “μεταξύ τους”, και μόνο 2 μικρά <= 1.60. Αλλιώς καθόλου.
CORE_DOUBLE_MAX_ODDS = 1.60

# System configs (όπως στα screenshots)
FUN_SYSTEM = {"k": 4, "n": 7, "columns": 35, "min_hits": 4, "stake_total": 35}
DRAW_SYSTEM = {"k": 2, "n": 5, "columns": 10, "min_hits": 2, "stake_total": 25}

# Simple stake tiers for Core singles (αν ήδη έχεις staking logic αλλού, το αφήνεις όπως είναι)
CORE_STAKE_HIGH = 40
CORE_STAKE_MID = 30
CORE_STAKE_LOW = 20


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _load_latest_thursday_report() -> Dict[str, Any]:
    """
    Prefer logs/thursday_report_v3.json.
    Fallback: latest logs/thursday_*.json
    """
    p1 = os.path.join(LOGS_DIR, THURSDAY_REPORT_BASENAME)
    if os.path.exists(p1):
        with open(p1, "r", encoding="utf-8") as f:
            return json.load(f)

    # fallback: latest thursday_*.json
    candidates = sorted(
        glob.glob(os.path.join(LOGS_DIR, "thursday_*.json")),
        key=lambda p: os.path.getmtime(p),
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"Could not find {p1} or any thursday_*.json inside {LOGS_DIR}")
    with open(candidates[0], "r", encoding="utf-8") as f:
        return json.load(f)


def _match_str(fx: Dict[str, Any]) -> str:
    home = fx.get("home", "")
    away = fx.get("away", "")
    return f"{home} – {away}"


def _pick_market_for_core(fx: Dict[str, Any]) -> Optional[Tuple[str, float, float]]:
    """
    Returns (market, offered_odds, model_score) or None.

    Core: παίρνουμε το καλύτερο μεταξύ Over/Under/1/X/2 με θετικό EV,
    αλλά πρακτικά στο dataset σου έχεις ήδη:
      - score_over
      - score_draw
      - ev_over / ev_under / ev_1 / ev_x / ev_2
    """
    # Build candidates from known fields (keep it robust)
    cand = []

    # Over 2.5
    if fx.get("offered_over_2_5") is not None and fx.get("ev_over") is not None:
        cand.append(("Over 2.5", float(fx["offered_over_2_5"]), float(fx["ev_over"])))

    # Under 2.5
    if fx.get("offered_under_2_5") is not None and fx.get("ev_under") is not None:
        cand.append(("Under 2.5", float(fx["offered_under_2_5"]), float(fx["ev_under"])))

    # 1X2
    if fx.get("offered_1") is not None and fx.get("ev_1") is not None:
        cand.append(("Home (1)", float(fx["offered_1"]), float(fx["ev_1"])))
    if fx.get("offered_x") is not None and fx.get("ev_x") is not None:
        cand.append(("Draw", float(fx["offered_x"]), float(fx["ev_x"])))
    if fx.get("offered_2") is not None and fx.get("ev_2") is not None:
        cand.append(("Away (2)", float(fx["offered_2"]), float(fx["ev_2"])))

    # Keep only positive EV
    cand = [c for c in cand if c[2] > 0]

    if not cand:
        return None

    # Prefer higher EV, tie-breaker: higher quality
    q = fixture_quality_score(fx)
    cand.sort(key=lambda x: (x[2], q), reverse=True)
    return cand[0]


def _core_stake_from_confidence(fx: Dict[str, Any]) -> int:
    flags = fx.get("flags", {}) or {}
    c = float(flags.get("confidence", 0.0) or 0.0)
    if c >= 0.67:
        return CORE_STAKE_HIGH
    if c >= 0.55:
        return CORE_STAKE_MID
    return CORE_STAKE_LOW


def _pick_core_singles(fixtures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build Core singles list (pre-match).
    Keep it tight: max tickets total enforced later (because double may exist).
    """
    rows = []
    for fx in fixtures:
        if not passes_quality_gate(fx, "core"):
            continue

        pick = _pick_market_for_core(fx)
        if not pick:
            continue

        market, odds, ev = pick
        rows.append(
            {
                "fixture_id": fx.get("fixture_id"),
                "date": fx.get("date"),
                "time_gr": fx.get("time"),
                "league": fx.get("league"),
                "match": _match_str(fx),
                "market": market,
                "odds": round(float(odds), 2),
                "stake": _core_stake_from_confidence(fx),
                "quality": round(fixture_quality_score(fx), 3),
                "ev": round(float(ev), 4),
            }
        )

    # Rank by EV then quality
    rows.sort(key=lambda r: (r["ev"], r["quality"]), reverse=True)

    # Leave room for optional double (we will cap later)
    # Here we just take top 7 for now; final cap is enforced after double decision.
    return rows[:CORE_MAX_TICKETS_TOTAL]


def _build_core_double(core_singles: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Double only if we have 2 singles with odds <= 1.60.
    Otherwise None.
    """
    small = [s for s in core_singles if float(s["odds"]) <= CORE_DOUBLE_MAX_ODDS]
    if len(small) < 2:
        return None

    a, b = small[0], small[1]
    comb_odds = float(a["odds"]) * float(b["odds"])

    return {
        "type": "Double",
        "legs": [
            {"match": a["match"], "market": a["market"], "odds": a["odds"]},
            {"match": b["match"], "market": b["market"], "odds": b["odds"]},
        ],
        "odds": round(comb_odds, 2),
        # stake: if you already have a rule, change here. Keeping conservative.
        "stake": min(20, int(a["stake"])),
    }


def _pick_fun_pool(fixtures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Fun: higher odds / spicy value, but still filtered by quality gate (0.60).
    We’ll pick 7 candidates for the 4/7 system pool.
    """
    rows = []
    for fx in fixtures:
        if not passes_quality_gate(fx, "fun"):
            continue

        # Fun tries to use best positive EV among 1/X/2/Over/Under,
        # but doesn't care as much about "boring" lines.
        pick = _pick_market_for_core(fx)  # reuse, but it's EV-based
        if not pick:
            continue
        market, odds, ev = pick

        # Fun wants bigger prices: keep odds >= 2.00 ideally
        if float(odds) < 2.0:
            continue

        rows.append(
            {
                "fixture_id": fx.get("fixture_id"),
                "date": fx.get("date"),
                "time_gr": fx.get("time"),
                "league": fx.get("league"),
                "match": _match_str(fx),
                "market": market,
                "odds": round(float(odds), 2),
                "quality": round(fixture_quality_score(fx), 3),
                "ev": round(float(ev), 4),
            }
        )

    rows.sort(key=lambda r: (r["ev"], r["quality"], r["odds"]), reverse=True)
    return rows[:FUN_SYSTEM["n"]]


def _pick_draw_pool(fixtures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Draw: αυστηρό. Αν δεν έχει, δεν παίζει.
    Quality gate 0.70 + draw_shape/ draw friendly / tight_game help.
    Pick 5 for 2/5 system pool.
    """
    rows = []
    for fx in fixtures:
        if not passes_quality_gate(fx, "draw"):
            continue

        flags = fx.get("flags", {}) or {}
        if not (flags.get("tight_game", False) and flags.get("draw_shape", False)):
            continue

        offered_x = fx.get("offered_x")
        ev_x = fx.get("ev_x")
        if offered_x is None or ev_x is None:
            continue
        if float(ev_x) <= 0:
            continue

        rows.append(
            {
                "fixture_id": fx.get("fixture_id"),
                "date": fx.get("date"),
                "time_gr": fx.get("time"),
                "league": fx.get("league"),
                "match": _match_str(fx),
                "market": "Draw",
                "odds": round(float(offered_x), 2),
                "quality": round(fixture_quality_score(fx), 3),
                "ev": round(float(ev_x), 4),
            }
        )

    rows.sort(key=lambda r: (r["ev"], r["quality"]), reverse=True)
    return rows[:DRAW_SYSTEM["n"]]


def main() -> int:
    th = _load_latest_thursday_report()
    fixtures = th.get("fixtures", []) or []

    core_singles = _pick_core_singles(fixtures)
    core_double = _build_core_double(core_singles)

    # Enforce max 7 tickets total for Core
    # If we have a double ticket, keep at most 6 singles.
    if core_double is not None:
        core_singles = core_singles[: max(0, CORE_MAX_TICKETS_TOTAL - 1)]
    else:
        core_singles = core_singles[:CORE_MAX_TICKETS_TOTAL]

    fun_pool = _pick_fun_pool(fixtures)
    draw_pool = _pick_draw_pool(fixtures)

    # Draw strict: if not enough, play none
    if len(draw_pool) < DRAW_SYSTEM["n"]:
        draw_pool = []

    out = {
        "generated_at": _now_iso(),
        "window": th.get("window"),
        "fixtures_total": th.get("fixtures_total", len(fixtures)),
        "quality_gate": {
            "core_min": 0.70,
            "fun_min": 0.60,
            "draw_min": 0.70,
        },
        "corebet": {
            "singles": core_singles,
            "double": core_double,
            "tickets_total": len(core_singles) + (1 if core_double else 0),
            "max_tickets_total": CORE_MAX_TICKETS_TOTAL,
        },
        "funbet": {
            "pool": fun_pool,
            "system": FUN_SYSTEM,
        },
        "drawbet": {
            "pool": draw_pool,
            "system": DRAW_SYSTEM if draw_pool else None,
        },
    }

    os.makedirs(LOGS_DIR, exist_ok=True)
    out_path = os.path.join(LOGS_DIR, FRIDAY_OUT_BASENAME)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
