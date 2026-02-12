# src/analysis/friday_shortlist_v3.py
"""
Friday shortlist v3 — NO SCALING + TOP-5 DRAWS + chronological ordering + fixture_id everywhere.

Guarantees:
- Reads Thursday schema correctly:
  probs: home_prob/draw_prob/away_prob/over_2_5_prob/under_2_5_prob
  odds:  offered_1/offered_x/offered_2/offered_over_2_5/offered_under_2_5
  value: value_pct_1/value_pct_x/value_pct_2/value_pct_over/value_pct_under
- No stake scaling. Ever.
- Core stake rule: odds <= 2.00 -> 40, else -> 30
- DrawBet: Top-5 Draws by probability (needs draw odds+prob)
- Outputs sorted chronologically (date, time_gr)
- Writes:
  - logs/friday_shortlist_v3.json
  - logs/friday_YYYY-MM-DD_HH-MM-SS.json
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -------------------------
# Path / root robustness
# -------------------------

def _find_project_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / "logs").is_dir() and (p / "src").is_dir():
            return p
    return start.parents[2] if len(start.parents) >= 3 else start


_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _find_project_root(_THIS_FILE.parent)
LOGS_DIR = PROJECT_ROOT / "logs"

for candidate in [PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "src" / "src"]:
    if candidate.is_dir():
        s = str(candidate)
        if s not in sys.path:
            sys.path.insert(0, s)


# -------------------------
# Quality gate import (NO __init__.py needed)
# -------------------------

@dataclass
class _QualityResult:
    score: float
    reasons: Tuple[str, ...]


def _load_quality_gate() -> Any:
    q_path = Path(__file__).resolve().parent / "quality_gate_v1.py"
    if not q_path.exists():
        return None
    import importlib.util
    spec = importlib.util.spec_from_file_location("quality_gate_v1", str(q_path))
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


_QMOD = _load_quality_gate()


def fixture_quality_score(fixture: Dict[str, Any]) -> _QualityResult:
    if _QMOD and hasattr(_QMOD, "fixture_quality_score"):
        try:
            qr = _QMOD.fixture_quality_score(fixture)
            score = float(getattr(qr, "score", 0.0))
            reasons = tuple(getattr(qr, "reasons", ()))
            return _QualityResult(score=score, reasons=reasons)
        except Exception:
            return _QualityResult(score=0.0, reasons=("quality_gate_error",))
    return _QualityResult(score=1.0, reasons=("quality_gate_missing",))


# -------------------------
# Helpers
# -------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _safe_float(v: Any) -> Optional[float]:
    if isinstance(v, (int, float)):
        return float(v)
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None

def _fmt_date(date_str: str) -> str:
    return str(date_str or "")

def _fmt_time(time_str: str) -> str:
    return str(time_str or "")

def _chrono_key(item: Dict[str, Any]) -> Tuple[str, str]:
    return (str(item.get("date") or ""), str(item.get("time_gr") or ""))

def _market_defs() -> List[Dict[str, str]]:
    return [
        {"label": "Home (1)", "odds": "offered_1", "prob": "home_prob", "value": "value_pct_1"},
        {"label": "Draw",     "odds": "offered_x", "prob": "draw_prob", "value": "value_pct_x"},
        {"label": "Away (2)", "odds": "offered_2", "prob": "away_prob", "value": "value_pct_2"},
        {"label": "Over 2.5", "odds": "offered_over_2_5",  "prob": "over_2_5_prob",  "value": "value_pct_over"},
        {"label": "Under 2.5","odds": "offered_under_2_5", "prob": "under_2_5_prob", "value": "value_pct_under"},
    ]

def _prob_points(prob: float) -> float:
    # convert prob in [0..1] into centered points scale
    return (float(prob) - 0.50) * 100.0

def _rank_score(value_pct: float, prob: float) -> float:
    # EV-first, with stability from prob
    return 0.60 * float(value_pct) + 0.40 * _prob_points(float(prob))

def _candidate_from_fixture(fx: Dict[str, Any], md: Dict[str, str]) -> Optional[Dict[str, Any]]:
    odds = _safe_float(fx.get(md["odds"]))
    prob = _safe_float(fx.get(md["prob"]))
    value_pct = _safe_float(fx.get(md["value"])) or 0.0

    if odds is None or prob is None:
        return None

    league = str(fx.get("league") or "")
    home = str(fx.get("home") or "")
    away = str(fx.get("away") or "")
    fixture_id = fx.get("fixture_id")

    date = _fmt_date(str(fx.get("date") or ""))
    time_gr = _fmt_time(str(fx.get("time") or ""))

    qr = fixture_quality_score(fx)

    return {
        "fixture_id": fixture_id,
        "date": date,
        "time_gr": time_gr,
        "league": league,
        "match": f"{home} – {away}",
        "market": md["label"],
        "odds": round(float(odds), 2),
        "prob": round(float(prob), 4),
        "value_pct": round(float(value_pct), 2),
        "rank_score": float(_rank_score(float(value_pct), float(prob))),
        "quality": round(float(qr.score), 3),
        "quality_reasons": list(qr.reasons),
    }

def _load_latest_thursday_report() -> Dict[str, Any]:
    p1 = LOGS_DIR / "thursday_report_v3.json"
    if p1.exists():
        return _read_json(p1)

    th_files = sorted(LOGS_DIR.glob("thursday_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if th_files:
        return _read_json(th_files[0])

    raise FileNotFoundError(f"Could not find thursday_report_v3.json or any thursday_*.json inside {LOGS_DIR}")


# -------------------------
# Stake rules (NO scaling)
# -------------------------

def core_stake_from_odds(odds: float) -> int:
    """
    Deterministic stake rule (NO scaling).
      - odds <= 2.00 -> 40
      - odds >  2.00 -> 30
    """
    return 40 if float(odds) <= 2.00 else 30


# -------------------------
# Selection logic helpers
# -------------------------

def _is_total_market(market: str) -> bool:
    return market in ("Over 2.5", "Under 2.5")

def _is_under(market: str) -> bool:
    return market == "Under 2.5"


# -------------------------
# Selection logic
# -------------------------

def _select_core(
    cands: List[Dict[str, Any]],
    strict_ok_by_id: Dict[Any, bool],
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], float]:
    """
    Core rules:
    - No Draw by default
    - Basic filters: odds window / min prob / min value_pct
    - STRICT odds gate (default ON via env)
    - Correlation caps:
        * totals (over/under) max = 4
        * under max = 1
        * prefer over vs under
    """
    require_strict = os.getenv("FRIDAY_CORE_REQUIRE_STRICT_OK", "true").lower() == "true"
    max_totals = int(os.getenv("FRIDAY_CORE_MAX_TOTALS", "4"))
    max_unders = int(os.getenv("FRIDAY_CORE_MAX_UNDERS", "1"))

    core = [c for c in cands if c["market"] != "Draw"]
    core = [c for c in core if 1.55 <= float(c["odds"]) <= 2.20 and float(c["prob"]) >= 0.50 and float(c["value_pct"]) >= 1.0]

    if require_strict:
        core = [c for c in core if bool(strict_ok_by_id.get(c.get("fixture_id"), False))]

    # Rank EV-first
    core.sort(key=lambda x: float(x["rank_score"]), reverse=True)

    singles: List[Dict[str, Any]] = []
    open_total = 0.0
    totals_cnt = 0
    unders_cnt = 0

    for c in core:
        # caps
        if _is_total_market(c["market"]):
            if totals_cnt >= max_totals:
                continue
            if _is_under(c["market"]) and unders_cnt >= max_unders:
                continue

        stake = core_stake_from_odds(float(c["odds"]))
        singles.append({
            "fixture_id": c.get("fixture_id"),
            "date": c["date"],
            "time_gr": c["time_gr"],
            "league": c["league"],
            "match": c["match"],
            "market": c["market"],
            "odds": c["odds"],
            "prob": c["prob"],
            "stake": stake,
            "quality": c["quality"],
        })
        open_total += stake

        if _is_total_market(c["market"]):
            totals_cnt += 1
            if _is_under(c["market"]):
                unders_cnt += 1

        if len(singles) == 7:
            break

    # Optional: Core double small-odds rule (kept as-is)
    eligible = [s for s in singles if float(s["odds"]) <= 1.60]
    eligible.sort(key=lambda s: float(s["odds"]))

    core_double = None
    if len(eligible) >= 2:
        a, b = eligible[0], eligible[1]
        combined_odds = round(float(a["odds"]) * float(b["odds"]), 2)
        core_double = {
            "legs": [
                {k: a.get(k) for k in ["fixture_id", "date", "time_gr", "league", "match", "market", "odds", "prob"]},
                {k: b.get(k) for k in ["fixture_id", "date", "time_gr", "league", "match", "market", "odds", "prob"]},
            ],
            "odds": combined_odds,
            "stake": 20,
            "label": "Double (Core small-odds only)",
        }
        open_total += 20

    singles.sort(key=_chrono_key)
    return singles, core_double, float(round(open_total, 2))

def _select_fun_system(cands: List[Dict[str, Any]], stake_total: float) -> Tuple[List[Dict[str, Any]], Dict[str, Any], float]:
    pool = [c for c in cands if c["market"] != "Draw"]
    pool = [c for c in pool if 1.90 <= float(c["odds"]) <= 3.60 and float(c["prob"]) >= 0.30 and float(c["value_pct"]) >= 1.0]
    pool.sort(key=lambda x: float(x["rank_score"]), reverse=True)
    pool = pool[:7]

    system = {"k": 4, "n": 7, "columns": 35, "min_hits": 4, "stake": float(stake_total)}

    system_pool = [{
        "fixture_id": c.get("fixture_id"),
        "date": c["date"],
        "time_gr": c["time_gr"],
        "league": c["league"],
        "match": c["match"],
        "market": c["market"],
        "odds": c["odds"],
        "prob": c["prob"],
        "quality": c["quality"],
    } for c in pool]

    system_pool.sort(key=_chrono_key)
    open_amt = float(stake_total) if system_pool else 0.0
    return system_pool, system, float(round(open_amt, 2))

def _select_draw_top5(all_cands: List[Dict[str, Any]], stake_total: float) -> Tuple[List[Dict[str, Any]], Dict[str, Any], float]:
    draws = [c for c in all_cands if c["market"] == "Draw" and c.get("prob") is not None and c.get("odds") is not None]
    draws = [c for c in draws if 3.00 <= float(c["odds"]) <= 5.50]
    draws.sort(key=lambda x: (float(x["prob"]), float(x["odds"])), reverse=True)
    draws = draws[:5]

    system = {"k": 2, "n": 5, "columns": 10, "min_hits": 2, "stake": float(stake_total), "mode": "top5_by_probability"}

    draw_pool = [{
        "fixture_id": c.get("fixture_id"),
        "date": c["date"],
        "time_gr": c["time_gr"],
        "league": c["league"],
        "match": c["match"],
        "market": "Draw",
        "odds": c["odds"],
        "prob": c["prob"],
        "quality": c["quality"],
    } for c in draws]

    draw_pool.sort(key=_chrono_key)
    open_amt = float(stake_total) if draw_pool else 0.0
    return draw_pool, system, float(round(open_amt, 2))


# -------------------------
# Sanity checks (fail fast)
# -------------------------

def _assert_friday_sanity(friday: Dict[str, Any]) -> None:
    def _all_lines() -> List[Dict[str, Any]]:
        return (
            (friday.get("core", {}) or {}).get("singles", []) +
            (friday.get("funbet", {}) or {}).get("system_pool", []) +
            (friday.get("drawbet", {}) or {}).get("system_pool", [])
        )

    # fixture_id everywhere
    for ln in _all_lines():
        fid = ln.get("fixture_id")
        if not isinstance(fid, int):
            raise ValueError(f"Missing/invalid fixture_id in line: {ln}")

    # chronological ordering inside each list
    for path in [("core", "singles"), ("funbet", "system_pool"), ("drawbet", "system_pool")]:
        lst = (friday.get(path[0], {}) or {}).get(path[1], [])
        if lst != sorted(lst, key=_chrono_key):
            raise ValueError(f"List not chronological: {path}")

    # stake rule check (Core)
    for ln in (friday.get("core", {}) or {}).get("singles", []):
        odds = float(ln["odds"])
        expected = 40 if odds <= 2.00 else 30
        if int(ln["stake"]) != expected:
            raise ValueError(f"Stake rule broken: odds={odds} stake={ln['stake']} expected={expected}")

    # draw pool not empty if draw candidates exist (best effort)
    dbg = friday.get("debug", {}) or {}
    draw_possible = bool((dbg.get("counts", {}) or {}).get("draw_candidates_possible", 0))
    draw_pool_n = len((friday.get("drawbet", {}) or {}).get("system_pool", []) or [])
    if draw_possible and draw_pool_n == 0:
        raise ValueError("DrawBet is empty but draw candidates exist.")


# -------------------------
# Main builder
# -------------------------

def build_friday_shortlist() -> Dict[str, Any]:
    th = _load_latest_thursday_report()
    fixtures = th.get("fixtures") or []
    if not isinstance(fixtures, list):
        raise ValueError("Thursday report has no fixtures list.")

    core_min_q = float(os.getenv("FRIDAY_CORE_MIN_QUALITY", "0.70"))
    fun_min_q  = float(os.getenv("FRIDAY_FUN_MIN_QUALITY", "0.60"))

    core_bankroll = float(os.getenv("CORE_BANKROLL_START", "800"))
    fun_bankroll  = float(os.getenv("FUN_BANKROLL_START", "400"))
    draw_bankroll = float(os.getenv("DRAW_BANKROLL_START", "300"))

    fun_stake_total  = float(os.getenv("FUN_SYSTEM_STAKE", "35"))
    draw_stake_total = float(os.getenv("DRAW_SYSTEM_STAKE", "25"))

    all_cands: List[Dict[str, Any]] = []
    q_by_id: Dict[Any, float] = {}
    strict_ok_by_id: Dict[Any, bool] = {}

    # per-fixture signals
    for fx in fixtures:
        fid = fx.get("fixture_id")
        qr = fixture_quality_score(fx)
        q_by_id[fid] = float(qr.score)

        flags = (fx.get("flags") or {})
        strict_ok_by_id[fid] = bool(flags.get("odds_strict_ok") is True)

    # build candidates
    for fx in fixtures:
        for md in _market_defs():
            c = _candidate_from_fixture(fx, md)
            if c:
                all_cands.append(c)

    # if any fixture_id missing in Thursday -> fail fast (should never happen)
    if any((c.get("fixture_id") is None) for c in all_cands):
        bad = [c for c in all_cands if c.get("fixture_id") is None][:3]
        raise ValueError(f"Some candidates have missing fixture_id. Examples: {bad}")

    core_cands = [c for c in all_cands if (q_by_id.get(c["fixture_id"], 0.0) >= core_min_q)]
    fun_cands  = [c for c in all_cands if (q_by_id.get(c["fixture_id"], 0.0) >= fun_min_q)]

    core_singles, core_double, core_open = _select_core(core_cands, strict_ok_by_id)
    fun_pool, fun_system, fun_open = _select_fun_system(fun_cands, fun_stake_total)
    draw_pool, draw_system, draw_open = _select_draw_top5(all_cands, draw_stake_total)

    # draw candidates possible (for sanity)
    draw_candidates_possible = 0
    for c in all_cands:
        if c["market"] == "Draw":
            if c.get("odds") is not None and 3.00 <= float(c["odds"]) <= 5.50 and c.get("prob") is not None:
                draw_candidates_possible += 1

    window = th.get("window") or {
        "from": str(th.get("start_date") or ""),
        "to": str(th.get("end_date") or ""),
        "hours": int(th.get("window_hours") or 72),
    }

    out = {
        "title": "Bombay Friday — Week",
        "generated_at": _utc_now_iso(),
        "window": window,

        "core": {
            "label": "CoreBet",
            "bankroll_start": core_bankroll,
            "max_singles": 7,
            "stake_rule": "odds<=2.00 -> 40, else -> 30 (NO scaling)",
            "rules": {
                "require_strict_ok": (os.getenv("FRIDAY_CORE_REQUIRE_STRICT_OK", "true").lower() == "true"),
                "max_totals": int(os.getenv("FRIDAY_CORE_MAX_TOTALS", "4")),
                "max_unders": int(os.getenv("FRIDAY_CORE_MAX_UNDERS", "1")),
                "totals_bias": "Over preferred, Under capped",
            },
            "singles": core_singles,
            "double": core_double,
            "doubles": ([core_double] if core_double else []),
            "open": round(core_open, 2),
            "after_open": round(core_bankroll - core_open, 2),
        },

        "funbet": {
            "label": "FunBet",
            "bankroll_start": fun_bankroll,
            "system_pool": fun_pool,
            "system": fun_system,
            "open": round(fun_open, 2),
            "after_open": round(fun_bankroll - fun_open, 2),
        },

        "drawbet": {
            "label": "DrawBet",
            "bankroll_start": draw_bankroll,
            "system_pool": draw_pool,
            "system": draw_system,
            "open": round(draw_open, 2),
            "after_open": round(draw_bankroll - draw_open, 2),
        },

        "debug": {
            "project_root": str(PROJECT_ROOT),
            "logs_dir": str(LOGS_DIR),
            "thresholds": {"core": core_min_q, "fun": fun_min_q},
            "counts": {
                "fixtures": len(fixtures),
                "candidates_total": len(all_cands),
                "core_singles": len(core_singles),
                "fun_pool": len(fun_pool),
                "draw_pool": len(draw_pool),
                "draw_candidates_possible": draw_candidates_possible,
            },
        },
    }

    # fail fast sanity
    _assert_friday_sanity(out)
    return out


def main() -> int:
    try:
        friday = build_friday_shortlist()

        _write_json(LOGS_DIR / "friday_shortlist_v3.json", friday)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        _write_json(LOGS_DIR / f"friday_{ts}.json", friday)

        print(json.dumps({
            "status": "ok",
            "saved": str(LOGS_DIR / "friday_shortlist_v3.json"),
            "generated_at": friday["generated_at"],
            "counts": friday["debug"]["counts"],
            "logs_dir": str(LOGS_DIR),
        }))
        return 0

    except Exception as e:
        print(json.dumps({
            "status": "error",
            "error": str(e),
            "logs_dir": str(LOGS_DIR),
        }))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
