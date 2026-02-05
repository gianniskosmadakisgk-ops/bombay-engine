# src/analysis/friday_shortlist_v3.py
"""
Friday shortlist v3 — FINAL LOCK (Stake v2 + looser Draw + Fun pulls from Core)

Changes locked:
- Core stake rule (simple):
    odds <= 1.80 => 40
    odds >  1.80 => 30
  (so: up to 2.00 it can be 40 or 30; above 2.00 stays 30)
- NO scaling / NO exposure cap modifications.
- DrawBet loosened so it can actually produce draws sometimes.
- FunBet pool = Fun candidates + Core candidates (dedup), and Fun min_odds lowered.

Still locked:
- Core singles max 7.
- Core double only if >=2 Core singles with odds <= 1.60, only those 2 legs.
- Double stake fixed 25.
- DrawBet is Draw-only; if none => open 0 and empty pool.
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
    return None

def _fmt_date(date_str: str) -> str:
    return str(date_str or "")

def _fmt_time(time_str: str) -> str:
    return str(time_str or "")

def _market_defs() -> List[Dict[str, str]]:
    return [
        {"label": "Home (1)", "odds": "offered_1", "prob": "home_prob", "value": "value_pct_1"},
        {"label": "Draw",     "odds": "offered_x", "prob": "draw_prob", "value": "value_pct_x"},
        {"label": "Away (2)", "odds": "offered_2", "prob": "away_prob", "value": "value_pct_2"},
        {"label": "Over 2.5", "odds": "offered_over_2_5",  "prob": "over_2_5_prob",  "value": "value_pct_over"},
        {"label": "Under 2.5","odds": "offered_under_2_5", "prob": "under_2_5_prob", "value": "value_pct_under"},
    ]


# -------------------------
# STAKES (LOCKED)
# -------------------------

CORE_STAKE_40_MAX_ODDS = float(os.getenv("CORE_STAKE_40_MAX_ODDS", "1.80"))
CORE_STAKE_40 = int(os.getenv("CORE_STAKE_40", "40"))
CORE_STAKE_30 = int(os.getenv("CORE_STAKE_30", "30"))

DOUBLE_STAKE = int(os.getenv("CORE_DOUBLE_STAKE", "25"))

def _core_stake_from_odds(odds: float) -> int:
    o = float(odds)
    return CORE_STAKE_40 if o <= CORE_STAKE_40_MAX_ODDS else CORE_STAKE_30


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

    rank_score = (value_pct * 1.0) + ((prob - 0.50) * 100.0 * 0.7)

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
        "rank_score": float(rank_score),
        "quality": round(float(qr.score), 3),
        "quality_reasons": list(qr.reasons),
        "raw": fx,
    }


def _load_latest_thursday_report() -> Dict[str, Any]:
    p1 = LOGS_DIR / "thursday_report_v3.json"
    if p1.exists():
        return _read_json(p1)

    th_files = sorted(LOGS_DIR.glob("thursday_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if th_files:
        return _read_json(th_files[0])

    raise FileNotFoundError(f"Could not find thursday_report_v3.json or any thursday_*.json inside {LOGS_DIR}")


def _load_tuesday_history() -> Tuple[Optional[Dict[str, Any]], str]:
    p = LOGS_DIR / "tuesday_history_v3.json"
    if p.exists():
        return _read_json(p), str(p)

    files = sorted(LOGS_DIR.glob("tuesday_history*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    if files:
        return _read_json(files[0]), str(files[0])

    return None, ""


def _bankrolls_from_history_or_env(hist: Optional[Dict[str, Any]]) -> Tuple[float, float, float, int]:
    core_default = float(os.getenv("CORE_BANKROLL_START", "800"))
    fun_default = float(os.getenv("FUN_BANKROLL_START", "400"))
    draw_default = float(os.getenv("DRAW_BANKROLL_START", "300"))

    week_no = 1
    if isinstance(hist, dict):
        wc = hist.get("week_count")
        if isinstance(wc, int) and wc >= 0:
            week_no = wc + 1

        core_cur = fun_cur = draw_cur = None
        if isinstance(hist.get("core"), dict):
            core_cur = hist["core"].get("bankroll_current")
        if isinstance(hist.get("funbet"), dict):
            fun_cur = hist["funbet"].get("bankroll_current")
        if isinstance(hist.get("drawbet"), dict):
            draw_cur = hist["drawbet"].get("bankroll_current")

        if isinstance(core_cur, (int, float)):
            core_default = float(core_cur)
        if isinstance(fun_cur, (int, float)):
            fun_default = float(fun_cur)
        if isinstance(draw_cur, (int, float)):
            draw_default = float(draw_cur)

    return core_default, fun_default, draw_default, week_no


# -------------------------
# Selection logic
# -------------------------

def _select_core(cands: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], float]:
    core = [c for c in cands if c["market"] != "Draw"]

    # Keep your sane band
    core_min_odds = float(os.getenv("CORE_MIN_ODDS", "1.55"))
    core_max_odds = float(os.getenv("CORE_MAX_ODDS", "2.20"))
    core_min_prob = float(os.getenv("CORE_MIN_PROB", "0.50"))
    core_min_value = float(os.getenv("CORE_MIN_VALUE", "1.0"))

    core = [c for c in core if core_min_odds <= c["odds"] <= core_max_odds and c["prob"] >= core_min_prob and c["value_pct"] >= core_min_value]
    core.sort(key=lambda x: x["rank_score"], reverse=True)
    core = core[:7]

    singles: List[Dict[str, Any]] = []
    open_total = 0.0

    for c in core:
        stake = _core_stake_from_odds(float(c["odds"]))
        singles.append({
            "date": c["date"],
            "time_gr": c["time_gr"],
            "league": c["league"],
            "match": c["match"],
            "market": c["market"],
            "odds": c["odds"],
            "stake": stake,
            "quality": c["quality"],
        })
        open_total += stake

    eligible = [s for s in singles if float(s["odds"]) <= 1.60]
    eligible.sort(key=lambda s: float(s["odds"]))

    core_double = None
    if len(eligible) >= 2:
        a, b = eligible[0], eligible[1]
        combined_odds = round(float(a["odds"]) * float(b["odds"]), 2)
        core_double = {
            "legs": [
                {k: a[k] for k in ["date", "time_gr", "league", "match", "market", "odds"]},
                {k: b[k] for k in ["date", "time_gr", "league", "match", "market", "odds"]},
            ],
            "odds": combined_odds,
            "stake": DOUBLE_STAKE,
            "label": "Double (Core small-odds only)",
        }
        open_total += DOUBLE_STAKE

    return singles, core_double, float(round(open_total, 2))


def _select_system(
    cands: List[Dict[str, Any]],
    *,
    n: int,
    min_odds: float,
    max_odds: float,
    min_prob: float,
    min_value: float,
    stake_total: float,
    allow_draw: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], float]:
    pool = cands[:]
    if not allow_draw:
        pool = [c for c in pool if c["market"] != "Draw"]

    pool = [c for c in pool if min_odds <= c["odds"] <= max_odds and c["prob"] >= min_prob and c["value_pct"] >= min_value]
    pool.sort(key=lambda x: x["rank_score"], reverse=True)
    pool = pool[:n]

    if n == 7:
        system = {"k": 4, "n": 7, "columns": 35, "min_hits": 4, "stake": stake_total}
    elif n == 5:
        system = {"k": 2, "n": 5, "columns": 10, "min_hits": 2, "stake": stake_total}
    else:
        system = {"k": max(1, n // 2), "n": n, "columns": None, "min_hits": max(1, n // 2), "stake": stake_total}

    system_pool = [{
        "date": c["date"],
        "time_gr": c["time_gr"],
        "league": c["league"],
        "match": c["match"],
        "market": c["market"],
        "odds": c["odds"],
        "stake": None,
        "quality": c["quality"],
    } for c in pool]

    open_amt = float(stake_total) if system_pool else 0.0
    return system_pool, system, float(round(open_amt, 2))


def _dedup_candidates(cands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for c in cands:
        key = (c.get("fixture_id"), c.get("market"))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


# -------------------------
# Main builder
# -------------------------

def build_friday_shortlist() -> Dict[str, Any]:
    th = _load_latest_thursday_report()
    fixtures = th.get("fixtures") or []
    if not isinstance(fixtures, list):
        raise ValueError("Thursday report has no fixtures list.")

    hist, hist_path = _load_tuesday_history()
    core_bankroll, fun_bankroll, draw_bankroll, week_no = _bankrolls_from_history_or_env(hist)

    # Quality thresholds (Draw loosened)
    core_min_q = float(os.getenv("FRIDAY_CORE_MIN_QUALITY", "0.70"))
    fun_min_q  = float(os.getenv("FRIDAY_FUN_MIN_QUALITY", "0.60"))
    draw_min_q = float(os.getenv("FRIDAY_DRAW_MIN_QUALITY", "0.58"))

    mdefs = _market_defs()
    all_cands: List[Dict[str, Any]] = []
    q_by_id: Dict[Any, float] = {}

    for fx in fixtures:
        qr = fixture_quality_score(fx)
        q_by_id[fx.get("fixture_id")] = float(qr.score)

    for fx in fixtures:
        for md in mdefs:
            c = _candidate_from_fixture(fx, md)
            if c:
                all_cands.append(c)

    core_cands = [c for c in all_cands if (q_by_id.get(c["fixture_id"], 0.0) >= core_min_q)]
    fun_cands  = [c for c in all_cands if (q_by_id.get(c["fixture_id"], 0.0) >= fun_min_q)]
    draw_cands = [c for c in all_cands if (q_by_id.get(c["fixture_id"], 0.0) >= draw_min_q)]

    # Core build
    core_singles, core_double, core_open = _select_core(core_cands)

    # FunBet: allow it to pull from Core candidates too
    fun_stake_total = float(os.getenv("FUN_SYSTEM_STAKE", "35"))
    fun_mix = _dedup_candidates(fun_cands + core_cands)

    fun_min_odds = float(os.getenv("FUN_MIN_ODDS", "1.90"))
    fun_max_odds = float(os.getenv("FUN_MAX_ODDS", "3.60"))
    fun_min_prob = float(os.getenv("FUN_MIN_PROB", "0.30"))
    fun_min_value = float(os.getenv("FUN_MIN_VALUE", "1.5"))

    fun_pool, fun_system, fun_open = _select_system(
        fun_mix,
        n=7,
        min_odds=fun_min_odds, max_odds=fun_max_odds,
        min_prob=fun_min_prob, min_value=fun_min_value,
        stake_total=fun_stake_total,
        allow_draw=False,
    )

    # DrawBet: loosened filters
    draw_stake_total = float(os.getenv("DRAW_SYSTEM_STAKE", "25"))
    draw_min_odds = float(os.getenv("DRAW_MIN_ODDS", "2.80"))
    draw_max_odds = float(os.getenv("DRAW_MAX_ODDS", "5.20"))
    draw_min_prob = float(os.getenv("DRAW_MIN_PROB", "0.21"))
    draw_min_value = float(os.getenv("DRAW_MIN_VALUE", "1.0"))

    draw_pool, draw_system, draw_open = _select_system(
        draw_cands,
        n=5,
        min_odds=draw_min_odds, max_odds=draw_max_odds,
        min_prob=draw_min_prob, min_value=draw_min_value,
        stake_total=draw_stake_total,
        allow_draw=True,
    )

    # Strict: Draw-only
    draw_pool = [x for x in draw_pool if x["market"] == "Draw"]
    if not draw_pool:
        draw_open = 0.0

    window = th.get("window") or {}
    if not window:
        window = {
            "start": str(th.get("start_date") or ""),
            "end": str(th.get("end_date") or ""),
            "hours": int(th.get("window_hours") or 72),
        }

    out = {
        "title": f"Bombay Friday — Week {week_no}",
        "generated_at": _utc_now_iso(),
        "window": window,

        "core": {
            "label": "CoreBet",
            "bankroll_start": core_bankroll,
            "max_singles": 7,
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
            "history_loaded": bool(hist_path),
            "history_path": hist_path,
            "thresholds": {"core": core_min_q, "fun": fun_min_q, "draw": draw_min_q},
            "counts": {
                "fixtures": len(fixtures),
                "candidates_total": len(all_cands),
                "core_singles": len(core_singles),
                "fun_pool": len(fun_pool),
                "draw_pool": len(draw_pool),
            },
            "stake_rule_core": {
                "<=1.80": 40,
                ">1.80": 30,
                "double_stake": DOUBLE_STAKE,
                "scaling": "OFF",
            },
            "draw_filters": {
                "min_odds": draw_min_odds,
                "max_odds": draw_max_odds,
                "min_prob": draw_min_prob,
                "min_value": draw_min_value,
            },
            "fun_filters": {
                "min_odds": fun_min_odds,
                "max_odds": fun_max_odds,
                "min_prob": fun_min_prob,
                "min_value": fun_min_value,
                "pool_source": "fun + core (dedup)",
            },
        },
    }
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
            "history_loaded": friday["debug"]["history_loaded"],
            "history_path": friday["debug"]["history_path"],
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
