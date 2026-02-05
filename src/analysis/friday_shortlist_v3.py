# src/analysis/friday_shortlist_v3.py
"""
Friday shortlist v3 — Quality Gate + your constraints — FIXED for Thursday schema
+ FIXED to use Tuesday History for bankroll continuity (no more "Week 1" by default)
+ NO stake scaling (no exposure-cap scaling)
+ FIXED stake rules: stake is decided ONLY by ODDS band (your rule)

Thursday schema expected (your engine):
- probs: home_prob/draw_prob/away_prob/over_2_5_prob/under_2_5_prob
- odds:  offered_1/offered_x/offered_2/offered_over_2_5/offered_under_2_5
- value: value_pct_1/value_pct_x/value_pct_2/value_pct_over/value_pct_under

Outputs:
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
    """
    Search upwards for a folder that contains logs/ and src/
    Render often runs from: /opt/render/project/src/src/analysis/...
    logs live in:        /opt/render/project/src/logs
    """
    for p in [start] + list(start.parents):
        if (p / "logs").is_dir() and (p / "src").is_dir():
            return p
    # fallback: assume .../src/src/analysis -> project root is 2 levels up
    return start.parents[2] if len(start.parents) >= 3 else start


_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _find_project_root(_THIS_FILE.parent)
LOGS_DIR = PROJECT_ROOT / "logs"
DATA_DIR = PROJECT_ROOT / "data"

# Add useful paths for imports
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
    """
    Load src/analysis/quality_gate_v1.py by path (works even without packages/__init__.py).
    """
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
    # If missing, do NOT kill Friday; treat as pass-through.
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


# -------------------------
# Tuesday history loading (bankroll continuity)
# -------------------------

def _load_latest_tuesday_history() -> Optional[Dict[str, Any]]:
    """
    We try to locate Tuesday history in common places.
    Priority:
      1) logs/tuesday_history_v3.json
      2) logs/tuesday_history_v3week*.json (latest)
      3) data/tuesday_history_v3.json
      4) data/tuesday_history_v3week*.json (latest)
    """
    candidates: List[Path] = []

    p1 = LOGS_DIR / "tuesday_history_v3.json"
    if p1.exists():
        candidates.append(p1)

    candidates.extend(sorted(LOGS_DIR.glob("tuesday_history_v3week*.json"), key=lambda p: p.stat().st_mtime, reverse=True))

    p2 = DATA_DIR / "tuesday_history_v3.json"
    if p2.exists():
        candidates.append(p2)

    candidates.extend(sorted(DATA_DIR.glob("tuesday_history_v3week*.json"), key=lambda p: p.stat().st_mtime, reverse=True))

    if not candidates:
        return None

    try:
        js = _read_json(candidates[0])
        if isinstance(js, dict):
            js["_source_path"] = str(candidates[0])
            return js
        return None
    except Exception:
        return None


def _bankrolls_from_history(history: Optional[Dict[str, Any]]) -> Tuple[float, float, float, int, Optional[str]]:
    """
    Returns (core_bankroll, fun_bankroll, draw_bankroll, week_no, source_path)
    week_no = (history.week_count + 1) if present, else 1
    """
    core_default = float(os.getenv("CORE_BANKROLL_START", "800"))
    fun_default = float(os.getenv("FUN_BANKROLL_START", "400"))
    draw_default = float(os.getenv("DRAW_BANKROLL_START", "300"))

    week_no = 1
    source_path = None

    if not history:
        return core_default, fun_default, draw_default, week_no, source_path

    source_path = str(history.get("_source_path") or "") or None

    wc = history.get("week_count")
    if isinstance(wc, int) and wc >= 1:
        week_no = wc + 1

    def grab(section: str, fallback: float) -> float:
        block = history.get(section)
        if isinstance(block, dict):
            bc = block.get("bankroll_current")
            if isinstance(bc, (int, float)):
                return float(bc)
        return fallback

    core_b = grab("core", core_default)
    fun_b = grab("funbet", fun_default)
    draw_b = grab("drawbet", draw_default)

    return core_b, fun_b, draw_b, week_no, source_path


# -------------------------
# Markets mapping (Thursday schema)
# -------------------------

def _market_defs() -> List[Dict[str, str]]:
    return [
        {"label": "Home (1)", "odds": "offered_1", "prob": "home_prob", "value": "value_pct_1"},
        {"label": "Draw",     "odds": "offered_x", "prob": "draw_prob", "value": "value_pct_x"},
        {"label": "Away (2)", "odds": "offered_2", "prob": "away_prob", "value": "value_pct_2"},
        {"label": "Over 2.5", "odds": "offered_over_2_5",  "prob": "over_2_5_prob",  "value": "value_pct_over"},
        {"label": "Under 2.5","odds": "offered_under_2_5", "prob": "under_2_5_prob", "value": "value_pct_under"},
    ]

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

    # ranking (simple and stable)
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


# -------------------------
# Stake rules (NO scaling) — YOUR rule
# -------------------------

def _stake_from_odds(odds: float) -> int:
    """
    Your rule (as you wrote it, cleaned):
      - < 1.70  -> 25
      - 1.70 .. < 1.90 -> 40
      - 1.90 .. < 2.05 -> 30
      - >= 2.05 -> 20
    """
    o = float(odds)
    if o < 1.70:
        return 25
    if o < 1.90:
        return 40
    if o < 2.05:
        return 30
    return 20


# -------------------------
# Selection logic
# -------------------------

def _select_core(cands: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], float]:
    # Core: avoid Draw
    core = [c for c in cands if c["market"] != "Draw"]

    # Filters (keep as-is)
    core = [c for c in core if 1.55 <= c["odds"] <= 2.20 and c["prob"] >= 0.50 and c["value_pct"] >= 1.0]
    core.sort(key=lambda x: x["rank_score"], reverse=True)
    core = core[:7]

    singles: List[Dict[str, Any]] = []
    open_total = 0.0

    for c in core:
        stake = _stake_from_odds(c["odds"])
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

    # Doubles rule: ONLY if 2 singles odds <= 1.60
    eligible = [s for s in singles if s["odds"] <= 1.60]
    eligible.sort(key=lambda s: s["odds"])

    core_double = None
    if len(eligible) >= 2:
        a, b = eligible[0], eligible[1]
        combined_odds = round(a["odds"] * b["odds"], 2)
        dbl_stake = float(os.getenv("CORE_DOUBLE_STAKE", "25"))  # you hinted "25" — default 25
        core_double = {
            "legs": [
                {k: a[k] for k in ["date", "time_gr", "league", "match", "market", "odds"]},
                {k: b[k] for k in ["date", "time_gr", "league", "match", "market", "odds"]},
            ],
            "odds": combined_odds,
            "stake": dbl_stake,
            "label": "Double (Core small-odds only)",
        }
        open_total += dbl_stake

    # IMPORTANT: NO scaling / NO exposure-cap cutting
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


# -------------------------
# Main builder
# -------------------------

def build_friday_shortlist() -> Dict[str, Any]:
    th = _load_latest_thursday_report()
    fixtures = th.get("fixtures") or []
    if not isinstance(fixtures, list):
        raise ValueError("Thursday report has no fixtures list.")

    # Load Tuesday history (optional)
    hist = _load_latest_tuesday_history()
    core_bankroll, fun_bankroll, draw_bankroll, week_no, hist_src = _bankrolls_from_history(hist)

    # Quality thresholds
    core_min_q = float(os.getenv("FRIDAY_CORE_MIN_QUALITY", "0.70"))
    fun_min_q = float(os.getenv("FRIDAY_FUN_MIN_QUALITY", "0.60"))
    draw_min_q = float(os.getenv("FRIDAY_DRAW_MIN_QUALITY", "0.70"))

    mdefs = _market_defs()

    all_cands: List[Dict[str, Any]] = []
    q_by_id: Dict[Any, float] = {}

    # Precompute fixture quality per fixture_id
    for fx in fixtures:
        qr = fixture_quality_score(fx)
        q_by_id[fx.get("fixture_id")] = float(qr.score)

    # Expand fixture -> candidates per market
    for fx in fixtures:
        for md in mdefs:
            c = _candidate_from_fixture(fx, md)
            if c:
                all_cands.append(c)

    core_cands = [c for c in all_cands if (q_by_id.get(c["fixture_id"], 0.0) >= core_min_q)]
    fun_cands  = [c for c in all_cands if (q_by_id.get(c["fixture_id"], 0.0) >= fun_min_q)]
    draw_cands = [c for c in all_cands if (q_by_id.get(c["fixture_id"], 0.0) >= draw_min_q)]

    # Build portfolios
    core_singles, core_double, core_open = _select_core(core_cands)

    fun_stake_total = float(os.getenv("FUN_SYSTEM_STAKE", "35"))
    fun_pool, fun_system, fun_open = _select_system(
        fun_cands,
        n=7, min_odds=2.00, max_odds=3.60, min_prob=0.32, min_value=2.0,
        stake_total=fun_stake_total, allow_draw=False,
    )

    draw_stake_total = float(os.getenv("DRAW_SYSTEM_STAKE", "25"))
    draw_pool, draw_system, draw_open = _select_system(
        draw_cands,
        n=5, min_odds=3.00, max_odds=4.50, min_prob=0.24, min_value=2.0,
        stake_total=draw_stake_total, allow_draw=True,
    )
    draw_pool = [x for x in draw_pool if x["market"] == "Draw"]
    if not draw_pool:
        draw_open = 0.0

    # Window
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
        "week_no": week_no,
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
            "stake_rule": {
                "<1.70": 25,
                "1.70-1.89": 40,
                "1.90-2.04": 30,
                ">=2.05": 20,
                "double_stake": float(os.getenv("CORE_DOUBLE_STAKE", "25")),
                "scaling": "OFF",
            },
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
            "history_source": hist_src,
            "thresholds": {"core": core_min_q, "fun": fun_min_q, "draw": draw_min_q},
            "counts": {
                "fixtures": len(fixtures),
                "candidates_total": len(all_cands),
                "core_singles": len(core_singles),
                "fun_pool": len(fun_pool),
                "draw_pool": len(draw_pool),
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
            "week_no": friday.get("week_no"),
            "counts": friday["debug"]["counts"],
            "logs_dir": str(LOGS_DIR),
            "history_source": friday["debug"].get("history_source"),
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
