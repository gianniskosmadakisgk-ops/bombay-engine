# filename: src/analysis/friday_shortlist_v3.py
"""
Friday shortlist v3 (Quality Gate + locked constraints + HISTORY-aware)

Key fixes:
- Render path-safe: finds PROJECT_ROOT and uses real <root>/logs
- Reads Tuesday history from <root>/logs/tuesday_history_v3.json (if exists)
- Week label/window carried from Thursday report (no Week=1 reset if history is found)

Locked business rules:
- Core singles: max 7 total.
- Core double: only if there are at least 2 Core singles with odds <= 1.60.
  Double uses ONLY those two smallest odds. Otherwise no double.
- FunBet: system pool (7) as before.
- DrawBet: strict. If no good draws => empty pool and open=0.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# -------------------------
# Path / import robustness
# -------------------------

def _find_project_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / "logs").is_dir() and (p / "src").is_dir():
            return p
        if (p / "logs").is_dir() and (p / "analysis").is_dir():
            return p
    return start.parents[1]

_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _find_project_root(_THIS_FILE.parent)
LOGS_DIR = PROJECT_ROOT / "logs"

# Make imports work whether code lives in PROJECT_ROOT/src or PROJECT_ROOT/src/src
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if (PROJECT_ROOT / "src").is_dir() and str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if (PROJECT_ROOT / "src" / "src").is_dir() and str(PROJECT_ROOT / "src" / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src" / "src"))

try:
    from analysis.quality_gate_v1 import fixture_quality_score
except Exception:
    def fixture_quality_score(fixture: Dict[str, Any]):  # type: ignore
        return type("QR", (), {"score": 0.0, "reasons": ("import_failed",)})

# -------------------------
# IO helpers
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

def _safe_float(v: Any, d: Optional[float] = None) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return d

def _parse_dt(dt_str: str) -> Optional[datetime]:
    if not dt_str:
        return None
    try:
        return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except Exception:
        return None

def _fmt_date(dt_utc: Optional[datetime]) -> str:
    return dt_utc.date().isoformat() if dt_utc else ""

def _fmt_time(dt_utc: Optional[datetime]) -> str:
    return dt_utc.strftime("%H:%M") if dt_utc else ""

# -------------------------
# Load Thursday / History
# -------------------------

def _load_latest_thursday_report() -> Dict[str, Any]:
    p1 = LOGS_DIR / "thursday_report_v3.json"
    if p1.exists():
        return _read_json(p1)

    th_files = sorted(LOGS_DIR.glob("thursday_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if th_files:
        return _read_json(th_files[0])

    raise FileNotFoundError(f"Could not find Thursday report inside {LOGS_DIR}")

def _load_tuesday_history() -> Dict[str, Any]:
    """
    IMPORTANT: We ONLY trust the real logs folder.
    If user uploaded history but backend saved it elsewhere, it will be invisible here.
    """
    p = LOGS_DIR / "tuesday_history_v3.json"
    if not p.exists():
        return {}
    try:
        return _read_json(p) or {}
    except Exception:
        return {}

def _bankrolls_from_history_or_defaults(hist: Dict[str, Any]) -> Tuple[float, float, float, str]:
    # defaults
    d_core = float(os.getenv("BANKROLL_CORE", "800"))
    d_fun  = float(os.getenv("BANKROLL_FUN", "400"))
    d_draw = float(os.getenv("BANKROLL_DRAW", "300"))

    src = "default"
    core = _safe_float(((hist.get("core") or {}).get("bankroll_current")), None)
    fun  = _safe_float(((hist.get("funbet") or {}).get("bankroll_current")), None)
    draw = _safe_float(((hist.get("drawbet") or {}).get("bankroll_current")), None)

    if core is not None or fun is not None or draw is not None:
        src = "history"

    return (core if core is not None else d_core,
            fun if fun is not None else d_fun,
            draw if draw is not None else d_draw,
            src)

# -------------------------
# Markets & candidates
# -------------------------

def _market_defs() -> List[Dict[str, str]]:
    return [
        {"code": "1",   "label": "Home (1)",   "offered": "offered_1",           "prob": "prob_1",           "value": "value_pct_1"},
        {"code": "X",   "label": "Draw",       "offered": "offered_x",           "prob": "prob_x",           "value": "value_pct_x"},
        {"code": "2",   "label": "Away (2)",   "offered": "offered_2",           "prob": "prob_2",           "value": "value_pct_2"},
        {"code": "O25", "label": "Over 2.5",   "offered": "offered_over_2_5",    "prob": "prob_over_2_5",    "value": "value_pct_over_2_5"},
        {"code": "U25", "label": "Under 2.5",  "offered": "offered_under_2_5",   "prob": "prob_under_2_5",   "value": "value_pct_under_2_5"},
    ]

def _get_num(fx: Dict[str, Any], key: str) -> Optional[float]:
    v = fx.get(key)
    if isinstance(v, (int, float)):
        return float(v)
    return None

def _candidate_from_fixture(fx: Dict[str, Any], md: Dict[str, str]) -> Optional[Dict[str, Any]]:
    odds = _get_num(fx, md["offered"])
    prob = _get_num(fx, md["prob"])
    value_pct = _get_num(fx, md["value"])
    if odds is None or prob is None:
        return None

    kickoff = _parse_dt(str(fx.get("kickoff_utc") or ""))
    league = str(fx.get("league") or "")
    home = str(fx.get("home") or "")
    away = str(fx.get("away") or "")
    fixture_id = fx.get("fixture_id")

    # ranking heuristic (keep simple)
    score = (value_pct or 0.0) * 0.6 + (prob - 0.5) * 100 * 0.4

    return {
        "fixture_id": fixture_id,
        "date": _fmt_date(kickoff),
        "time_gr": _fmt_time(kickoff),
        "league": league,
        "match": f"{home} – {away}",
        "market_code": md["code"],
        "market": md["label"],
        "odds": round(float(odds), 2),
        "prob": round(float(prob), 4),
        "value_pct": round(float(value_pct or 0.0), 2),
        "rank_score": float(score),
        "quality": None,
        "quality_reasons": [],
        "raw": fx,
    }

# -------------------------
# Selection rules (locked)
# -------------------------

def _select_core(cands: List[Dict[str, Any]], bankroll_start: float) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], float]:
    # Core excludes Draw
    core = [c for c in cands if c["market_code"] != "X"]

    # Basic gates (you can tune later)
    core = [c for c in core if 1.55 <= c["odds"] <= 2.20 and c["prob"] >= 0.50 and c["value_pct"] >= 1.0]
    core.sort(key=lambda x: x["rank_score"], reverse=True)
    core = core[:7]  # LOCKED

    singles: List[Dict[str, Any]] = []
    open_total = 0.0

    for c in core:
        strength = (c["prob"] - 0.5) * 100 + c["value_pct"]
        if strength >= 20:
            stake = 40
        elif strength >= 12:
            stake = 30
        else:
            stake = 20

        singles.append({
            "fixture_id": c["fixture_id"],
            "date": c["date"],
            "time_gr": c["time_gr"],
            "league": c["league"],
            "match": c["match"],
            "market_code": c["market_code"],
            "market": c["market"],
            "odds": c["odds"],
            "stake": stake,
            "quality": c["quality"],
        })
        open_total += stake

    # LOCKED: double ONLY if at least 2 singles with odds <= 1.60 (pick 2 smallest odds)
    eligible = [s for s in singles if float(s["odds"]) <= 1.60]
    eligible.sort(key=lambda s: float(s["odds"]))

    core_double = None
    if len(eligible) >= 2:
        a, b = eligible[0], eligible[1]
        combined_odds = round(float(a["odds"]) * float(b["odds"]), 2)
        core_double = {
            "legs": [
                {k: a[k] for k in ["fixture_id","date","time_gr","league","match","market_code","market","odds"]},
                {k: b[k] for k in ["fixture_id","date","time_gr","league","match","market_code","market","odds"]},
            ],
            "combo_odds": combined_odds,
            "stake": 20,
            "tag": "core_double_small_only",
        }
        open_total += 20

    # Exposure cap (keep)
    exposure_cap_pct = float(os.getenv("CORE_EXPOSURE_CAP_PCT", "0.30"))
    cap = bankroll_start * exposure_cap_pct
    if cap > 0 and open_total > cap and singles:
        factor = cap / open_total
        new_total = 0.0
        for s in singles:
            s["stake"] = max(5, int(round(s["stake"] * factor / 5) * 5))
            new_total += s["stake"]
        open_total = new_total + (core_double["stake"] if core_double else 0.0)

    return singles, core_double, round(open_total, 2)

def _select_system(
    cands: List[Dict[str, Any]],
    *,
    n: int,
    min_odds: float,
    max_odds: float,
    min_prob: float,
    min_value: float,
    stake_total: float,
    allow_draw: bool
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], float]:
    pool = cands[:]
    if not allow_draw:
        pool = [c for c in pool if c["market_code"] != "X"]

    pool = [c for c in pool if min_odds <= c["odds"] <= max_odds and c["prob"] >= min_prob and c["value_pct"] >= min_value]
    pool.sort(key=lambda x: x["rank_score"], reverse=True)
    pool = pool[:n]

    # system metadata (simple)
    if n == 7:
        system = {"label": "4/7", "k": 4, "n": 7, "columns": 35, "min_hits": 4, "stake": float(stake_total)}
    elif n == 5:
        system = {"label": "2/5", "k": 2, "n": 5, "columns": 10, "min_hits": 2, "stake": float(stake_total)}
    else:
        system = {"label": f"{max(1,n//2)}/{n}", "k": max(1, n // 2), "n": n, "columns": None, "min_hits": max(1, n // 2), "stake": float(stake_total)}

    system_pool = [{
        "fixture_id": c["fixture_id"],
        "date": c["date"],
        "time_gr": c["time_gr"],
        "league": c["league"],
        "match": c["match"],
        "market_code": c["market_code"],
        "market": c["market"],
        "odds": c["odds"],
        "quality": c["quality"],
    } for c in pool]

    open_amt = float(stake_total if system_pool else 0.0)
    return system_pool, system, round(open_amt, 2)

# -------------------------
# Build Friday
# -------------------------

def build_friday_shortlist() -> Dict[str, Any]:
    th = _load_latest_thursday_report()
    fixtures = th.get("fixtures") or []
    if not isinstance(fixtures, list):
        raise ValueError("Thursday report has no fixtures list.")

    # thresholds
    core_min_q = float(os.getenv("FRIDAY_CORE_MIN_QUALITY", "0.70"))
    fun_min_q  = float(os.getenv("FRIDAY_FUN_MIN_QUALITY", "0.60"))
    draw_min_q = float(os.getenv("FRIDAY_DRAW_MIN_QUALITY", "0.70"))

    mdefs = _market_defs()
    all_cands: List[Dict[str, Any]] = []

    q_by_id: Dict[Any, float] = {}
    for fx in fixtures:
        qr = fixture_quality_score(fx)
        q_by_id[fx.get("fixture_id")] = float(getattr(qr, "score", 0.0))

    for fx in fixtures:
        qr = fixture_quality_score(fx)
        q = float(getattr(qr, "score", 0.0))
        reasons = list(getattr(qr, "reasons", ()))

        for md in mdefs:
            c = _candidate_from_fixture(fx, md)
            if not c:
                continue
            c["quality"] = round(q, 3)
            c["quality_reasons"] = reasons
            all_cands.append(c)

    # quality gate per portfolio
    core_cands = [c for c in all_cands if q_by_id.get(c["fixture_id"], 0.0) >= core_min_q]
    fun_cands  = [c for c in all_cands if q_by_id.get(c["fixture_id"], 0.0) >= fun_min_q]
    draw_cands = [c for c in all_cands if q_by_id.get(c["fixture_id"], 0.0) >= draw_min_q]

    # bankrolls from history
    hist = _load_tuesday_history()
    core_bankroll, fun_bankroll, draw_bankroll, bankroll_src = _bankrolls_from_history_or_defaults(hist)

    # Core
    core_singles, core_double, core_open = _select_core(core_cands, core_bankroll)

    # Fun (keep stake as-is from env)
    fun_stake_total = float(os.getenv("FUN_SYSTEM_STAKE", "35"))
    fun_pool, fun_system, fun_open = _select_system(
        fun_cands,
        n=7, min_odds=2.00, max_odds=3.60, min_prob=0.32, min_value=2.0,
        stake_total=fun_stake_total, allow_draw=False
    )

    # Draw strict
    draw_stake_total = float(os.getenv("DRAW_SYSTEM_STAKE", "25"))
    draw_pool, draw_system, draw_open = _select_system(
        draw_cands,
        n=5, min_odds=3.00, max_odds=4.50, min_prob=0.24, min_value=2.0,
        stake_total=draw_stake_total, allow_draw=True
    )
    draw_pool = [x for x in draw_pool if x["market_code"] == "X"]
    if not draw_pool:
        draw_open = 0.0

    window = th.get("window") or {}

    out = {
        "generated_at": _utc_now_iso(),
        "window": window,
        "core": {
            "label": "CoreBet",
            "bankroll_start": round(core_bankroll, 2),
            "bankroll_source": bankroll_src,
            "max_singles": 7,
            "singles": core_singles,
            "double": core_double,
            "doubles": ([core_double] if core_double else []),
            "open": round(core_open, 2),
            "after_open": round(core_bankroll - core_open, 2),
        },
        "funbet": {
            "label": "FunBet",
            "bankroll_start": round(fun_bankroll, 2),
            "bankroll_source": bankroll_src,
            "system_pool": fun_pool,
            "system": fun_system,
            "open": round(fun_open, 2),
            "after_open": round(fun_bankroll - fun_open, 2),
        },
        "drawbet": {
            "label": "DrawBet",
            "bankroll_start": round(draw_bankroll, 2),
            "bankroll_source": bankroll_src,
            "system_pool": draw_pool,
            "system": draw_system,
            "open": round(draw_open, 2),
            "after_open": round(draw_bankroll - draw_open, 2),
        },
        "debug": {
            "project_root": str(PROJECT_ROOT),
            "logs_dir": str(LOGS_DIR),
            "history_seen": bool((LOGS_DIR / "tuesday_history_v3.json").exists()),
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

        out_main = LOGS_DIR / "friday_shortlist_v3.json"
        _write_json(out_main, friday)

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        _write_json(LOGS_DIR / f"friday_{ts}.json", friday)

        print(json.dumps({
            "status": "ok",
            "saved": str(out_main),
            "generated_at": friday["generated_at"],
            "history_seen": friday["debug"]["history_seen"],
            "logs_dir": str(LOGS_DIR),
        }))
        return 0
    except Exception as e:
        print(json.dumps({"status": "error", "error": str(e), "logs_dir": str(LOGS_DIR)}))
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
