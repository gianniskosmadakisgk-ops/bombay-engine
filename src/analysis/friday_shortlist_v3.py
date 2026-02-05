# src/analysis/friday_shortlist_v3.py
"""
Friday shortlist v3.2 — Locked for deploy

What’s included now:
- Reads Tuesday history (week_count + bankroll_current) and prints correct Week label.
- NO stake scaling / NO exposure cap (you asked: fixed stakes).
- Core stake = based on odds only (simple locked rule).
- DrawBet = ALWAYS Top-5 draws by probability (if odds+prob exist).
- Chronological ordering for all outputs (date+time).
- FunBet prefers to include some Core picks when they qualify.

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
# Quality gate import by path
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

def _sf(v: Any) -> Optional[float]:
    if isinstance(v, (int, float)):
        return float(v)
    return None

def _fmt_date(s: str) -> str:
    return str(s or "")

def _fmt_time(s: str) -> str:
    return str(s or "")

def _sort_dt(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def key(x):
        return (str(x.get("date") or ""), str(x.get("time_gr") or ""))
    return sorted(items, key=key)

def _market_defs() -> List[Dict[str, str]]:
    return [
        {"label": "Home (1)", "odds": "offered_1", "prob": "home_prob", "value": "value_pct_1"},
        {"label": "Draw",     "odds": "offered_x", "prob": "draw_prob", "value": "value_pct_x"},
        {"label": "Away (2)", "odds": "offered_2", "prob": "away_prob", "value": "value_pct_2"},
        {"label": "Over 2.5", "odds": "offered_over_2_5",  "prob": "over_2_5_prob",  "value": "value_pct_over"},
        {"label": "Under 2.5","odds": "offered_under_2_5", "prob": "under_2_5_prob", "value": "value_pct_under"},
    ]

def _candidate_from_fixture(fx: Dict[str, Any], md: Dict[str, str]) -> Optional[Dict[str, Any]]:
    odds = _sf(fx.get(md["odds"]))
    prob = _sf(fx.get(md["prob"]))
    value_pct = _sf(fx.get(md["value"])) or 0.0

    if odds is None or prob is None:
        return None

    league = str(fx.get("league") or "")
    home = str(fx.get("home") or "")
    away = str(fx.get("away") or "")
    fixture_id = fx.get("fixture_id")

    date = _fmt_date(str(fx.get("date") or ""))
    time_gr = _fmt_time(str(fx.get("time") or ""))

    # ranking: value + prob edge (stable)
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

    raise FileNotFoundError(f"Could not find thursday_report_v3.json or thursday_*.json inside {LOGS_DIR}")

def _load_tuesday_history() -> Optional[Dict[str, Any]]:
    # Let this be configurable; default is what most deployments use.
    hp = os.getenv("TUESDAY_HISTORY_PATH", "data/tuesday_history_v3.json")
    p = (PROJECT_ROOT / hp) if not Path(hp).is_absolute() else Path(hp)
    if p.exists():
        return _read_json(p)
    # also try logs/
    p2 = LOGS_DIR / "tuesday_history_v3.json"
    if p2.exists():
        return _read_json(p2)
    return None


# -------------------------
# Stake rules (NO scaling)
# -------------------------

def _core_stake_from_odds(odds: float) -> int:
    # Locked simple rule for deploy:
    # <=1.80 -> 40
    # <=2.00 -> 30
    # >2.00  -> 30
    if odds <= 1.80:
        return 40
    if odds <= 2.00:
        return 30
    return 30


# -------------------------
# Selection logic
# -------------------------

def _select_core(cands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Core: avoid Draw markets
    core = [c for c in cands if c["market"] != "Draw"]

    # Core filters (keep your existing “sane” bounds)
    core = [c for c in core if 1.55 <= c["odds"] <= 2.20 and c["prob"] >= 0.50 and c["value_pct"] >= 1.0]
    core.sort(key=lambda x: x["rank_score"], reverse=True)
    core = core[:7]  # hard cap

    singles: List[Dict[str, Any]] = []
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
            "prob": c["prob"],
        })

    return _sort_dt(singles)


def _select_funbet(cands: List[Dict[str, Any]], core_singles: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any], float]:
    # Fun filters (your old ones)
    pool = [c for c in cands if c["market"] != "Draw"]
    pool = [c for c in pool if 2.00 <= c["odds"] <= 3.60 and c["prob"] >= 0.32 and c["value_pct"] >= 2.0]
    pool.sort(key=lambda x: x["rank_score"], reverse=True)

    # Prefer to include some Core picks if they qualify for fun constraints
    core_as_cands = []
    core_set = {(x["match"], x["market"]) for x in core_singles}
    for c in pool:
        if (c["match"], c["market"]) in core_set:
            core_as_cands.append(c)

    selected: List[Dict[str, Any]] = []
    # add up to 3 core-overlap picks first
    for c in core_as_cands[:3]:
        selected.append(c)

    # fill remaining up to 7
    for c in pool:
        if len(selected) >= 7:
            break
        if c in selected:
            continue
        selected.append(c)

    system = {"k": 4, "n": 7, "columns": 35, "min_hits": 4, "stake": float(os.getenv("FUN_SYSTEM_STAKE", "35"))}
    out_pool = [{
        "date": c["date"],
        "time_gr": c["time_gr"],
        "league": c["league"],
        "match": c["match"],
        "market": c["market"],
        "odds": c["odds"],
        "quality": c["quality"],
        "prob": c["prob"],
    } for c in selected]

    out_pool = _sort_dt(out_pool)
    open_amt = system["stake"] if out_pool else 0.0
    return out_pool, system, float(round(open_amt, 2))


def _select_draw_top5(all_cands: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any], float]:
    # Always Top-5 draws by probability, as long as odds+prob exist.
    draws = [c for c in all_cands if c["market"] == "Draw"]
    draws = [c for c in draws if c.get("odds") is not None and c.get("prob") is not None]

    # optional sanity range (keeps junk out)
    draws = [c for c in draws if 2.80 <= float(c["odds"]) <= 6.00]

    draws.sort(key=lambda x: float(x["prob"]), reverse=True)
    draws = draws[:5]

    system = {"k": 2, "n": 5, "columns": 10, "min_hits": 2, "stake": float(os.getenv("DRAW_SYSTEM_STAKE", "25"))}

    out_pool = [{
        "date": c["date"],
        "time_gr": c["time_gr"],
        "league": c["league"],
        "match": c["match"],
        "market": "Draw",
        "odds": c["odds"],
        "quality": c["quality"],
        "prob": c["prob"],
    } for c in draws]

    out_pool = _sort_dt(out_pool)
    open_amt = system["stake"] if out_pool else 0.0
    return out_pool, system, float(round(open_amt, 2))


# -------------------------
# Main builder
# -------------------------

def build_friday_shortlist() -> Dict[str, Any]:
    th = _load_latest_thursday_report()
    fixtures = th.get("fixtures") or []
    if not isinstance(fixtures, list):
        raise ValueError("Thursday report has no fixtures list.")

    # Quality thresholds
    core_min_q = float(os.getenv("FRIDAY_CORE_MIN_QUALITY", "0.70"))
    fun_min_q  = float(os.getenv("FRIDAY_FUN_MIN_QUALITY", "0.60"))

    # Load Tuesday history for week + bankrolls
    hist = _load_tuesday_history() or {}
    week_count = int(hist.get("week_count") or 0)
    week_no = week_count + 1

    # Default bankrolls from history if present, else env, else fallback
    def bankroll_from(hist_key: str, env_key: str, fallback: float) -> float:
        try:
            if isinstance(hist.get(hist_key), dict) and hist[hist_key].get("bankroll_current") is not None:
                return float(hist[hist_key]["bankroll_current"])
        except Exception:
            pass
        return float(os.getenv(env_key, str(fallback)))

    core_bankroll = bankroll_from("core", "CORE_BANKROLL_START", 800)
    fun_bankroll  = bankroll_from("funbet", "FUN_BANKROLL_START", 400)
    draw_bankroll = bankroll_from("drawbet", "DRAW_BANKROLL_START", 300)

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

    core_singles = _select_core(core_cands)
    core_open = float(sum(int(x["stake"]) for x in core_singles))

    fun_pool, fun_system, fun_open = _select_funbet(fun_cands, core_singles)
    draw_pool, draw_system, draw_open = _select_draw_top5(all_cands)

    window = th.get("window") or {}
    if not window:
        window = {
            "from": str(th.get("start_date") or ""),
            "to": str(th.get("end_date") or ""),
            "hours": int(th.get("window_hours") or 72),
        }

    out = {
        "title": f"Bombay Friday — Week {week_no}",
        "week_no": week_no,
        "generated_at": _utc_now_iso(),
        "window": window,

        "core": {
            "label": "CoreBet",
            "bankroll_start": round(core_bankroll, 2),
            "singles": core_singles,
            "open": round(core_open, 2),
            "after_open": round(core_bankroll - core_open, 2),
            "stake_rule": "odds<=1.80→40, <=2.00→30, >2.00→30 (NO scaling)",
        },

        "funbet": {
            "label": "FunBet",
            "bankroll_start": round(fun_bankroll, 2),
            "system_pool": fun_pool,
            "system": fun_system,
            "open": round(fun_open, 2),
            "after_open": round(fun_bankroll - fun_open, 2),
        },

        "drawbet": {
            "label": "DrawBet",
            "bankroll_start": round(draw_bankroll, 2),
            "system_pool": draw_pool,
            "system": draw_system,
            "open": round(draw_open, 2),
            "after_open": round(draw_bankroll - draw_open, 2),
            "rule": "Top-5 Draws by probability (odds+prob required)",
        },

        "debug": {
            "project_root": str(PROJECT_ROOT),
            "logs_dir": str(LOGS_DIR),
            "quality_thresholds": {"core": core_min_q, "fun": fun_min_q},
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
