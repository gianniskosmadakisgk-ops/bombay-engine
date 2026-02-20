# src/analysis/friday_shortlist_v3.py
"""
Friday shortlist v3 — HISTORY-AWARE + env-driven + NO SCALING + TOP-5 DRAWS
+ chronological ordering + fixture_id everywhere + draw tempo filter.

Key guarantees:
- Writes canonical: logs/friday_shortlist_v3.json
- Reads Thursday:     logs/thursday_report_v3.json
- Reads History:      logs/tuesday_history_v3.json (or env TUESDAY_HISTORY_PATH)
- Core stake rule: odds <= 2.00 -> 40, else -> 30  (NO scaling)
- Core singles target: FRIDAY_CORE_TARGET_SINGLES (default 7)
- Core totals cap: FRIDAY_CORE_MAX_TOTALS (default 4) and under cap: FRIDAY_CORE_MAX_UNDERS (default 1)
- Core strict odds gate: FRIDAY_CORE_REQUIRE_STRICT_OK (default true) uses Thursday flags.odds_strict_ok
- Ranking weights: FRIDAY_RANK_W_VALUE / FRIDAY_RANK_W_PROB
- DrawBet: top-5 by probability within odds window AND total_lambda <= FRIDAY_DRAW_MAX_TOTAL_LAMBDA (default 2.9)
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
    # fallback (Render usually: /opt/render/project/src)
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

def _sf_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return float(default)

def _si_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return int(default)

def _sb_env(name: str, default: bool) -> bool:
    v = os.getenv(name, "true" if default else "false").strip().lower()
    return v in ("1", "true", "yes", "y", "on")

def _chrono_key(item: Dict[str, Any]) -> Tuple[str, str]:
    return (str(item.get("date") or ""), str(item.get("time_gr") or ""))

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

def _prob_points(prob: float) -> float:
    return (float(prob) - 0.50) * 100.0

def _rank_score(value_pct: float, prob: float) -> float:
    wv = _sf_env("FRIDAY_RANK_W_VALUE", 0.60)
    wp = _sf_env("FRIDAY_RANK_W_PROB", 0.40)
    if wv < 0: wv = 0.0
    if wp < 0: wp = 0.0
    s = wv + wp
    if s <= 0:
        wv, wp = 0.60, 0.40
        s = 1.0
    wv /= s
    wp /= s
    return (wv * float(value_pct)) + (wp * _prob_points(float(prob)))

def _is_total_market(market: str) -> bool:
    return market in ("Over 2.5", "Under 2.5")

def _is_under(market: str) -> bool:
    return market == "Under 2.5"


# -------------------------
# Loaders
# -------------------------

def _load_latest_thursday_report() -> Dict[str, Any]:
    p1 = LOGS_DIR / "thursday_report_v3.json"
    if p1.exists():
        return _read_json(p1)

    th_files = sorted(LOGS_DIR.glob("thursday_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if th_files:
        return _read_json(th_files[0])

    raise FileNotFoundError(f"Could not find thursday_report_v3.json or any thursday_*.json inside {LOGS_DIR}")

def _load_history() -> Tuple[Dict[str, Any], bool, str]:
    """
    Expected schema (your file):
      { "week_count": int, "bankroll_current": {"core":float,"fun":float,"draw":float}, ... }
    """
    hp = os.getenv("TUESDAY_HISTORY_PATH", str(LOGS_DIR / "tuesday_history_v3.json"))
    p = Path(hp)
    if p.exists():
        try:
            js = _read_json(p)
            return js, True, str(p)
        except Exception:
            return {}, False, str(p)
    return {}, False, str(p)


# -------------------------
# Stake rules (NO scaling)
# -------------------------

def core_stake_from_odds(odds: float) -> int:
    return 40 if float(odds) <= 2.00 else 30


# -------------------------
# Candidate builder
# -------------------------

def _candidate_from_fixture(fx: Dict[str, Any], md: Dict[str, str]) -> Optional[Dict[str, Any]]:
    odds = _safe_float(fx.get(md["odds"]))
    prob = _safe_float(fx.get(md["prob"]))
    value_pct = _safe_float(fx.get(md["value"])) or 0.0
    if odds is None or prob is None:
        return None

    fixture_id = fx.get("fixture_id")
    league = str(fx.get("league") or "")
    home = str(fx.get("home") or "")
    away = str(fx.get("away") or "")

    date = _fmt_date(str(fx.get("date") or ""))
    time_gr = _fmt_time(str(fx.get("time") or ""))

    total_lambda = _safe_float(fx.get("total_lambda"))  # needed for draw tempo filter
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
        "total_lambda": None if total_lambda is None else round(float(total_lambda), 3),
    }


# -------------------------
# Selection logic
# -------------------------

def _select_core(cands: List[Dict[str, Any]], strict_ok_by_id: Dict[Any, bool]) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], float]:
    odds_min = _sf_env("FRIDAY_CORE_ODDS_MIN", 1.55)
    odds_max = _sf_env("FRIDAY_CORE_ODDS_MAX", 2.20)
    min_prob = _sf_env("FRIDAY_CORE_MIN_PROB", 0.50)
    min_val  = _sf_env("FRIDAY_CORE_MIN_VALUE_PCT", 1.0)

    target_singles = _si_env("FRIDAY_CORE_TARGET_SINGLES", 7)
    max_totals = _si_env("FRIDAY_CORE_MAX_TOTALS", 4)
    max_unders = _si_env("FRIDAY_CORE_MAX_UNDERS", 1)
    require_strict = _sb_env("FRIDAY_CORE_REQUIRE_STRICT_OK", True)

    core = [c for c in cands if c["market"] != "Draw"]
    core = [c for c in core if odds_min <= float(c["odds"]) <= odds_max and float(c["prob"]) >= min_prob and float(c["value_pct"]) >= min_val]

    if require_strict:
        core = [c for c in core if bool(strict_ok_by_id.get(c.get("fixture_id"), False))]

    # bias Over slightly on ties
    core.sort(key=lambda x: (float(x["rank_score"]), 1 if x["market"] == "Over 2.5" else 0), reverse=True)

    singles: List[Dict[str, Any]] = []
    open_total = 0.0
    totals_cnt = 0
    unders_cnt = 0

    # PASS 1: totals first (up to cap)
    for c in core:
        if len(singles) >= target_singles:
            break
        if not _is_total_market(c["market"]):
            continue
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
        totals_cnt += 1
        if _is_under(c["market"]):
            unders_cnt += 1

    # PASS 2: fill with 1X2 (Home/Away), no duplicate fixture_id
    for c in core:
        if len(singles) >= target_singles:
            break
        if _is_total_market(c["market"]) or c["market"] == "Draw":
            continue
        if any(s["fixture_id"] == c.get("fixture_id") for s in singles):
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

    # Double (unchanged)
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
    pool_size = _si_env("FRIDAY_FUN_POOL_SIZE", 7)
    max_totals = _si_env("FRIDAY_FUN_MAX_TOTALS", 4)
    max_unders = _si_env("FRIDAY_FUN_MAX_UNDERS", 1)

    pool = [c for c in cands if c["market"] != "Draw"]
    pool = [c for c in pool if 1.90 <= float(c["odds"]) <= 3.60 and float(c["prob"]) >= 0.30 and float(c["value_pct"]) >= 1.0]
    pool.sort(key=lambda x: (float(x["rank_score"]), 1 if x["market"] == "Over 2.5" else 0), reverse=True)

    system_pool: List[Dict[str, Any]] = []
    totals_cnt = 0
    unders_cnt = 0
    seen_fixtures = set()

    for c in pool:
        if len(system_pool) >= pool_size:
            break
        fid = c.get("fixture_id")
        if fid in seen_fixtures:
            continue

        if _is_total_market(c["market"]):
            if totals_cnt >= max_totals:
                continue
            if _is_under(c["market"]) and unders_cnt >= max_unders:
                continue

        system_pool.append({
            "fixture_id": fid,
            "date": c["date"],
            "time_gr": c["time_gr"],
            "league": c["league"],
            "match": c["match"],
            "market": c["market"],
            "odds": c["odds"],
            "prob": c["prob"],
            "quality": c["quality"],
        })
        seen_fixtures.add(fid)

        if _is_total_market(c["market"]):
            totals_cnt += 1
            if _is_under(c["market"]):
                unders_cnt += 1

    system = {"k": 4, "n": 7, "columns": 35, "min_hits": 4, "stake": float(stake_total)}
    system_pool.sort(key=_chrono_key)
    open_amt = float(stake_total) if system_pool else 0.0
    return system_pool, system, float(round(open_amt, 2))


def _select_draw_top5(all_cands: List[Dict[str, Any]], stake_total: float) -> Tuple[List[Dict[str, Any]], Dict[str, Any], float]:
    dmin = _sf_env("FRIDAY_DRAW_ODDS_MIN", 3.00)
    dmax = _sf_env("FRIDAY_DRAW_ODDS_MAX", 5.50)
    max_tl = _sf_env("FRIDAY_DRAW_MAX_TOTAL_LAMBDA", 2.90)  # <-- the tempo filter

    draws = [c for c in all_cands if c["market"] == "Draw" and c.get("prob") is not None and c.get("odds") is not None]
    draws = [c for c in draws if dmin <= float(c["odds"]) <= dmax]

    # apply tempo filter if total_lambda exists
    filtered = []
    for c in draws:
        tl = c.get("total_lambda")
        if tl is None:
            # keep it, but mark as unknown (Friday shouldn't die)
            filtered.append(c)
        else:
            if float(tl) <= float(max_tl):
                filtered.append(c)

    filtered.sort(key=lambda x: (float(x["prob"]), float(x["odds"])), reverse=True)
    filtered = filtered[:5]

    system = {
        "k": 2, "n": 5, "columns": 10, "min_hits": 2, "stake": float(stake_total),
        "mode": "top5_by_probability",
        "draw_max_total_lambda": float(max_tl),
    }

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
        "total_lambda": c.get("total_lambda"),
    } for c in filtered]

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

    for ln in _all_lines():
        fid = ln.get("fixture_id")
        if not isinstance(fid, int):
            raise ValueError(f"Missing/invalid fixture_id in line: {ln}")

    for path in [("core", "singles"), ("funbet", "system_pool"), ("drawbet", "system_pool")]:
        lst = (friday.get(path[0], {}) or {}).get(path[1], [])
        if lst != sorted(lst, key=_chrono_key):
            raise ValueError(f"List not chronological: {path}")

    for ln in (friday.get("core", {}) or {}).get("singles", []):
        odds = float(ln["odds"])
        expected = 40 if odds <= 2.00 else 30
        if int(ln["stake"]) != expected:
            raise ValueError(f"Stake rule broken: odds={odds} stake={ln['stake']} expected={expected}")

    max_totals = _si_env("FRIDAY_CORE_MAX_TOTALS", 4)
    max_unders = _si_env("FRIDAY_CORE_MAX_UNDERS", 1)
    core_singles = (friday.get("core", {}) or {}).get("singles", []) or []
    totals = [x for x in core_singles if _is_total_market(x.get("market", ""))]
    unders = [x for x in core_singles if _is_under(x.get("market", ""))]
    if len(totals) > max_totals:
        raise ValueError(f"Core totals cap broken: totals={len(totals)} max={max_totals}")
    if len(unders) > max_unders:
        raise ValueError(f"Core under cap broken: unders={len(unders)} max={max_unders}")


# -------------------------
# Main builder
# -------------------------

def build_friday_shortlist() -> Dict[str, Any]:
    th = _load_latest_thursday_report()
    fixtures = th.get("fixtures") or []
    if not isinstance(fixtures, list):
        raise ValueError("Thursday report has no fixtures list.")

    # history (bankroll continuity + week numbering)
    hist, hist_ok, hist_path = _load_history()
    hist_week = int(hist.get("week_count") or 0) if isinstance(hist, dict) else 0
    wk = hist_week + 1 if hist_week > 0 else 1

    bc = (hist.get("bankroll_current") or {}) if isinstance(hist, dict) else {}
    hist_core = _safe_float(bc.get("core"))
    hist_fun  = _safe_float(bc.get("fun"))
    hist_draw = _safe_float(bc.get("draw"))

    # fallbacks if history missing
    core_bankroll = hist_core if hist_core is not None else _sf_env("CORE_BANKROLL_START", 800.0)
    fun_bankroll  = hist_fun  if hist_fun  is not None else _sf_env("FUN_BANKROLL_START", 400.0)
    draw_bankroll = hist_draw if hist_draw is not None else _sf_env("DRAW_BANKROLL_START", 300.0)

    core_min_q = _sf_env("FRIDAY_CORE_MIN_QUALITY", 0.70)
    fun_min_q  = _sf_env("FRIDAY_FUN_MIN_QUALITY", 0.60)

    fun_stake_total  = _sf_env("FUN_SYSTEM_STAKE", 35.0)
    draw_stake_total = _sf_env("DRAW_SYSTEM_STAKE", 25.0)

    all_cands: List[Dict[str, Any]] = []
    q_by_id: Dict[Any, float] = {}
    strict_ok_by_id: Dict[Any, bool] = {}

    for fx in fixtures:
        fid = fx.get("fixture_id")
        qr = fixture_quality_score(fx)
        q_by_id[fid] = float(qr.score)

        flags = (fx.get("flags") or {})
        strict_ok_by_id[fid] = bool(flags.get("odds_strict_ok") is True)

    for fx in fixtures:
        for md in _market_defs():
            c = _candidate_from_fixture(fx, md)
            if c:
                all_cands.append(c)

    if any((c.get("fixture_id") is None) for c in all_cands):
        bad = [c for c in all_cands if c.get("fixture_id") is None][:3]
        raise ValueError(f"Some candidates have missing fixture_id. Examples: {bad}")

    core_cands = [c for c in all_cands if (q_by_id.get(c["fixture_id"], 0.0) >= core_min_q)]
    fun_cands  = [c for c in all_cands if (q_by_id.get(c["fixture_id"], 0.0) >= fun_min_q)]

    core_singles, core_double, core_open = _select_core(core_cands, strict_ok_by_id)
    fun_pool, fun_system, fun_open = _select_fun_system(fun_cands, fun_stake_total)
    draw_pool, draw_system, draw_open = _select_draw_top5(all_cands, draw_stake_total)

    window = th.get("window") or {
        "from": str(th.get("start_date") or ""),
        "to": str(th.get("end_date") or ""),
        "hours": int(th.get("window_hours") or 72),
    }

    out = {
        "title": f"Bombay Friday — Week {wk}",
        "week": wk,
        "generated_at": _utc_now_iso(),
        "window": window,

        "history": {
            "loaded": bool(hist_ok),
            "path": str(hist_path),
            "week_count_in_history": hist_week,
            "bankroll_current": {"core": core_bankroll, "fun": fun_bankroll, "draw": draw_bankroll},
        },

        "core": {
            "label": "CoreBet",
            "bankroll_start": float(core_bankroll),
            "max_singles": _si_env("FRIDAY_CORE_TARGET_SINGLES", 7),
            "stake_rule": "odds<=2.00 -> 40, else -> 30 (NO scaling)",
            "singles": core_singles,
            "double": core_double,
            "doubles": ([core_double] if core_double else []),
            "open": round(core_open, 2),
            "after_open": round(float(core_bankroll) - core_open, 2),
        },

        "funbet": {
            "label": "FunBet",
            "bankroll_start": float(fun_bankroll),
            "system_pool": fun_pool,
            "system": fun_system,
            "open": round(fun_open, 2),
            "after_open": round(float(fun_bankroll) - fun_open, 2),
        },

        "drawbet": {
            "label": "DrawBet",
            "bankroll_start": float(draw_bankroll),
            "system_pool": draw_pool,
            "system": draw_system,
            "open": round(draw_open, 2),
            "after_open": round(float(draw_bankroll) - draw_open, 2),
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
            },
        },
    }

    _assert_friday_sanity(out)
    return out


def main() -> int:
    try:
        friday = build_friday_shortlist()

        # canonical v3 path (THIS is what your API endpoint expects)
        canonical = LOGS_DIR / "friday_shortlist_v3.json"
        _write_json(canonical, friday)

        # archive
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        _write_json(LOGS_DIR / f"friday_{ts}.json", friday)

        print(json.dumps({
            "status": "ok",
            "saved": str(canonical),
            "generated_at": friday["generated_at"],
            "week": friday.get("week"),
            "history_loaded": (friday.get("history") or {}).get("loaded", False),
            "history_path": (friday.get("history") or {}).get("path", ""),
            "counts": (friday.get("debug") or {}).get("counts", {}),
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
