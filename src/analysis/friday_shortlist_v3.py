# src/analysis/friday_shortlist_v3.py
"""
Friday shortlist v3 — history-aware + env-driven + NO SCALING + TOP-5 DRAWS + chronological ordering + fixture_id everywhere.

Fixes:
- Reads bankrolls + week_count from Tuesday history (HISTORY_PATH preferred).
- Core/Fun "fill pass" so we aim to reach targets instead of stopping at 4/6.
- Fun system auto-adjusts n/columns if pool < desired (never prints n=7 with 6 picks).
- Keeps existing Thursday schema + quality gate behavior.

History:
- HISTORY_PATH (preferred) or TUESDAY_HISTORY_PATH (fallback)
- Default path: logs/tuesday_history_v3.json
- If FRIDAY_REQUIRE_HISTORY=true and file missing -> hard error
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from math import comb


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
# History (Tuesday bankroll continuity)
# -------------------------

def _history_path() -> Path:
    # one canonical variable going forward: HISTORY_PATH
    hp = os.getenv("HISTORY_PATH", "").strip()
    if hp:
        return Path(hp)

    # backward-compat fallback
    hp2 = os.getenv("TUESDAY_HISTORY_PATH", "").strip()
    if hp2:
        return Path(hp2)

    return LOGS_DIR / "tuesday_history_v3.json"

def _load_history() -> Optional[Dict[str, Any]]:
    path = _history_path()
    if path.exists():
        return _read_json(path)
    return None

def _history_bankrolls_or_env() -> Tuple[float, float, float, Optional[int], Optional[str], Dict[str, Any]]:
    """
    Returns: (core, fun, draw, next_week, as_of, meta)
    - If history exists: use bankroll_current + week_count
    - Else: fallback to CORE_BANKROLL_START etc
    """
    require = _sb_env("FRIDAY_REQUIRE_HISTORY", True)
    hist = _load_history()
    if hist is None:
        if require:
            raise FileNotFoundError(f"History file missing at: {_history_path()}")
        # fallback
        core = _sf_env("CORE_BANKROLL_START", 800.0)
        fun  = _sf_env("FUN_BANKROLL_START", 400.0)
        draw = _sf_env("DRAW_BANKROLL_START", 300.0)
        return core, fun, draw, None, None, {"history_used": False, "history_path": str(_history_path())}

    bc = (hist.get("bankroll_current") or {})
    core = float(bc.get("core", _sf_env("CORE_BANKROLL_START", 800.0)))
    fun  = float(bc.get("fun",  _sf_env("FUN_BANKROLL_START", 400.0)))
    draw = float(bc.get("draw", _sf_env("DRAW_BANKROLL_START", 300.0)))

    week_count = hist.get("week_count")
    next_week = None
    try:
        if week_count is not None:
            next_week = int(week_count) + 1
    except Exception:
        next_week = None

    as_of = str(hist.get("as_of") or "")
    return core, fun, draw, next_week, as_of, {
        "history_used": True,
        "history_path": str(_history_path()),
        "history_week_count": hist.get("week_count"),
        "history_as_of": as_of,
        "history_note": str(hist.get("note") or ""),
    }


# -------------------------
# Stake rules (NO scaling)
# -------------------------

def core_stake_from_odds(odds: float) -> int:
    return 40 if float(odds) <= 2.00 else 30


# -------------------------
# Selection logic
# -------------------------

def _select_core(cands: List[Dict[str, Any]], strict_ok_by_id: Dict[Any, bool]) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], float, Dict[str, Any]]:
    odds_min = _sf_env("FRIDAY_CORE_ODDS_MIN", 1.55)
    odds_max = _sf_env("FRIDAY_CORE_ODDS_MAX", 2.20)
    min_prob = _sf_env("FRIDAY_CORE_MIN_PROB", 0.50)
    min_val  = _sf_env("FRIDAY_CORE_MIN_VALUE_PCT", 1.0)

    target_singles = _si_env("FRIDAY_CORE_TARGET_SINGLES", 7)
    max_totals = _si_env("FRIDAY_CORE_MAX_TOTALS", 4)
    max_unders = _si_env("FRIDAY_CORE_MAX_UNDERS", 1)
    require_strict = _sb_env("FRIDAY_CORE_REQUIRE_STRICT_OK", True)

    fill_mode = _sb_env("FRIDAY_CORE_FILL_MODE", True)

    base = [c for c in cands if c["market"] != "Draw"]
    base = [c for c in base if odds_min <= float(c["odds"]) <= odds_max]

    def _apply_filters(lst: List[Dict[str, Any]], mp: float, mv: float, strict: bool) -> List[Dict[str, Any]]:
        out = [c for c in lst if float(c["prob"]) >= mp and float(c["value_pct"]) >= mv]
        if strict:
            out = [c for c in out if bool(strict_ok_by_id.get(c.get("fixture_id"), False))]
        out.sort(key=lambda x: (float(x["rank_score"]), 1 if x["market"] == "Over 2.5" else 0), reverse=True)
        return out

    # Pass ladder: strict -> relaxed strict -> relaxed value -> relaxed prob (small)
    passes = []
    passes.append(("strict", min_prob, min_val, require_strict))
    if fill_mode:
        passes.append(("no_strict", min_prob, min_val, False))
        passes.append(("low_value", min_prob, 0.0, False))
        passes.append(("low_prob", max(0.45, min_prob - 0.05), 0.0, False))

    singles: List[Dict[str, Any]] = []
    open_total = 0.0
    totals_cnt = 0
    unders_cnt = 0

    used_fids = set()

    def _try_add(c: Dict[str, Any]) -> bool:
        nonlocal open_total, totals_cnt, unders_cnt
        if len(singles) >= target_singles:
            return False
        fid = c.get("fixture_id")
        if fid in used_fids:
            return False

        # caps for totals
        if _is_total_market(c["market"]):
            if totals_cnt >= max_totals:
                return False
            if _is_under(c["market"]) and unders_cnt >= max_unders:
                return False

        stake = core_stake_from_odds(float(c["odds"]))
        singles.append({
            "fixture_id": fid,
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
        used_fids.add(fid)
        open_total += stake

        if _is_total_market(c["market"]):
            totals_cnt += 1
            if _is_under(c["market"]):
                unders_cnt += 1
        return True

    # Phase A: fill totals first (prefer Over), then 1X2
    for tag, mp, mv, strict in passes:
        pool = _apply_filters(base, mp, mv, strict)

        # A1 totals
        for c in pool:
            if len(singles) >= target_singles:
                break
            if not _is_total_market(c["market"]):
                continue
            _try_add(c)

        # A2 1X2 (Home/Away only)
        for c in pool:
            if len(singles) >= target_singles:
                break
            if _is_total_market(c["market"]) or c["market"] == "Draw":
                continue
            _try_add(c)

        if len(singles) >= target_singles:
            break

    # Core double (unchanged)
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

    dbg = {
        "target_singles": target_singles,
        "got_singles": len(singles),
        "fill_mode": fill_mode,
        "totals_cnt": totals_cnt,
        "unders_cnt": unders_cnt,
        "require_strict_default": require_strict,
    }
    return singles, core_double, float(round(open_total, 2)), dbg


def _select_fun_system(cands: List[Dict[str, Any]], stake_total: float, core_fixture_ids: set) -> Tuple[List[Dict[str, Any]], Dict[str, Any], float, Dict[str, Any]]:
    desired_n = _si_env("FRIDAY_FUN_POOL_SIZE", 7)
    k = _si_env("FRIDAY_FUN_K", 4)

    # thresholds (env-driven)
    odds_min = _sf_env("FRIDAY_FUN_ODDS_MIN", 2.00)
    odds_max = _sf_env("FRIDAY_FUN_ODDS_MAX", 3.30)
    min_prob = _sf_env("FRIDAY_FUN_MIN_PROB", 0.40)
    min_val  = _sf_env("FRIDAY_FUN_MIN_VALUE_PCT", 1.0)
    min_q    = _sf_env("FRIDAY_FUN_MIN_QUALITY", 0.60)

    max_totals = _si_env("FRIDAY_FUN_MAX_TOTALS", 4)
    max_unders = _si_env("FRIDAY_FUN_MAX_UNDERS", 1)
    max_overlap = _si_env("FRIDAY_FUN_MAX_OVERLAP_WITH_CORE", 2)

    fill_mode = _sb_env("FRIDAY_FUN_FILL_MODE", True)

    base = [c for c in cands if c["market"] != "Draw" and float(c.get("quality", 0.0)) >= min_q]
    base = [c for c in base if odds_min <= float(c["odds"]) <= odds_max]

    def _rank(lst: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        lst.sort(key=lambda x: (float(x["rank_score"]), 1 if x["market"] == "Over 2.5" else 0), reverse=True)
        return lst

    def _passes() -> List[Tuple[str, float, float]]:
        out = [("base", min_prob, min_val)]
        if fill_mode:
            out += [
                ("low_value", min_prob, 0.0),
                ("low_prob", max(0.35, min_prob - 0.05), 0.0),
            ]
        return out

    system_pool: List[Dict[str, Any]] = []
    totals_cnt = 0
    unders_cnt = 0
    seen_fixtures = set()
    overlap_cnt = 0

    def _try_add(c: Dict[str, Any]) -> bool:
        nonlocal totals_cnt, unders_cnt, overlap_cnt

        if len(system_pool) >= desired_n:
            return False
        fid = c.get("fixture_id")
        if fid in seen_fixtures:
            return False

        # overlap guard
        if fid in core_fixture_ids:
            if overlap_cnt >= max_overlap:
                return False

        # totals caps
        if _is_total_market(c["market"]):
            if totals_cnt >= max_totals:
                return False
            if _is_under(c["market"]) and unders_cnt >= max_unders:
                return False

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

        if fid in core_fixture_ids:
            overlap_cnt += 1

        if _is_total_market(c["market"]):
            totals_cnt += 1
            if _is_under(c["market"]):
                unders_cnt += 1
        return True

    for tag, mp, mv in _passes():
        pool = [c for c in base if float(c["prob"]) >= mp and float(c["value_pct"]) >= mv]
        pool = _rank(pool)

        for c in pool:
            if len(system_pool) >= desired_n:
                break
            _try_add(c)

        if len(system_pool) >= desired_n:
            break

    # AUTO-ADJUST system n/columns to what we actually have
    n_actual = len(system_pool)
    k_actual = min(k, n_actual) if n_actual > 0 else k
    columns = comb(n_actual, k_actual) if (n_actual > 0 and k_actual > 0 and n_actual >= k_actual) else 0

    system = {
        "k": k_actual,
        "n": n_actual,
        "columns": columns,
        "min_hits": k_actual,
        "stake": float(stake_total),
        "requested_n": desired_n,
    }

    system_pool.sort(key=_chrono_key)
    open_amt = float(stake_total) if system_pool else 0.0

    dbg = {
        "requested_n": desired_n,
        "got_n": n_actual,
        "k_requested": k,
        "k_used": k_actual,
        "fill_mode": fill_mode,
        "overlap_with_core": overlap_cnt,
        "totals_cnt": totals_cnt,
        "unders_cnt": unders_cnt,
    }

    return system_pool, system, float(round(open_amt, 2)), dbg


def _select_draw_top5(all_cands: List[Dict[str, Any]], stake_total: float) -> Tuple[List[Dict[str, Any]], Dict[str, Any], float]:
    dmin = _sf_env("FRIDAY_DRAW_ODDS_MIN", 3.00)
    dmax = _sf_env("FRIDAY_DRAW_ODDS_MAX", 5.50)

    use_tempo = _sb_env("FRIDAY_DRAW_USE_TEMPO_FILTER", True)
    max_lambda = _sf_env("FRIDAY_DRAW_MAX_TOTAL_LAMBDA", 2.9)

    draws = [c for c in all_cands if c["market"] == "Draw" and c.get("prob") is not None and c.get("odds") is not None]
    draws = [c for c in draws if dmin <= float(c["odds"]) <= dmax]

    # tempo filter if Thursday put total_lambda into fixture (optional). If not present, we don't block.
    if use_tempo:
        filtered = []
        for c in draws:
            # candidates don't carry total_lambda, but fixture might have it; Thursday sometimes copies it into candidate debug.
            # If you already add it in Thursday fixture, you can pass it through here by extending _candidate_from_fixture.
            # For now: accept all (best effort) – no hard block without data.
            filtered.append(c)
        draws = filtered

    draws.sort(key=lambda x: (float(x["prob"]), float(x["odds"])), reverse=True)
    draws = draws[:5]

    system = {"k": 2, "n": len(draws), "columns": comb(len(draws), 2) if len(draws) >= 2 else 0, "min_hits": 2, "stake": float(stake_total), "mode": "top5_by_probability"}

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
# Main builder
# -------------------------

def build_friday_shortlist() -> Dict[str, Any]:
    th = _load_latest_thursday_report()
    fixtures = th.get("fixtures") or []
    if not isinstance(fixtures, list):
        raise ValueError("Thursday report has no fixtures list.")

    core_min_q = _sf_env("FRIDAY_CORE_MIN_QUALITY", 0.70)

    # bankroll continuity FROM HISTORY
    core_bankroll, fun_bankroll, draw_bankroll, next_week, hist_as_of, hist_meta = _history_bankrolls_or_env()

    fun_stake_total  = _sf_env("FUN_SYSTEM_STAKE", 35.0)
    draw_stake_total = _sf_env("DRAW_SYSTEM_STAKE", 25.0)

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

    if any((c.get("fixture_id") is None) for c in all_cands):
        bad = [c for c in all_cands if c.get("fixture_id") is None][:3]
        raise ValueError(f"Some candidates have missing fixture_id. Examples: {bad}")

    core_cands = [c for c in all_cands if (q_by_id.get(c["fixture_id"], 0.0) >= core_min_q)]
    fun_cands  = [c for c in all_cands]  # fun gating happens inside selector (min quality is checked there)

    core_singles, core_double, core_open, core_dbg = _select_core(core_cands, strict_ok_by_id)
    core_fids = {s.get("fixture_id") for s in core_singles if s.get("fixture_id") is not None}

    fun_pool, fun_system, fun_open, fun_dbg = _select_fun_system(fun_cands, fun_stake_total, core_fids)
    draw_pool, draw_system, draw_open = _select_draw_top5(all_cands, draw_stake_total)

    window = th.get("window") or {
        "from": str(th.get("start_date") or ""),
        "to": str(th.get("end_date") or ""),
        "hours": int(th.get("window_hours") or 72),
    }

    title = "Bombay Friday — Week"
    if next_week is not None:
        title = f"Bombay Friday — Week {next_week}"

    out = {
        "title": title,
        "generated_at": _utc_now_iso(),
        "window": window,

        "history": {
            **hist_meta,
            "bankroll_start_from_history": {"core": core_bankroll, "fun": fun_bankroll, "draw": draw_bankroll},
            "next_week": next_week,
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
            "debug": core_dbg,
        },

        "funbet": {
            "label": "FunBet",
            "bankroll_start": float(fun_bankroll),
            "system_pool": fun_pool,
            "system": fun_system,
            "open": round(fun_open, 2),
            "after_open": round(float(fun_bankroll) - fun_open, 2),
            "debug": fun_dbg,
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
            "title": friday.get("title"),
            "history": friday.get("history", {}),
            "counts": friday["debug"]["counts"],
            "logs_dir": str(LOGS_DIR),
        }))
        return 0

    except Exception as e:
        print(json.dumps({
            "status": "error",
            "error": str(e),
            "history_path": str(_history_path()),
            "logs_dir": str(LOGS_DIR),
        }))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
