# src/analysis/friday_shortlist_v3.py
"""
Friday shortlist v3 — env-driven + history-first bankrolls + NO SCALING + TOP-5 DRAWS
+ chronological ordering + fixture_id everywhere + copy_play.

Key fixes:
- Bankrolls come from tuesday_history_v3.json FIRST (continuity), env starts are fallback only.
  Override only if FRIDAY_FORCE_ENV_BANKROLL=true.
- Core target singles is enforced with controlled fallback (relax strict_ok only, then small prob relax).
- Fun pool tries to fill to pool_size with a 2nd pass if caps/uniques block it.
- Adds copy_play field for easy copy/paste.

Still guarantees:
- No stake scaling. Ever.
- Core stake rule: odds <= 2.00 -> 40, else -> 30
- DrawBet tempo filter: total_lambda <= FRIDAY_DRAW_MAX_TOTAL_LAMBDA (default 2.90)
- Core totals cap: FRIDAY_CORE_MAX_TOTALS (default 4) and under cap: FRIDAY_CORE_MAX_UNDERS (default 1)
- Under guard: confidence>=FRIDAY_CORE_UNDER_MIN_CONF AND prob_instability<=FRIDAY_CORE_UNDER_MAX_INSTAB
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

def _load_latest_thursday_report() -> Dict[str, Any]:
    p1 = LOGS_DIR / "thursday_report_v3.json"
    if p1.exists():
        return _read_json(p1)

    th_files = sorted(LOGS_DIR.glob("thursday_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if th_files:
        return _read_json(th_files[0])

    raise FileNotFoundError(f"Could not find thursday_report_v3.json or any thursday_*.json inside {LOGS_DIR}")

def _load_history_bankrolls() -> Tuple[Optional[Dict[str, float]], str]:
    """
    Returns (bankrolls, source_string)
    bankrolls format: {"core":..., "fun":..., "draw":...}
    """
    # allow explicit path override
    env_path = os.getenv("TUESDAY_HISTORY_PATH", "").strip()
    candidates: List[Path] = []
    if env_path:
        candidates.append(Path(env_path))

    # standard names in logs/
    candidates += [
        LOGS_DIR / "tuesday_history_v3_next.json",
        LOGS_DIR / "tuesday_history_v3.json",
    ]

    for p in candidates:
        try:
            if p.exists():
                js = _read_json(p)
                bc = js.get("bankroll_current") if isinstance(js, dict) else None
                if isinstance(bc, dict):
                    core = _safe_float(bc.get("core"))
                    fun = _safe_float(bc.get("fun"))
                    draw = _safe_float(bc.get("draw"))
                    if core is not None and fun is not None and draw is not None:
                        return {"core": float(core), "fun": float(fun), "draw": float(draw)}, f"history:{p.name}"
        except Exception:
            continue

    return None, "history:missing"

def _bankroll_starts() -> Tuple[float, float, float, str]:
    """
    History-first bankrolls (continuity).
    Env bankrolls are fallback ONLY unless FRIDAY_FORCE_ENV_BANKROLL=true.
    """
    force_env = _sb_env("FRIDAY_FORCE_ENV_BANKROLL", False)
    hist, src = _load_history_bankrolls()
    if (not force_env) and hist:
        return hist["core"], hist["fun"], hist["draw"], src

    core = _sf_env("CORE_BANKROLL_START", 800.0)
    fun  = _sf_env("FUN_BANKROLL_START", 400.0)
    draw = _sf_env("DRAW_BANKROLL_START", 300.0)
    return core, fun, draw, ("env" if force_env else "env:fallback")


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

    qr = fixture_quality_score(fx)
    total_lambda = _safe_float(fx.get("total_lambda"))

    flags = (fx.get("flags") or {})
    conf = _safe_float(flags.get("confidence"))
    inst = _safe_float(flags.get("prob_instability"))
    strict_ok = bool(flags.get("odds_strict_ok") is True)

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
        # extra signals
        "total_lambda": total_lambda,
        "confidence": conf,
        "prob_instability": inst,
        "strict_ok": strict_ok,
    }


# -------------------------
# Stake rules (NO scaling)
# -------------------------

def core_stake_from_odds(odds: float) -> int:
    return 40 if float(odds) <= 2.00 else 30


# -------------------------
# Guardrails
# -------------------------

def _under_guard_ok(c: Dict[str, Any]) -> bool:
    conf_min = _sf_env("FRIDAY_CORE_UNDER_MIN_CONF", 0.65)
    inst_max = _sf_env("FRIDAY_CORE_UNDER_MAX_INSTAB", 0.18)
    conf = _safe_float(c.get("confidence"))
    inst = _safe_float(c.get("prob_instability"))
    if conf is None or inst is None:
        return False
    return (conf >= conf_min) and (inst <= inst_max)


# -------------------------
# Selection logic
# -------------------------

def _select_core(cands: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], float, Dict[str, Any]]:
    odds_min = _sf_env("FRIDAY_CORE_ODDS_MIN", 1.55)
    odds_max = _sf_env("FRIDAY_CORE_ODDS_MAX", 2.20)
    min_prob = _sf_env("FRIDAY_CORE_MIN_PROB", 0.50)
    min_val  = _sf_env("FRIDAY_CORE_MIN_VALUE_PCT", 1.0)

    target_singles = _si_env("FRIDAY_CORE_TARGET_SINGLES", 7)
    max_totals = _si_env("FRIDAY_CORE_MAX_TOTALS", 4)
    max_unders = _si_env("FRIDAY_CORE_MAX_UNDERS", 1)
    require_strict = _sb_env("FRIDAY_CORE_REQUIRE_STRICT_OK", True)

    # controlled fallback knobs (tiny, not chaos)
    fallback_min_prob = _sf_env("FRIDAY_CORE_FALLBACK_MIN_PROB", 0.48)

    debug = {"fill_mode": "base", "fallback_used": False}

    def eligible_list(require_strict_local: bool, min_prob_local: float) -> List[Dict[str, Any]]:
        core = [c for c in cands if c["market"] != "Draw"]
        core = [c for c in core if odds_min <= float(c["odds"]) <= odds_max and float(c["prob"]) >= min_prob_local and float(c["value_pct"]) >= min_val]
        if require_strict_local:
            core = [c for c in core if bool(c.get("strict_ok", False))]
        core.sort(key=lambda x: (float(x["rank_score"]), 1 if x["market"] == "Over 2.5" else 0), reverse=True)
        return core

    # try base first
    core = eligible_list(require_strict, min_prob)

    def build_from(core_list: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], float]:
        singles: List[Dict[str, Any]] = []
        open_total = 0.0
        totals_cnt = 0
        unders_cnt = 0

        # PASS 1: totals first (up to cap), under guard + cap
        for c in core_list:
            if len(singles) >= target_singles:
                break
            if not _is_total_market(c["market"]):
                continue
            if totals_cnt >= max_totals:
                continue

            if _is_under(c["market"]):
                if unders_cnt >= max_unders:
                    continue
                if not _under_guard_ok(c):
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

        # PASS 2: fill with 1X2 (Home/Away), avoid duplicate fixture_id
        for c in core_list:
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

        singles.sort(key=_chrono_key)
        return singles, float(round(open_total, 2))

    singles, open_total = build_from(core)

    # FALLBACK 1: relax strict_ok only (still same thresholds)
    if len(singles) < target_singles and require_strict:
        debug["fill_mode"] = "fallback_relax_strict"
        debug["fallback_used"] = True
        core2 = eligible_list(False, min_prob)
        singles, open_total = build_from(core2)

    # FALLBACK 2: tiny relax prob (still no chaos)
    if len(singles) < target_singles:
        debug["fill_mode"] = "fallback_relax_prob"
        debug["fallback_used"] = True
        core3 = eligible_list(False, fallback_min_prob)
        singles, open_total = build_from(core3)

    # Core double: same rule
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
        open_total = float(round(open_total + 20, 2))

    return singles, core_double, open_total, debug


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

    def try_add(c: Dict[str, Any], enforce_caps: bool) -> bool:
        nonlocal totals_cnt, unders_cnt
        if len(system_pool) >= pool_size:
            return False
        fid = c.get("fixture_id")
        if fid in seen_fixtures:
            return False

        if enforce_caps and _is_total_market(c["market"]):
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
        if _is_total_market(c["market"]):
            totals_cnt += 1
            if _is_under(c["market"]):
                unders_cnt += 1
        return True

    # PASS 1: with caps
    for c in pool:
        if len(system_pool) >= pool_size:
            break
        try_add(c, enforce_caps=True)

    # PASS 2: fill to 7 even if caps block it (still unique fixtures)
    for c in pool:
        if len(system_pool) >= pool_size:
            break
        try_add(c, enforce_caps=False)

    system = {"k": 4, "n": pool_size, "columns": 35, "min_hits": 4, "stake": float(stake_total)}
    system_pool.sort(key=_chrono_key)
    open_amt = float(stake_total) if system_pool else 0.0
    return system_pool, system, float(round(open_amt, 2))


def _select_draw_top5(all_cands: List[Dict[str, Any]], stake_total: float) -> Tuple[List[Dict[str, Any]], Dict[str, Any], float]:
    dmin = _sf_env("FRIDAY_DRAW_ODDS_MIN", 3.00)
    dmax = _sf_env("FRIDAY_DRAW_ODDS_MAX", 5.50)
    max_tl = _sf_env("FRIDAY_DRAW_MAX_TOTAL_LAMBDA", 2.90)

    draws = [
        c for c in all_cands
        if c["market"] == "Draw"
        and c.get("prob") is not None
        and c.get("odds") is not None
        and c.get("total_lambda") is not None
    ]
    draws = [c for c in draws if dmin <= float(c["odds"]) <= dmax and float(c["total_lambda"]) <= max_tl]
    draws.sort(key=lambda x: (float(x["prob"]), float(x["odds"])), reverse=True)
    draws = draws[:5]

    system = {"k": 2, "n": 5, "columns": 10, "min_hits": 2, "stake": float(stake_total), "mode": "top5_by_probability", "tempo_filter": f"total_lambda<={max_tl}"}

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
# Sanity checks
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


# -------------------------
# copy_play
# -------------------------

def _build_copy_play(friday: Dict[str, Any]) -> List[str]:
    lines: List[str] = []

    core = (friday.get("core", {}) or {}).get("singles", []) or []
    for s in core:
        lines.append(f"CORE {s['stake']}€ | {s['date']} {s['time_gr']} | {s['league']} | {s['match']} | {s['market']} @ {s['odds']}")

    cd = (friday.get("core", {}) or {}).get("double")
    if isinstance(cd, dict) and cd.get("legs"):
        lines.append(f"CORE DOUBLE {cd.get('stake')}€ | ODDS {cd.get('odds')}")

    fun = (friday.get("funbet", {}) or {}).get("system_pool", []) or []
    if fun:
        lines.append(f"FUN SYSTEM 4/7 | STAKE {((friday.get('funbet', {}) or {}).get('system') or {}).get('stake')}€")
        for s in fun:
            lines.append(f"FUN | {s['date']} {s['time_gr']} | {s['league']} | {s['match']} | {s['market']} @ {s['odds']}")

    dr = (friday.get("drawbet", {}) or {}).get("system_pool", []) or []
    if dr:
        lines.append(f"DRAW SYSTEM 2/5 | STAKE {((friday.get('drawbet', {}) or {}).get('system') or {}).get('stake')}€")
        for s in dr:
            lines.append(f"DRAW | {s['date']} {s['time_gr']} | {s['league']} | {s['match']} | Draw @ {s['odds']}")

    return lines


# -------------------------
# Main builder
# -------------------------

def build_friday_shortlist() -> Dict[str, Any]:
    th = _load_latest_thursday_report()
    fixtures = th.get("fixtures") or []
    if not isinstance(fixtures, list):
        raise ValueError("Thursday report has no fixtures list.")

    core_min_q = _sf_env("FRIDAY_CORE_MIN_QUALITY", 0.70)
    fun_min_q  = _sf_env("FRIDAY_FUN_MIN_QUALITY", 0.60)

    core_bankroll, fun_bankroll, draw_bankroll, bankroll_src = _bankroll_starts()

    fun_stake_total  = _sf_env("FUN_SYSTEM_STAKE", 35.0)
    draw_stake_total = _sf_env("DRAW_SYSTEM_STAKE", 25.0)

    all_cands: List[Dict[str, Any]] = []
    q_by_id: Dict[Any, float] = {}

    for fx in fixtures:
        fid = fx.get("fixture_id")
        qr = fixture_quality_score(fx)
        q_by_id[fid] = float(qr.score)

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

    core_singles, core_double, core_open, core_debug = _select_core(core_cands)
    fun_pool, fun_system, fun_open = _select_fun_system(fun_cands, fun_stake_total)
    draw_pool, draw_system, draw_open = _select_draw_top5(all_cands, draw_stake_total)

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
            "max_singles": _si_env("FRIDAY_CORE_TARGET_SINGLES", 7),
            "stake_rule": "odds<=2.00 -> 40, else -> 30 (NO scaling)",
            "singles": core_singles,
            "double": core_double,
            "doubles": ([core_double] if core_double else []),
            "open": round(core_open, 2),
            "after_open": round(core_bankroll - core_open, 2),
            "debug": core_debug,
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
            "bankroll_source": bankroll_src,
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

    out["copy_play"] = _build_copy_play(out)

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
            "bankroll_source": friday["debug"]["bankroll_source"],
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
