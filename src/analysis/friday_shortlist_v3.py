# src/analysis/friday_shortlist_v3.py
"""
Friday shortlist v4 — history-aware + style-aware + tiered Core/Fun + SuperFun from Core+Fun.

What it does
------------
- Reads bankrolls + week_count from Tuesday history.
- Builds CoreBet and FunBet from Thursday report.
- Uses more of the Thursday data directly in selection:
  - odds_match grade
  - tight-game filter
  - team_value_ratio filter
  - tempo/style validation for Over
  - slow-favourite trap downgrade
- Core is tiered:
  - STRONG -> 40 stake
  - TRB    -> 30 stake
- Fun is tiered:
  - A / B
- SuperFun is built ONLY from Core + Fun (no extra selection).
- Core always has priority in SuperFun.
- SuperFun targets:
  - 12 -> 8/12
  - 11 -> 7/11
  - 10 -> 6/10
  - never below 10 picks

User-agreed ranges
------------------
- Core: 1.55 -> 1.90
- Fun : 1.80 -> 2.40
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from math import comb
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
# Quality gate import
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


def _ss_env(name: str, default: str) -> str:
    v = os.getenv(name, "").strip()
    return v if v else default


def _fmt_date(date_str: str) -> str:
    return str(date_str or "")


def _fmt_time(time_str: str) -> str:
    return str(time_str or "")


def _chrono_key(item: Dict[str, Any]) -> Tuple[str, str]:
    return (str(item.get("date") or ""), str(item.get("time_gr") or ""))


def _prob_points(prob: float) -> float:
    return (float(prob) - 0.50) * 100.0


def _rank_score(value_pct: float, prob: float) -> float:
    wv = _sf_env("FRIDAY_RANK_W_VALUE", 0.60)
    wp = _sf_env("FRIDAY_RANK_W_PROB", 0.40)
    if wv < 0:
        wv = 0.0
    if wp < 0:
        wp = 0.0
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


def _prob_gap(fx: Dict[str, Any]) -> float:
    hp = _safe_float(fx.get("home_prob")) or 0.0
    ap = _safe_float(fx.get("away_prob")) or 0.0
    return abs(hp - ap)


def _odds_grade_rank(grade: str) -> int:
    g = (grade or "").strip().upper()
    return {"A": 4, "B": 3, "C": 2, "D": 1}.get(g, 0)


def _grade_at_least(grade: str, minimum: str) -> bool:
    return _odds_grade_rank(grade) >= _odds_grade_rank(minimum)


def _normalize_team_name(name: str) -> str:
    s = (name or "").strip().lower()
    repl = {
        "á": "a", "à": "a", "ä": "a", "â": "a",
        "é": "e", "è": "e", "ë": "e", "ê": "e",
        "í": "i", "ì": "i", "ï": "i", "î": "i",
        "ó": "o", "ò": "o", "ö": "o", "ô": "o",
        "ú": "u", "ù": "u", "ü": "u", "û": "u",
        "ç": "c", "ñ": "n",
    }
    for a, b in repl.items():
        s = s.replace(a, b)
    s = s.replace(".", " ").replace("-", " ")
    s = s.replace(" fc ", " ").replace(" cf ", " ").replace(" ac ", " ")
    s = " ".join(s.split())
    return s


def _style_path() -> Path:
    sp = os.getenv("TEAM_STYLE_METRICS_PATH", "").strip()
    if sp:
        return Path(sp)
    return LOGS_DIR / "team_style_metrics.json"


@lru_cache(maxsize=1)
def _load_style_metrics() -> Dict[str, Any]:
    path = _style_path()
    if not path.exists():
        return {}

    try:
        data = _read_json(path)
    except Exception:
        return {}

    out: Dict[str, Any] = {}

    if isinstance(data, dict):
        teams = data.get("teams")
        if isinstance(teams, list):
            for row in teams:
                if not isinstance(row, dict):
                    continue
                team = str(row.get("team") or row.get("name") or "").strip()
                if team:
                    out[_normalize_team_name(team)] = row
        else:
            for k, v in data.items():
                if isinstance(v, dict):
                    out[_normalize_team_name(str(k))] = v

    elif isinstance(data, list):
        for row in data:
            if not isinstance(row, dict):
                continue
            team = str(row.get("team") or row.get("name") or "").strip()
            if team:
                out[_normalize_team_name(team)] = row

    return out


def _team_style_row(team_name: str) -> Dict[str, Any]:
    return _load_style_metrics().get(_normalize_team_name(team_name), {})


def _fixture_style_profile(fx: Dict[str, Any]) -> Dict[str, Any]:
    home = str(fx.get("home") or "")
    away = str(fx.get("away") or "")
    flags = fx.get("flags") or {}

    home_style = _team_style_row(home)
    away_style = _team_style_row(away)

    tempo_home = _safe_float(fx.get("tempo_index_home")) or 0.0
    tempo_away = _safe_float(fx.get("tempo_index_away")) or 0.0
    tempo_total = _safe_float(fx.get("tempo_total_mult")) or 1.0

    home_block = _safe_float(home_style.get("defensive_block"))
    away_block = _safe_float(away_style.get("defensive_block"))
    home_attack = _safe_float(home_style.get("attacking_intensity"))
    away_attack = _safe_float(away_style.get("attacking_intensity"))
    home_transition = _safe_float(home_style.get("transition_speed"))
    away_transition = _safe_float(away_style.get("transition_speed"))

    # safe fallbacks from Thursday flags / tempo
    if home_block is None:
        home_block = 0.70 if flags.get("home_shape") is False else 0.50
    if away_block is None:
        away_block = 0.70 if flags.get("away_shape") is False else 0.50
    if home_attack is None:
        home_attack = 0.70 if flags.get("home_shape") else 0.50
    if away_attack is None:
        away_attack = 0.70 if flags.get("away_shape") else 0.50
    if home_transition is None:
        home_transition = tempo_home
    if away_transition is None:
        away_transition = tempo_away

    return {
        "tempo_home": float(tempo_home),
        "tempo_away": float(tempo_away),
        "tempo_total": float(tempo_total),
        "home_block": float(home_block),
        "away_block": float(away_block),
        "home_attack": float(home_attack),
        "away_attack": float(away_attack),
        "home_transition": float(home_transition),
        "away_transition": float(away_transition),
    }


def _over_style_ok(fx: Dict[str, Any]) -> bool:
    sp = _fixture_style_profile(fx)
    over_prob = _safe_float(fx.get("over_2_5_prob")) or 0.0

    if sp["tempo_total"] >= 1.02:
        return True
    if over_prob >= 0.60:
        return True
    if sp["tempo_total"] >= 1.00 and (sp["tempo_home"] + sp["tempo_away"]) >= 4.20:
        return True

    return False


def _slow_favourite_trap(fx: Dict[str, Any], market: str) -> bool:
    if market not in ("Home (1)", "Away (2)"):
        return False

    sp = _fixture_style_profile(fx)
    home_prob = _safe_float(fx.get("home_prob")) or 0.0
    away_prob = _safe_float(fx.get("away_prob")) or 0.0

    if market == "Home (1)" and home_prob >= 0.50:
        return (
            sp["tempo_home"] < 2.00 and
            sp["tempo_total"] < 1.00 and
            sp["away_block"] >= 0.70
        )

    if market == "Away (2)" and away_prob >= 0.50:
        return (
            sp["tempo_away"] < 2.00 and
            sp["tempo_total"] < 1.00 and
            sp["home_block"] >= 0.70
        )

    return False


def _passes_tight_game_filter(fx: Dict[str, Any], market: str, mode: str) -> bool:
    if market not in ("Home (1)", "Away (2)"):
        return True

    gap = _prob_gap(fx)
    if mode == "core":
        return gap >= 0.12
    return gap >= 0.07


def _passes_team_value_filter(fx: Dict[str, Any], market: str, value_pct: float) -> bool:
    if market not in ("Home (1)", "Away (2)"):
        return True

    ratio = _safe_float(fx.get("team_value_ratio")) or 1.0
    hp = _safe_float(fx.get("home_prob")) or 0.0
    ap = _safe_float(fx.get("away_prob")) or 0.0

    # Only filter underdog 1X2 bets when squad-value gap is large.
    is_underdog = False
    if market == "Home (1)" and hp < ap:
        is_underdog = True
    if market == "Away (2)" and ap < hp:
        is_underdog = True

    if not is_underdog:
        return True

    if ratio > 1.70 and float(value_pct) < 35.0:
        return False
    return True


def _market_defs() -> List[Dict[str, str]]:
    # No draw selection for now.
    return [
        {"label": "Home (1)", "odds": "offered_1", "prob": "home_prob", "value": "value_pct_1"},
        {"label": "Away (2)", "odds": "offered_2", "prob": "away_prob", "value": "value_pct_2"},
        {"label": "Over 2.5", "odds": "offered_over_2_5", "prob": "over_2_5_prob", "value": "value_pct_over"},
        {"label": "Under 2.5", "odds": "offered_under_2_5", "prob": "under_2_5_prob", "value": "value_pct_under"},
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

    qr = fixture_quality_score(fx)
    odds_match = fx.get("odds_match") or {}
    odds_grade = str(odds_match.get("grade") or "").upper()
    odds_score = _safe_float(odds_match.get("score")) or 0.0

    slow_trap = _slow_favourite_trap(fx, md["label"])
    over_ok = True
    if md["label"] == "Over 2.5":
        over_ok = _over_style_ok(fx)

    rank = float(_rank_score(float(value_pct), float(prob)))
    if slow_trap:
        rank *= 0.80

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
        "rank_score": float(rank),
        "quality": round(float(qr.score), 3),
        "quality_reasons": list(qr.reasons),
        "odds_match_grade": odds_grade,
        "odds_match_score": float(odds_score),
        "team_value_ratio": round(_safe_float(fx.get("team_value_ratio")) or 1.0, 3),
        "prob_gap": round(_prob_gap(fx), 4),
        "slow_fav_trap": bool(slow_trap),
        "over_style_ok": bool(over_ok),
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
    hp = os.getenv("HISTORY_PATH", "").strip()
    if hp:
        return Path(hp)

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
    require = _sb_env("FRIDAY_REQUIRE_HISTORY", True)
    hist = _load_history()
    if hist is None:
        if require:
            raise FileNotFoundError(f"History file missing at: {_history_path()}")
        core = _sf_env("CORE_BANKROLL_START", 800.0)
        fun = _sf_env("FUN_BANKROLL_START", 400.0)
        draw = _sf_env("DRAW_BANKROLL_START", 300.0)
        return core, fun, draw, None, None, {"history_used": False, "history_path": str(_history_path())}

    bc = (hist.get("bankroll_current") or {})
    core = float(bc.get("core", _sf_env("CORE_BANKROLL_START", 800.0)))
    fun = float(bc.get("fun", _sf_env("FUN_BANKROLL_START", 400.0)))
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
# Stake rules
# -------------------------

def core_stake_from_tier(tier: str) -> int:
    return 40 if tier == "STRONG" else 30


# -------------------------
# Core stake boost (optional)
# -------------------------

def _apply_core_stake_boost(singles: List[Dict[str, Any]], open_total: float) -> Tuple[List[Dict[str, Any]], float, Dict[str, Any]]:
    enabled = _sb_env("FRIDAY_CORE_BOOST_ENABLED", False)
    min_open = _sf_env("FRIDAY_CORE_MIN_OPEN", 0.0)
    max_stake = _si_env("FRIDAY_CORE_BOOST_MAX_STAKE", 50)
    step = _si_env("FRIDAY_CORE_BOOST_STEP", 5)
    if step <= 0:
        step = 5

    dbg = {
        "enabled": enabled,
        "min_open": float(min_open),
        "max_stake": int(max_stake),
        "step": int(step),
        "boost_applied": False,
        "before_open": float(round(open_total, 2)),
        "after_open": float(round(open_total, 2)),
        "delta_requested": 0.0,
        "delta_applied": 0.0,
        "capped_out": False,
    }

    if (not enabled) or (min_open <= 0) or (not singles):
        return singles, float(round(open_total, 2)), dbg

    if float(open_total) >= float(min_open):
        return singles, float(round(open_total, 2)), dbg

    target = float(min_open)
    delta = float(target - float(open_total))
    dbg["delta_requested"] = float(round(delta, 2))

    idx = 0
    n = len(singles)
    stuck_rounds = 0

    while delta > 1e-9:
        advanced = False

        ln = singles[idx]
        cur = int(ln.get("stake") or 0)
        if cur < max_stake:
            add = min(step, max_stake - cur)
            ln["stake"] = cur + add
            open_total += add
            delta -= add
            advanced = True

        idx = (idx + 1) % n

        if not advanced:
            stuck_rounds += 1
            if stuck_rounds >= n:
                dbg["capped_out"] = True
                break
        else:
            stuck_rounds = 0

        if float(open_total) >= target:
            break

    dbg["boost_applied"] = True
    dbg["after_open"] = float(round(open_total, 2))
    dbg["delta_applied"] = float(round(dbg["after_open"] - dbg["before_open"], 2))
    return singles, float(round(open_total, 2)), dbg


# -------------------------
# Selection logic
# -------------------------

def _select_core(
    fixture_rows: List[Dict[str, Any]],
    candidate_rows: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], float, Dict[str, Any]]:
    odds_min = _sf_env("FRIDAY_CORE_ODDS_MIN", 1.55)
    odds_max = _sf_env("FRIDAY_CORE_ODDS_MAX", 1.90)
    target_singles = _si_env("FRIDAY_CORE_TARGET_SINGLES", 7)
    max_totals = _si_env("FRIDAY_CORE_MAX_TOTALS", 4)
    max_unders = _si_env("FRIDAY_CORE_MAX_UNDERS", 1)

    # stricter selection by agreement
    strong_prob = _sf_env("FRIDAY_CORE_STRONG_MIN_PROB", 0.53)
    strong_val = _sf_env("FRIDAY_CORE_STRONG_MIN_VALUE_PCT", 3.0)
    strong_q = _sf_env("FRIDAY_CORE_STRONG_MIN_QUALITY", 0.70)

    trb_prob = _sf_env("FRIDAY_CORE_TRB_MIN_PROB", 0.50)
    trb_val = _sf_env("FRIDAY_CORE_TRB_MIN_VALUE_PCT", 1.0)
    trb_q = _sf_env("FRIDAY_CORE_TRB_MIN_QUALITY", 0.65)

    fixture_by_id: Dict[Any, Dict[str, Any]] = {fx.get("fixture_id"): fx for fx in fixture_rows}

    base = []
    for c in candidate_rows:
        if not (odds_min <= float(c["odds"]) <= odds_max):
            continue
        if not _grade_at_least(str(c.get("odds_match_grade") or ""), "B"):
            continue

        fx = fixture_by_id.get(c.get("fixture_id"))
        if not fx:
            continue

        if not _passes_tight_game_filter(fx, c["market"], "core"):
            continue
        if not _passes_team_value_filter(fx, c["market"], float(c["value_pct"])):
            continue
        if c["market"] == "Over 2.5" and not bool(c.get("over_style_ok", True)):
            continue

        base.append(c)

    strong: List[Dict[str, Any]] = []
    trb: List[Dict[str, Any]] = []

    for c in base:
        q = float(c["quality"])
        p = float(c["prob"])
        v = float(c["value_pct"])
        slow_trap = bool(c.get("slow_fav_trap", False))

        if p >= strong_prob and v >= strong_val and q >= strong_q and not slow_trap:
            row = dict(c)
            row["tier"] = "STRONG"
            row["risk_tag"] = "A"
            strong.append(row)
        elif p >= trb_prob and v >= trb_val and q >= trb_q:
            row = dict(c)
            row["tier"] = "TRB"
            row["risk_tag"] = "TRB"
            trb.append(row)

    strong.sort(key=lambda x: (float(x["rank_score"]), float(x["quality"])), reverse=True)
    trb.sort(key=lambda x: (float(x["rank_score"]), float(x["quality"])), reverse=True)

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

        if _is_total_market(c["market"]):
            if totals_cnt >= max_totals:
                return False
            if _is_under(c["market"]) and unders_cnt >= max_unders:
                return False

        stake = core_stake_from_tier(str(c.get("tier") or "TRB"))

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
            "tier": c["tier"],
            "risk_tag": c["risk_tag"],
            "slow_fav_trap": bool(c.get("slow_fav_trap", False)),
            "odds_match_grade": c.get("odds_match_grade"),
            "value_pct": c.get("value_pct"),
        })
        used_fids.add(fid)
        open_total += stake

        if _is_total_market(c["market"]):
            totals_cnt += 1
            if _is_under(c["market"]):
                unders_cnt += 1
        return True

    # Strong first, then TRB. Totals first as in current architecture.
    ordered_groups = [strong, trb]
    for grp in ordered_groups:
        for c in grp:
            if len(singles) >= target_singles:
                break
            if _is_total_market(c["market"]):
                _try_add(c)
        for c in grp:
            if len(singles) >= target_singles:
                break
            if not _is_total_market(c["market"]):
                _try_add(c)

    singles, open_total, boost_dbg = _apply_core_stake_boost(singles, open_total)

    # keep optional double logic
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
        "strong_candidates": len(strong),
        "trb_candidates": len(trb),
        "totals_cnt": totals_cnt,
        "unders_cnt": unders_cnt,
        "stake_boost": boost_dbg,
    }
    return singles, core_double, float(round(open_total, 2)), dbg


def _soft_prob_bucket(c: Dict[str, Any]) -> int:
    p = float(c.get("prob") or 0.0)
    if p >= 0.53:
        return 3
    if p >= 0.50:
        return 2
    return 1


def _select_fun_system(
    fixture_rows: List[Dict[str, Any]],
    candidate_rows: List[Dict[str, Any]],
    stake_total: float,
    core_fixture_ids: set,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], float, Dict[str, Any]]:
    desired_n = _si_env("FRIDAY_FUN_POOL_SIZE", 6)
    k = _si_env("FRIDAY_FUN_K", 3)

    odds_min = _sf_env("FRIDAY_FUN_ODDS_MIN", 1.80)
    odds_max = _sf_env("FRIDAY_FUN_ODDS_MAX", 2.40)

    a_prob = _sf_env("FRIDAY_FUN_A_MIN_PROB", 0.45)
    a_val = _sf_env("FRIDAY_FUN_A_MIN_VALUE_PCT", 5.0)
    a_q = _sf_env("FRIDAY_FUN_A_MIN_QUALITY", 0.65)

    b_prob = _sf_env("FRIDAY_FUN_B_MIN_PROB", 0.40)
    b_val = _sf_env("FRIDAY_FUN_B_MIN_VALUE_PCT", 1.0)
    b_q = _sf_env("FRIDAY_FUN_B_MIN_QUALITY", 0.60)

    max_overlap = _si_env("FRIDAY_FUN_MAX_OVERLAP_WITH_CORE", 2)

    fixture_by_id: Dict[Any, Dict[str, Any]] = {fx.get("fixture_id"): fx for fx in fixture_rows}

    base = []
    for c in candidate_rows:
        if not (odds_min <= float(c["odds"]) <= odds_max):
            continue
        if not _grade_at_least(str(c.get("odds_match_grade") or ""), "C"):
            continue

        fx = fixture_by_id.get(c.get("fixture_id"))
        if not fx:
            continue

        if not _passes_tight_game_filter(fx, c["market"], "fun"):
            continue
        if not _passes_team_value_filter(fx, c["market"], float(c["value_pct"])):
            continue
        if c["market"] == "Over 2.5" and not bool(c.get("over_style_ok", True)):
            continue

        base.append(c)

    tier_a: List[Dict[str, Any]] = []
    tier_b: List[Dict[str, Any]] = []

    for c in base:
        q = float(c["quality"])
        p = float(c["prob"])
        v = float(c["value_pct"])
        slow_trap = bool(c.get("slow_fav_trap", False))

        if p >= a_prob and v >= a_val and q >= a_q and not slow_trap:
            row = dict(c)
            row["tier"] = "A"
            row["risk_tag"] = "A"
            tier_a.append(row)
        elif p >= b_prob and v >= b_val and q >= b_q:
            row = dict(c)
            row["tier"] = "B"
            row["risk_tag"] = "B"
            tier_b.append(row)

    # soft preference for healthier probability distribution, but not hard forcing
    tier_a.sort(key=lambda x: (_soft_prob_bucket(x), float(x["rank_score"]), float(x["quality"])), reverse=True)
    tier_b.sort(key=lambda x: (_soft_prob_bucket(x), float(x["rank_score"]), float(x["quality"])), reverse=True)

    system_pool: List[Dict[str, Any]] = []
    seen_fixtures = set()
    overlap_cnt = 0

    def _try_add(c: Dict[str, Any]) -> bool:
        nonlocal overlap_cnt

        if len(system_pool) >= desired_n:
            return False

        fid = c.get("fixture_id")
        if fid in seen_fixtures:
            return False

        if fid in core_fixture_ids and overlap_cnt >= max_overlap:
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
            "tier": c["tier"],
            "risk_tag": c["risk_tag"],
            "slow_fav_trap": bool(c.get("slow_fav_trap", False)),
            "odds_match_grade": c.get("odds_match_grade"),
            "value_pct": c.get("value_pct"),
        })
        seen_fixtures.add(fid)

        if fid in core_fixture_ids:
            overlap_cnt += 1
        return True

    for grp in [tier_a, tier_b]:
        for c in grp:
            if len(system_pool) >= desired_n:
                break
            _try_add(c)

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
        "tier_a_candidates": len(tier_a),
        "tier_b_candidates": len(tier_b),
        "overlap_with_core": overlap_cnt,
    }

    return system_pool, system, float(round(open_amt, 2)), dbg


# -------------------------
# SuperFun (Core + Fun only)
# -------------------------

def _build_candidate_index(all_cands: List[Dict[str, Any]]) -> Dict[Tuple[Any, str], Dict[str, Any]]:
    idx: Dict[Tuple[Any, str], Dict[str, Any]] = {}
    for c in all_cands:
        idx[(c.get("fixture_id"), str(c.get("market")))] = c
    return idx


def _select_draw_superfun(
    core_singles: List[Dict[str, Any]],
    fun_pool: List[Dict[str, Any]],
    cand_index: Dict[Tuple[Any, str], Dict[str, Any]],
    stake_per_column: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], float, Dict[str, Any]]:
    n_max = _si_env("SUPERFUN_N_MAX", 12)
    n_min = _si_env("SUPERFUN_MIN_TOTAL", 10)
    offset = _si_env("SUPERFUN_HITS_OFFSET", 4)
    min_k = _si_env("SUPERFUN_MIN_K", 6)

    def _build_row(item: Dict[str, Any], source: str) -> Optional[Dict[str, Any]]:
        fid = item.get("fixture_id")
        market = str(item.get("market") or "")
        if fid is None or not market:
            return None

        cand = cand_index.get((fid, market))
        rank = float(cand.get("rank_score", 0.0)) if cand else 0.0

        return {
            "fixture_id": fid,
            "date": item.get("date"),
            "time_gr": item.get("time_gr"),
            "league": item.get("league"),
            "match": item.get("match"),
            "market": market,
            "odds": float(item.get("odds") or 0.0),
            "prob": float(item.get("prob") or 0.0),
            "quality": float(item.get("quality") or 0.0),
            "source": source,
            "rank_score": rank,
            "tier": str(item.get("tier") or "B"),
            "risk_tag": str(item.get("risk_tag") or "B"),
        }

    core_rows: List[Dict[str, Any]] = []
    seen_core = set()
    for s in core_singles:
        row = _build_row(s, "core")
        if row is None:
            continue
        fid = row["fixture_id"]
        if fid in seen_core:
            continue
        seen_core.add(fid)
        core_rows.append(row)

    fun_best_by_fid: Dict[Any, Dict[str, Any]] = {}
    for s in fun_pool:
        row = _build_row(s, "fun")
        if row is None:
            continue
        fid = row["fixture_id"]

        if fid in seen_core:
            continue

        prev = fun_best_by_fid.get(fid)
        if prev is None:
            fun_best_by_fid[fid] = row
        else:
            # prefer A over B, then rank
            prev_t = 2 if str(prev.get("tier")) == "A" else 1
            row_t = 2 if str(row.get("tier")) == "A" else 1
            if row_t > prev_t:
                fun_best_by_fid[fid] = row
            elif row_t == prev_t and float(row["rank_score"]) > float(prev.get("rank_score", 0.0)) + 1e-9:
                fun_best_by_fid[fid] = row

    fun_rows = list(fun_best_by_fid.values())
    fun_rows.sort(
        key=lambda x: (
            2 if str(x.get("tier")) == "A" else 1,
            float(x.get("rank_score", 0.0)),
            float(x.get("quality", 0.0)),
        ),
        reverse=True,
    )

    if n_max > 0:
        fun_slots = max(0, n_max - len(core_rows))
        fun_rows = fun_rows[:fun_slots]

    picks = core_rows + fun_rows
    picks.sort(key=_chrono_key)
    n = len(picks)

    if n < n_min:
        system = {
            "mode": "superfun_from_core_plus_fun",
            "n": n,
            "k": 0,
            "columns": 0,
            "stake_per_column": float(stake_per_column),
            "stake_total": 0.0,
            "target": None,
            "error": f"not_enough_picks_min_{n_min}",
        }
        return [], system, 0.0, {
            "core_kept": len(core_rows),
            "fun_candidates_unique": len(fun_best_by_fid),
            "fun_kept": len(fun_rows),
            "n_after_cap": n,
            "reason": f"need_at_least_{n_min}",
        }

    k = n - int(offset)
    if k < int(min_k):
        k = int(min_k)
    if k > n:
        k = n

    columns = comb(n, k) if (n >= k and k > 0) else 0
    open_amt = float(round(columns * float(stake_per_column), 2))

    pool_out = []
    for p in picks:
        pool_out.append({
            "fixture_id": p["fixture_id"],
            "date": p.get("date"),
            "time_gr": p.get("time_gr"),
            "league": p.get("league"),
            "match": p.get("match"),
            "market": p.get("market"),
            "odds": round(float(p.get("odds") or 0.0), 2),
            "prob": round(float(p.get("prob") or 0.0), 4),
            "quality": round(float(p.get("quality") or 0.0), 3),
            "source": p.get("source"),
            "tier": p.get("tier"),
            "risk_tag": p.get("risk_tag"),
        })

    system = {
        "mode": "superfun_from_core_plus_fun",
        "n": n,
        "k": k,
        "columns": columns,
        "min_hits": k,
        "stake_per_column": float(stake_per_column),
        "stake_total": open_amt,
        "target": f"{k}/{n}",
        "rules": {
            "n_max": int(n_max),
            "n_min": int(n_min),
            "hits_offset": int(offset),
            "min_k": int(min_k),
            "unique_by_fixture": True,
            "core_priority": True,
        },
    }

    dbg = {
        "core_kept": len(core_rows),
        "fun_candidates_unique": len(fun_best_by_fid),
        "fun_kept": len(fun_rows),
        "n_after_cap": n,
        "k": k,
        "columns": columns,
        "stake_per_column": float(stake_per_column),
        "open": open_amt,
    }

    return pool_out, system, open_amt, dbg


# -------------------------
# Main builder
# -------------------------

def build_friday_shortlist() -> Dict[str, Any]:
    th = _load_latest_thursday_report()
    fixtures = th.get("fixtures") or []
    if not isinstance(fixtures, list):
        raise ValueError("Thursday report has no fixtures list.")

    core_bankroll, fun_bankroll, draw_bankroll, next_week, _hist_as_of, hist_meta = _history_bankrolls_or_env()

    fun_stake_total = _sf_env("FUN_SYSTEM_STAKE", 35.0)
    superfun_stake_per_col = _sf_env("SUPERFUN_STAKE_PER_COLUMN", 0.10)
    draw_mode = _ss_env("FRIDAY_DRAW_MODE", "superfun").lower()

    all_cands: List[Dict[str, Any]] = []

    for fx in fixtures:
        for md in _market_defs():
            c = _candidate_from_fixture(fx, md)
            if c:
                all_cands.append(c)

    if any((c.get("fixture_id") is None) for c in all_cands):
        bad = [c for c in all_cands if c.get("fixture_id") is None][:3]
        raise ValueError(f"Some candidates have missing fixture_id. Examples: {bad}")

    core_singles, core_double, core_open, core_dbg = _select_core(fixtures, all_cands)
    core_fids = {s.get("fixture_id") for s in core_singles if s.get("fixture_id") is not None}

    fun_pool, fun_system, fun_open, fun_dbg = _select_fun_system(fixtures, all_cands, fun_stake_total, core_fids)

    cand_index = _build_candidate_index(all_cands)
    if draw_mode == "superfun":
        draw_pool, draw_system, draw_open, draw_dbg = _select_draw_superfun(
            core_singles=core_singles,
            fun_pool=fun_pool,
            cand_index=cand_index,
            stake_per_column=float(superfun_stake_per_col),
        )
    else:
        draw_pool, draw_system, draw_open, draw_dbg = [], {"mode": "disabled"}, 0.0, {"reason": "FRIDAY_DRAW_MODE!=superfun"}

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
            "stake_rule": "STRONG=40, TRB=30 + optional boost if open too low",
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
            "after_open": round(float(draw_bankroll) - float(draw_open), 2),
            "debug": draw_dbg,
        },

        "debug": {
            "project_root": str(PROJECT_ROOT),
            "logs_dir": str(LOGS_DIR),
            "style_metrics_loaded": bool(_load_style_metrics()),
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
