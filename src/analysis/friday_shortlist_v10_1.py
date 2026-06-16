# src/analysis/friday_shortlist_v10_1.py
"""
Bombay Friday Shortlist V10.1

Reads:
- logs/thursday_report_v3.json
- logs/tuesday_history_v3.json

Uses:
- selection_rules_v10.py
- bankroll_policy_v10.py

Writes:
- logs/friday_shortlist_v10_1.json

Does NOT overwrite V3.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# PATHS
# ============================================================

def find_project_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / "logs").is_dir() and (p / "src").is_dir():
            return p
    return start.parents[2] if len(start.parents) >= 3 else start


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = find_project_root(THIS_FILE.parent)
LOGS_DIR = PROJECT_ROOT / "logs"

for p in [PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "src" / "analysis"]:
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


from bankroll_policy_v10 import (  # noqa
    apply_core_exposure_cap,
    build_system_bet,
    normalize_history_bankrolls,
    bankroll_policy_summary,
)

from selection_rules_v10 import (  # noqa
    select_core,
    select_fun,
    select_superfun,
    selection_rules_summary,
)


# ============================================================
# HELPERS
# ============================================================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def sf(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def odds_grade_rank(grade: str) -> int:
    g = str(grade or "").strip().upper()
    return {"A": 4, "B": 3, "C": 2, "D": 1}.get(g, 0)


def load_thursday() -> Dict[str, Any]:
    path = LOGS_DIR / "thursday_report_v3.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing Thursday report: {path}")
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError("Thursday report is not a dict.")
    return data


def history_path() -> Path:
    return LOGS_DIR / "tuesday_history_v3.json"


def load_history() -> Optional[Dict[str, Any]]:
    path = history_path()
    if not path.exists():
        return None
    data = read_json(path)
    return data if isinstance(data, dict) else None


# ============================================================
# CANDIDATES FROM THURSDAY
# ============================================================

MARKET_DEFS = [
    {
        "market": "Home (1)",
        "odds": "offered_1",
        "prob": "home_prob",
        "value": "value_pct_1",
        "ev": "ev_1",
    },
    {
        "market": "Away (2)",
        "odds": "offered_2",
        "prob": "away_prob",
        "value": "value_pct_2",
        "ev": "ev_2",
    },
    {
        "market": "Over 2.5",
        "odds": "offered_over_2_5",
        "prob": "over_2_5_prob",
        "value": "value_pct_over",
        "ev": "ev_over",
    },
    {
        "market": "Under 2.5",
        "odds": "offered_under_2_5",
        "prob": "under_2_5_prob",
        "value": "value_pct_under",
        "ev": "ev_under",
    },
]


def quality_from_fixture(fx: Dict[str, Any]) -> float:
    flags = fx.get("flags") or {}
    odds_match = fx.get("odds_match") or {}

    if not bool(odds_match.get("matched")):
        return 0.0

    q = sf(odds_match.get("score"), 0.0)

    conf = sf(flags.get("confidence"), 0.50)
    if conf < 0.40:
        q *= 0.70
    elif conf < 0.50:
        q *= 0.85

    if flags.get("value_missing") is True:
        q *= 0.85
    if flags.get("history_missing") is True:
        q *= 0.85
    if flags.get("style_missing") is True:
        q *= 0.85

    instability = flags.get("prob_instability")
    if instability is not None:
        gap = sf(instability)
        if gap > 0.25:
            q *= 0.80
        elif gap > 0.18:
            q *= 0.90

    return round(max(0.0, min(1.0, q)), 4)


def confirmation_score(candidate: Dict[str, Any]) -> float:
    prob = sf(candidate.get("prob"))
    quality = sf(candidate.get("quality"))
    value = sf(candidate.get("value_pct"))
    grade = odds_grade_rank(str(candidate.get("odds_match_grade") or "")) / 4.0

    style = 1.0
    if candidate.get("market") == "Over 2.5" and candidate.get("over_style_ok") is False:
        style = 0.45
    if candidate.get("slow_fav_trap") is True:
        style = 0.55

    value_norm = max(0.0, min(1.0, value / 15.0))

    score = (
        0.32 * prob
        + 0.28 * quality
        + 0.18 * grade
        + 0.12 * style
        + 0.10 * value_norm
    )

    return round(max(0.0, min(1.0, score)), 4)


def candidate_from_fixture(fx: Dict[str, Any], md: Dict[str, str]) -> Optional[Dict[str, Any]]:
    odds = fx.get(md["odds"])
    prob = fx.get(md["prob"])

    if odds is None or prob is None:
        return None

    odds = sf(odds)
    prob = sf(prob)

    if odds <= 1.0 or prob <= 0:
        return None

    flags = fx.get("flags") or {}
    odds_match = fx.get("odds_match") or {}

    home_prob = sf(fx.get("home_prob"))
    away_prob = sf(fx.get("away_prob"))

    row = {
        "fixture_id": fx.get("fixture_id"),
        "date": str(fx.get("date") or ""),
        "time_gr": str(fx.get("time") or ""),
        "league": str(fx.get("league") or ""),
        "match": f"{fx.get('home')} – {fx.get('away')}",
        "market": md["market"],
        "odds": round(odds, 2),
        "prob": round(prob, 4),
        "value_pct": round(sf(fx.get(md["value"])), 2),
        "ev": fx.get(md["ev"]),
        "quality": quality_from_fixture(fx),
        "odds_match_grade": str(odds_match.get("grade") or "").upper(),
        "odds_match_score": sf(odds_match.get("score")),
        "prob_gap": round(abs(home_prob - away_prob), 4),
        "team_value_ratio": fx.get("team_value_ratio"),
        "slow_fav_trap": False,
        "over_style_ok": True,
        "flags": {
            "tight_game": flags.get("tight_game"),
            "home_shape": flags.get("home_shape"),
            "away_shape": flags.get("away_shape"),
            "over_good_shape": flags.get("over_good_shape"),
            "under_elite": flags.get("under_elite"),
            "value_missing": flags.get("value_missing"),
            "history_missing": flags.get("history_missing"),
            "style_missing": flags.get("style_missing"),
        },
    }

    market = row["market"]

    if market == "Over 2.5":
        row["over_style_ok"] = bool(
            flags.get("over_good_shape")
            or sf(fx.get("tempo_total_mult"), 1.0) >= 1.00
            or sf(fx.get("over_2_5_prob")) >= 0.58
        )

    if market in ("Home (1)", "Away (2)"):
        row["slow_fav_trap"] = bool(
            flags.get("tight_game") is True
            and row["prob_gap"] < 0.12
            and row["value_pct"] < 8.0
        )

    row["confirmation_score"] = confirmation_score(row)

    return row


def build_candidates(thursday: Dict[str, Any]) -> List[Dict[str, Any]]:
    fixtures = thursday.get("fixtures") or []
    if not isinstance(fixtures, list):
        raise ValueError("Thursday report has no fixtures list.")

    out: List[Dict[str, Any]] = []

    for fx in fixtures:
        if not isinstance(fx, dict):
            continue

        for md in MARKET_DEFS:
            c = candidate_from_fixture(fx, md)
            if c is not None:
                out.append(c)

    return out


# ============================================================
# OUTPUT ADAPTERS
# ============================================================

def clean_line(row: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "fixture_id",
        "date",
        "time_gr",
        "league",
        "match",
        "market",
        "odds",
        "prob",
        "value_pct",
        "ev",
        "quality",
        "confirmation_score",
        "selection_score",
        "money_rating",
        "stake",
        "odds_match_grade",
        "prob_gap",
        "slow_fav_trap",
        "over_style_ok",
        "source",
    ]

    return {k: row.get(k) for k in keys if k in row}


def portfolio_after_open(bankroll: float, open_amount: float) -> float:
    return round(sf(bankroll) - sf(open_amount), 2)


def total_open(core_open: float, fun_open: float, superfun_open: float) -> float:
    return round(sf(core_open) + sf(fun_open) + sf(superfun_open), 2)


# ============================================================
# MAIN BUILDER
# ============================================================

def build_friday_v10_1() -> Dict[str, Any]:
    thursday = load_thursday()
    history = load_history()

    bankrolls = normalize_history_bankrolls(history)

    candidates = build_candidates(thursday)

    core_raw, core_debug_select = select_core(candidates)
    core_lines, core_money_debug = apply_core_exposure_cap(core_raw)
    core_open = sf(core_money_debug.get("open"))

    core_fixture_ids = {x.get("fixture_id") for x in core_lines if x.get("fixture_id") is not None}

    fun_lines_raw, fun_debug_select = select_fun(candidates, core_fixture_ids=core_fixture_ids)
    fun_system = build_system_bet(fun_lines_raw, mode="fun")
    fun_open = sf(fun_system.get("stake_total"))

    superfun_lines_raw, superfun_debug_select = select_superfun(core_lines, fun_lines_raw)
    superfun_system = build_system_bet(superfun_lines_raw, mode="superfun")
    superfun_open = sf(superfun_system.get("stake_total"))

    hist_week = None
    if isinstance(history, dict):
        try:
            hist_week = int(history.get("week_count"))
        except Exception:
            hist_week = None

    next_week = (hist_week + 1) if hist_week is not None else 1

    out = {
        "title": f"Bombay Friday V10.1 — Week {next_week}",
        "version": "friday_shortlist_v10_1",
        "generated_at": utc_now_iso(),
        "window": thursday.get("window") or {},
        "source_thursday_generated_at": thursday.get("generated_at"),

        "history": {
            "history_used": isinstance(history, dict),
            "history_path": str(history_path()),
            "history_week_count": hist_week,
            "next_week": next_week,
            "bankroll_start_from_history": bankrolls,
        },

        "core": {
            "label": "CoreBet",
            "mode": "singles",
            "bankroll_start": bankrolls["core"],
            "singles": [clean_line(x) for x in core_lines],
            "open": round(core_open, 2),
            "after_open": portfolio_after_open(bankrolls["core"], core_open),
            "debug": {
                "selection": core_debug_select,
                "money": core_money_debug,
            },
        },

        "funbet": {
            "label": "FunBet",
            "mode": "system",
            "bankroll_start": bankrolls["fun"],
            "system_pool": [clean_line(x) for x in fun_lines_raw],
            "system": fun_system,
            "open": round(fun_open, 2),
            "after_open": portfolio_after_open(bankrolls["fun"], fun_open),
            "debug": {
                "selection": fun_debug_select,
            },
        },

        "superfun": {
            "label": "SuperFun",
            "mode": "system_lottery",
            "bankroll_start": bankrolls["superfun"],
            "system_pool": [clean_line(x) for x in superfun_lines_raw],
            "system": superfun_system,
            "open": round(superfun_open, 2),
            "after_open": portfolio_after_open(bankrolls["superfun"], superfun_open),
            "debug": {
                "selection": superfun_debug_select,
            },
        },

        "drawbet": {
            "label": "SuperFun",
            "deprecated_alias": True,
            "bankroll_start": bankrolls["superfun"],
            "system_pool": [clean_line(x) for x in superfun_lines_raw],
            "system": superfun_system,
            "open": round(superfun_open, 2),
            "after_open": portfolio_after_open(bankrolls["superfun"], superfun_open),
        },

        "summary": {
            "core_picks": len(core_lines),
            "fun_picks": len(fun_lines_raw),
            "superfun_picks": len(superfun_lines_raw),
            "core_open": round(core_open, 2),
            "fun_open": round(fun_open, 2),
            "superfun_open": round(superfun_open, 2),
            "total_open": total_open(core_open, fun_open, superfun_open),
        },

        "debug": {
            "project_root": str(PROJECT_ROOT),
            "logs_dir": str(LOGS_DIR),
            "candidates_total": len(candidates),
            "fixtures_total": thursday.get("fixtures_total"),
            "bankroll_policy": bankroll_policy_summary(),
            "selection_rules": selection_rules_summary(),
        },
    }

    return out


def main() -> int:
    try:
        out = build_friday_v10_1()

        save_path = LOGS_DIR / "friday_shortlist_v10_1.json"
        write_json(save_path, out)

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        backup_path = LOGS_DIR / f"friday_v10_1_{ts}.json"
        write_json(backup_path, out)

        print(json.dumps({
            "status": "ok",
            "saved": str(save_path),
            "backup": str(backup_path),
            "generated_at": out.get("generated_at"),
            "title": out.get("title"),
            "summary": out.get("summary"),
        }, ensure_ascii=False))
        return 0

    except Exception as e:
        print(json.dumps({
            "status": "error",
            "error": str(e),
            "logs_dir": str(LOGS_DIR),
            "history_path": str(history_path()),
        }, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
