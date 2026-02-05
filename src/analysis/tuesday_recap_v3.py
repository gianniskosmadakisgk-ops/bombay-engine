# src/analysis/tuesday_recap_v3.py
"""
Tuesday Recap v3 — Deploy version

Reads:
- logs/friday_shortlist_v3.json  (what was played)
- logs/tuesday_results_v3.json   (settled results you provide)

Writes:
- data/tuesday_history_v3.json   (rolling bankroll history)
- logs/tuesday_recap_v3.json     (human-friendly recap output)

Results file format (logs/tuesday_results_v3.json):
{
  "week_no": 3,
  "core": [
    {"match":"...", "market":"Over 2.5", "odds":1.97, "stake":30, "won": true},
    ...
  ],
  "funbet": {
    "stake": 35,
    "hits": 4,
    "payout": 0
  },
  "drawbet": {
    "stake": 25,
    "hits": 2,
    "payout": 0
  }
}

If results file is missing, script outputs a “pending recap skeleton” and exits OK.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = PROJECT_ROOT / "logs"

HISTORY_PATH = os.getenv("TUESDAY_HISTORY_PATH", "data/tuesday_history_v3.json")
HISTORY_FILE = (PROJECT_ROOT / HISTORY_PATH) if not Path(HISTORY_PATH).is_absolute() else Path(HISTORY_PATH)

FRIDAY_FILE = LOGS_DIR / "friday_shortlist_v3.json"
RESULTS_FILE = LOGS_DIR / "tuesday_results_v3.json"
RECAP_OUT = LOGS_DIR / "tuesday_recap_v3.json"

def _read_json(p: Path) -> Any:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def _write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def _roi(profit: float, stake: float) -> float:
    if stake <= 0:
        return 0.0
    return round((profit / stake) * 100.0, 2)

def _init_history() -> Dict[str, Any]:
    return {
        "week_count": 0,
        "updated_at": _utc_now_iso(),
        "core": {"bankroll_current": float(os.getenv("CORE_BANKROLL_START", "800")), "stake": 0.0, "profit": 0.0},
        "funbet": {"bankroll_current": float(os.getenv("FUN_BANKROLL_START", "400")), "stake": 0.0, "profit": 0.0},
        "drawbet": {"bankroll_current": float(os.getenv("DRAW_BANKROLL_START", "300")), "stake": 0.0, "profit": 0.0},
        "weeks": {}
    }

def _load_history() -> Dict[str, Any]:
    if HISTORY_FILE.exists():
        return _read_json(HISTORY_FILE)
    return _init_history()

def _key_for_week(friday: Dict[str, Any]) -> str:
    w = (friday.get("window") or {})
    return str(w.get("from") or "") or friday.get("generated_at", "")[:10]

def _tick(won: Optional[bool]) -> str:
    if won is True:
        return "✅"
    if won is False:
        return "❌"
    return "⏳"

def main() -> int:
    if not FRIDAY_FILE.exists():
        _write_json(RECAP_OUT, {"status": "error", "error": "missing friday_shortlist_v3.json", "at": _utc_now_iso()})
        print("missing friday_shortlist_v3.json")
        return 1

    friday = _read_json(FRIDAY_FILE)
    hist = _load_history()

    week_no = int(friday.get("week_no") or (int(hist.get("week_count") or 0) + 1))
    week_key = _key_for_week(friday)

    if not RESULTS_FILE.exists():
        # Pending skeleton recap
        recap = {
            "status": "pending_results",
            "week_no": week_no,
            "week_key": week_key,
            "message": "Add logs/tuesday_results_v3.json to settle bets and compute ROI.",
            "at": _utc_now_iso(),
            "friday_snapshot": {
                "core_open": friday.get("core", {}).get("open"),
                "fun_open": friday.get("funbet", {}).get("open"),
                "draw_open": friday.get("drawbet", {}).get("open"),
            }
        }
        _write_json(RECAP_OUT, recap)
        print("pending results")
        return 0

    results = _read_json(RESULTS_FILE)

    # ---------------- Core settlement ----------------
    core_rows = results.get("core") or []
    core_stake = 0.0
    core_profit = 0.0
    core_list = []

    for r in core_rows:
        odds = float(r.get("odds") or 0)
        stake = float(r.get("stake") or 0)
        won = r.get("won")
        core_stake += stake
        if won is True:
            # profit = stake*(odds-1)
            core_profit += stake * (odds - 1.0)
        elif won is False:
            core_profit -= stake

        core_list.append({
            "tick": _tick(won),
            "match": r.get("match"),
            "market": r.get("market"),
            "odds": odds,
            "stake": stake,
            "won": won,
        })

    # ---------------- Fun / Draw settlement (system-level) ----------------
    # Here we keep it simple: you provide payout (net return) or hits; if payout missing, assume 0 return.
    fun = results.get("funbet") or {}
    draw = results.get("drawbet") or {}

    fun_stake = float(fun.get("stake") or friday.get("funbet", {}).get("open") or 0.0)
    fun_payout = float(fun.get("payout") or 0.0)  # total return back
    fun_profit = fun_payout - fun_stake

    draw_stake = float(draw.get("stake") or friday.get("drawbet", {}).get("open") or 0.0)
    draw_payout = float(draw.get("payout") or 0.0)
    draw_profit = draw_payout - draw_stake

    # ---------------- Update cumulative history ----------------
    def bump(fund_key: str, stake: float, profit: float):
        hist[fund_key]["stake"] = float(hist[fund_key].get("stake") or 0.0) + stake
        hist[fund_key]["profit"] = float(hist[fund_key].get("profit") or 0.0) + profit
        hist[fund_key]["bankroll_current"] = float(hist[fund_key].get("bankroll_current") or 0.0) + profit

    bump("core", core_stake, core_profit)
    bump("funbet", fun_stake, fun_profit)
    bump("drawbet", draw_stake, draw_profit)

    hist["week_count"] = max(int(hist.get("week_count") or 0), week_no)
    hist["updated_at"] = _utc_now_iso()

    hist["weeks"][week_key] = {
        "week_no": week_no,
        "window": friday.get("window"),
        "updated_at": _utc_now_iso(),
        "core": {"stake": round(core_stake, 2), "profit": round(core_profit, 2), "pending": 0, "roi_pct": _roi(core_profit, core_stake)},
        "funbet": {"stake": round(fun_stake, 2), "profit": round(fun_profit, 2), "pending": 0, "roi_pct": _roi(fun_profit, fun_stake), "hits": fun.get("hits")},
        "drawbet": {"stake": round(draw_stake, 2), "profit": round(draw_profit, 2), "pending": 0, "roi_pct": _roi(draw_profit, draw_stake), "hits": draw.get("hits")},
        "core_bets": core_list,
    }

    _write_json(HISTORY_FILE, hist)

    recap = {
        "status": "ok",
        "week_no": week_no,
        "week_key": week_key,
        "at": _utc_now_iso(),
        "core": {
            "stake": round(core_stake, 2),
            "profit": round(core_profit, 2),
            "roi_pct": _roi(core_profit, core_stake),
            "score": f"{sum(1 for x in core_rows if x.get('won') is True)}/{len(core_rows)}",
            "bets": core_list,
        },
        "funbet": {
            "stake": round(fun_stake, 2),
            "profit": round(fun_profit, 2),
            "roi_pct": _roi(fun_profit, fun_stake),
            "hits": fun.get("hits"),
        },
        "drawbet": {
            "stake": round(draw_stake, 2),
            "profit": round(draw_profit, 2),
            "roi_pct": _roi(draw_profit, draw_stake),
            "hits": draw.get("hits"),
        },
        "bankrolls": {
            "core_current": round(hist["core"]["bankroll_current"], 2),
            "fun_current": round(hist["funbet"]["bankroll_current"], 2),
            "draw_current": round(hist["drawbet"]["bankroll_current"], 2),
        },
        "cumulative": {
            "core_roi_pct": _roi(float(hist["core"]["profit"]), float(hist["core"]["stake"])),
            "fun_roi_pct": _roi(float(hist["funbet"]["profit"]), float(hist["funbet"]["stake"])),
            "draw_roi_pct": _roi(float(hist["drawbet"]["profit"]), float(hist["drawbet"]["stake"])),
        }
    }
    _write_json(RECAP_OUT, recap)
        # ---------------- Export for next week (download/upload) ----------------
    export_name = f"tuesday_history_export_week_{week_no}.json"
    export_path = LOGS_DIR / export_name
    _write_json(export_path, hist)

    # Print a stable marker so you can grep it from logs
    print(f"HISTORY_EXPORT: {export_path}")

    print("ok")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
