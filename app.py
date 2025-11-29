from flask import Flask, request, jsonify
import subprocess
import requests
import json
import os
import sys
from datetime import datetime

# === Real-time logging fix for Render ===
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)

app = Flask(__name__)

# === ENVIRONMENT CONFIG ===
CHAT_FORWARD_URL = os.getenv(
    "CHAT_FORWARD_URL",
    "https://bombay-engine.onrender.com/chat_forward",
)
LOCAL_CHAT_URL = os.getenv(
    "LOCAL_CHAT_URL",
    "https://api.openai.com/v1/bombay/chat",  # placeholder
)


# === Utility: Send structured data to chat ===
def send_to_chat(title: str, data: dict):
    """Send structured data (summary/report) to chat or local forward."""
    payload = {"message": f"📊 {title}", "data": data}

    # Remote forward (προς ChatGPT)
    try:
        print(f"📤 Sending report to chat: {title}")
        resp = requests.post(CHAT_FORWARD_URL, json=payload, timeout=25)
        print(f"✅ Forward to CHAT_FORWARD_URL -> {resp.status_code}")
    except Exception as e:
        print(f"⚠️ Chat forward error: {e}")

    # Local forward (αν ποτέ χρειαστεί)
    try:
        requests.post(LOCAL_CHAT_URL, json=payload, timeout=10)
    except Exception as e:
        print(f"💬 Local chat forward skipped ({e})")


# === MAIN ROUTE: Handle chat commands ===
@app.route("/chat_command", methods=["POST"])
def chat_command():
    try:
        print("📩 Received /chat_command request")
        data = request.get_json(force=True)
        command = (data.get("command", "") or "").lower().strip()
        print(f"🧭 Command detected: {command}")

        # === Identify script based on command ===
        if "thursday" in command:
            script = "thursday_analysis_v1.py"
            label = "Thursday Analysis"
            report_file = "logs/thursday_report_v1.json"

        elif "friday" in command:
            script = "friday_shortlist_v2.py"
            label = "Friday Shortlist v2"
            report_file = "logs/friday_shortlist_v2.json"

        elif "tuesday" in command:
            script = "tuesday_recap_v2.py"
            label = "Tuesday Recap v2"
            report_file = "logs/tuesday_recap_v2.json"

        else:
            return jsonify({"error": "❓ Unknown command"}), 400

        # === Run script ===
        print(f"🚀 Running {label} ({script})")
        env = os.environ.copy()

        result = subprocess.run(
            ["python3", script],
            cwd="/opt/render/project/src",
            env=env,
            capture_output=True,
            text=True,
        )

        print("----- SCRIPT OUTPUT START -----")
        print(result.stdout)
        print("----- SCRIPT OUTPUT END -----")

        if result.stderr:
            print("⚠️ SCRIPT ERRORS:")
            print(result.stderr)

        # === Load JSON output ===
        if os.path.exists(os.path.join("/opt/render/project/src", report_file)):
            path = os.path.join("/opt/render/project/src", report_file)
        else:
            # fallback αν τρέχουμε local
            path = report_file

        report_data = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                report_data = json.load(f)
        else:
            report_data = {"info": "⚠️ No report file found.", "report_file": report_file}

        # === Extract Kelly picks (v1 ή v2 format) ===
        kelly_picks = []
        if isinstance(report_data.get("kelly"), dict):
            kelly_picks = report_data["kelly"].get("picks", []) or []
        elif isinstance(report_data.get("fraction_kelly"), dict):
            kelly_picks = report_data["fraction_kelly"].get("picks", []) or []

        # === Compose summary ===
        summary_lines = []
        summary_lines.append(f"✅ {label} ολοκληρώθηκε επιτυχώς.")
        summary_lines.append(f"📅 {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

        # --- Wallet summaries (Tuesday recap v2 ή Friday shortlist v2) ---
        if "wallets" in report_data:
            # Tuesday recap v2
            summary_lines.append("\n💼 Wallets recap:")
            for w in report_data["wallets"]:
                summary_lines.append(
                    f"- {w.get('Wallet')}: "
                    f"Staked {w.get('Staked')}, "
                    f"Profit {w.get('Profit')}, "
                    f"Final {w.get('Final')}, "
                    f"Yield {w.get('Yield%')}"
                )
        elif "bankroll_status" in report_data:
            # Friday shortlist v2
            summary_lines.append("\n💼 Bankroll status (Friday):")
            for w in report_data["bankroll_status"]:
                summary_lines.append(
                    f"- {w.get('Wallet')}: "
                    f"Before {w.get('Before')}, "
                    f"After {w.get('After')}, "
                    f"Open Bets {w.get('Open Bets')}"
                )

        # --- Meta info (αν υπάρχει) ---
        if isinstance(report_data.get("meta"), dict):
            meta = report_data["meta"]
            meta_msg = []
            if "fixtures_total" in meta:
                meta_msg.append(f"fixtures_total={meta['fixtures_total']}")
            if "draw_singles" in meta:
                meta_msg.append(f"draw_singles={meta['draw_singles']}")
            if "over_singles" in meta:
                meta_msg.append(f"over_singles={meta['over_singles']}")
            if "kelly_picks" in meta:
                meta_msg.append(f"kelly_picks={meta['kelly_picks']}")
            if "funbet_draw_cols" in meta:
                meta_msg.append(f"FunBetDrawCols={meta['funbet_draw_cols']}")
            if "funbet_over_cols" in meta:
                meta_msg.append(f"FunBetOverCols={meta['funbet_over_cols']}")
            if meta_msg:
                summary_lines.append("\n📌 Meta: " + ", ".join(meta_msg))

        # --- Kelly picks preview ---
        if kelly_picks:
            summary_lines.append("\n🎯 Top Kelly Picks:")
            for i, pick in enumerate(kelly_picks[:10], 1):
                summary_lines.append(
                    f"{i}. {pick.get('match','N/A')} | "
                    f"{pick.get('market','-').upper()} | "
                    f"Fair: {pick.get('fair','-')} | "
                    f"Offered: {pick.get('offered','-')} | "
                    f"Diff: {pick.get('diff','-')} | "
                    f"Stake: €{pick.get('stake (€)','-')}"
                )
        else:
            summary_lines.append("\n⚠️ Δεν εντοπίστηκαν Kelly picks σε αυτό το report.")

        summary = "\n".join(summary_lines)

        # === Send to chat ===
        send_to_chat(label, {"summary": summary})
        send_to_chat(f"{label} – Full Report", report_data)

        return jsonify({"status": "ok", "message": f"{label} executed and sent"})

    except Exception as e:
        print(f"❌ Error executing command: {e}")
        send_to_chat("Error", {"error": str(e)})
        return jsonify({"error": str(e)}), 500


# === Manual Run Routes ===
@app.route("/run/thursday", methods=["GET"])
def run_thursday():
    return run_script("thursday_analysis_v1.py", "Thursday Analysis")


@app.route("/run/friday", methods=["GET"])
def run_friday():
    return run_script("friday_shortlist_v2.py", "Friday Shortlist v2")


@app.route("/run/tuesday", methods=["GET"])
def run_tuesday():
    return run_script("tuesday_recap_v2.py", "Tuesday Recap v2")


def run_script(script_name: str, label: str):
    try:
        print(f"🚀 Manual trigger: Running {label} ({script_name})...")
        result = subprocess.run(
            ["python3", script_name],
            cwd="/opt/render/project/src",
            capture_output=True,
            text=True,
        )
        print("----- SCRIPT OUTPUT START -----")
        print(result.stdout)
        print("----- SCRIPT OUTPUT END -----")

        if result.stderr:
            print("⚠️ SCRIPT ERRORS:")
            print(result.stderr)

        return jsonify({"status": "ok", "message": f"{label} executed."}), 200
    except Exception as e:
        print(f"❌ Manual run error: {e}")
        return jsonify({"error": str(e)}), 500


# === Chat forward endpoint (για debug) ===
@app.route("/chat_forward", methods=["POST"])
def chat_forward():
    try:
        data = request.get_json()
        print("💬 Incoming chat message:", json.dumps(data, indent=2, ensure_ascii=False))
        return jsonify({"status": "received"}), 200
    except Exception as e:
        print(f"⚠️ Error in chat_forward: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


# === Healthcheck ===
@app.route("/healthcheck", methods=["GET"])
def healthcheck():
    return jsonify({"message": "Server running", "status": "ok"})


# === Entry point ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"🟢 Starting Bombay Engine Flask Server on port {port}...")
    app.run(host="0.0.0.0", port=port, use_reloader=False)
