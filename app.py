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


# === Utility: Send structured data to chat (προαιρετικό) ===
def send_to_chat(title, data):
    """
    Στέλνει report σε εξωτερικό listener (αν το χρειαστείς).
    Δεν είναι απαραίτητο για το Custom GPT, αλλά το κρατάμε.
    """
    payload = {"message": f"📊 {title}", "data": data}
    try:
        print(f"📤 Sending report to chat: {title}")
        resp = requests.post(CHAT_FORWARD_URL, json=payload, timeout=25)
        print(f"✅ Forward to CHAT_FORWARD_URL -> {resp.status_code}")
    except Exception as e:
        print(f"⚠️ Chat forward error: {e}")

    try:
        requests.post(LOCAL_CHAT_URL, json=payload, timeout=10)
    except Exception as e:
        print(f"💬 Local chat forward skipped ({e})")


# === MAIN ROUTE: Handle chat commands ===
@app.route("/chat_command", methods=["POST"])
def chat_command():
    """
    Κεντρικό endpoint που θα καλεί ο Agent Bombay (Custom GPT Action).
    Περιμένει JSON: { "command": "run friday shortlist" } κτλ.
    """
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
            # Χρησιμοποιούμε την v2 έκδοση
            script = "friday_shortlist_v2.py"
            label = "Friday Shortlist"
            report_file = "logs/friday_shortlist_v2.json"

        elif "tuesday" in command:
            script = "tuesday_recap_v2.py"
            label = "Tuesday Recap"
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
        report_data = {}
        if os.path.exists(report_file):
            try:
                with open(report_file, "r", encoding="utf-8") as f:
                    report_data = json.load(f)
            except Exception as e:
                print(f"⚠️ Error reading report file: {e}")
                report_data = {"error": f"Failed to read {report_file}"}
        else:
            report_data = {
                "error": f"⚠️ Report file not found: {report_file}"
            }

        # === Handle Kelly picks (διάφορες δομές) ===
        # Friday v1: "fraction_kelly": {"picks": [...]}
        # Friday v2: "kelly": {"picks": [...]}
        kelly_block = (
            report_data.get("kelly")
            or report_data.get("fraction_kelly")
            or {}
        )
        if isinstance(kelly_block, dict):
            kelly_picks = kelly_block.get("picks", [])
        else:
            kelly_picks = []

        # === Compose summary (generic) ===
        summary_lines = []
        summary_lines.append(f"✅ {label} ολοκληρώθηκε επιτυχώς.")
        summary_lines.append(
            f"📅 {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
        )

        # Αν είναι Friday Shortlist, δείχνουμε και λίγη Kelly πληροφορία
        if "Friday" in label:
            if kelly_picks:
                summary_lines.append("\n🎯 Top 10 Kelly Picks:")
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
                summary_lines.append(
                    "\n⚠️ Δεν εντοπίστηκαν Kelly picks σε αυτό το report."
                )

        summary = "\n".join(summary_lines)

        # === Optional: forward to external listener (αν θες) ===
        try:
            send_to_chat(label, {"summary": summary})
            send_to_chat(f"{label} – Full Report", report_data)
        except Exception as e:
            print(f"⚠️ Error forwarding to chat: {e}")

        # === IMPORTANT: structured response για τον Agent Bombay ===
        return jsonify(
            {
                "status": "ok",
                "label": label,
                "summary": summary,
                "report": report_data,
            }
        )

    except Exception as e:
        print(f"❌ Error executing command: {e}")
        try:
            send_to_chat("Error", {"error": str(e)})
        except Exception:
            pass
        return jsonify({"error": str(e)}), 500


# === Manual Run Routes (για browser / health checks) ===
@app.route("/run/thursday", methods=["GET"])
def run_thursday():
    return run_script("thursday_analysis_v1.py", "Thursday Analysis")


@app.route("/run/friday", methods=["GET"])
def run_friday():
    return run_script("friday_shortlist_v2.py", "Friday Shortlist")


@app.route("/run/tuesday", methods=["GET"])
def run_tuesday():
    return run_script("tuesday_recap_v2.py", "Tuesday Recap")


def run_script(script_name, label):
    try:
        print(f"🚀 Manual trigger: Running {label}...")
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
           
