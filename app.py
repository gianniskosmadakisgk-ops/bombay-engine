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
def send_to_chat(title, data):
    """Στέλνει δομημένα δεδομένα (summary/report) στο chat forward."""
    payload = {"message": f"📊 {title}", "data": data}

    # Κύριο forward (Render → ChatGPT)
    try:
        print(f"📤 Sending report to chat: {title}")
        resp = requests.post(CHAT_FORWARD_URL, json=payload, timeout=25)
        print(f"✅ Forward to CHAT_FORWARD_URL -> {resp.status_code}")
    except Exception as e:
        print(f"⚠️ Chat forward error: {e}")

    # Local / placeholder forward (αν ποτέ το χρειαστούμε)
    try:
        requests.post(LOCAL_CHAT_URL, json=payload, timeout=10)
    except Exception as e:
        print(f"💬 Local chat forward skipped ({e})")


# === Helper: Run a script safely ===
def run_script(script_name: str) -> subprocess.CompletedProcess:
    """
    Τρέχει ένα Python script μέσα στο Render project folder.
    ΔΕΝ καλεί τίποτα μόνο του στο startup – μόνο όταν ζητηθεί route.
    """
    print(f"🚀 Running script: {script_name}")
    env = os.environ.copy()

    result = subprocess.run(
        ["python3", script_name],
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

    return result


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
            # ➜ Χρησιμοποιούμε πλέον το v2
            script = "friday_shortlist_v2.py"
            label = "Friday Shortlist (v2)"
            report_file = "logs/friday_shortlist_v2.json"
        elif "tuesday" in command:
            script = "tuesday_recap.py"
            label = "Tuesday Recap"
            report_file = "logs/tuesday_recap_v1.json"
        else:
            return jsonify({"error": "❓ Unknown command"}), 400

        # === Run script ===
        result = run_script(script)

        # === Load JSON output (αν υπάρχει) ===
        if os.path.exists(report_file):
            with open(report_file, "r", encoding="utf-8") as f:
                report_data = json.load(f)
        else:
            report_data = {"info": "⚠️ No report file found."}

        # === Kelly picks (νέα δομή: report['kelly']['picks']) ===
        kelly_data = report_data.get("kelly", {})
        if isinstance(kelly_data, dict):
            kelly_picks = kelly_data.get("picks", []) or []
        else:
            kelly_picks = []

        # === Summary κειμενάκι για το Chat ===
        summary = f"✅ {label} ολοκληρώθηκε.\n"
        summary += f"📅 {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n"

        if kelly_picks:
            summary += "\n🎯 Top 10 Kelly Picks:\n"
            for i, pick in enumerate(kelly_picks[:10], 1):
                summary += (
                    f"\n{i}. {pick.get('match','N/A')} "
                    f"| {pick.get('league','-')} "
                    f"| {pick.get('market','-').upper()} "
                    f"| Fair: {pick.get('fair','-')} "
                    f"| Offered: {pick.get('offered','-')} "
                    f"| Diff: {pick.get('diff','-')} "
                    f"| Stake: €{pick.get('stake (€)','-')}"
                )
        else:
            summary += "\n⚠️ Δεν εντοπίστηκαν Kelly picks σε αυτό το report."

        # === Στέλνουμε στο chat ===
        send_to_chat(label, {"summary": summary})
        send_to_chat(f"{label} – Full Report", report_data)

        return jsonify(
            {
                "status": "ok",
                "message": f"{label} executed and sent",
                "stderr": result.stderr,
            }
        )

    except Exception as e:
        print(f"❌ Error executing command: {e}")
        send_to_chat("Error", {"error": str(e)})
        return jsonify({"error": str(e)}), 500


# === Manual Run Routes (browser / Postman) ===
@app.route("/run/thursday", methods=["GET"])
def run_thursday():
    try:
        result = run_script("thursday_analysis_v1.py")
        return jsonify({"status": "ok", "message": "Thursday Analysis executed."}), 200
    except Exception as e:
        print(f"❌ Manual Thursday run error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/run/friday", methods=["GET"])
def run_friday():
    try:
        result = run_script("friday_shortlist_v2.py")
        return jsonify({"status": "ok", "message": "Friday Shortlist (v2) executed."}), 200
    except Exception as e:
        print(f"❌ Manual Friday run error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/run/tuesday", methods=["GET"])
def run_tuesday():
    try:
        result = run_script("tuesday_recap.py")
        return jsonify({"status": "ok", "message": "Tuesday Recap executed."}), 200
    except Exception as e:
        print(f"❌ Manual Tuesday run error: {e}")
        return jsonify({"error": str(e)}), 500


# === Chat forward endpoint ===
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
    # ΠΟΛΥ ΣΗΜΑΝΤΙΚΟ: Δεν τρέχουμε κανένα Thursday/Friday εδώ.
    app.run(host="0.0.0.0", port=port, use_reloader=False)
