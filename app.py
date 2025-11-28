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
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

app = Flask(__name__)

CHAT_FORWARD_URL = "https://bombay-engine.onrender.com/chat_forward"

# === Utility: Send structured data to chat ===
def send_to_chat(title, data):
    """Send structured data to chat (logging-safe)"""
    try:
        print(f"📤 Sending report to chat: {title}")
        response = requests.post(
            CHAT_FORWARD_URL,
            json={"message": f"📊 {title}", "data": data},
            timeout=25
        )
        print(f"✅ Chat forward status: {response.status_code}")
    except Exception as e:
        print(f"⚠️ Chat forward error: {e}")

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
            script = "friday_shortlist_v1.py"
            label = "Friday Shortlist"
            report_file = "logs/friday_shortlist_v1.json"
        elif "tuesday" in command:
            script = "tuesday_recap.py"
            label = "Tuesday Recap"
            report_file = "logs/tuesday_recap_v1.json"
        else:
            return jsonify({"error": "❓ Unknown command"}), 400

        # === Run corresponding script ===
        print(f"🚀 Running {label} ({script})")
        env = os.environ.copy()

        result = subprocess.run(
            ["python3", script],
            cwd="/opt/render/project/src",
            env=env,
            capture_output=True,
            text=True
        )

        print("----- SCRIPT OUTPUT START -----")
        print(result.stdout)
        print("----- SCRIPT OUTPUT END -----")

        if result.stderr:
            print("⚠️ SCRIPT ERRORS:")
            print(result.stderr)

        # === Load JSON output if exists ===
        report_data = {}
        if os.path.exists(report_file):
            with open(report_file, "r", encoding="utf-8") as f:
                report_data = json.load(f)
        else:
            report_data = {"info": "⚠️ No report file found."}

        # === Handle Kelly picks safely ===
        kelly_data = report_data.get("fraction_kelly", {})
        kelly_picks = kelly_data.get("picks", []) if isinstance(kelly_data, dict) else []

        # === Compose chat summary ===
        summary = f"✅ {label} ολοκληρώθηκε.\n\n"
        summary += f"📅 {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n"

        if kelly_picks:
            summary += f"\n🎯 Top 10 Kelly Value Picks:\n"
            for i, k in enumerate(kelly_picks[:10], 1):
                summary += (
                    f"\n{i}. {k.get('match','N/A')} | {k.get('market','-').upper()} | "
                    f"Fair: {k.get('fair','-')} | Offered: {k.get('offered','-')} | "
                    f"Diff: {k.get('diff%','-')}% | Stake: €{k.get('stake (€)','-')}"
                )
        else:
            summary += "\n⚠️ Δεν εντοπίστηκαν Kelly picks σε αυτό το report."

        # === Send to chat ===
        send_to_chat(label, {"summary": summary})
        send_to_chat(f"{label} – Full Report", report_data)

        return jsonify({"status": "ok", "message": f"{label} executed and sent"})

    except Exception as e:
        print(f"❌ Error executing command: {e}")
        send_to_chat("Error", {"error": str(e)})
        return jsonify({"error": str(e)}), 500

# === Chat forward endpoint ===
@app.route("/chat_forward", methods=["POST"])
def chat_forward():
    try:
        data = request.get_json()
        print("💬 Incoming chat message:", data)
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
