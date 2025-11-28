from flask import Flask, request, jsonify
import subprocess
import requests
import json
import os
import sys

# -----------------------------------------------------------
# 🧠 Real-time logging fix for Render
# -----------------------------------------------------------
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

# -----------------------------------------------------------
# Flask Setup
# -----------------------------------------------------------
app = Flask(__name__)

CHAT_FORWARD_URL = "https://bombay-engine.onrender.com/chat_forward"

# -----------------------------------------------------------
# Helper: Send full data to Chat
# -----------------------------------------------------------
def send_to_chat(title, data):
    try:
        print(f"📤 Sending full report to chat: {title}")
        response = requests.post(
            CHAT_FORWARD_URL,
            json={"message": f"📊 {title}", "data": data},
            timeout=20
        )
        print(f"✅ Chat forward status: {response.status_code}")
    except Exception as e:
        print(f"⚠️ Chat forward error: {e}")

# -----------------------------------------------------------
# Chat Command Endpoint
# -----------------------------------------------------------
@app.route("/chat_command", methods=["POST"])
def chat_command():
    try:
        print("📩 Received /chat_command request")

        data = request.get_json(force=True)
        command = (data.get("command", "") or "").lower().strip()
        print(f"🧭 Command detected: {command}")

        if "thursday" in command:
            script = "thursday_analysis_v1.py"
            label = "Thursday Analysis"
            report_file = "logs/thursday_output.json"
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

        print(f"🚀 Executing {label} ({script})")

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

        # -----------------------------------------------------------
        # Read JSON output if exists
        # -----------------------------------------------------------
        report_data = {}
        if os.path.exists(report_file):
            with open(report_file, "r", encoding="utf-8") as f:
                report_data = json.load(f)
        else:
            report_data = {"info": "⚠️ No report file found."}

        # -----------------------------------------------------------
        # Prepare compact chat summary
        # -----------------------------------------------------------
        summary = report_data.get("summary", {})
        draw_picks = report_data.get("draw_picks", [])
        over_picks = report_data.get("over_picks", [])
        kelly_picks = report_data.get("fraction_kelly", [])

        full_text = f"""
✅ {label} ολοκληρώθηκε.

📊 Summary:
Draw Picks: {len(draw_picks)}
Over Picks: {len(over_picks)}
Kelly Picks: {len(kelly_picks)}

🎯 Top 5 Kelly Value Picks:
"""
        for i, k in enumerate(kelly_picks[:5], 1):
            full_text += (
                f"\n{i}. {k['match']} | {k['market'].upper()} | "
                f"Fair: {k['fair']} | Offered: {k['offered']} | "
                f"Diff: {k['diff%']}% | Stake: €{k['stake (€)']}"
            )

        # -----------------------------------------------------------
        # Send compact summary first
        # -----------------------------------------------------------
        send_to_chat(label, {"summary": full_text.strip()})

        # -----------------------------------------------------------
        # Then send full JSON report (all matches)
        # -----------------------------------------------------------
        send_to_chat(f"{label} – Full Report", report_data)

        return jsonify({"status": "ok", "message": f"{label} executed and sent"})

    except Exception as e:
        print(f"⚠️ Error executing command: {e}")
        send_to_chat("Error", {"error": str(e)})
        return jsonify({"error": str(e)}), 500

# -----------------------------------------------------------
# Chat Forward Endpoint
# -----------------------------------------------------------
@app.route("/chat_forward", methods=["POST"])
def chat_forward():
    try:
        data = request.get_json()
        print("💬 Incoming chat message:", data)
        return jsonify({"status": "received"}), 200
    except Exception as e:
        print(f"⚠️ Error in chat_forward: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

# -----------------------------------------------------------
# Healthcheck
# -----------------------------------------------------------
@app.route("/healthcheck", methods=["GET"])
def healthcheck():
    return jsonify({"message": "Server running", "status": "ok"})

# -----------------------------------------------------------
# Main Entry
# -----------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"🟢 Starting Bombay Engine Flask Server on port {port}...")
    app.run(host="0.0.0.0", port=port, use_reloader=False)
