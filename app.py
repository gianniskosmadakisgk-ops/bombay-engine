from flask import Flask, jsonify, request
import os
import requests

app = Flask(__name__)

# ---- HEALTH & ROOT ----
@app.route('/')
def home():
    return "✅ Bombay Engine is running and connected."

@app.route('/health')
def health():
    return jsonify({"status": "ok", "service": "bombay-engine"})

# ---- THURSDAY ANALYSIS ----
@app.route('/thursday-analysis', methods=['GET'])
def thursday_analysis():
    """
    Run Thursday analytics pipeline and return data summary.
    This connects with your internal Bombay Engine pipeline.
    """
    try:
        # External analysis engine endpoint
        url = "https://bombay-engine.onrender.com/run_thursday_analysis"
        response = requests.get(url, timeout=60)

        if response.status_code == 200:
            return jsonify({
                "status": "success",
                "analysis_result": response.json()
            })
        else:
            return jsonify({
                "status": "error",
                "message": f"Engine returned status {response.status_code}",
                "details": response.text
            }), 500

    except Exception as e:
        return jsonify({
            "status": "failed",
            "error": str(e)
        }), 500


# ---- MAIN ----
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
