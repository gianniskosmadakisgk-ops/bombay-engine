from flask import Flask, jsonify, request
import os
import requests

app = Flask(__name__)

# ---- ROUTES ----

@app.route('/')
def home():
    return "✅ Bombay Engine is running and connected."

@app.route('/health')
def health():
    return jsonify({"status": "ok", "service": "bombay-engine"})

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Example endpoint for football data analysis.
    This will later call your logic for draws/over predictions.
    """
    data = request.get_json(force=True)
    # Dummy response for now
    return jsonify({
        "message": "Data received successfully.",
        "data_preview": str(data)[:100]
    })

# ---- MAIN ----

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
