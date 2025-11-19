from flask import Flask, jsonify, send_file
import os
import datetime
import json

app = Flask(__name__)

# -------------------------------
# Thursday Analysis Endpoint
# -------------------------------
@app.route("/thursday-analysis", methods=["GET"])
def thursday_analysis():
    return jsonify({
        "status": "Thursday Analysis complete",
        "timestamp": datetime.datetime.utcnow().isoformat()
    })

# -------------------------------
# Friday Shortlist Endpoint
# -------------------------------
@app.route("/friday-shortlist", methods=["GET"])
def friday_shortlist():
    return jsonify({
        "status": "Friday Shortlist ready",
        "timestamp": datetime.datetime.utcnow().isoformat()
    })

# -------------------------------
# Tuesday Recap Endpoint
# -------------------------------
@app.route("/tuesday-recap", methods=["GET"])
def tuesday_recap():
    return jsonify({
        "status": "Tuesday Recap completed",
        "timestamp": datetime.datetime.utcnow().isoformat()
    })

# -------------------------------
# Serve the OpenAPI YAML
# -------------------------------
@app.route("/openapi.yaml", methods=["GET"])
def serve_openapi():
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "openapi.yaml")
    if os.path.exists(file_path):
        return send_file(file_path, mimetype="text/yaml")
    else:
        return jsonify({"error": f"File not found at {file_path}"}), 404

# -------------------------------
# Health check route
# -------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Bombay Engine API active", "status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
