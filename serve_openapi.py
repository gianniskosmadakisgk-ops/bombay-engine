from flask import Flask, jsonify, send_from_directory
import os
import datetime

app = Flask(__name__)

# Thursday Analysis Endpoint
@app.route("/thursday-analysis", methods=["GET"])
def thursday_analysis():
    return jsonify({
        "status": "Thursday Analysis complete",
        "timestamp": datetime.datetime.utcnow().isoformat()
    })

# Friday Shortlist Endpoint
@app.route("/friday-shortlist", methods=["GET"])
def friday_shortlist():
    return jsonify({
        "status": "Friday Shortlist ready",
        "timestamp": datetime.datetime.utcnow().isoformat()
    })

# Tuesday Recap Endpoint
@app.route("/tuesday-recap", methods=["GET"])
def tuesday_recap():
    return jsonify({
        "status": "Tuesday Recap completed",
        "timestamp": datetime.datetime.utcnow().isoformat()
    })

# Serve the OpenAPI YAML
@app.route("/openapi.yaml", methods=["GET"])
def serve_openapi():
    directory = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(directory, "openapi.yaml", mimetype="text/yaml")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
