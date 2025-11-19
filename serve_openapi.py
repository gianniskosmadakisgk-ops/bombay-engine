from flask import Flask, jsonify, Response
import datetime

app = Flask(__name__)

# Endpoints
@app.route("/thursday-analysis", methods=["GET"])
def thursday_analysis():
    return jsonify({
        "status": "Thursday Analysis complete",
        "timestamp": datetime.datetime.utcnow().isoformat()
    })

@app.route("/friday-shortlist", methods=["GET"])
def friday_shortlist():
    return jsonify({
        "status": "Friday Shortlist ready",
        "timestamp": datetime.datetime.utcnow().isoformat()
    })

@app.route("/tuesday-recap", methods=["GET"])
def tuesday_recap():
    return jsonify({
        "status": "Tuesday Recap completed",
        "timestamp": datetime.datetime.utcnow().isoformat()
    })

# Serve the OpenAPI YAML dynamically
@app.route("/openapi.yaml", methods=["GET"])
def serve_openapi():
    yaml_content = """openapi: 3.0.1
info:
  title: Bombay Engine API
  version: "1.0.0"
  description: API endpoints for internal automation.
servers:
  - url: https://bombay-engine.onrender.com
paths:
  /thursday-analysis:
    get:
      summary: Thursday Analysis
      responses:
        '200':
          description: Thursday report ready.
  /friday-shortlist:
    get:
      summary: Friday Shortlist
      responses:
        '200':
          description: Friday shortlist done.
  /tuesday-recap:
    get:
      summary: Tuesday Recap
      responses:
        '200':
          description: Recap complete.
"""
    return Response(yaml_content, mimetype="text/yaml")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
