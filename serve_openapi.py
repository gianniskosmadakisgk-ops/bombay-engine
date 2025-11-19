from flask import Flask, jsonify, send_file
import os
import datetime

app = Flask(__name__)

# ------------------------------
# Thursday Analysis Endpoint
# ------------------------------
@app.route('/thursday-analysis', methods=['GET'])
def thursday_analysis():
    return jsonify({
        "status": "Thursday Analysis complete",
        "timestamp": datetime.datetime.utcnow().isoformat()
    })

# ------------------------------
# Friday Shortlist Endpoint
# ------------------------------
@app.route('/friday-shortlist', methods=['GET'])
def friday_shortlist():
    return jsonify({
        "status": "Friday Shortlist ready",
        "timestamp": datetime.datetime.utcnow().isoformat()
    })

# ------------------------------
# Tuesday Recap Endpoint
# ------------------------------
@app.route('/tuesday-recap', methods=['GET'])
def tuesday_recap():
    return jsonify({
        "status": "Tuesday Recap completed",
        "timestamp": datetime.datetime.utcnow().isoformat()
    })

# ------------------------------
# Serve OpenAPI YAML
# ------------------------------
@app.route('/openapi.yaml', methods=['GET'])
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
      operationId: runThursdayAnalysis
      summary: Thursday Analysis
      responses:
        '200':
          description: Thursday report ready.
  /friday-shortlist:
    get:
      operationId: runFridayShortlist
      summary: Friday Shortlist
      responses:
        '200':
          description: Friday shortlist done.
  /tuesday-recap:
    get:
      operationId: runTuesdayRecap
      summary: Tuesday Recap
      responses:
        '200':
          description: Recap complete.
"""
    return app.response_class(yaml_content, mimetype='text/yaml')

# ------------------------------
# Health Check Root
# ------------------------------
@app.route('/', methods=['GET'])
def health():
    return jsonify({"status": "Bombay Engine live", "ok": True})

# ------------------------------
# Run App
# ------------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
