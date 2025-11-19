from flask import Flask, send_file, jsonify
import os

app = Flask(__name__)

@app.route("/openapi.yaml", methods=["GET"])
def serve_openapi():
    file_path = os.path.join(os.path.dirname(__file__), "openapi.yaml")
    if os.path.exists(file_path):
        return send_file(file_path, mimetype="text/yaml")
    else:
        return jsonify({"error": "openapi.yaml not found"}), 404

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "OpenAPI route active ✅",
        "available": ["/openapi.yaml"]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
