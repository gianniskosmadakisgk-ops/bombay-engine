from flask import Flask, jsonify
from simulate import run_simulation
from friday import run_friday_simulation

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"status": "Bombay Engine active"})

@app.route("/simulate")
def simulate():
    return jsonify(run_simulation())

@app.route("/friday")
def friday():
    return jsonify(run_friday_simulation())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
