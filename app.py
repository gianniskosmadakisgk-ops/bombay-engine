from flask import Flask, jsonify
import datetime
import json
from friday import runFridayShortlist
from thursday import runThursdayAnalysis
from tuesday import runTuesdayRecap

app = Flask(__name__)

# ====== ΕΝΤΟΛΕΣ ΠΟΥ ΤΡΕΧΟΥΝ ΤΗΝ ΑΝΑΛΥΣΗ ======

@app.route("/thursday-analysis", methods=["GET"])
def thursday_analysis():
    try:
        data = runThursdayAnalysis()
        return jsonify({
            "status": "Thursday Analysis Complete",
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "analysis": data
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/friday-shortlist", methods=["GET"])
def friday_shortlist():
    try:
        data = runFridayShortlist()
        return jsonify({
            "status": "Friday Shortlist Complete",
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "shortlist": data
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/tuesday-recap", methods=["GET"])
def tuesday_recap():
    try:
        data = runTuesdayRecap()
        return jsonify({
            "status": "Tuesday Recap Complete",
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "recap": data
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ====== ΡΙΖΑ ΓΙΑ ΕΛΕΓΧΟ ======
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Bombay Engine API is active.",
        "available_endpoints": [
            "/thursday-analysis",
            "/friday-shortlist",
            "/tuesday-recap"
        ],
        "timestamp": datetime.datetime.utcnow().isoformat()
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
