"""Flask API.
  POST /train    {"test": false}                       -> trains all models
  POST /predict  {"country": "all", "date": "2019-06-01"} -> 30-day revenue forecast
  GET  /predict?country=france&date=2019-06-01          -> same, via query string
  GET  /monitor                                         -> performance report
  GET  /health
"""
import os
from flask import Flask, jsonify, request
import model
from monitoring import performance_report

app = Flask(__name__)


def _params():
    body = request.get_json(silent=True) or {}
    return {**request.args.to_dict(), **body}


def _flag(v):
    return str(v).lower() in ("1", "true", "yes")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/train", methods=["POST"])
def train():
    p = _params()
    try:
        results = model.model_train(data_dir=p.get("data_dir"), test=_flag(p.get("test", False)),
                                    top_n=int(p.get("top_n", 10)), plot_dir=p.get("plot_dir"))
    except (FileNotFoundError, ValueError) as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"trained": len(results), "results": results})


@app.route("/predict", methods=["GET", "POST"])
def predict():
    p = _params()
    if not p.get("country") or not p.get("date"):
        return jsonify({"error": "country and date are required"}), 400
    try:
        out = model.model_predict(p["country"], p["date"], test=_flag(p.get("test", False)),
                                  data_dir=p.get("data_dir"))
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify(out)


@app.route("/monitor")
def monitor():
    return jsonify(performance_report(test=_flag(request.args.get("test", False))))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
