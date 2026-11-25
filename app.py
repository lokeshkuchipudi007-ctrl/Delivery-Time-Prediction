"""Flask app that serves delivery time predictions."""
import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

MODEL_PATH = Path(__file__).parent / "model.pkl"

NUMERIC_RANGES = {
    "distance_km": (0.1, 100),
    "prep_time_min": (0, 120),
    "courier_experience_years": (0, 40),
}
CATEGORICAL_OPTIONS = {
    "traffic": ["low", "medium", "high"],
    "weather": ["clear", "rain", "storm"],
}
FEATURES = list(NUMERIC_RANGES) + list(CATEGORICAL_OPTIONS)

app = Flask(__name__)
_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "model.pkl not found. Run `python train_model.py` first."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def parse_features(data):
    """Validate raw input and return a one-row DataFrame. Raises ValueError."""
    row = {}
    for name, (lo, hi) in NUMERIC_RANGES.items():
        raw = data.get(name)
        if raw in (None, ""):
            raise ValueError(f"'{name}' is required")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"'{name}' must be a number")
        if not lo <= value <= hi:
            raise ValueError(f"'{name}' must be between {lo} and {hi}")
        row[name] = value
    for name, options in CATEGORICAL_OPTIONS.items():
        value = str(data.get(name, "")).strip().lower()
        if value not in options:
            raise ValueError(f"'{name}' must be one of {options}")
        row[name] = value
    return pd.DataFrame([row], columns=FEATURES)


def predict_minutes(data):
    X = parse_features(data)
    return round(float(get_model().predict(X)[0]), 1)


@app.get("/")
def index():
    return render_template("index.html", options=CATEGORICAL_OPTIONS, form={})


@app.post("/")
def index_post():
    form = request.form.to_dict()
    try:
        minutes = predict_minutes(form)
        return render_template(
            "index.html", options=CATEGORICAL_OPTIONS, form=form, prediction=minutes
        )
    except ValueError as exc:
        return (
            render_template(
                "index.html", options=CATEGORICAL_OPTIONS, form=form, error=str(exc)
            ),
            400,
        )


@app.post("/api/predict")
def api_predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error="Request body must be a JSON object"), 400
    try:
        return jsonify(delivery_time_min=predict_minutes(data))
    except ValueError as exc:
        return jsonify(error=str(exc)), 400


@app.get("/health")
def health():
    return jsonify(status="ok", model_loaded=MODEL_PATH.exists())


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=os.environ.get("FLASK_DEBUG") == "1",
    )
