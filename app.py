from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request


app = Flask(__name__)

MODEL_PATH = Path("student_performance_model.pkl")

FEATURES = [
    "Sleep_Hours",
    "Stress_Level",
    "Previous_Exam_Scores",
    "Study_Hours"
]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "student_performance_model.pkl was not found. "
            "Run the training pipeline first."
        )

    return joblib.load(MODEL_PATH)


@app.get("/")
def health_check():
    return jsonify({
        "status": "ok",
        "service": "student-performance-prediction"
    })


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({
            "error": "JSON request body is required"
        }), 400

    missing_fields = [
        feature for feature in FEATURES
        if feature not in data
    ]

    if missing_fields:
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields
        }), 400

    sample = pd.DataFrame([{
        feature: data[feature]
        for feature in FEATURES
    }])

    model = load_model()

    prediction = float(model.predict(sample)[0])

    return jsonify({
        "predicted_performance_score": prediction
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
