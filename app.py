
import logging
import os
import time
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request, g


app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("student-performance-api")

MODEL_PATH = Path("student_performance_model.pkl")

FEATURES = [
    "Sleep_Hours",
    "Stress_Level",
    "Previous_Exam_Scores",
    "Study_Hours"
]


@app.before_request
def log_request_start():
    g.request_start_time = time.time()
    g.request_id = os.urandom(8).hex()

    logger.info(
        "Request started | id=%s method=%s path=%s",
        g.request_id,
        request.method,
        request.path
    )


@app.after_request
def log_request_end(response):
    duration = time.time() - getattr(
        g, "request_start_time", time.time()
    )

    logger.info(
        "Request completed | id=%s method=%s path=%s status=%s duration_ms=%.2f",
        getattr(g, "request_id", "unknown"),
        request.method,
        request.path,
        response.status_code,
        duration * 1000
    )
    return response



@app.errorhandler(404)
def handle_not_found(error):
    logger.info(
        "Resource not found | id=%s path=%s",
        getattr(g, "request_id", "unknown"),
        request.path
    )
    return jsonify({
        "error": "Not found"
    }), 404


@app.errorhandler(Exception)
def handle_unexpected_error(error):
    from werkzeug.exceptions import HTTPException

    if isinstance(error, HTTPException):
        return error

    logger.error(
        "Unexpected application error | id=%s type=%s",
        getattr(g, "request_id", "unknown"),
        type(error).__name__
    )

    return jsonify({
        "error": "Internal server error",
        "request_id": getattr(g, "request_id", "unknown")
    }), 500



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

    try:
        sample = pd.DataFrame([{
            feature: data[feature]
            for feature in FEATURES
        }])

        model = load_model()
        prediction = float(model.predict(sample)[0])

        return jsonify({
            "predicted_performance_score": prediction
        })

    except Exception:
        logger.exception(
            "Prediction failed | id=%s",
            getattr(g, "request_id", "unknown")
        )
        return jsonify({
            "error": "Prediction could not be completed",
            "request_id": getattr(g, "request_id", "unknown")
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )
