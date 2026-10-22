
import logging
import os
import time
from pathlib import Path

import joblib
import pandas as pd

from flask import (
    Flask,
    Response,
    g,
    jsonify,
    request,
)

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)


# ==========================================
# 1. FLASK APPLICATION
# ==========================================

app = Flask(__name__)


# ==========================================
# 2. LOGGING CONFIGURATION
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger("student-performance-api")


# ==========================================
# 3. MODEL CONFIGURATION
# ==========================================

MODEL_PATH = Path(__file__).resolve().parent / "student_performance_model.pkl"

FEATURES = [
    "Sleep_Hours",
    "Stress_Level",
    "Previous_Exam_Scores",
    "Study_Hours",
]


# ==========================================
# 4. PROMETHEUS METRICS
# ==========================================

HTTP_REQUESTS = Counter(
    "student_api_http_requests_total",
    "Total HTTP requests received",
    ["method", "path", "status"],
)

HTTP_LATENCY = Histogram(
    "student_api_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"],
)

PREDICTIONS = Counter(
    "student_api_predictions_total",
    "Total prediction attempts",
    ["result"],
)

MODEL_LOADED = Gauge(
    "student_api_model_loaded",
    "Whether the trained model is loaded",
)


# ==========================================
# 5. MODEL LOADING
# ==========================================

def load_model():
    if not MODEL_PATH.is_file():
        MODEL_LOADED.set(0)

        logger.error(
            "Model file missing | path=%s",
            MODEL_PATH.name,
        )

        raise FileNotFoundError(
            "student_performance_model.pkl was not found. "
            "Run the training pipeline first."
        )

    try:
        model = joblib.load(MODEL_PATH)

        MODEL_LOADED.set(1)

        logger.info(
            "Model loaded successfully | file=%s",
            MODEL_PATH.name,
        )

        return model

    except Exception:
        MODEL_LOADED.set(0)

        logger.exception(
            "Model loading failed | file=%s",
            MODEL_PATH.name,
        )

        raise


# Check model readiness when the API starts.
# Prediction requests will continue loading the
# model using the existing project behavior.

try:
    load_model()
except Exception:
    logger.warning(
        "Model unavailable at startup; "
        "prediction requests may fail."
    )


# ==========================================
# 6. REQUEST START LOGGING
# ==========================================

@app.before_request
def log_request_start():
    g.request_start_time = time.perf_counter()
    g.request_id = os.urandom(8).hex()

    logger.info(
        "Request started | id=%s method=%s path=%s",
        g.request_id,
        request.method,
        request.path,
    )


# ==========================================
# 7. REQUEST END LOGGING + METRICS
# ==========================================

@app.after_request
def log_request_end(response):
    duration = time.perf_counter() - getattr(
        g,
        "request_start_time",
        time.perf_counter(),
    )

    logger.info(
        "Request completed | "
        "id=%s method=%s path=%s "
        "status=%s duration_ms=%.2f",
        getattr(g, "request_id", "unknown"),
        request.method,
        request.path,
        response.status_code,
        duration * 1000,
    )

    # Exclude Prometheus scrapes from
    # ordinary application traffic metrics.

    if request.path != "/metrics":
        HTTP_REQUESTS.labels(
            method=request.method,
            path=request.path,
            status=str(response.status_code),
        ).inc()

        HTTP_LATENCY.labels(
            method=request.method,
            path=request.path,
        ).observe(duration)

    return response


# ==========================================
# 8. ERROR HANDLING - 404
# ==========================================

@app.errorhandler(404)
def handle_not_found(error):
    logger.info(
        "Resource not found | id=%s path=%s",
        getattr(g, "request_id", "unknown"),
        request.path,
    )

    return jsonify({
        "error": "Not found",
    }), 404


# ==========================================
# 9. ERROR HANDLING - UNEXPECTED ERRORS
# ==========================================

@app.errorhandler(Exception)
def handle_unexpected_error(error):
    from werkzeug.exceptions import HTTPException

    if isinstance(error, HTTPException):
        return error

    logger.exception(
        "Unexpected application error | id=%s",
        getattr(g, "request_id", "unknown"),
    )

    return jsonify({
        "error": "Internal server error",
        "request_id": getattr(
            g,
            "request_id",
            "unknown",
        ),
    }), 500


# ==========================================
# 10. HEALTH CHECK ENDPOINT
# ==========================================

@app.get("/")
def health_check():
    return jsonify({
        "status": "ok",
        "service": "student-performance-prediction",
    })


# ==========================================
# 11. PREDICTION ENDPOINT
# ==========================================

@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({
            "error": "JSON request body is required",
        }), 400

    missing_fields = [
        feature
        for feature in FEATURES
        if feature not in data
    ]

    if missing_fields:
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields,
        }), 400

    try:
        sample = pd.DataFrame([{
            feature: data[feature]
            for feature in FEATURES
        }])

        model = load_model()

        prediction = float(
            model.predict(sample)[0]
        )

        # Count successful predictions.
        PREDICTIONS.labels(
            result="success"
        ).inc()

        logger.info(
            "Prediction succeeded | id=%s",
            getattr(g, "request_id", "unknown"),
        )

        return jsonify({
            "predicted_performance_score": prediction,
        })

    except Exception:
        # Count prediction failures.
        PREDICTIONS.labels(
            result="error"
        ).inc()

        logger.exception(
            "Prediction failed | id=%s",
            getattr(g, "request_id", "unknown"),
        )

        return jsonify({
            "error": "Prediction could not be completed",
            "request_id": getattr(
                g,
                "request_id",
                "unknown",
            ),
        }), 500


# ==========================================
# 12. PROMETHEUS METRICS ENDPOINT
# ==========================================

@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        content_type=CONTENT_TYPE_LATEST,
    )


# ==========================================
# 13. START FLASK APPLICATION
# ==========================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
    )
