
import json
import os
import unittest

import joblib
import pandas as pd


class TestMLPipeline(unittest.TestCase):

    def test_dataset_created(self):
        self.assertTrue(os.path.exists("SrudentPerformanceScore.csv"))

    def test_model_created(self):
        self.assertTrue(
            os.path.exists("student_performance_model.pkl")
        )

    def test_metrics_created(self):
        self.assertTrue(os.path.exists("metrics.json"))

    def test_metrics_are_valid(self):
        with open("metrics.json", "r") as file:
            metrics = json.load(file)

        self.assertIn("mae", metrics)
        self.assertIn("rmse", metrics)
        self.assertIn("r2", metrics)

        self.assertGreaterEqual(metrics["mae"], 0)
        self.assertGreaterEqual(metrics["rmse"], 0)

    def test_model_prediction(self):
        model = joblib.load("student_performance_model.pkl")

        sample = pd.DataFrame([{
            "Sleep_Hours": 7.5,
            "Stress_Level": 5.0,
            "Previous_Exam_Scores": 75.0,
            "Study_Hours": 5.0
        }])

        prediction = model.predict(sample)[0]

        self.assertIsInstance(float(prediction), float)

    def test_prediction_is_reasonable(self):
        model = joblib.load("student_performance_model.pkl")

        sample = pd.DataFrame([{
            "Sleep_Hours": 7.5,
            "Stress_Level": 5.0,
            "Previous_Exam_Scores": 75.0,
            "Study_Hours": 5.0
        }])

        prediction = model.predict(sample)[0]

        self.assertGreaterEqual(prediction, 0)
        self.assertLessEqual(prediction, 100)


if __name__ == "__main__":
    unittest.main()
