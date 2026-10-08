import unittest

from app import app


class TestPredictionApplication(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json()["status"],
            "ok"
        )

    def test_prediction(self):
        response = self.client.post(
            "/predict",
            json={
                "Sleep_Hours": 7.5,
                "Stress_Level": 5.0,
                "Previous_Exam_Scores": 75.0,
                "Study_Hours": 5.0
            }
        )

        self.assertEqual(response.status_code, 200)

        result = response.get_json()

        self.assertIn(
            "predicted_performance_score",
            result
        )

        prediction = result["predicted_performance_score"]

        self.assertIsInstance(prediction, float)

        self.assertGreaterEqual(prediction, 0)
        self.assertLessEqual(prediction, 100)

    def test_missing_field_validation(self):
        response = self.client.post(
            "/predict",
            json={
                "Sleep_Hours": 7.5,
                "Stress_Level": 5.0
            }
        )

        self.assertEqual(response.status_code, 400)

        self.assertIn(
            "missing_fields",
            response.get_json()
        )


if __name__ == "__main__":
    unittest.main()
