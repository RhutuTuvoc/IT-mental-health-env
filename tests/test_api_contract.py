import unittest

from fastapi.testclient import TestClient

from app import app
from server.app import session_store


class ApiContractTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.other_client = TestClient(app)
        session_store.clear()

    def test_step_requires_reset(self):
        response = self.client.post(
            "/step",
            json={"response": "Structured response", "task_id": "burnout_detection"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Call /reset before /step", response.json()["detail"])

    def test_step_rejects_task_id_mismatch(self):
        reset_response = self.client.post("/reset", json={})
        self.assertEqual(reset_response.status_code, 200)

        response = self.client.post(
            "/step",
            json={"response": "Structured response", "task_id": "intervention_plan"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("task_id mismatch", response.json()["detail"])

    def test_cookie_session_keeps_clients_isolated(self):
        first_reset = self.client.post("/reset", json={})
        second_reset = self.other_client.post("/reset", json={})

        self.assertEqual(first_reset.status_code, 200)
        self.assertEqual(second_reset.status_code, 200)

        first_step = self.client.post(
            "/step",
            json={"response": "Structured response", "task_id": "burnout_detection"},
        )
        self.assertEqual(first_step.status_code, 200)

        first_state = self.client.get("/state")
        second_state = self.other_client.get("/state")

        self.assertEqual(first_state.status_code, 200)
        self.assertEqual(second_state.status_code, 200)
        self.assertEqual(first_state.json()["step_count"], 1)
        self.assertEqual(second_state.json()["step_count"], 0)
        self.assertNotEqual(first_state.json()["session_id"], second_state.json()["session_id"])


if __name__ == "__main__":
    unittest.main()
