"""Concurrency contracts: isolated thresholds, bounded CPU work and responsive health."""

from concurrent.futures import ThreadPoolExecutor
import io
from threading import Event, Lock
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from backend.config import INFERENCE_CONCURRENCY
from backend.main import app, advisory_engine, inference_engine


class ApiConcurrencyTests(unittest.TestCase):
    @staticmethod
    def image():
        stream = io.BytesIO()
        Image.new("RGB", (64, 64), "green").save(stream, format="PNG")
        return stream.getvalue()

    @staticmethod
    def prediction():
        return {"predicted_class": "Tomato_Early_blight", "confidence": .5, "crop": "Tomato", "latency_ms": 1.}

    def test_thresholds_are_isolated_and_cpu_work_is_bounded(self):
        state = {"active": 0, "peak": 0}
        lock = Lock()
        original = advisory_engine.threshold

        def predict(_):
            with lock:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            time.sleep(.02)
            with lock:
                state["active"] -= 1
            return self.prediction()

        with patch.object(inference_engine, "predict", side_effect=predict), TestClient(app) as client:
            image = self.image()

            def request(index):
                threshold = .1 if index % 2 == 0 else .9
                response = client.post("/api/diagnose", params={"custom_threshold": threshold},
                                       files={"file": ("leaf.png", image, "image/png")})
                self.assertEqual(response.status_code, 200)
                self.assertIn("processing;dur=", response.headers.get("server-timing", ""))
                payload = response.json()
                self.assertEqual(payload["prediction"]["threshold_applied"], threshold)
                self.assertEqual(payload["diagnosis_status"], "CONFIRMED" if threshold == .1 else "UNCERTAIN")

            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(request, range(32)))
        self.assertLessEqual(state["peak"], INFERENCE_CONCURRENCY)
        self.assertEqual(advisory_engine.threshold, original)

    def test_health_does_not_wait_for_inference(self):
        started, release = Event(), Event()

        def predict(_):
            started.set()
            release.wait(timeout=2)
            return self.prediction()

        with patch.object(inference_engine, "predict", side_effect=predict), TestClient(app) as client:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(client.post, "/api/diagnose",
                                     files={"file": ("leaf.png", self.image(), "image/png")})
                try:
                    self.assertTrue(started.wait(timeout=1))
                    before = time.perf_counter()
                    self.assertEqual(client.get("/api/health").status_code, 200)
                    self.assertLess(time.perf_counter() - before, .5)
                finally:
                    release.set()
                self.assertEqual(future.result(timeout=2).status_code, 200)


if __name__ == "__main__":
    unittest.main()
