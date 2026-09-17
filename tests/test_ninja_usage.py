import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import app
import model
import ninja_usage as usage


class NinjaUsageTests(unittest.TestCase):
    def setUp(self):
        with usage._LOCK:
            usage._USAGE.clear()

    def test_per_key_rolling_window_and_nonnegative_remaining(self):
        with patch.object(usage.time, "time", return_value=10000):
            for _ in range(52):
                usage.record_request("test-key-a")
            self.assertEqual(usage.usage_snapshot("test-key-a")["remaining"], 0)
            self.assertEqual(usage.usage_snapshot("test-key-b")["remaining"], 50)
            self.assertNotIn("test-key-a", usage._USAGE)
        with patch.object(usage.time, "time", return_value=13599):
            self.assertEqual(usage.usage_snapshot("test-key-a")["next_release_seconds"], 1)
            usage.record_request("test-key-a")
        with patch.object(usage.time, "time", return_value=13600):
            self.assertEqual(usage.usage_snapshot("test-key-a")["remaining"], 49)

    def test_retry_after_seconds_date_and_invalid_header(self):
        with patch.object(usage.time, "time", return_value=10000):
            usage.record_response("test", SimpleNamespace(status_code=429, headers={"Retry-After": "120"}))
            self.assertEqual(usage.usage_snapshot("test")["retry_seconds"], 120)
            usage.record_response("test", SimpleNamespace(status_code=429, headers={"Retry-After": "invalid"}))
            self.assertEqual(usage.usage_snapshot("test")["retry_seconds"], 120)
        with patch.object(usage.time, "time", return_value=0):
            usage.record_response("date", SimpleNamespace(
                status_code=429, headers={"Retry-After": "Thu, 01 Jan 1970 00:02:00 GMT"}))
            self.assertEqual(usage.usage_snapshot("date")["retry_seconds"], 120)
        with patch.object(usage.time, "time", return_value=10121):
            self.assertEqual(usage.usage_snapshot("test")["retry_seconds"], 0)

    def test_weather_attempts_include_retries_and_server_throttle(self):
        session = Mock()
        session.get.side_effect = [
            SimpleNamespace(status_code=503, headers={}, text="Unavailable"),
            SimpleNamespace(status_code=429, headers={"Retry-After": "300"}, text="Throttled"),
        ]
        with patch.object(model.time, "sleep"), self.assertRaises(model.WeatherDataError):
            model._request_ninja_year(model.WeatherConfig(51, 0), 2025, "test",
                                      session=session, max_retries=1)
        snapshot = usage.usage_snapshot("test")
        self.assertEqual(snapshot["used"], 2)
        self.assertGreater(snapshot["retry_seconds"], 295)

    def test_counter_callback_does_not_call_ninja(self):
        application = app.create_app()
        callback = application.callback_map["ninja-usage.children"]["callback"].__wrapped__
        usage.record_request("test")
        with patch.object(app, "_resolve_ninja_api_key", return_value="test"), \
                patch.object(model.requests.Session, "get") as request:
            displayed = callback(1, "ninja", None)
            self.assertIn("~49/50", displayed.children[0])
            self.assertIn("Synthetic uses no calls", callback(2, "demo", None).children[0])
            request.assert_not_called()
        with patch.object(app, "_resolve_ninja_api_key", return_value=None):
            self.assertIn("enter an API key", callback(3, "ninja", None))
