"""Tests for the /api/humanize model fallback chain and quota refund.

Standard library only (unittest), no network: the OpenRouter client (httpx.AsyncClient used by
app.services.llm_humanizer) is replaced by a scripted fake.

Run from backend/:  python -m unittest discover -s tests -t . -v
"""
import asyncio
import json
import os
import sqlite3
import tempfile
import time
import unittest
from unittest import mock

# Must be set before importing the app: DB_PATH is read at import time.
_TMP = tempfile.TemporaryDirectory()
os.environ["DB_PATH"] = os.path.join(_TMP.name, "test.db")
os.environ["OPENROUTER_API_KEY"] = "test-key-not-real"
os.environ["RATE_LIMIT_PER_DAY"] = "3"
os.environ.pop("HUMANIZE_MODELS", None)

import httpx  # noqa: E402

from app.services import llm_humanizer  # noqa: E402
from app.services import rate_limiter  # noqa: E402

M1, M2, M3, M4 = llm_humanizer.DEFAULT_MODELS
USER_TEXT = (
    "Artificial intelligence has transformed the way businesses operate. Furthermore, it is important "
    "to note that organizations must adapt quickly to remain competitive in an evolving landscape."
)


def ok_body(text: str = "Rewritten text that sounds like a person wrote it, honestly.") -> dict:
    content = json.dumps({"humanized_text": text, "changes_made": ["shorter sentences"]})
    return {"choices": [{"message": {"content": content}}]}


class FakeOpenRouter:
    """Scripted stand-in for httpx.AsyncClient. script: model -> {"status", "body", "delay"}."""

    def __init__(self, script: dict):
        self.script = script
        self.calls: list[str] = []
        self.timeouts: list[float] = []

    def client_class(self):
        fake = self

        class _Client:
            def __init__(self, timeout=None, **_):
                fake.timeouts.append(timeout)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def post(self, url, json=None, headers=None):
                model = json["model"]
                fake.calls.append(model)
                step = fake.script[model]
                if step.get("delay"):
                    await asyncio.sleep(step["delay"])
                return httpx.Response(step["status"], json=step.get("body", {}))

        return _Client


def patch_client(fake: FakeOpenRouter):
    return mock.patch.object(llm_humanizer.httpx, "AsyncClient", fake.client_class())


class HumanizeChainTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, fake: FakeOpenRouter):
        with patch_client(fake), self.assertLogs(llm_humanizer.logger, level="INFO") as logs:
            result, ok = await llm_humanizer.humanize_text(USER_TEXT, "en")
        lines = [r.getMessage() for r in logs.records]
        # Privacy: never log the user text, the prompt or the model output.
        for line in lines:
            self.assertNotIn("Artificial intelligence", line)
            self.assertNotIn("Rewritten text", line)
        return result, ok, [l for l in lines if l.startswith("llm_request")]

    async def test_success_on_first_model(self):
        fake = FakeOpenRouter({M1: {"status": 200, "body": ok_body()}})
        result, ok, lines = await self._run(fake)
        self.assertTrue(ok)
        self.assertEqual(result["changes_made"], ["shorter sentences"])
        self.assertEqual(fake.calls, [M1])
        self.assertEqual(lines, [f"llm_request endpoint=humanize model={M1} attempts=1 errors=-"])

    async def test_404_on_first_then_success_on_second_without_waiting(self):
        fake = FakeOpenRouter({
            M1: {"status": 404, "body": {"error": {"message": "No endpoints found"}}},
            M2: {"status": 200, "body": ok_body()},
        })
        start = time.monotonic()
        result, ok, lines = await self._run(fake)
        self.assertLess(time.monotonic() - start, 1.0, "404 must skip to the next model without delay")
        self.assertTrue(ok)
        self.assertEqual(fake.calls, [M1, M2])
        self.assertEqual(lines, [f"llm_request endpoint=humanize model={M2} attempts=2 errors={M1}:http_404"])

    async def test_400_is_skipped_and_other_errors_fall_through(self):
        fake = FakeOpenRouter({
            M1: {"status": 400},
            M2: {"status": 502},
            M3: {"status": 200, "body": {"choices": [{"message": {"content": "not json at all"}}]}},
            M4: {"status": 200, "body": ok_body()},
        })
        _, ok, lines = await self._run(fake)
        self.assertTrue(ok)
        self.assertEqual(fake.calls, [M1, M2, M3, M4])
        self.assertEqual(
            lines,
            [f"llm_request endpoint=humanize model={M4} attempts=4 errors={M1}:http_400,{M2}:http_502,{M3}:malformed"],
        )

    async def test_all_models_fail(self):
        fake = FakeOpenRouter({m: {"status": 404} for m in llm_humanizer.DEFAULT_MODELS})
        result, ok, lines = await self._run(fake)
        self.assertIsNone(result)
        self.assertFalse(ok)
        self.assertEqual(fake.calls, list(llm_humanizer.DEFAULT_MODELS))
        errors = ",".join(f"{m}:http_404" for m in llm_humanizer.DEFAULT_MODELS)
        self.assertEqual(lines, [f"llm_request endpoint=humanize model=none attempts=4 errors={errors}"])

    async def test_time_budget_exhausted_stops_the_chain(self):
        # Every model hangs. Scaled-down budget: 0.6s total, 0.25s per model, no attempt under 0.15s.
        fake = FakeOpenRouter({m: {"status": 200, "body": ok_body(), "delay": 5} for m in llm_humanizer.DEFAULT_MODELS})
        with mock.patch.multiple(
            llm_humanizer, TOTAL_BUDGET_SECONDS=0.6, PER_MODEL_TIMEOUT_SECONDS=0.25, MIN_ATTEMPT_SECONDS=0.15
        ):
            start = time.monotonic()
            result, ok, lines = await self._run(fake)
            elapsed = time.monotonic() - start
        self.assertIsNone(result)
        self.assertFalse(ok)
        self.assertLess(elapsed, 0.6 + 0.2, "the chain must not run past its time budget")
        # 0.25 + 0.25 = 0.5s used, 0.1s left < 0.15s minimum: models 3 and 4 are never called.
        self.assertEqual(fake.calls, [M1, M2])
        self.assertEqual(
            lines, [f"llm_request endpoint=humanize model=none attempts=2 errors={M1}:timeout,{M2}:timeout"]
        )
        # Each attempt got min(per-model timeout, remaining budget).
        self.assertTrue(all(t <= 0.25 for t in fake.timeouts))

    async def test_models_from_env(self):
        with mock.patch.dict(os.environ, {"HUMANIZE_MODELS": " a/one:free , b/two:free,a/one:free,, "}):
            self.assertEqual(llm_humanizer.get_models(), ["a/one:free", "b/two:free"])
        with mock.patch.dict(os.environ, {"HUMANIZE_MODELS": ""}):
            self.assertEqual(llm_humanizer.get_models(), list(llm_humanizer.DEFAULT_MODELS))


class HumanizeEndpointQuotaTests(unittest.TestCase):
    """End-to-end through FastAPI: when every model fails the use is refunded."""

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from app.main import app

        cls.client_cm = TestClient(app)
        cls.client = cls.client_cm.__enter__()  # runs startup: creates the tables in the temp DB

    @classmethod
    def tearDownClass(cls):
        cls.client_cm.__exit__(None, None, None)

    def setUp(self):
        with sqlite3.connect(os.environ["DB_PATH"]) as db:
            db.execute("DELETE FROM rate_limits")

    def _count(self, ip="testclient"):
        with sqlite3.connect(os.environ["DB_PATH"]) as db:
            row = db.execute("SELECT count FROM rate_limits WHERE ip = ?", (ip,)).fetchone()
        return row[0] if row else None

    def test_all_models_fail_returns_same_error_and_refunds_quota(self):
        from app.routers.analyze import HUMANIZE_FALLBACK

        fake = FakeOpenRouter({m: {"status": 404} for m in llm_humanizer.DEFAULT_MODELS})
        with patch_client(fake):
            res = self.client.post("/api/humanize", json={"text": USER_TEXT, "ui_language": "en"})
        self.assertEqual(res.status_code, 503)
        self.assertEqual(res.json(), {"error": HUMANIZE_FALLBACK["en"]["error"]})
        self.assertEqual(self._count(), 0, "the failed request must not consume a use")
        self.assertEqual(res.headers["X-RateLimit-Remaining"], "3")

    def test_success_keeps_the_use(self):
        fake = FakeOpenRouter({M1: {"status": 200, "body": ok_body()}})
        with patch_client(fake):
            res = self.client.post("/api/humanize", json={"text": USER_TEXT, "ui_language": "en"})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(self._count(), 1)
        self.assertEqual(res.headers["X-RateLimit-Remaining"], "2")

    def test_refund_never_goes_below_zero(self):
        day = "2026-01-01"
        with sqlite3.connect(os.environ["DB_PATH"]) as db:
            db.execute("INSERT INTO rate_limits (ip, date, count) VALUES ('1.2.3.4', ?, 0)", (day,))
        remaining = asyncio.run(rate_limiter.refund_rate_limit("1.2.3.4", day=day))
        self.assertEqual(remaining, 3)
        self.assertEqual(self._count("1.2.3.4"), 0)


if __name__ == "__main__":
    unittest.main()
