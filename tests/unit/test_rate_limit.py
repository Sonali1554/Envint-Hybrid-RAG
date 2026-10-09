"""Unit test: the LLM client surfaces a rate-limit (HTTP 429) as a typed error."""

import pytest

from app.llm.client import LLMRateLimitError, OllamaBackend


class _FakeResponse:
    def __init__(self, status_code):
        self.status_code = status_code

    def raise_for_status(self):
        pass

    def json(self):
        return {}


class _FakeRequests:
    """Stand-in for the `requests` module that always returns HTTP 429."""

    def get(self, *a, **k):
        return _FakeResponse(200)

    def post(self, *a, **k):
        return _FakeResponse(429)


def test_ollama_429_raises_rate_limit_error():
    backend = OllamaBackend()
    backend._requests = _FakeRequests()  # inject the fake transport
    with pytest.raises(LLMRateLimitError):
        backend.generate("any query", [])
