"""The OpenAI-compatible client, against simulated Groq and Gemini responses."""

import asyncio
import json

import httpx
import pytest

from autodidact.envs.session import ToolSession
from autodidact.llm.base import LLMError, QuotaExhausted
from autodidact.llm.cache import ResponseCache
from autodidact.llm.openai_compat import OpenAICompatLLM, retry_after_seconds
from autodidact.llm.schema import to_openai_tools

TOOL_REPLY = {
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": None,
                "reasoning": "secret chain of thought",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_product", "arguments": '{"sku": "BEV-0001"}'},
                        "extra_content": {"google": {"thought_signature": "SIG123"}},
                    }
                ],
            }
        }
    ],
    "usage": {"prompt_tokens": 1200, "completion_tokens": 30},
}


def make_client(handler, *, cache=None, rpm=0, **kw):
    sleeps: list[float] = []

    async def fake_sleep(s):
        sleeps.append(s)

    llm = OpenAICompatLLM(
        provider="test",
        base_url="https://example.test/v1",
        api_key="k-123",
        model="m",
        rpm=rpm,
        cache=cache,
        transport=httpx.MockTransport(handler),
        sleep=fake_sleep,
        **kw,
    )
    return llm, sleeps


def run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------- schemas


def test_mcp_schemas_are_reduced_to_a_portable_subset():
    async def go():
        async with ToolSession(1) as s:
            return to_openai_tools(await s.tool_specs())

    tools = run(go())
    text = json.dumps(tools)
    for banned in ("$ref", "$defs", "anyOf", '"title"', '"default"'):
        assert banned not in text
    by_name = {t["function"]["name"]: t["function"] for t in tools}
    lines = by_name["create_purchase_order"]["parameters"]["properties"]["lines"]
    assert set(lines["items"]["properties"]) == {"sku", "qty"}
    page = by_name["list_products"]["parameters"]["properties"]["page_size"]
    assert page == {"type": "integer", "description": "Default: 10."}
    assert by_name["list_products"]["parameters"]["properties"]["cursor"]["type"] == "string"


# --------------------------------------------------------------------- parsing


def test_tool_calls_parsed_and_assistant_message_kept_with_signature():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers["authorization"]
        return httpx.Response(200, json=TOOL_REPLY)

    llm, _ = make_client(handler, reasoning_effort="low")
    resp = run(llm.chat([{"role": "user", "content": "hi"}], [{"type": "function"}]))
    assert resp.tool_calls[0].name == "get_product"
    assert resp.tool_calls[0].arguments == {"sku": "BEV-0001"}
    tc = resp.message["tool_calls"][0]
    assert tc["extra_content"]["google"]["thought_signature"] == "SIG123"
    assert "reasoning" not in resp.message
    assert resp.usage.prompt_tokens == 1200 and resp.usage.requests == 1
    assert seen["auth"] == "Bearer k-123"
    assert seen["body"]["reasoning_effort"] == "low"
    assert seen["body"]["tool_choice"] == "auto"
    assert "temperature" not in seen["body"]  # provider default unless configured


def test_missing_ids_and_bad_json_arguments():
    reply = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {"type": "function", "function": {"name": "a", "arguments": "{not json"}},
                        {"type": "function", "function": {"name": "b", "arguments": ""}},
                    ],
                }
            }
        ]
    }
    llm, _ = make_client(lambda r: httpx.Response(200, json=reply))
    resp = run(llm.chat([], []))
    a, b = resp.tool_calls
    assert a.id == "call_0" and resp.message["tool_calls"][0]["id"] == "call_0"
    assert a.parse_error and a.arguments == {}
    assert b.arguments == {} and b.parse_error is None


# ----------------------------------------------------------------- rate limits


def test_short_rate_limit_waits_then_retries():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, headers={"retry-after": "3"}, text="Rate limit (RPM)")
        return httpx.Response(200, json=TOOL_REPLY)

    llm, sleeps = make_client(handler)
    run(llm.chat([], []))
    assert len(calls) == 2 and 3.0 in sleeps


def test_groq_per_minute_limit_retries_after_parsed_delay():
    body = (
        '{"error":{"message":"Rate limit reached for model `openai/gpt-oss-120b` on tokens '
        'per minute (TPM): Limit 8000, Used 7900. Please try again in 2.5s."}}'
    )
    calls = []

    def handler(request):
        calls.append(1)
        return (
            httpx.Response(429, text=body)
            if len(calls) == 1
            else httpx.Response(200, json=TOOL_REPLY)
        )

    llm, sleeps = make_client(handler)
    run(llm.chat([], []))
    assert sleeps == [2.5]


@pytest.mark.parametrize(
    "body",
    [
        # Groq daily token limit
        '{"error":{"message":"Rate limit reached for model `openai/gpt-oss-120b` on tokens per day '
        '(TPD): Limit 200000, Used 199500. Please try again in 7m12.5s."}}',
        # Gemini free-tier daily request quota
        '{"error":{"code":429,"status":"RESOURCE_EXHAUSTED","details":[{"violations":[{"quotaId":'
        '"GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]},{"retryDelay":"39s"}]}}',
    ],
)
def test_daily_limits_stop_immediately(body):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(429, text=body)

    llm, sleeps = make_client(handler)
    with pytest.raises(QuotaExhausted):
        run(llm.chat([], []))
    assert len(calls) == 1 and sleeps == []


def test_retry_after_parsing():
    def resp(text, headers=None):
        return httpx.Response(429, text=text, headers=headers or {})

    assert retry_after_seconds(resp("", {"retry-after": "12"})) == 12
    assert retry_after_seconds(resp('{"retryDelay": "39s"}')) == 39
    assert retry_after_seconds(resp("Please try again in 1m2.5s.")) == 62.5
    assert retry_after_seconds(resp("Please try again in 1h2m3s")) == 3723
    assert retry_after_seconds(resp("Please try again in 850ms")) == 0.85
    assert retry_after_seconds(resp("nothing")) is None


def test_client_errors_fail_fast_and_server_errors_retry():
    llm, _ = make_client(lambda r: httpx.Response(400, text="bad schema"))
    with pytest.raises(LLMError, match="HTTP 400"):
        run(llm.chat([], []))

    calls = []

    def flaky(request):
        calls.append(1)
        return (
            httpx.Response(503, text="busy")
            if len(calls) < 3
            else httpx.Response(200, json=TOOL_REPLY)
        )

    llm, _ = make_client(flaky)
    run(llm.chat([], []))
    assert len(calls) == 3

    llm, _ = make_client(lambda r: httpx.Response(500, text="down"), max_retries=2)
    with pytest.raises(LLMError, match="gave up after 3 attempts"):
        run(llm.chat([], []))


def test_pacing_spaces_requests():
    llm, sleeps = make_client(lambda r: httpx.Response(200, json=TOOL_REPLY), rpm=6)

    async def two():
        await llm.chat([{"role": "user", "content": "a"}], [])
        await llm.chat([{"role": "user", "content": "b"}], [])

    run(two())
    assert sleeps and 9.5 < max(sleeps) <= 10.0  # 6 per minute = one every 10 s


# ----------------------------------------------------------------------- cache


def test_cache_replays_identical_requests_but_not_new_samples(tmp_path):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(200, json=TOOL_REPLY)

    cache = ResponseCache(tmp_path / "c.sqlite3")
    llm, _ = make_client(handler, cache=cache)
    msgs = [{"role": "user", "content": "same"}]
    first = run(llm.chat(msgs, []))
    again = run(llm.chat(msgs, []))
    assert len(calls) == 1 and not first.cached and again.cached
    assert again.usage.requests == 0 and again.message == first.message
    run(llm.chat(msgs, [], sample=1))
    assert len(calls) == 2 and len(cache) == 2


def test_a_request_can_override_reasoning_effort():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=TOOL_REPLY)

    llm, _ = make_client(handler, reasoning_effort="low")
    run(llm.chat([{"role": "user", "content": "a"}], []))
    run(llm.chat([{"role": "user", "content": "b"}], [], effort="medium"))
    assert [b["reasoning_effort"] for b in bodies] == ["low", "medium"]


def test_persistent_overload_raises_a_clean_stop_after_gentle_backoff():
    from autodidact.llm.base import ProviderUnavailable, StopRun

    overloaded = '{"error": {"code": 503, "message": "high demand", "status": "UNAVAILABLE"}}'
    llm, sleeps = make_client(lambda r: httpx.Response(503, text=overloaded))
    with pytest.raises(ProviderUnavailable) as info:
        run(llm.chat([], []))
    assert isinstance(info.value, StopRun)
    assert sleeps == [5, 10, 20, 40, 60, 60, 60, 60]  # about 5 minutes, none after the last try
