"""Command line: doctor (with a simulated provider), eval without keys, report, show."""

import asyncio
import json

import httpx
from test_eval_runner import STOCK, StockPolicy

from autodidact import cli
from autodidact.eval.runner import evaluate
from autodidact.llm import providers
from autodidact.llm.openai_compat import OpenAICompatLLM


def fake_provider(monkeypatch, handler):
    """Make ``make_llm`` talk to an in-memory HTTP handler instead of the internet."""
    real = providers.make_llm

    def patched(provider=None, model=None, **kw):
        llm = real(provider, model, **kw)
        return OpenAICompatLLM(
            provider=llm.provider,
            base_url="https://x.test/v1",
            api_key="k",
            model=llm.model,
            rpm=0,
            transport=httpx.MockTransport(handler),
        )

    monkeypatch.setattr(cli, "make_llm", patched)


def test_doctor_runs_a_two_turn_tool_round_trip(monkeypatch, capsys):
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test-key-123456")
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if len(bodies) == 1:
            msg = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "t1",
                        "type": "function",
                        "function": {"name": "add", "arguments": '{"a": 17, "b": 25}'},
                        "extra_content": {"google": {"thought_signature": "SIG"}},
                    }
                ],
            }
        else:
            msg = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "t2",
                        "type": "function",
                        "function": {"name": "finish", "arguments": '{"answer": "42"}'},
                    }
                ],
            }
        return httpx.Response(
            200,
            json={
                "choices": [{"message": msg}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 5},
            },
        )

    fake_provider(monkeypatch, handler)
    assert cli.main(["doctor", "--provider", "gemini"]) == 0
    out = capsys.readouterr().out
    assert "multi-turn tool calling works" in out and "AIza…3456" in out
    assert "AIza-test-key-123456" not in out  # the key itself is never printed
    sent_back = bodies[1]["messages"][2]
    assert sent_back["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == "SIG"
    assert bodies[1]["messages"][3] == {"role": "tool", "tool_call_id": "t1", "content": "42"}


def test_doctor_without_keys_explains_what_to_do(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert cli.main(["doctor"]) == 1
    out = capsys.readouterr().out
    assert "set GEMINI_API_KEY" in out and "aistudio.google.com" in out


def test_eval_without_key_fails_cleanly(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert cli.main(["eval", "--provider", "groq", "--per-template", "1"]) == 1
    assert "GROQ_API_KEY is not set" in capsys.readouterr().out


def test_dotenv_is_loaded(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    (tmp_path / ".env").write_text('# comment\nGROQ_MODEL="openai/gpt-oss-20b"\n')
    cli.load_dotenv()
    assert providers.resolve("groq")[2] == "openai/gpt-oss-20b"


def test_report_and_show(tmp_path, capsys):
    run_dir = tmp_path / "run"
    asyncio.run(evaluate(StockPolicy(), STOCK[:2], run_dir, condition="docs-only"))
    assert cli.main(["report", str(run_dir)]) == 0
    assert "PASS RATE" in capsys.readouterr().out
    assert cli.main(["show", str(run_dir), STOCK[0].id]) == 0
    out = capsys.readouterr().out
    assert "get_product" in out and "PASS" in out
