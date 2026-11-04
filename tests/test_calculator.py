"""The agent's calculator: exact, money-friendly, and impossible to abuse."""

import asyncio

import pytest
from conftest import ScriptedLLM

from autodidact.agent.calculator import calculate
from autodidact.agent.loop import run_episode
from autodidact.envs.session import ToolSession


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("1800*104 + 48350*98", "4925500"),
        ("11455650/100", "114556.5"),
        ("0.1 + 0.2", "0.3"),  # exact, unlike floating point
        ("round(2.675, 2)", "2.68"),  # half-up, like money; Python's round gives 2.67
        ("round(46600*112/100)", "52192"),
        ("sum(1, 2, 3) / 3", "2"),
        ("max(3, 9) - abs(-2)", "7"),
        ("-(4 - 10) % 4", "2"),
        ("17 // 5", "3"),
        ("₹193.50 * 3", "580.5"),
    ],
)
def test_arithmetic(expression, expected):
    assert calculate(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo hi')",
        "open('x')",
        "2 ** 999999",
        "().__class__",
        "[1, 2]",
        "x + 1",
        "lambda: 1",
        "1/0",
        "9" * 100,
        "1" * 3000,
        "round(1.5, 0.5)",
        "1,24,350",
        "",
    ],
)
def test_refuses_anything_but_arithmetic(expression):
    assert calculate(expression).startswith("ERROR:")


def test_agent_can_use_the_calculator():
    llm = ScriptedLLM(
        [
            {"calls": [("calculate", {"expression": "(19350 * 3) / 100"})]},
            lambda msgs: {"calls": [("finish", {"answer": msgs[-1]["content"]})]},
        ]
    )

    async def go():
        async with ToolSession(1) as s:
            return await run_episode(llm, s, "What is 3 x Rs 193.50?")

    episode = asyncio.run(go())
    assert episode.answer == "580.5"
    assert episode.turns[0].calls[0].ok and episode.tool_errors == 0
