"""StockRoom runs as a normal MCP server process that any MCP client can use."""

import asyncio
import json
import sys

from mcp import Client, StdioServerParameters


def test_stdio_server_round_trip():
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "autodidact.envs.inventory.server", "--seed", "1"]
    )

    async def go():
        async with Client(params) as client:
            tools = await client.list_tools()
            assert len(tools.tools) == 23
            result = await client.call_tool("list_categories", {})
            names = {c["name"] for c in json.loads(result.content[0].text)["items"]}
            assert "Dairy" in names

    asyncio.run(go())


def test_stdio_server_refuses_hidden_test_world():
    import subprocess

    from autodidact.envs.inventory.world import TEST_WORLD_SEED

    proc = subprocess.run(
        [sys.executable, "-m", "autodidact.envs.inventory.server", "--seed", str(TEST_WORLD_SEED)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode != 0
    assert "reserved for hidden evaluation" in proc.stderr
